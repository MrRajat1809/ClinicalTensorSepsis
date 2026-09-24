"""Normalize units and screen measurements, preserving explicit rejection reasons."""

import hashlib
import json
import math
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("dataset_config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DATASET_VERSION = CONFIG["dataset_version"]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "work"
METRICS_DIR = BASE_DIR / "outputs" / "mimiciv"
REGISTRY_FILE = BASE_DIR / "src" / "common" / "feature_registry.json"
RULES_FILE = BASE_DIR / "src" / "common" / "measurement_rules.json"
SCHEMA_VERSION = "2.0.0"


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized_unit(value):
    return value.strip().lower().replace("µ", "u").replace("μ", "u")


def configure_rules(connection, registry, rules):
    units = {entry["name"]: entry["canonical_unit"] for entry in registry["temporal_features"]}
    units.update(registry["evidence_features"])
    mapping = []
    seen = set()
    for source, features in registry["mimiciv_sources"].items():
        for feature, itemids in features.items():
            unit = units[feature]
            limits = rules["bounds"].get(feature)
            if unit is not None and (
                limits is None or len(limits) != 2
                or not all(math.isfinite(value) for value in limits)
                or limits[0] >= limits[1]
            ):
                raise ValueError(f"Missing or invalid bounds: {feature}")
            for itemid in itemids:
                key = (source, itemid)
                if key in seen:
                    raise ValueError(f"Duplicate source mapping: {key}")
                seen.add(key)
                mapping.append((
                    source, itemid, feature, unit,
                    limits[0] if limits else None, limits[1] if limits else None,
                ))
    connection.execute("""
        CREATE TEMP TABLE feature_map (
            source_table VARCHAR, itemid INTEGER, feature VARCHAR,
            canonical_unit VARCHAR, bound_min DOUBLE, bound_max DOUBLE
        )
    """)
    connection.executemany("INSERT INTO feature_map VALUES (?, ?, ?, ?, ?, ?)", mapping)
    conversions = {}

    def add(feature, unit, factor=1.0, offset=0.0, weight_operation="none"):
        key = (feature, normalized_unit(unit))
        if key in conversions:
            raise ValueError(f"Duplicate conversion rule: {key}")
        if weight_operation not in ("none", "divide", "multiply"):
            raise ValueError(f"Unsupported weight operation: {weight_operation}")
        if not math.isfinite(factor) or factor <= 0 or not math.isfinite(offset):
            raise ValueError(f"Invalid conversion: {key}")
        conversions[key] = (feature, key[1], factor, offset, weight_operation)

    for feature in {row[2] for row in mapping if row[3] is not None}:
        for unit in rules["unit_aliases"][units[feature]]:
            add(feature, unit)
    for conversion in rules["extra_conversions"]:
        for feature in conversion["features"]:
            for unit in conversion["units"]:
                add(feature, unit, conversion["factor"], conversion.get("offset", 0),
                    conversion.get("weight_operation", "none"))
    connection.execute("""
        CREATE TEMP TABLE conversions (
            feature VARCHAR, normalized_unit VARCHAR, factor DOUBLE,
            unit_offset DOUBLE, weight_operation VARCHAR
        )
    """)
    connection.executemany("INSERT INTO conversions VALUES (?, ?, ?, ?, ?)", list(conversions.values()))


def validate_inputs(connection):
    connection.execute("""
        SELECT source_db, source_table, subject_id, hadm_id, stay_id, itemid,
            feature, canonical_unit, raw_value, raw_valuenum, raw_unit, event_time,
            event_end_time, patient_weight_kg, source_status, interval_status,
            linkage_status, is_rewritten_or_cancelled, specimen_type,
            specimen_type_conflict, arterial_specimen_required,
            within_icu_interval, within_hospital_interval,
            after_recorded_hospital_death
        FROM raw_events LIMIT 0
    """)
    invalid_windows = connection.execute("""
        SELECT COUNT(*) FROM windows
        WHERE subject_id IS NULL OR hadm_id IS NULL OR stay_id IS NULL
            OR icu_intime IS NULL OR icu_outtime IS NULL OR icu_intime >= icu_outtime
            OR hospital_admittime IS NULL OR hospital_dischtime IS NULL
            OR hospital_admittime >= hospital_dischtime
            OR window_start IS NULL OR window_end IS NULL OR window_start >= window_end
    """).fetchone()[0]
    duplicate_windows = connection.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT stay_id) FROM windows
    """).fetchone()[0]
    if invalid_windows or duplicate_windows:
        raise ValueError("Invalid or duplicate extraction windows; rerun stage 04")
    invalid_rows = connection.execute("""
        SELECT COUNT(*) FROM raw_events AS raw
        LEFT JOIN feature_map AS mapping
            ON raw.source_table = mapping.source_table AND raw.itemid = mapping.itemid
        LEFT JOIN windows AS extraction_window
            ON raw.stay_id = extraction_window.stay_id AND raw.subject_id = extraction_window.subject_id
            AND raw.hadm_id = extraction_window.hadm_id
        WHERE mapping.itemid IS NULL OR extraction_window.stay_id IS NULL
            OR raw.source_db IS DISTINCT FROM 'MIMIC-IV'
            OR raw.feature IS DISTINCT FROM mapping.feature
            OR raw.canonical_unit IS DISTINCT FROM mapping.canonical_unit
            OR raw.arterial_specimen_required IS DISTINCT FROM (raw.feature IN ('pao2', 'paco2'))
    """).fetchone()[0]
    if invalid_rows:
        raise ValueError(f"{invalid_rows:,} rows disagree with the registry or extraction windows")


def cleaning_query(weight_bounds):
    weight_min, weight_max = map(float, weight_bounds)
    if not (math.isfinite(weight_min) and math.isfinite(weight_max) and 0 < weight_min < weight_max):
        raise ValueError("Invalid weight bounds")
    return f"""
        WITH prepared AS (
            SELECT raw.*,
                mapping.bound_min, mapping.bound_max,
                LOWER(REPLACE(REPLACE(TRIM(COALESCE(raw.raw_unit, '')), 'µ', 'u'), 'μ', 'u'))
                    AS normalized_unit,
                raw.canonical_unit IS NULL AS is_evidence,
                raw.source_table IN ('inputevents', 'procedureevents') AS is_interval,
                raw.event_time BETWEEN extraction_window.icu_intime AND extraction_window.icu_outtime AS in_icu,
                raw.event_time BETWEEN extraction_window.hospital_admittime AND extraction_window.hospital_dischtime AS in_hospital,
                raw.event_time BETWEEN extraction_window.window_start AND extraction_window.window_end AS in_envelope,
                raw.event_time > extraction_window.hospital_deathtime AS after_death,
                CASE WHEN raw.source_table IN ('inputevents', 'procedureevents')
                    THEN GREATEST(raw.event_time, extraction_window.icu_intime, extraction_window.window_start)
                    ELSE raw.event_time END AS effective_start_time,
                CASE WHEN raw.source_table IN ('inputevents', 'procedureevents')
                    THEN LEAST(raw.event_end_time, extraction_window.icu_outtime, extraction_window.window_end,
                        extraction_window.hospital_deathtime)
                    ELSE NULL END AS effective_end_time
            FROM raw_events AS raw
            JOIN feature_map AS mapping
                ON raw.source_table = mapping.source_table AND raw.itemid = mapping.itemid
            JOIN windows AS extraction_window ON raw.stay_id = extraction_window.stay_id
        ), matched AS (
            SELECT prepared.*, conversions.factor, conversions.unit_offset, conversions.weight_operation,
                CASE
                    WHEN is_evidence THEN 'not_numeric_evidence'
                    WHEN feature = 'fio2' AND itemid IN (223835, 50816)
                        AND normalized_unit IN ('', '%', 'percent')
                        AND raw_valuenum BETWEEN 0.2 AND 1 THEN 'mimic_fio2_fraction_recording'
                    WHEN feature = 'fio2' AND itemid IN (223835, 50816)
                        AND normalized_unit IN ('', '%', 'percent')
                        AND raw_valuenum BETWEEN 20 AND 100 THEN 'mimic_fio2_percent_recording'
                    WHEN feature = 'fio2' AND normalized_unit IN ('', '%', 'percent')
                        THEN 'ambiguous_fio2'
                    WHEN normalized_unit = '' AND feature IN
                        ('gcs_eye', 'gcs_verbal', 'gcs_motor', 'inr', 'ph') THEN 'intrinsic_item_unit'
                    WHEN feature = 'mchc' AND itemid = 51249 AND normalized_unit = '%'
                        THEN 'mimic_mchc_percent_label'
                    WHEN feature = 'rbc' AND itemid = 51279 AND normalized_unit = 'm/ul'
                        THEN 'mimic_rbc_million_per_ul'
                    WHEN feature = 'norepinephrine' AND normalized_unit = 'mg/kg/min'
                        AND patient_weight_kg = 1 THEN 'ambiguous_norepinephrine_unit'
                    WHEN conversions.factor IS NULL THEN 'unsupported_unit'
                    WHEN conversions.weight_operation <> 'none' AND NOT COALESCE(
                        ISFINITE(patient_weight_kg)
                        AND patient_weight_kg BETWEEN {weight_min} AND {weight_max}, FALSE)
                        THEN 'missing_or_invalid_weight'
                    WHEN conversions.weight_operation <> 'none' THEN 'documented_weight_conversion'
                    WHEN conversions.factor = 1 AND conversions.unit_offset = 0 THEN 'canonical_or_alias'
                    ELSE 'linear_unit_conversion'
                END AS unit_rule
            FROM prepared LEFT JOIN conversions USING (feature, normalized_unit)
        ), converted AS (
            SELECT * EXCLUDE (factor, unit_offset, weight_operation),
                CASE
                    WHEN unit_rule IN ('mimic_fio2_fraction_recording', 'intrinsic_item_unit',
                        'mimic_mchc_percent_label', 'mimic_rbc_million_per_ul') THEN raw_valuenum
                    WHEN unit_rule = 'mimic_fio2_percent_recording' THEN raw_valuenum / 100.0
                    WHEN unit_rule IN ('canonical_or_alias', 'linear_unit_conversion',
                        'documented_weight_conversion') THEN
                        (raw_valuenum * factor + unit_offset) *
                        CASE weight_operation WHEN 'divide' THEN 1.0 / patient_weight_kg
                            WHEN 'multiply' THEN patient_weight_kg ELSE 1.0 END
                    ELSE NULL
                END AS converted_value,
                CASE
                    WHEN COALESCE(is_rewritten_or_cancelled, FALSE)
                        OR REGEXP_MATCHES(LOWER(COALESCE(source_status, '')), 'rewritten|cancel')
                        THEN 'rewritten_or_cancelled'
                    WHEN event_time IS NULL THEN 'missing_event_time'
                    WHEN is_interval AND (event_end_time IS NULL OR event_end_time <= event_time
                        OR interval_status IS DISTINCT FROM 'valid_interval') THEN 'invalid_interval'
                    WHEN COALESCE(after_death, FALSE) THEN 'after_recorded_death'
                    WHEN is_interval AND effective_start_time >= effective_end_time THEN 'no_valid_interval_overlap'
                    WHEN NOT is_interval AND NOT COALESCE(in_envelope, FALSE) THEN 'outside_extraction_window'
                    WHEN NOT is_interval AND source_table = 'labevents'
                        AND NOT COALESCE(in_icu OR in_hospital, FALSE) THEN 'outside_care_intervals'
                    WHEN NOT is_interval AND source_table <> 'labevents'
                        AND NOT COALESCE(in_icu, FALSE) THEN 'outside_icu_interval'
                    WHEN source_table = 'inputevents'
                        AND LOWER(TRIM(COALESCE(source_status, ''))) = 'bolus' THEN 'bolus_not_continuous_rate'
                    ELSE 'accepted'
                END AS record_qc_status
            FROM matched
        ), assessed AS (
            SELECT *,
                CASE
                    WHEN is_evidence AND (source_table = 'procedureevents'
                        OR NULLIF(TRIM(raw_value), '') IS NOT NULL) THEN 'evidence_only'
                    WHEN is_evidence THEN 'missing_evidence'
                    WHEN feature = 'gcs_verbal' AND REGEXP_MATCHES(
                        LOWER(COALESCE(raw_value, '')), 'ett|trach|intubat|unable')
                        THEN 'gcs_verbal_unassessable'
                    WHEN REGEXP_MATCHES(UPPER(TRIM(COALESCE(raw_value, ''))),
                        '^[<>≤≥]|^(LESS THAN|GREATER THAN)') THEN 'censored_numeric'
                    WHEN raw_valuenum IS NULL THEN 'missing_numeric'
                    WHEN NOT ISFINITE(raw_valuenum) THEN 'nonfinite_numeric'
                    WHEN unit_rule IN ('unsupported_unit', 'ambiguous_fio2',
                        'ambiguous_norepinephrine_unit', 'missing_or_invalid_weight') THEN unit_rule
                    WHEN converted_value IS NULL OR NOT ISFINITE(converted_value) THEN 'nonfinite_conversion'
                    WHEN arterial_specimen_required AND COALESCE(specimen_type_conflict, FALSE)
                        THEN 'conflicting_specimen'
                    WHEN arterial_specimen_required AND NULLIF(TRIM(specimen_type), '') IS NULL
                        THEN 'unknown_arterial_specimen'
                    WHEN arterial_specimen_required AND UPPER(TRIM(specimen_type)) NOT IN
                        ('ART.', 'ART', 'ARTERIAL') THEN 'nonarterial_specimen'
                    WHEN feature IN ('gcs_eye', 'gcs_verbal', 'gcs_motor')
                        AND converted_value <> FLOOR(converted_value) THEN 'noninteger_score'
                    WHEN converted_value < bound_min OR converted_value > bound_max THEN 'outside_bounds'
                    ELSE 'accepted'
                END AS value_qc_status
            FROM converted
        )
        SELECT * EXCLUDE (is_evidence, is_interval, in_icu, in_hospital, in_envelope, after_death),
            CASE WHEN record_qc_status <> 'accepted' THEN record_qc_status
                ELSE value_qc_status END AS qc_status,
            record_qc_status = 'accepted' AND value_qc_status = 'accepted' AS numeric_usable,
            record_qc_status = 'accepted' AND value_qc_status = 'evidence_only' AS evidence_usable,
            CASE WHEN record_qc_status = 'accepted' AND value_qc_status = 'accepted'
                THEN converted_value ELSE NULL END AS valuenum,
            canonical_unit AS valueuom,
            unit_rule IN ('mimic_fio2_fraction_recording', 'mimic_fio2_percent_recording',
                'intrinsic_item_unit', 'mimic_mchc_percent_label', 'mimic_rbc_million_per_ul')
                AS source_unit_exception,
            is_interval AND record_qc_status = 'accepted' AND (
                event_time <> effective_start_time OR event_end_time <> effective_end_time)
                AS interval_clipped,
            {sql_string(SCHEMA_VERSION)} AS cleaning_schema_version
        FROM assessed
    """


def clean_temporal_data(processed_dir=PROCESSED_DIR, metrics_dir=METRICS_DIR,
                        registry_file=REGISTRY_FILE, rules_file=RULES_FILE):
    started = time.perf_counter()
    processed_dir, metrics_dir = Path(processed_dir), Path(metrics_dir)
    raw_file = processed_dir / "events_raw.parquet"
    window_file = processed_dir / "extraction_windows.parquet"
    extraction_report = metrics_dir / "04_extraction.json"
    for path in (raw_file, window_file, extraction_report, Path(registry_file), Path(rules_file)):
        if not path.is_file():
            raise FileNotFoundError(f"Required cleaning input not found: {path}")
    registry, rules = read_json(registry_file), read_json(rules_file)
    registry["mimiciv_sources"] = CONFIG["source_mapping"]["mimiciv_sources"]
    if read_json(extraction_report)["dataset_config_sha256"] != hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest():
        raise ValueError("Dataset source mapping changed since extraction; rerun stage 04")
    if read_json(extraction_report)["registry_sha256"] != hashlib.sha256(Path(registry_file).read_bytes()).hexdigest():
        raise ValueError("Feature registry changed since extraction; rerun stage 04")
    if registry["schema_version"] != "1.0.0" or rules["schema_version"] != "1.0.0":
        raise ValueError("Unsupported registry/rules schema")
    if any(
        rules["bounds"][name][1] != upper for name, upper in rules["revised_upper_limits"].items()
    ):
        raise ValueError("Extreme-value quarantine rules must match the declared bounds")
    with tempfile.TemporaryDirectory(prefix=".clean_", dir=processed_dir) as temporary:
        staging = Path(temporary)
        output_file = staging / "events_clean.parquet"
        with duckdb.connect() as connection:
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(f"SET temp_directory = {sql_string((staging / 'spill').as_posix())}")
            connection.execute(f"CREATE VIEW raw_events AS SELECT * FROM read_parquet({sql_string(raw_file.as_posix())})")
            connection.execute(f"CREATE VIEW windows AS SELECT * FROM read_parquet({sql_string(window_file.as_posix())})")
            configure_rules(connection, registry, rules)
            validate_inputs(connection)
            raw_count = connection.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
            connection.execute(f"""
                COPY ({cleaning_query(rules["weight_kg_bounds"])})
                TO {sql_string(output_file.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            connection.execute(f"CREATE VIEW cleaned AS SELECT * FROM read_parquet({sql_string(output_file.as_posix())})")
            summary = dict(zip((
                "raw_rows", "stays_with_rows", "accepted_numeric_rows", "usable_evidence_rows",
                "withheld_rows", "converted_numeric_rows", "source_unit_exception_rows",
                "clipped_valid_intervals", "accepted_inferred_lab_links",
                "accepted_rows_outside_hospital_interval",
            ), connection.execute("""
                SELECT COUNT(*), COUNT(DISTINCT stay_id),
                    COUNT(*) FILTER (WHERE numeric_usable),
                    COUNT(*) FILTER (WHERE evidence_usable),
                    COUNT(*) FILTER (WHERE NOT numeric_usable AND NOT evidence_usable),
                    COUNT(*) FILTER (WHERE numeric_usable AND valuenum IS DISTINCT FROM raw_valuenum),
                    COUNT(*) FILTER (WHERE source_unit_exception),
                    COUNT(*) FILTER (WHERE interval_clipped),
                    COUNT(*) FILTER (WHERE numeric_usable AND linkage_status = 'inferred_hospital_time'),
                    COUNT(*) FILTER (WHERE (numeric_usable OR evidence_usable) AND within_hospital_interval = FALSE)
                FROM cleaned
            """).fetchone()))
            if summary["raw_rows"] != raw_count:
                raise AssertionError("Cleaning changed the number of rows")
            invalid_output = connection.execute("""
                SELECT COUNT(*) FROM cleaned WHERE
                    (valuenum IS NOT NULL) IS DISTINCT FROM numeric_usable
                    OR (numeric_usable AND (NOT ISFINITE(valuenum)
                        OR valuenum < bound_min OR valuenum > bound_max))
                    OR (evidence_usable AND valuenum IS NOT NULL)
            """).fetchone()[0]
            if invalid_output:
                raise AssertionError(f"{invalid_output:,} cleaned rows violate QC invariants")
            statuses = dict(connection.execute("""
                SELECT qc_status, COUNT(*) FROM cleaned GROUP BY qc_status ORDER BY qc_status
            """).fetchall())
            value_statuses = dict(connection.execute("""
                SELECT value_qc_status, COUNT(*) FROM cleaned GROUP BY value_qc_status ORDER BY value_qc_status
            """).fetchall())
            accepted_features = {row[0] for row in connection.execute(
                "SELECT DISTINCT feature FROM cleaned WHERE numeric_usable"
            ).fetchall()}
            missing_features = [
                entry["name"] for entry in registry["temporal_features"]
                if not entry["derived"] and entry["name"] not in accepted_features
            ]
            profile_file = staging / "measurement_qc.csv"
            connection.execute(f"""
                COPY (
                    SELECT source_table, feature, itemid, raw_unit, canonical_unit, unit_rule,
                        record_qc_status, value_qc_status, qc_status, COUNT(*) AS rows,
                        COUNT(DISTINCT stay_id) AS stays,
                        COUNT(*) FILTER (WHERE numeric_usable) AS accepted_numeric_rows,
                        COUNT(*) FILTER (WHERE evidence_usable) AS usable_evidence_rows,
                        MIN(raw_valuenum) FILTER (WHERE ISFINITE(raw_valuenum)) AS raw_min,
                        MAX(raw_valuenum) FILTER (WHERE ISFINITE(raw_valuenum)) AS raw_max,
                        MIN(valuenum) AS accepted_min, MAX(valuenum) AS accepted_max
                    FROM cleaned
                    GROUP BY source_table, feature, itemid, raw_unit, canonical_unit, unit_rule,
                        record_qc_status, value_qc_status, qc_status
                    ORDER BY source_table, feature, itemid, raw_unit, qc_status
                ) TO {sql_string(profile_file.as_posix())} (HEADER, DELIMITER ',')
            """)
        report = {
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION, "duckdb_version": duckdb.__version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_version": registry["schema_version"], "rules_version": rules["schema_version"],
            "rules_revision": rules["rules_revision"],
            "registry_sha256": hashlib.sha256(Path(registry_file).read_bytes()).hexdigest(),
            "rules_sha256": hashlib.sha256(Path(rules_file).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input": {"path": str(raw_file), "size_bytes": raw_file.stat().st_size},
            "summary": summary, "qc_status_counts": statuses,
            "value_qc_status_counts": value_statuses,
            "direct_features_without_accepted_numeric_rows": missing_features,
            "pending_derived_features": [entry["name"] for entry in registry["temporal_features"] if entry["derived"]],
            "policy": {
                "accepted_numeric": "Use valuenum/numeric_usable; converted_value is diagnostic only",
                "raw_rows": "Preserved, including invalid, censored, nonarterial and text records",
                "bounds": rules["bounds_policy"],
                "unknown_units": "Quarantine; only explicitly enumerated unit conversions and source exceptions",
                "gcs": "ETT/tracheostomy/unassessable verbal responses are missing, never an observed score of 1 or imputed normal",
                "specimen": "PaO2/PaCO2 require explicit arterial labels without specimen conflicts",
                "points": "Labs: hospital OR ICU interval; ICU sources: ICU interval; no points after recorded death",
                "intervals": "Require valid endpoints; clip effective intervals to ICU, extraction envelope and recorded death",
                "hospital_disagreement": "Do not automatically exclude valid ICU evidence solely on inconsistent hospital timestamps",
                "inferred_lab_links": "Retained when within care intervals; linkage_status remains available for sensitivity exclusion",
                "censoring": "Inequality results are not exact measurements; preserve raw limit but withhold valuenum",
                "text_and_procedures": "Use evidence_usable and raw text/effective intervals; no fabricated numeric ventilation flag",
                "urine": "Recorded mL, not mL/hour; irrigation handling and interval coverage remain downstream",
                "weight": {"fallback": None, "allowed_kg_for_conversion": rules["weight_kg_bounds"]},
            },
        }
        report_file = staging / "05_cleaning.json"
        report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for artifact in (profile_file, report_file):
            artifact.replace(metrics_dir / artifact.name)
        output_file.replace(processed_dir / output_file.name)
    print(f"[05] Complete v{DATASET_VERSION}; {time.perf_counter() - started:.1f}s; report: {metrics_dir}/05_cleaning.json", flush=True)
    return report


if __name__ == "__main__":
    print(f"[05] Normalize units and screen measurements, preserving explicit rejection reasons. v{DATASET_VERSION}", flush=True)
    clean_temporal_data()
