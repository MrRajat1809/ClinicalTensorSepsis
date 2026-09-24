"""Extract source measurements and evidence across infection-candidate windows."""

import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("dataset_config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DATASET_VERSION = CONFIG["dataset_version"]
MIMIC_DIR = BASE_DIR / "data" / "raw" / "mimiciv" / "3.1"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "work"
METRICS_DIR = BASE_DIR / "outputs" / "mimiciv"
REGISTRY_FILE = BASE_DIR / "src" / "common" / "feature_registry.json"
SCHEMA_VERSION = "2.0.0"
SOURCE_MODULES = {
    "chartevents": "icu", "labevents": "hosp", "inputevents": "icu",
    "outputevents": "icu", "procedureevents": "icu",
}
OPTIONAL_FIELDS = {
    "event_end_time": "TIMESTAMP", "source_event_id": "BIGINT", "specimen_id": "BIGINT",
    "order_id": "BIGINT", "link_order_id": "BIGINT", "patient_weight_kg": "DOUBLE",
    "raw_amount": "DOUBLE", "raw_amount_unit": "VARCHAR",
    "source_status": "VARCHAR", "order_category": "VARCHAR", "source_warning": "VARCHAR",
}


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def load_registry(path):
    registry_bytes = Path(path).read_bytes()
    registry = json.loads(registry_bytes)
    registry["mimiciv_sources"] = CONFIG["source_mapping"]["mimiciv_sources"]
    features = registry["temporal_features"]
    names = [feature["name"] for feature in features]
    if len(names) != 50 or len(set(names)) != 50:
        raise ValueError("Expected 50 uniquely named temporal features")
    units = {feature["name"]: feature["canonical_unit"] for feature in features}
    units.update(registry["evidence_features"])
    rows = []
    mapped_features = set()
    for source_table in SOURCE_MODULES:
        seen = set()
        for feature, itemids in registry["mimiciv_sources"][source_table].items():
            if feature not in units:
                raise ValueError(f"Missing unit definition: {feature}")
            for itemid in itemids:
                if itemid in seen:
                    raise ValueError(f"Duplicate mapping: {source_table}/{itemid}")
                seen.add(itemid)
                rows.append((source_table, itemid, feature, units[feature], feature in ("pao2", "paco2")))
                mapped_features.add(feature)
    missing = [feature["name"] for feature in features if not feature["derived"] and feature["name"] not in mapped_features]
    if missing:
        raise ValueError(f"Missing extraction mappings: {missing}")
    return registry, rows, hashlib.sha256(registry_bytes).hexdigest()


def extraction_query(source_table, path):
    interval_source = source_table in ("inputevents", "procedureevents")
    lab_source = source_table == "labevents"
    timestamp = "events.starttime" if interval_source else "events.charttime"
    fields = {name: f"CAST(NULL AS {dtype})" for name, dtype in OPTIONAL_FIELDS.items()}
    types = {
        "subject_id": "BIGINT", "hadm_id": "BIGINT", "itemid": "INTEGER",
        "storetime": "TIMESTAMP",
    }
    if not lab_source:
        types["stay_id"] = "BIGINT"
    if interval_source:
        types.update({
            "starttime": "TIMESTAMP", "endtime": "TIMESTAMP", "orderid": "BIGINT",
            "linkorderid": "BIGINT", "patientweight": "DOUBLE",
            "statusdescription": "VARCHAR", "ordercategorydescription": "VARCHAR",
        })
        fields.update({
            "event_end_time": "events.endtime", "order_id": "events.orderid",
            "link_order_id": "events.linkorderid", "patient_weight_kg": "events.patientweight",
            "source_status": "events.statusdescription", "order_category": "events.ordercategorydescription",
        })
        interval_status = """CASE WHEN events.starttime IS NULL THEN 'missing_start'
            WHEN events.endtime IS NULL THEN 'missing_end'
            WHEN events.endtime <= events.starttime THEN 'nonpositive_interval'
            ELSE 'valid_interval' END"""
        time_filter = """(
            (events.starttime < windows.window_end AND events.endtime > windows.window_start
                AND events.endtime > events.starttime)
            OR (COALESCE(events.starttime, events.endtime) BETWEEN windows.window_start AND windows.window_end
                AND (events.starttime IS NULL OR events.endtime IS NULL OR events.endtime <= events.starttime))
        )"""
    else:
        types["charttime"] = "TIMESTAMP"
        interval_status = "'point'"
        time_filter = f"{timestamp} BETWEEN windows.window_start AND windows.window_end"

    if source_table == "inputevents":
        types.update({"rate": "DOUBLE", "rateuom": "VARCHAR", "amount": "DOUBLE", "amountuom": "VARCHAR"})
        raw_value, raw_valuenum, raw_unit = "CAST(events.rate AS VARCHAR)", "events.rate", "events.rateuom"
        fields.update({"raw_amount": "events.amount", "raw_amount_unit": "events.amountuom"})
    elif source_table in ("outputevents", "procedureevents"):
        types.update({"value": "DOUBLE", "valueuom": "VARCHAR"})
        raw_value, raw_valuenum, raw_unit = "CAST(events.value AS VARCHAR)", "events.value", "events.valueuom"
    else:
        types.update({"value": "VARCHAR", "valuenum": "DOUBLE", "valueuom": "VARCHAR"})
        raw_value, raw_valuenum, raw_unit = "events.value", "events.valuenum", "events.valueuom"
        if lab_source:
            types.update({"labevent_id": "BIGINT", "specimen_id": "BIGINT", "flag": "VARCHAR"})
            fields.update({"source_event_id": "events.labevent_id", "specimen_id": "events.specimen_id", "source_warning": "events.flag"})
        else:
            types["warning"] = "VARCHAR"
            fields["source_warning"] = "events.warning"

    if lab_source:
        linkage = """events.subject_id = windows.subject_id AND (
            events.hadm_id = windows.hadm_id OR (
                events.hadm_id IS NULL AND events.charttime BETWEEN
                    windows.hospital_admittime AND windows.hospital_dischtime
            ))"""
        link_status = "CASE WHEN events.hadm_id IS NULL THEN 'inferred_hospital_time' ELSE 'exact_admission' END"
    else:
        linkage = """events.stay_id = windows.stay_id AND events.subject_id = windows.subject_id
            AND events.hadm_id = windows.hadm_id"""
        link_status = "'exact_stay'"
    type_sql = ", ".join(f"{sql_string(name)}: {sql_string(dtype)}" for name, dtype in types.items())
    field_sql = ",\n".join(f"{expression} AS {name}" for name, expression in fields.items())
    return f"""
        SELECT 'MIMIC-IV' AS source_db, {sql_string(source_table)} AS source_table,
            windows.subject_id, windows.hadm_id, windows.stay_id,
            events.hadm_id AS source_hadm_id, events.itemid, mapping.feature,
            mapping.canonical_unit, mapping.arterial_specimen_required,
            {timestamp} AS event_time, events.storetime,
            {raw_value} AS raw_value, {raw_valuenum} AS raw_valuenum, {raw_unit} AS raw_unit,
            {field_sql},
            {interval_status} AS interval_status, {link_status} AS linkage_status,
            COALESCE(LOWER(TRIM({fields['source_status']})) IN
                ('rewritten', 'cancelled', 'canceled'), FALSE) AS is_rewritten_or_cancelled,
            {timestamp} BETWEEN windows.icu_intime AND windows.icu_outtime AS within_icu_interval,
            {timestamp} BETWEEN windows.hospital_admittime AND windows.hospital_dischtime
                AS within_hospital_interval,
            {timestamp} > windows.hospital_deathtime AS after_recorded_hospital_death
        FROM read_csv_auto({sql_string(path.as_posix())}, types={{{type_sql}}}) events
        INNER JOIN windows ON {linkage}
        INNER JOIN item_mapping mapping ON events.itemid = mapping.itemid
            AND mapping.source_table = {sql_string(source_table)}
        WHERE {time_filter}
    """


def extract_temporal_data(mimic_dir=MIMIC_DIR, processed_dir=PROCESSED_DIR,
                          metrics_dir=METRICS_DIR, registry_file=REGISTRY_FILE):
    started = time.perf_counter()
    mimic_dir, processed_dir, metrics_dir = map(Path, (mimic_dir, processed_dir, metrics_dir))
    registry, mapping_rows, registry_hash = load_registry(registry_file)
    cohort_file = processed_dir / "phenotypes.parquet"
    candidate_file = processed_dir / "infection_candidates.parquet"
    source_paths = {
        name: mimic_dir / module / f"{name}.csv.gz" for name, module in SOURCE_MODULES.items()
    }
    for path in (cohort_file, candidate_file, *source_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required extraction input not found: {path}")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".extract_", dir=processed_dir) as temporary:
        staging = Path(temporary)
        with duckdb.connect(":memory:") as connection:
            connection.execute("SET threads = 4")
            connection.execute(f"SET temp_directory = {sql_string((staging / 'duckdb_spill').as_posix())}")
            connection.execute(f"""
                CREATE TEMP TABLE cohort AS
                SELECT subject_id, hadm_id, stay_id, icu_intime, icu_outtime,
                    hospital_admittime, hospital_dischtime, hospital_deathtime
                FROM read_parquet({sql_string(cohort_file.as_posix())})
            """)
            count, subjects, stays, missing_keys = connection.execute("""
                SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT stay_id),
                    COUNT(*) FILTER (WHERE hadm_id IS NULL) FROM cohort
            """).fetchone()
            if count != subjects or count != stays or missing_keys:
                raise ValueError("Expected one non-null patient/admission/stay per cohort row")
            connection.execute(f"""
                CREATE TEMP TABLE candidates AS
                SELECT subject_id, hadm_id, stay_id, suspected_infection_time,
                    suspected_infection_time_upper_bound
                FROM read_parquet({sql_string(candidate_file.as_posix())})
            """)
            invalid_candidates = connection.execute("""
                SELECT COUNT(*) FROM candidates LEFT JOIN cohort USING (stay_id)
                WHERE cohort.stay_id IS NULL OR candidates.subject_id IS DISTINCT FROM cohort.subject_id
                   OR candidates.hadm_id IS DISTINCT FROM cohort.hadm_id
                   OR suspected_infection_time IS NULL OR suspected_infection_time_upper_bound IS NULL
                   OR suspected_infection_time_upper_bound < suspected_infection_time
            """).fetchone()[0]
            if invalid_candidates:
                raise ValueError(f"Invalid or unmatched infection candidates: {invalid_candidates}")
            connection.execute("""
                CREATE TEMP TABLE windows AS
                WITH limits AS (
                    SELECT stay_id,
                        MIN(suspected_infection_time) - INTERVAL 72 HOUR AS window_start,
                        MAX(suspected_infection_time_upper_bound) + INTERVAL 48 HOUR AS window_end,
                        COUNT(*) AS infection_candidate_count
                    FROM candidates GROUP BY stay_id
                )
                SELECT cohort.*, limits.* EXCLUDE (stay_id) FROM cohort INNER JOIN limits USING (stay_id)
            """)
            if connection.execute("SELECT COUNT(*) FROM windows").fetchone()[0] != count:
                raise ValueError("Every cohort stay must have at least one infection candidate")
            connection.execute("""
                CREATE TEMP TABLE item_mapping (
                    source_table VARCHAR, itemid INTEGER, feature VARCHAR,
                    canonical_unit VARCHAR, arterial_specimen_required BOOLEAN
                )
            """)
            connection.executemany("INSERT INTO item_mapping VALUES (?, ?, ?, ?, ?)", mapping_rows)
            parts = []
            for source_table, path in source_paths.items():
                print(f"    Scanning {source_table}...", flush=True)
                part = staging / f"{source_table}.parquet"
                query = extraction_query(source_table, path)
                connection.execute(f"COPY ({query}) TO {sql_string(part.as_posix())} (FORMAT PARQUET)")
                parts.append(sql_string(part.as_posix()))
            connection.execute(
                f"CREATE TEMP VIEW raw_events AS SELECT * FROM read_parquet([{', '.join(parts)}])"
            )
            connection.execute("""
                CREATE TEMP TABLE specimen_info AS
                SELECT subject_id, specimen_id,
                    CASE WHEN COUNT(DISTINCT NULLIF(UPPER(TRIM(raw_value)), '')) = 1
                         THEN MIN(NULLIF(UPPER(TRIM(raw_value)), '')) END AS specimen_type,
                    COUNT(DISTINCT NULLIF(UPPER(TRIM(raw_value)), '')) > 1 AS specimen_type_conflict
                FROM raw_events WHERE source_table = 'labevents' AND feature = 'specimen_type'
                    AND specimen_id IS NOT NULL
                GROUP BY subject_id, specimen_id
            """)
            raw_output = staging / "events_raw.parquet"
            connection.execute(f"""
                COPY (
                    SELECT raw_events.*, specimen_info.specimen_type,
                        COALESCE(specimen_info.specimen_type_conflict, FALSE) AS specimen_type_conflict
                    FROM raw_events LEFT JOIN specimen_info USING (subject_id, specimen_id)
                ) TO {sql_string(raw_output.as_posix())} (FORMAT PARQUET)
            """)
            connection.execute(
                f"CREATE TEMP VIEW extracted AS SELECT * FROM read_parquet({sql_string(raw_output.as_posix())})"
            )
            summary = dict(zip(
                ("raw_rows", "stays_with_raw_events", "text_only_rows", "inferred_lab_links",
                 "invalid_or_incomplete_intervals", "rewritten_or_cancelled_rows",
                 "rows_outside_hospital_interval", "rows_after_recorded_hospital_death",
                 "arterial_target_rows_without_specimen_type", "specimen_type_conflict_rows"),
                connection.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT stay_id),
                        COUNT(*) FILTER (WHERE raw_valuenum IS NULL AND raw_value IS NOT NULL),
                        COUNT(*) FILTER (WHERE linkage_status = 'inferred_hospital_time'),
                        COUNT(*) FILTER (WHERE interval_status NOT IN ('point', 'valid_interval')),
                        COUNT(*) FILTER (WHERE is_rewritten_or_cancelled),
                        COUNT(*) FILTER (WHERE within_hospital_interval = FALSE),
                        COUNT(*) FILTER (WHERE after_recorded_hospital_death),
                        COUNT(*) FILTER (WHERE arterial_specimen_required AND specimen_type IS NULL),
                        COUNT(*) FILTER (WHERE specimen_type_conflict)
                    FROM extracted
                """).fetchone(),
            ))
            summary["cohort_stays"] = count
            summary["stays_without_raw_events"] = count - summary["stays_with_raw_events"]
            sources = dict(connection.execute(
                "SELECT source_table, COUNT(*) FROM extracted GROUP BY source_table ORDER BY source_table"
            ).fetchall())
            observed_features = {row[0] for row in connection.execute(
                "SELECT DISTINCT feature FROM extracted WHERE ISFINITE(raw_valuenum)"
            ).fetchall()}
            missing_features = [
                feature["name"] for feature in registry["temporal_features"]
                if not feature["derived"] and feature["name"] not in observed_features
            ]
            window_file = staging / "extraction_windows.parquet"
            connection.execute(
                f"COPY (SELECT * FROM windows ORDER BY stay_id) TO {sql_string(window_file.as_posix())} (FORMAT PARQUET)"
            )
        report = {
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION, "source_database": "MIMIC-IV", "source_version": "3.1",
            "duckdb_version": duckdb.__version__, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry_version": registry["schema_version"], "registry_sha256": registry_hash,
            "dataset_config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "target_temporal_features": [feature["name"] for feature in registry["temporal_features"]],
            "policy": {
                "values": "raw; canonical_unit is a target only; no valuenum column until cleaning",
                "window": "one encompassing interval per stay, across all candidates: min SIT -72h to max upper bound +48h",
                "point_boundaries": "inclusive extraction envelope; later hourly bins must be half-open",
                "treatment_intervals": "positive-duration overlap; malformed intervals retained only when an endpoint is in the envelope",
                "missing_admission_labs": "subject match inside documented hospital interval; inferred linkage flagged",
                "specimen": "exact subject/specimen ID match within extracted evidence; conflicts left unknown",
                "ventilation": "invasive/noninvasive procedures and text evidence remain separate; no binary inference",
                "quality_control": "invalid units, intervals, rewritten records and non-arterial blood gases must be adjudicated in cleaning/adjudication",
                "weight_fallback": None,
            },
            "summary": summary, "source_rows": sources, "direct_features_without_numeric_rows": missing_features,
            "source_files": {name: {"path": str(path), "size_bytes": path.stat().st_size} for name, path in source_paths.items()},
        }
        report_file = staging / "04_extraction.json"
        report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for artifact in (raw_output, window_file):
            artifact.replace(processed_dir / artifact.name)
        report_file.replace(metrics_dir / report_file.name)

    print(f"[04] Complete v{DATASET_VERSION}; {time.perf_counter() - started:.1f}s; report: {metrics_dir}/04_extraction.json", flush=True)
    return report


if __name__ == "__main__":
    print(f"[04] Extract source measurements and evidence across infection-candidate windows. v{DATASET_VERSION}", flush=True)
    extract_temporal_data()
