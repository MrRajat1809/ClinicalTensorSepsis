"""Build the observed 24-hour tensor from canonical measurements and clinical evidence."""

import hashlib
import importlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/processed/eicu"
PROCESSED_DIR = DATA_DIR / "work"
METRICS_DIR = BASE_DIR / "outputs/eicu"
CONFIG_PATH = Path(__file__).with_name("dataset_config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
TENSOR_SCHEMA_PATH = BASE_DIR / "src/common/tensor_schema.json"
TENSOR_SCHEMA = json.loads(TENSOR_SCHEMA_PATH.read_text(encoding="utf-8"))
if TENSOR_SCHEMA["schema_version"] != CONFIG["shared_schema_version"]:
    raise ValueError("Dataset and shared tensor schema versions differ")
if TENSOR_SCHEMA["policy"].keys() & CONFIG["tensor"].keys():
    raise ValueError("Dataset configuration must not override shared tensor settings")
CONFIG["tensor"] = dict(TENSOR_SCHEMA["policy"], **CONFIG["tensor"])
CONFIG["imputation"]["observed_only_features"] = TENSOR_SCHEMA["observed_only_features"]
DATASET_VERSION = CONFIG["dataset_version"]
COMMON_DIR = BASE_DIR / "src" / "common"
INPUTS = {
    "final": "sepsis_cohort.parquet",
    "cleaned": "events_clean.parquet",
    "respiratory": "respiratory_evidence.parquet",
    "gcs": "gcs_evidence.parquet",
    "urine": "urine_evidence.parquet",
}


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_list(values):
    return ", ".join(sql_string(value) for value in values)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(processed_dir, source_metrics_dir, common_dir):
    audit_path = source_metrics_dir / "cohort_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("audit_version") != "1.0.0":
        raise ValueError("A passing supported cohort audit is required")
    if not audit.get("checks") or any(check.get("passed") is not True for check in audit["checks"]):
        raise ValueError("Cohort checks are missing or failed")
    paths = {name: processed_dir / filename for name, filename in INPUTS.items()}
    paths.update({
        "registry": common_dir / "feature_registry.json",
        "measurement_rules": common_dir / "measurement_rules.json",
        "sofa_policy": CONFIG_PATH,
        "infection_policy": CONFIG_PATH,
    })
    manifest = {}
    for name, path in paths.items():
        digest = file_hash(path)
        if digest != audit.get("input_manifest", {}).get(name, {}).get("sha256"):
            raise ValueError(f"Input changed or absent from audit: {name}; rerun cohort QC")
        manifest[name] = {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
    policy_path = CONFIG_PATH
    policy = dict(CONFIG["tensor"], release_version=DATASET_VERSION)
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    sofa_policy = json.loads(paths["sofa_policy"].read_text(encoding="utf-8"))["sofa_policy"]
    names = [entry["name"] for entry in registry["temporal_features"]]
    if len(names) != 50 or len(set(names)) != 50 or set(policy["aggregation"]) != set(names):
        raise ValueError("Expected exactly 50 uniquely mapped temporal channels")
    if policy["schema_version"] != "1.0.0" or policy["hours"] != 24 or policy["tensor_dtype"] != "float64":
        raise ValueError("Unsupported tensor policy")
    if set(policy["aggregation"].values()) - {
        "min", "max", "mean", "coherent_gcs", "paired_min", "validated_volume_sum",
        "concurrent_peak", "any_invasive_with_coverage",
    }:
        raise ValueError("Unsupported aggregation")
    derived_aggregation = {
        "gcs_eye": "coherent_gcs", "gcs_verbal": "coherent_gcs", "gcs_motor": "coherent_gcs",
        "pf_ratio": "paired_min", "urine_output": "validated_volume_sum",
        "neq": "concurrent_peak", "vent": "any_invasive_with_coverage",
    }
    for name, aggregation in policy["aggregation"].items():
        if name in derived_aggregation:
            if aggregation != derived_aggregation[name]:
                raise ValueError(f"Unsupported derived aggregation: {name}")
        elif aggregation not in {"min", "max", "mean"}:
            raise ValueError(f"Unsupported direct aggregation: {name}")
    if len({entry["name"] for entry in policy["static_features"]}) != len(policy["static_features"]):
        raise ValueError("Duplicate static features")
    for name, vocabulary in policy["category_vocabularies"].items():
        if len({value.strip().upper() for value in vocabulary}) != len(vocabulary):
            raise ValueError(f"Duplicate category vocabulary: {name}")
    manifest["tensor_policy"] = {"path": str(policy_path), "sha256": file_hash(policy_path)}
    manifest["cohort_audit"] = {"path": str(audit_path), "sha256": file_hash(audit_path)}
    return registry, policy, sofa_policy, manifest


def create_bins(connection):
    connection.execute("""
        CREATE TEMP TABLE patients AS
        SELECT ROW_NUMBER() OVER (ORDER BY stay_id) - 1 AS patient_index, *,
            CASE WHEN hospital_expire_flag = 1 AND hospital_deathtime IS NULL
                THEN sepsis_onset_time
                ELSE LEAST(sepsis_onset_time + INTERVAL 24 HOUR, followup_end) END AS observation_end,
            hospital_expire_flag = 1 AND hospital_deathtime IS NULL AS temporal_withheld_unknown_death_time
        FROM final
    """)
    bad = connection.execute("""
        SELECT COUNT(*) FROM patients WHERE stay_id IS NULL OR subject_id IS NULL OR hadm_id IS NULL
            OR sepsis_onset_time IS NULL OR observation_end IS NULL
            OR followup_end IS NULL OR icu_outtime IS NULL
            OR observation_end < sepsis_onset_time
            OR followup_end IS DISTINCT FROM LEAST(icu_outtime, hospital_deathtime)
            OR (hospital_expire_flag IS NOT NULL AND hospital_expire_flag NOT IN (0, 1))
    """).fetchone()[0]
    duplicates = connection.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT subject_id) + COUNT(*) - COUNT(DISTINCT stay_id) FROM patients
    """).fetchone()[0]
    if bad or duplicates:
        raise ValueError("Invalid cohort identifiers, observation endpoints, or mortality labels")
    connection.execute("""
        CREATE TEMP TABLE bins AS
        SELECT patient_index, stay_id, time_step,
            sepsis_onset_time + time_step * INTERVAL 1 HOUR AS bin_start,
            LEAST(sepsis_onset_time + (time_step + 1) * INTERVAL 1 HOUR, observation_end) AS bin_end,
            GREATEST(0, EPOCH(LEAST(sepsis_onset_time + (time_step + 1) * INTERVAL 1 HOUR,
                observation_end) - (sepsis_onset_time + time_step * INTERVAL 1 HOUR))) AS exposure_seconds
        FROM patients CROSS JOIN RANGE(24) AS hours(time_step)
    """)
    connection.execute("CREATE TEMP VIEW active_bins AS SELECT * FROM bins WHERE exposure_seconds > 0")


def boundary_contract_check(connection):
    """Small deterministic boundary cases, executed as part of the build."""
    actual = connection.execute("""
        WITH offsets(seconds, expected) AS (
            VALUES (-1, NULL::INTEGER), (0, 0), (3599, 0), (3600, 1),
                   (86399, 23), (86400, NULL::INTEGER)
        ) SELECT COUNT(*) FROM offsets WHERE
            (CASE WHEN seconds >= 0 AND seconds < 86400 THEN FLOOR(seconds / 3600.0)::INTEGER END)
            IS DISTINCT FROM expected
    """).fetchone()[0]
    intervals = connection.execute("""
        WITH examples(start_second, end_second, expected_overlap) AS (
            VALUES (-60, 60, 60), (-60, 0, 0), (0, 3600, 3600),
                   (3600, 7200, 0), (3000, 4000, 600)
        ) SELECT COUNT(*) FROM examples WHERE
            GREATEST(0, LEAST(end_second, 3600) - GREATEST(start_second, 0)) <> expected_overlap
    """).fetchone()[0]
    if actual or intervals:
        raise AssertionError("Hourly point/interval boundary contract failed")


def build_point_values(connection):
    connection.execute("""
        CREATE TEMP TABLE direct_bins AS
        WITH observations AS (
            SELECT DISTINCT event.stay_id, event.event_time, event.source_table,
                event.itemid, event.specimen_id, event.feature, event.valuenum,
                rules.feature_index, rules.aggregation
            FROM cleaned AS event JOIN feature_rules AS rules USING (feature)
            JOIN patients AS patient USING (stay_id)
            WHERE event.numeric_usable AND rules.aggregation IN ('min', 'max', 'mean')
                AND event.event_time >= patient.sepsis_onset_time AND event.event_time < patient.observation_end
        )
        SELECT bins.patient_index, bins.stay_id, bins.time_step, observations.feature_index,
            observations.feature, CASE observations.aggregation
                WHEN 'min' THEN MIN(valuenum) WHEN 'max' THEN MAX(valuenum) ELSE AVG(valuenum) END AS value,
            COUNT(*) AS evidence_count
        FROM observations JOIN active_bins AS bins ON observations.stay_id = bins.stay_id
            AND observations.event_time >= bins.bin_start AND observations.event_time < bins.bin_end
        GROUP BY bins.patient_index, bins.stay_id, bins.time_step,
            observations.feature_index, observations.feature, observations.aggregation
    """)
    connection.execute("""
        CREATE TEMP TABLE chosen_gcs AS
        SELECT bins.patient_index, bins.stay_id, bins.time_step, gcs.event_time, gcs.gcs_total,
            gcs.observed_components
        FROM gcs JOIN active_bins AS bins ON gcs.stay_id = bins.stay_id
            AND gcs.event_time >= bins.bin_start AND gcs.event_time < bins.bin_end
        WHERE gcs.observed_components > 0
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY bins.stay_id, bins.time_step
            ORDER BY gcs.gcs_total IS NULL, gcs.gcs_total NULLS LAST, gcs.event_time DESC
        ) = 1
    """)
    connection.execute("""
        CREATE TEMP TABLE gcs_bins AS
        SELECT chosen.patient_index, chosen.stay_id, chosen.time_step, rules.feature_index, event.feature,
            CASE WHEN COUNT(*) = COUNT(event.valuenum) AND MIN(event.valuenum) = MAX(event.valuenum)
                THEN MIN(event.valuenum) END AS value,
            1::BIGINT AS evidence_count
        FROM chosen_gcs AS chosen JOIN cleaned AS event
            ON event.stay_id = chosen.stay_id AND event.event_time = chosen.event_time
        JOIN feature_rules AS rules ON event.feature = rules.feature
        WHERE rules.aggregation = 'coherent_gcs' AND event.record_qc_status = 'accepted'
        GROUP BY chosen.patient_index, chosen.stay_id, chosen.time_step, rules.feature_index, event.feature
    """)
    connection.execute("""
        CREATE TEMP TABLE pf_bins AS
        SELECT bins.patient_index, bins.stay_id, bins.time_step, rules.feature_index, 'pf_ratio' AS feature,
            MIN(respiratory.pf_ratio) AS value, COUNT(*) AS evidence_count
        FROM respiratory JOIN active_bins AS bins ON respiratory.stay_id = bins.stay_id
            AND respiratory.event_time >= bins.bin_start AND respiratory.event_time < bins.bin_end
        CROSS JOIN (SELECT feature_index FROM feature_rules WHERE feature = 'pf_ratio') AS rules
        WHERE respiratory.pf_ratio IS NOT NULL
        GROUP BY bins.patient_index, bins.stay_id, bins.time_step, rules.feature_index
    """)
    connection.execute("""
        CREATE TEMP TABLE urine_bins AS
        SELECT bins.patient_index, bins.stay_id, bins.time_step, rules.feature_index, 'urine_output' AS feature,
            CASE WHEN COUNT(*) = COUNT(urine.urine_ml) THEN ROUND(SUM(urine.urine_ml), 6) END AS value,
            COUNT(urine.urine_ml) AS evidence_count,
            COUNT(*) FILTER (WHERE urine.urine_ml IS NULL) AS invalid_times
        FROM urine JOIN active_bins AS bins ON urine.stay_id = bins.stay_id
            AND urine.event_time >= bins.bin_start AND urine.event_time < bins.bin_end
        CROSS JOIN (SELECT feature_index FROM feature_rules WHERE feature = 'urine_output') AS rules
        GROUP BY bins.patient_index, bins.stay_id, bins.time_step, rules.feature_index
    """)


def create_segments(connection, intervals, prefix):
    connection.execute(f"""
        CREATE TEMP TABLE {prefix}_segments AS
        WITH relevant AS (
            SELECT DISTINCT bins.* FROM active_bins AS bins JOIN {intervals} AS treatment
                ON bins.stay_id = treatment.stay_id AND treatment.interval_start < bins.bin_end
                AND treatment.interval_end > bins.bin_start
        ), edges AS (
            SELECT patient_index, stay_id, time_step, bin_start AS endpoint FROM relevant
            UNION SELECT patient_index, stay_id, time_step, bin_end FROM relevant
            UNION
            SELECT bins.patient_index, bins.stay_id, bins.time_step,
                GREATEST(treatment.interval_start, bins.bin_start)
            FROM relevant AS bins JOIN {intervals} AS treatment ON bins.stay_id = treatment.stay_id
                AND treatment.interval_start < bins.bin_end AND treatment.interval_end > bins.bin_start
            UNION
            SELECT bins.patient_index, bins.stay_id, bins.time_step,
                LEAST(treatment.interval_end, bins.bin_end)
            FROM relevant AS bins JOIN {intervals} AS treatment ON bins.stay_id = treatment.stay_id
                AND treatment.interval_start < bins.bin_end AND treatment.interval_end > bins.bin_start
        ), segments AS (
            SELECT *, LEAD(endpoint) OVER (
                PARTITION BY stay_id, time_step ORDER BY endpoint
            ) AS next_endpoint FROM edges
        )
        SELECT patient_index, stay_id, time_step, endpoint AS segment_start, next_endpoint AS segment_end
        FROM segments WHERE next_endpoint > endpoint
    """)


def build_neq(connection, policy):
    coefficients = policy["neq"]["coefficients"]
    if set(coefficients) != {"norepinephrine", "epinephrine", "phenylephrine", "dopamine", "vasopressin"}:
        raise ValueError("Unsupported NEQ drug set")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in coefficients.values()):
        raise ValueError("Invalid NEQ coefficients")
    connection.execute("CREATE TEMP TABLE neq_coefficients (feature VARCHAR, coefficient DOUBLE)")
    connection.executemany("INSERT INTO neq_coefficients VALUES (?, ?)", list(coefficients.items()))
    connection.execute("""
        CREATE TEMP TABLE unresolved_neq_stays AS
        SELECT DISTINCT event.stay_id FROM cleaned AS event JOIN neq_coefficients USING (feature)
        JOIN patients AS patient USING (stay_id)
        WHERE event.source_table = 'inputevents'
            AND (event.record_qc_status IN ('invalid_interval', 'missing_event_time')
                OR (event.record_qc_status = 'conflicting_infusion_rates'
                    AND (event.event_time IS NULL OR event.event_end_time IS NULL
                        OR event.event_end_time <= event.event_time)))
    """)
    connection.execute("""
        CREATE TEMP TABLE drug_intervals AS
        SELECT DISTINCT event.stay_id, event.feature, event.valuenum, event.numeric_usable,
            coefficients.coefficient, GREATEST(event.effective_start_time, patient.sepsis_onset_time) AS interval_start,
            LEAST(event.effective_end_time, patient.observation_end) AS interval_end
        FROM cleaned AS event JOIN neq_coefficients AS coefficients USING (feature)
        JOIN patients AS patient USING (stay_id)
        WHERE event.source_table = 'inputevents'
            AND event.record_qc_status IN ('accepted', 'conflicting_infusion_rates')
            AND event.effective_start_time < patient.observation_end
            AND event.effective_end_time > patient.sepsis_onset_time
    """)
    create_segments(connection, "drug_intervals", "neq")
    connection.execute("""
        CREATE TEMP TABLE neq_segment_values AS
        WITH drug_rates AS (
            SELECT segments.*, treatment.feature, MAX(treatment.valuenum) * MAX(treatment.coefficient) AS contribution,
                COUNT(DISTINCT treatment.valuenum) > 1
                    OR COUNT(*) FILTER (WHERE treatment.feature IS NOT NULL AND NOT treatment.numeric_usable) > 0 AS invalid_rate
            FROM neq_segments AS segments LEFT JOIN drug_intervals AS treatment
                ON segments.stay_id = treatment.stay_id AND treatment.interval_start <= segments.segment_start
                AND treatment.interval_end >= segments.segment_end
            GROUP BY segments.patient_index, segments.stay_id, segments.time_step,
                segments.segment_start, segments.segment_end, treatment.feature
        )
        SELECT patient_index, stay_id, time_step, segment_start, segment_end,
            EPOCH(segment_end - segment_start) AS duration_seconds,
            BOOL_OR(invalid_rate) AS invalid_rate,
            CASE WHEN NOT BOOL_OR(invalid_rate) THEN SUM(contribution) END AS neq
        FROM drug_rates GROUP BY patient_index, stay_id, time_step, segment_start, segment_end
    """)
    connection.execute("""
        CREATE TEMP TABLE neq_bins AS
        SELECT segments.patient_index, segments.stay_id, segments.time_step, rules.feature_index, 'neq' AS feature,
            CASE WHEN NOT BOOL_OR(segments.invalid_rate) AND unresolved.stay_id IS NULL
                THEN MAX(segments.neq) END AS value,
            COUNT(segments.neq) AS evidence_count,
            COALESCE(SUM(segments.duration_seconds) FILTER (WHERE segments.neq IS NOT NULL), 0) AS known_seconds,
            COALESCE(SUM(segments.duration_seconds) FILTER (WHERE segments.invalid_rate), 0) AS invalid_seconds,
            unresolved.stay_id IS NOT NULL AS unresolved_interval
        FROM neq_segment_values AS segments
        LEFT JOIN unresolved_neq_stays AS unresolved USING (stay_id)
        CROSS JOIN (SELECT feature_index FROM feature_rules WHERE feature = 'neq') AS rules
        GROUP BY segments.patient_index, segments.stay_id, segments.time_step, rules.feature_index, unresolved.stay_id
    """)


def build_ventilation(connection, sofa_policy):
    freshness = sofa_policy["support_lookback_hours"]
    if not isinstance(freshness, int) or freshness <= 0:
        raise ValueError("Invalid support freshness")
    connection.execute(f"""
        CREATE TEMP TABLE support_intervals AS
        WITH labels AS (
            SELECT stay_id, event_time, feature,
                CASE WHEN COUNT(*) = COUNT(*) FILTER (WHERE evidence_usable)
                    AND COUNT(DISTINCT LOWER(TRIM(raw_value))) = 1 THEN MIN(LOWER(TRIM(raw_value))) END AS label,
                COUNT(DISTINCT LOWER(TRIM(raw_value))) > 1
                    OR COUNT(*) FILTER (WHERE NOT evidence_usable) > 0 AS ambiguous
            FROM cleaned WHERE feature IN ('oxygen_device', 'ventilator_mode') AND record_qc_status = 'accepted'
            GROUP BY stay_id, event_time, feature
        ), durations AS (
            SELECT *, LEAD(event_time) OVER (PARTITION BY stay_id, feature ORDER BY event_time) AS next_time
            FROM labels
        ), combined AS (
            SELECT stay_id, feature, label, ambiguous, event_time AS interval_start,
                LEAST(next_time, event_time + INTERVAL {freshness} HOUR) AS interval_end FROM durations
            UNION ALL
            SELECT stay_id, feature, NULL::VARCHAR, FALSE,
                effective_start_time, effective_end_time
            FROM cleaned WHERE feature IN ('vent_invasive', 'vent_noninvasive') AND evidence_usable
        )
        SELECT combined.stay_id, combined.feature, combined.label, combined.ambiguous,
            GREATEST(combined.interval_start, patient.sepsis_onset_time) AS interval_start,
            LEAST(combined.interval_end, patient.observation_end) AS interval_end
        FROM combined JOIN patients AS patient USING (stay_id)
        WHERE combined.interval_start < patient.observation_end AND combined.interval_end > patient.sepsis_onset_time
    """)
    create_segments(connection, "support_intervals", "vent")
    connection.execute(f"""
        CREATE TEMP TABLE vent_segment_values AS
        WITH evidence AS (
            SELECT segments.*,
                COALESCE(BOOL_OR(support.ambiguous), FALSE) AS ambiguous,
                COALESCE(BOOL_OR(support.feature = 'vent_invasive'), FALSE) AS invasive_procedure,
                COALESCE(BOOL_OR(support.feature = 'oxygen_device'
                    AND support.label IN ('endotracheal tube', 'tracheostomy tube')), FALSE) AS airway,
                COALESCE(BOOL_OR(support.feature = 'ventilator_mode'
                    AND support.label IN ({sql_list(sofa_policy["active_ventilator_modes"])})), FALSE) AS active_mode,
                COALESCE(BOOL_OR(support.feature = 'vent_noninvasive'
                    OR (support.feature = 'oxygen_device' AND support.label IN ({sql_list(sofa_policy["noninvasive_devices"])}))
                    OR (support.feature = 'ventilator_mode' AND support.label IN ({sql_list(sofa_policy["noninvasive_modes"])}))), FALSE) AS noninvasive,
                COALESCE(BOOL_OR(
                    (support.feature = 'oxygen_device' AND support.label IN ({sql_list(sofa_policy["nonventilated_devices"])}))
                    OR (support.feature = 'ventilator_mode' AND support.label IN ('standby', 'ambient'))), FALSE) AS off_support
            FROM vent_segments AS segments LEFT JOIN support_intervals AS support
                ON segments.stay_id = support.stay_id AND support.interval_start <= segments.segment_start
                AND support.interval_end >= segments.segment_end
            GROUP BY segments.patient_index, segments.stay_id, segments.time_step,
                segments.segment_start, segments.segment_end
        ), classified AS (
            SELECT *, ambiguous OR ((invasive_procedure OR (airway AND active_mode))
                AND (noninvasive OR off_support)) AS conflicting
            FROM evidence
        )
        SELECT *, EPOCH(segment_end - segment_start) AS duration_seconds,
            CASE WHEN conflicting THEN NULL WHEN invasive_procedure OR (airway AND active_mode) THEN 1
                WHEN noninvasive OR off_support THEN 0 ELSE NULL END AS vent
        FROM classified
    """)
    connection.execute("""
        CREATE TEMP TABLE vent_bins AS
        SELECT segments.patient_index, segments.stay_id, segments.time_step, rules.feature_index, 'vent' AS feature,
            CASE WHEN MAX(segments.vent) = 1 THEN 1.0
                WHEN COUNT(*) = COUNT(segments.vent) AND MAX(segments.vent) = 0 THEN 0.0 END AS value,
            COUNT(segments.vent) AS evidence_count,
            COALESCE(SUM(duration_seconds) FILTER (WHERE segments.vent IS NOT NULL), 0) AS known_seconds,
            COALESCE(SUM(duration_seconds) FILTER (WHERE conflicting), 0) AS conflict_seconds
        FROM vent_segment_values AS segments
        CROSS JOIN (SELECT feature_index FROM feature_rules WHERE feature = 'vent') AS rules
        GROUP BY segments.patient_index, segments.stay_id, segments.time_step, rules.feature_index
    """)


def collect_static(connection, policy):
    static_features = [entry["name"] for entry in policy["static_features"]]
    selected_fields = ["stay_id", "subject_id", "hadm_id", *static_features, "hospital_expire_flag"]
    cursor = connection.execute("SELECT " + ", ".join(selected_fields) + " FROM patients ORDER BY patient_index")
    records = cursor.fetchall()
    matrix = np.full((len(records), len(static_features)), np.nan, dtype=np.float64)
    labels = np.full(len(records), np.nan, dtype=np.float64)
    stay_ids, subject_ids, hadm_ids = (np.empty(len(records), dtype=np.int64) for _ in range(3))
    vocabularies = {
        name: {label.strip().upper(): index for index, label in enumerate(labels)}
        for name, labels in policy["category_vocabularies"].items()
    }
    missing_tokens = set(policy["missing_category_labels"])
    unknown = {}
    for row_index, record in enumerate(records):
        stay_ids[row_index], subject_ids[row_index], hadm_ids[row_index] = record[:3]
        for column_index, feature in enumerate(static_features):
            value = record[column_index + 3]
            if feature in vocabularies:
                token = "" if value is None else str(value).strip().upper()
                if token in vocabularies[feature]:
                    matrix[row_index, column_index] = vocabularies[feature][token]
                elif token not in missing_tokens:
                    unknown.setdefault(feature, {}).setdefault(token, 0)
                    unknown[feature][token] += 1
            elif value is not None and np.isfinite(value):
                matrix[row_index, column_index] = value
        if record[-1] is not None:
            labels[row_index] = record[-1]
    return matrix, labels, stay_ids, subject_ids, hadm_ids, unknown


def build_tensor(processed_dir=PROCESSED_DIR, metrics_dir=METRICS_DIR, common_dir=COMMON_DIR,
                 output_dir=None, source_metrics_dir=None):
    started = time.perf_counter()
    processed_dir, metrics_dir, common_dir = Path(processed_dir), Path(metrics_dir), Path(common_dir)
    output_dir = Path(output_dir) if output_dir else DATA_DIR
    source_metrics_dir = Path(source_metrics_dir) if source_metrics_dir else metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    qc = importlib.import_module("09_eicu_master_qc")
    cohort_audit = qc.audit_cohort(processed_dir=processed_dir, metrics_dir=metrics_dir, common_dir=common_dir,
        adjudication_report=source_metrics_dir / "06_sepsis.json")
    if not cohort_audit["passed"]:
        raise ValueError("Cohort checks failed; see cohort_audit.json")
    registry, policy, sofa_policy, input_manifest = load_contract(processed_dir, metrics_dir, common_dir)
    feature_names = [entry["name"] for entry in registry["temporal_features"]]
    units = [entry["canonical_unit"] for entry in registry["temporal_features"]]
    static_names = [entry["name"] for entry in policy["static_features"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tensor_", dir=output_dir) as temporary:
        staging = Path(temporary)
        with duckdb.connect() as connection:
            connection.execute("SET threads = 4")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(f"SET temp_directory = {sql_string((staging / 'spill').as_posix())}")
            for name, filename in INPUTS.items():
                connection.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_string((processed_dir / filename).as_posix())})")
            connection.execute("CREATE TEMP TABLE feature_rules (feature_index INTEGER, feature VARCHAR, aggregation VARCHAR)")
            connection.executemany("INSERT INTO feature_rules VALUES (?, ?, ?)", [
                (index, feature, policy["aggregation"][feature]) for index, feature in enumerate(feature_names)
            ])
            boundary_contract_check(connection)
            create_bins(connection)
            build_point_values(connection)
            build_neq(connection, policy)
            build_ventilation(connection, sofa_policy)
            connection.execute("""
                CREATE TEMP TABLE hourly AS
                SELECT * FROM direct_bins WHERE value IS NOT NULL
                UNION ALL SELECT * FROM gcs_bins WHERE value IS NOT NULL
                UNION ALL SELECT * FROM pf_bins WHERE value IS NOT NULL
                UNION ALL SELECT * EXCLUDE (invalid_times) FROM urine_bins WHERE value IS NOT NULL
                UNION ALL SELECT * EXCLUDE (known_seconds, invalid_seconds, unresolved_interval) FROM neq_bins WHERE value IS NOT NULL
                UNION ALL SELECT * EXCLUDE (known_seconds, conflict_seconds) FROM vent_bins WHERE value IS NOT NULL
            """)
            invalid = connection.execute("""
                SELECT COUNT(*) FROM hourly LEFT JOIN active_bins USING (patient_index, stay_id, time_step)
                WHERE active_bins.stay_id IS NULL OR NOT ISFINITE(value) OR evidence_count <= 0
                    OR feature_index NOT BETWEEN 0 AND 49 OR time_step NOT BETWEEN 0 AND 23
            """).fetchone()[0]
            duplicate = connection.execute("""
                SELECT COUNT(*) FROM (
                    SELECT patient_index, time_step, feature_index FROM hourly
                    GROUP BY patient_index, time_step, feature_index HAVING COUNT(*) > 1
                )
            """).fetchone()[0]
            if invalid or duplicate:
                raise AssertionError(f"Invalid/duplicate tensor cells: {invalid}/{duplicate}")
            measurement_rules = json.loads((common_dir / "measurement_rules.json").read_text(encoding="utf-8"))
            cell_bounds = dict(measurement_rules["bounds"])
            cell_bounds.update(policy["derived_bounds"])
            # Recorded per-event urine limits do not apply to an hourly sum.
            cell_bounds["urine_output"] = [0, None]
            connection.execute("CREATE TEMP TABLE cell_bounds (feature VARCHAR, lower_bound DOUBLE, upper_bound DOUBLE)")
            connection.executemany("INSERT INTO cell_bounds VALUES (?, ?, ?)", [
                (name, *cell_bounds[name]) for name in feature_names
            ])
            invalid_bounds = connection.execute("""
                SELECT COUNT(*) FROM hourly JOIN cell_bounds USING (feature)
                WHERE value < lower_bound OR (upper_bound IS NOT NULL AND value > upper_bound + 1e-10)
                    OR (feature IN ('vent', 'gcs_eye', 'gcs_verbal', 'gcs_motor') AND value <> FLOOR(value))
            """).fetchone()[0]
            if invalid_bounds:
                raise AssertionError(f"Observed tensor cells outside their declared domain: {invalid_bounds}")
            static, labels, stay_ids, subject_ids, hadm_ids, unknown_categories = collect_static(connection, policy)
            tensor = np.full((len(stay_ids), 24, 50), np.nan, dtype=np.float64)
            counts = np.zeros(tensor.shape, dtype=np.uint32)
            cursor = connection.execute("SELECT patient_index, time_step, feature_index, value, evidence_count FROM hourly")
            while True:
                records = cursor.fetchmany(65536)
                if not records:
                    break
                batch = np.asarray(records, dtype=np.float64)
                patient_index, time_step, feature_index = (batch[:, column].astype(np.int64) for column in range(3))
                if np.any(batch[:, 4] > np.iinfo(np.uint32).max):
                    raise OverflowError("Evidence count exceeds uint32")
                tensor[patient_index, time_step, feature_index] = batch[:, 3]
                counts[patient_index, time_step, feature_index] = batch[:, 4].astype(np.uint32)
            exposure = np.asarray(connection.execute(
                "SELECT exposure_seconds FROM bins ORDER BY patient_index, time_step"
            ).fetchnumpy()["exposure_seconds"], dtype=np.float64).reshape(len(stay_ids), 24)
            structural = exposure == 0
            missing = np.isnan(tensor)
            within_followup_missing = missing & ~structural[:, :, None]
            if np.any(np.isfinite(tensor) & structural[:, :, None]) or not np.array_equal(counts > 0, ~missing):
                raise AssertionError("Observation, missingness and structural masks disagree")
            if np.any(np.isinf(tensor)) or np.any((exposure < 0) | (exposure > 3600)):
                raise AssertionError("Invalid tensor values or bin exposures")
            if np.any((exposure[:, :-1] < 3600) & (exposure[:, 1:] > 0)):
                raise AssertionError("Follow-up exposure must be a contiguous prefix")
            # Verify the actual populated array, not just the SQL dimensions.
            cursor = connection.execute("SELECT patient_index, time_step, feature_index, value, evidence_count FROM hourly")
            while records := cursor.fetchmany(65536):
                batch = np.asarray(records, dtype=np.float64)
                pi, ti, fi = (batch[:, column].astype(np.int64) for column in range(3))
                if not np.array_equal(tensor[pi, ti, fi], batch[:, 3]) or not np.array_equal(counts[pi, ti, fi], batch[:, 4]):
                    raise AssertionError("Hourly evidence does not match saved tensor cells/counts")
            coverage = {
                name: np.zeros((len(stay_ids), 24), dtype=np.float64)
                for name in ("neq_known_seconds", "neq_invalid_seconds", "vent_known_seconds", "vent_conflict_seconds", "urine_invalid_times")
            }
            for table, fields in (
                ("neq_bins", {"known_seconds": "neq_known_seconds", "invalid_seconds": "neq_invalid_seconds"}),
                ("vent_bins", {"known_seconds": "vent_known_seconds", "conflict_seconds": "vent_conflict_seconds"}),
                ("urine_bins", {"invalid_times": "urine_invalid_times"}),
            ):
                cursor = connection.execute(f"SELECT patient_index, time_step, {', '.join(fields)} FROM {table}")
                for record in cursor.fetchall():
                    for offset, name in enumerate(fields.values(), 2):
                        coverage[name][record[0], record[1]] = record[offset]
            for name, values in coverage.items():
                if name != "urine_invalid_times" and np.any(values > exposure + 0.000001):
                    raise AssertionError(f"Interval coverage exceeds follow-up: {name}")
            summary = {
                "patients": len(stay_ids), "temporal_features": 50, "static_context_features": len(static_names),
                "observed_cells": int((~missing).sum()),
                "within_followup_missing_cells": int(within_followup_missing.sum()),
                "structurally_unavailable_hours": int(structural.sum()),
                "partial_followup_hours": int(((exposure > 0) & (exposure < 3600)).sum()),
                "patients_with_incomplete_24h_followup": int((exposure.sum(axis=1) < 86400).sum()),
                "missing_mortality_labels": int(np.isnan(labels).sum()),
                "patients_with_temporal_data_withheld_unknown_death_time": connection.execute(
                    "SELECT COUNT(*) FROM patients WHERE temporal_withheld_unknown_death_time"
                ).fetchone()[0],
                "patients_without_observed_temporal_cells": int((~np.isfinite(tensor).any(axis=(1, 2))).sum()),
                "neq_unresolved_interval_stays": connection.execute("SELECT COUNT(*) FROM unresolved_neq_stays").fetchone()[0],
                "neq_withheld_hours": connection.execute("SELECT COUNT(*) FROM neq_bins WHERE invalid_seconds > 0 OR unresolved_interval").fetchone()[0],
                "urine_withheld_hours": connection.execute("SELECT COUNT(*) FROM urine_bins WHERE invalid_times > 0").fetchone()[0],
                "vent_conflict_hours": connection.execute("SELECT COUNT(*) FROM vent_bins WHERE conflict_seconds > 0").fetchone()[0],
            }
            connection.execute(f"COPY (SELECT * FROM patients ORDER BY patient_index) TO {sql_string((staging / 'cohort.parquet').as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")

        support = dict(observation_counts=counts, exposure_seconds=exposure,
            stay_ids=stay_ids, subject_ids=subject_ids, hadm_ids=hadm_ids, labels=labels,
            features=np.asarray(feature_names, dtype="U"), units=np.asarray(units, dtype="U"),
            static=static, static_features=np.asarray(static_names, dtype="U"),
            static_predictor_mask=np.asarray([entry["predictor_default"] for entry in policy["static_features"]], dtype=bool),
            **coverage)
        np.save(staging / "tensor_observed.npy", tensor, allow_pickle=False)
        np.savez_compressed(staging / "tensor_support.npz", **support)
        artifacts = {name: {"sha256": file_hash(staging / name), "size_bytes": (staging / name).stat().st_size}
                     for name in ("tensor_observed.npy", "tensor_support.npz", "cohort.parquet")}
        provenance = {}
        paths = list(processed_dir.glob("*.parquet"))
        paths += [source_metrics_dir / name for name in
                  ("01_base.json", "02_infection.json", "03_phenotypes.json", "04_extraction.json", "05_cleaning.json", "06_sepsis.json")]
        paths += [source_metrics_dir / name for name in
                  ("culture_qc.csv", "antimicrobial_qc.csv", "measurement_qc.csv", "source_mapping_qc.csv", "evidence_qc.csv")]
        paths += [Path(__file__).with_name(name) for name in
                  ("01_eicu_base_cohort.py", "02_eicu_infection_cohort.py", "03_eicu_phenotypes.py",
                   "04_eicu_temporal_extract.py", "05_eicu_temporal_clean.py", "06_eicu_sepsis3_cohort.py", "07_eicu_tensor_builder.py",
                   "09_eicu_master_qc.py")]
        paths += [CONFIG_PATH, *[common_dir / name for name in
                  ("feature_registry.json", "measurement_rules.json", "tensor_schema.json")]]
        for path in paths:
            provenance[path.resolve().relative_to(BASE_DIR).as_posix()] = {
                "sha256": file_hash(path), "size_bytes": path.stat().st_size}
        input_manifest = {name: dict(entry, path=Path(entry["path"]).resolve().relative_to(BASE_DIR).as_posix())
                          for name, entry in input_manifest.items()}
        report = dict(schema_version="1.0.0", dataset_version=DATASET_VERSION,
            generated_at_utc=datetime.now(timezone.utc).isoformat(), tensor_shape=list(tensor.shape),
            duckdb_version=duckdb.__version__, numpy_version=np.__version__,
            summary=summary, policy=policy, input_manifest=input_manifest, output_manifest=artifacts,
            provenance=provenance, unknown_static_categories=unknown_categories,
            acceptance_checks={name: True for name in (
                "matching_cohort_audit", "identifiers_labels_order", "point_interval_boundaries",
                "unique_finite_hourly_cells", "declared_cell_domains", "tensor_evidence_agreement",
                "missingness_counts_exposure", "intervention_coverage")},
            mask_contract={"missing": "isnan(tensor_observed)",
                "structural": "exposure_seconds == 0; never impute",
                "within_followup_missing": "isnan(tensor_observed) & (exposure_seconds > 0)[..., None]",
                "label_known": "isfinite(labels)", "static_missing": "isnan(static)"})
        manifest = dict(schema_version="1.0.0", dataset_version=DATASET_VERSION,
                        shared_schema_version=TENSOR_SCHEMA["schema_version"],
                        source_specific_limits=CONFIG['eicu'],
                        source_database=CONFIG["source_database"], source_version=CONFIG["source_version"], observed=report)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for filename in artifacts:
            (staging / filename).replace(output_dir / filename)
        (staging / "manifest.json").replace(output_dir / "manifest.json")
    print(f"[07] Complete: {tensor.shape}; observed cells={summary['observed_cells']:,}; {time.perf_counter() - started:.1f}s", flush=True)
    return report

if __name__ == "__main__":
    print(f"[07] Building observed tensor v{DATASET_VERSION}", flush=True)
    build_tensor()
