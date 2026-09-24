"""Select adults at their first ICU stay, before eligibility filtering."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_VERSION = json.loads(Path(__file__).with_name("dataset_config.json").read_text(encoding="utf-8"))["dataset_version"]
MIMIC_DIR = BASE_DIR / "data" / "raw" / "mimiciv" / "3.1"
OUT_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "work"
METRICS_DIR = BASE_DIR / "outputs" / "mimiciv"


def sql_path(path):
    return "'" + Path(path).as_posix().replace("'", "''") + "'"


def validate_keys(connection, table, primary_key, required_keys):
    missing_predicate = " OR ".join(f"{column} IS NULL" for column in required_keys)
    total, unique, missing = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT {primary_key}), "
        f"COUNT(*) FILTER (WHERE {missing_predicate}) FROM {table}"
    ).fetchone()
    if total == 0 or total != unique or missing:
        raise ValueError(
            f"Invalid {table}: rows={total}, unique {primary_key}={unique}, "
            f"rows with missing required keys={missing}"
        )


def build_base_cohort(mimic_dir=MIMIC_DIR, out_dir=OUT_DIR, metrics_dir=METRICS_DIR):
    start_time = time.perf_counter()
    mimic_dir, out_dir, metrics_dir = map(Path, (mimic_dir, out_dir, metrics_dir))
    sources = {
        "source_icu": mimic_dir / "icu" / "icustays.csv.gz",
        "source_patients": mimic_dir / "hosp" / "patients.csv.gz",
        "source_admissions": mimic_dir / "hosp" / "admissions.csv.gz",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required MIMIC-IV source not found: {path}")


    with duckdb.connect(database=":memory:") as connection:
        connection.execute(f"""
            CREATE TEMP TABLE source_icu AS
            SELECT subject_id, hadm_id, stay_id, intime, outtime, los, first_careunit
            FROM read_csv_auto({sql_path(sources['source_icu'])}, types={{
                'subject_id': 'BIGINT', 'hadm_id': 'BIGINT', 'stay_id': 'BIGINT',
                'intime': 'TIMESTAMP', 'outtime': 'TIMESTAMP', 'los': 'DOUBLE'
            }})
        """)
        connection.execute(f"""
            CREATE TEMP TABLE source_patients AS
            SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod
            FROM read_csv_auto({sql_path(sources['source_patients'])}, types={{
                'subject_id': 'BIGINT', 'anchor_age': 'INTEGER',
                'anchor_year': 'INTEGER', 'dod': 'DATE'
            }})
        """)
        connection.execute(f"""
            CREATE TEMP TABLE source_admissions AS
            SELECT subject_id, hadm_id, admittime, dischtime, deathtime,
                   race, admission_type, hospital_expire_flag
            FROM read_csv_auto({sql_path(sources['source_admissions'])}, types={{
                'subject_id': 'BIGINT', 'hadm_id': 'BIGINT',
                'admittime': 'TIMESTAMP', 'dischtime': 'TIMESTAMP',
                'deathtime': 'TIMESTAMP', 'hospital_expire_flag': 'INTEGER'
            }})
        """)
        validate_keys(connection, "source_icu", "stay_id", ["subject_id", "hadm_id", "stay_id", "intime"])
        validate_keys(connection, "source_patients", "subject_id", ["subject_id"])
        validate_keys(connection, "source_admissions", "hadm_id", ["subject_id", "hadm_id"])

        connection.execute("""
            CREATE TEMP TABLE eligibility AS
            WITH ranked_icu AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY subject_id ORDER BY intime, stay_id
                ) AS icu_seq
                FROM source_icu
            ), joined AS (
                SELECT
                    ranked.subject_id, ranked.hadm_id, ranked.stay_id,
                    patients.gender,
                    patients.anchor_age + (
                        EXTRACT(YEAR FROM admissions.admittime) - patients.anchor_year
                    ) AS age,
                    admissions.race, admissions.admission_type, ranked.first_careunit,
                    ranked.intime AS icu_intime, ranked.outtime AS icu_outtime,
                    EPOCH(ranked.outtime - ranked.intime) / 86400.0 AS icu_los_days,
                    admissions.hospital_expire_flag, patients.dod,
                    admissions.hospital_expire_flag = 1 AND admissions.deathtime IS NULL
                        AS death_time_unavailable,
                    admissions.admittime AS hospital_admittime,
                    admissions.dischtime AS hospital_dischtime,
                    admissions.deathtime AS hospital_deathtime,
                    patients.anchor_age, patients.anchor_year, patients.anchor_year_group,
                    patients.anchor_age = 91 AS age_is_deidentified,
                    ranked.los AS icu_los_days_recorded,
                    EPOCH(ranked.outtime - ranked.intime) / 3600.0 AS icu_los_hours,
                    ranked.outtime >= ranked.intime + INTERVAL 24 HOUR AS icu_los_ge_24h,
                    ranked.icu_seq,
                    patients.subject_id IS NOT NULL AND admissions.hadm_id IS NOT NULL
                        AS has_linked_records,
                    COALESCE(ranked.outtime > ranked.intime, FALSE) AS valid_icu_interval,
                    COALESCE(admissions.dischtime > admissions.admittime, FALSE)
                        AS valid_hospital_interval,
                    COALESCE(age >= 18, FALSE) AS is_adult,
                    NULLIF(TRIM(admissions.admission_type), '') IS NOT NULL
                        AS has_admission_type,
                    COALESCE(UPPER(TRIM(admissions.admission_type)) = 'ELECTIVE', FALSE)
                        AS is_elective_admission,
                    COALESCE(UPPER(TRIM(admissions.admission_type)) =
                        'SURGICAL SAME DAY ADMISSION', FALSE) AS is_same_day_surgical_admission
                FROM ranked_icu ranked
                LEFT JOIN source_patients patients ON ranked.subject_id = patients.subject_id
                LEFT JOIN source_admissions admissions
                    ON ranked.hadm_id = admissions.hadm_id
                    AND ranked.subject_id = admissions.subject_id
            ), reasons AS (
                SELECT *, CASE
                    WHEN icu_seq != 1 THEN 'not_first_icu_stay'
                    WHEN NOT has_linked_records THEN 'missing_linked_records'
                    WHEN NOT valid_icu_interval THEN 'invalid_icu_interval'
                    WHEN NOT valid_hospital_interval THEN 'invalid_hospital_interval'
                    WHEN age IS NULL THEN 'missing_age'
                    WHEN NOT is_adult THEN 'age_under_18'
                    WHEN NOT has_admission_type THEN 'missing_admission_type'
                    WHEN is_elective_admission THEN 'elective_admission'
                    ELSE 'included'
                END AS eligibility_reason
                FROM joined
            )
            SELECT *, eligibility_reason = 'included' AS included_in_base_cohort
            FROM reasons
        """)

        reason_counts = dict(connection.execute(
            "SELECT eligibility_reason, COUNT(*) FROM eligibility GROUP BY eligibility_reason"
        ).fetchall())
        total = connection.execute("SELECT COUNT(*) FROM source_icu").fetchone()[0]
        if sum(reason_counts.values()) != total:
            raise ValueError("Eligibility rows do not reconcile with input ICU stays")

        attrition = [{"step": "all_icu_stays", "removed": 0, "remaining": total}]
        remaining = total
        for reason in (
            "not_first_icu_stay", "missing_linked_records", "invalid_icu_interval",
            "invalid_hospital_interval", "missing_age", "age_under_18",
            "missing_admission_type", "elective_admission",
        ):
            removed = reason_counts.get(reason, 0)
            remaining -= removed
            attrition.append({"step": reason, "removed": removed, "remaining": remaining})

        connection.execute("""
            CREATE TEMP TABLE base_cohort AS
            SELECT * FROM eligibility WHERE included_in_base_cohort
        """)
        count, subjects, stays = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT stay_id) FROM base_cohort"
        ).fetchone()
        if count == 0 or count != remaining or count != subjects or count != stays:
            raise ValueError("Base cohort is empty, duplicated, or inconsistent with attrition")

        summary_names = (
            "stays_under_24h", "stays_at_least_24h", "short_stays_with_hospital_death",
            "deidentified_age_stays", "missing_mortality_labels", "icu_outside_hospital_interval",
        )
        summary_values = connection.execute("""
            SELECT
                COUNT(*) FILTER (WHERE NOT icu_los_ge_24h),
                COUNT(*) FILTER (WHERE icu_los_ge_24h),
                COUNT(*) FILTER (WHERE NOT icu_los_ge_24h AND hospital_expire_flag = 1),
                COUNT(*) FILTER (WHERE age_is_deidentified),
                COUNT(*) FILTER (WHERE hospital_expire_flag IS NULL),
                COUNT(*) FILTER (WHERE icu_intime < hospital_admittime OR icu_outtime > hospital_dischtime)
            FROM base_cohort
        """).fetchone()
        summary = dict(zip(summary_names, summary_values))
        admission_types = dict(connection.execute(
            "SELECT admission_type, COUNT(*) FROM base_cohort GROUP BY admission_type ORDER BY admission_type"
        ).fetchall())
        invalid_labels = connection.execute(
            "SELECT COUNT(*) FROM base_cohort WHERE hospital_expire_flag NOT IN (0, 1)"
        ).fetchone()[0]
        if invalid_labels:
            raise ValueError(f"Found {invalid_labels} invalid hospital mortality labels")

        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "base_cohort.parquet"
        eligibility_file = out_dir / "base_eligibility.parquet"
        metrics_file = metrics_dir / "01_base.json"
        connection.execute(
            f"COPY (SELECT * FROM base_cohort ORDER BY subject_id, stay_id) "
            f"TO {sql_path(out_file)} (FORMAT PARQUET)"
        )
        connection.execute(
            f"COPY (SELECT * FROM eligibility ORDER BY subject_id, icu_seq) "
            f"TO {sql_path(eligibility_file)} (FORMAT PARQUET)"
        )

    report = {
        "dataset_version": DATASET_VERSION,
        "source_database": "MIMIC-IV", "source_version": "3.1",
        "cohort_specification_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
        "policy": {
            "first_stay": "first ICU stay per subject before eligibility filtering",
            "minimum_age": 18, "minimum_icu_duration_hours": None,
            "excluded_admission_types": ["ELECTIVE"],
            "missing_admission_type": "exclude", "same_day_surgical_admission": "retain and flag",
            "age": "anchor_age + admission_year - anchor_year; anchor_age 91 is deidentified",
            "duration": "exact ICU outtime minus intime; 24h flag is admission-relative",
        },
        "input_files": {
            name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
            for name, path in sources.items()
        },
        "attrition": attrition, "base_cohort_stays": count,
        "summary": summary, "included_admission_types": admission_types,
    }
    metrics_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[01] Complete v{DATASET_VERSION}; {time.perf_counter() - start_time:.1f}s; report: {metrics_dir}/01_base.json", flush=True)
    return report


if __name__ == "__main__":
    print(f"[01] Select adults at their first ICU stay, before eligibility filtering. v{DATASET_VERSION}", flush=True)
    build_base_cohort()
