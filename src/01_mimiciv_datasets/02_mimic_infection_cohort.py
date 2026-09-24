"""Identify suspected infection from IV prescriptions and diagnostic cultures."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_VERSION = json.loads(Path(__file__).with_name("dataset_config.json").read_text(encoding="utf-8"))["dataset_version"]
MIMIC_DIR = BASE_DIR / "data" / "raw" / "mimiciv" / "3.1"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "work"
METRICS_DIR = BASE_DIR / "outputs" / "mimiciv"
INFECTION_SCHEMA_VERSION = "2.1.0"
INFECTION_POLICY_FILE = Path(__file__).with_name("dataset_config.json")
ANTIMICROBIAL_MAP_VERSION = "1.0.0"
ANTIMICROBIAL_NAMES = (
    "amikacin", "amikin", "amoxicillin", "ampicillin", "unasyn", "augmentin",
    "azithromycin", "zithromax", "aztreonam", "azactam",
    "cefazolin", "ancef", "kefzol", "cefepime", "maxipime", "cefotaxime", "claforan",
    "cefotetan", "cefotan", "cefoxitin", "mefoxin", "ceftaroline", "teflaro",
    "ceftazidime", "fortaz", "tazicef", "avibactam", "avycaz", "ceftolozane", "zerbaxa",
    "ceftriaxone", "rocephin", "cefuroxime", "zinacef", "cephalothin", "cephapirin",
    "chloramphenicol", "ciprofloxacin", "cipro", "clindamycin", "cleocin",
    "colistin", "colistimethate", "polymyxin", "daptomycin", "cubicin",
    "doripenem", "doribax", "doxycycline", "vibramycin", "ertapenem", "invanz",
    "erythromycin", "erythrocin", "gentamicin", "garamycin", "imipenem", "primaxin",
    "levofloxacin", "levaquin", "linezolid", "zyvox", "meropenem", "merrem",
    "metronidazole", "flagyl", "minocycline", "minocin", "moxifloxacin", "avelox",
    "nafcillin", "oxacillin", "penicillin", "pfizerpen", "piperacillin", "zosyn",
    "rifampin", "rifadin", "streptomycin", "sulfamethoxazole", "trimethoprim",
    "bactrim", "septra", "smz-tmp", "quinupristin", "dalfopristin", "synercid",
    "ticarcillin", "timentin", "tigecycline", "tygacil", "tobramycin",
    "vancomycin", "vancocin", "telavancin", "vibativ",
    "amphotericin", "ambisome", "abelcet", "amphotec",
    "anidulafungin", "eraxis", "caspofungin", "cancidas",
    "fluconazole", "diflucan", "micafungin", "mycamine",
    "voriconazole", "vfend", "posaconazole", "noxafil",
)
IV_ROUTES = ("IV", "IV DRIP", "IV PIGGYBACK", "IV PUSH", "IV BOLUS")


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def build_infection_cohort(mimic_dir=MIMIC_DIR, processed_dir=PROCESSED_DIR, metrics_dir=METRICS_DIR,
                           infection_policy_file=INFECTION_POLICY_FILE):
    started = time.perf_counter()
    mimic_dir, processed_dir, metrics_dir = map(Path, (mimic_dir, processed_dir, metrics_dir))
    inputs = {
        "base": processed_dir / "base_cohort.parquet",
        "prescriptions": mimic_dir / "hosp/prescriptions.csv.gz",
        "cultures": mimic_dir / "hosp/microbiologyevents.csv.gz",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    paths = {name: sql_string(path.as_posix()) for name, path in inputs.items()}
    policy_bytes = Path(infection_policy_file).read_bytes()
    culture_policy = json.loads(policy_bytes)["infection_policy"]
    if culture_policy["schema_version"] != "1.0.0":
        raise ValueError("Unsupported infection policy")
    normalize = lambda value: " ".join(value.upper().split())
    culture_names = [normalize(value) for value in culture_policy["diagnostic_culture_test_names"]]
    if not culture_names or len(set(culture_names)) != len(culture_names):
        raise ValueError("Diagnostic culture mapping must be nonempty and unique")
    allowed_tests = ", ".join(sql_string(value) for value in culture_names)
    excluded_specimens = ", ".join(sql_string(normalize(value)) for value in culture_policy["excluded_specimen_names"])
    pattern = "(^|[^a-z])(" + "|".join(re.escape(name) for name in ANTIMICROBIAL_NAMES) + ")([^a-z]|$)"
    routes = ", ".join(sql_string(route) for route in IV_ROUTES)

    with duckdb.connect(":memory:") as connection:
        connection.execute(f"CREATE TEMP TABLE base AS SELECT * FROM read_parquet({paths['base']})")
        base_count, subjects, stays = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT stay_id) FROM base"
        ).fetchone()
        if not base_count or base_count != subjects or base_count != stays:
            raise ValueError("Base cohort must contain unique, non-null patients and stays")
        previous_path = processed_dir / "infection_cohort.parquet"
        connection.execute(f"""
            CREATE TEMP TABLE prescription_audit AS
            SELECT base.stay_id, prescriptions.starttime, prescriptions.stoptime,
                   prescriptions.drug, UPPER(TRIM(prescriptions.route)) AS route,
                   CASE
                       WHEN NOT REGEXP_MATCHES(LOWER(COALESCE(drug, '')), {sql_string(pattern)})
                           THEN 'unmapped_name'
                       WHEN UPPER(TRIM(COALESCE(drug_type, ''))) = 'BASE'
                           OR REGEXP_MATCHES(LOWER(drug), 'cream|desensiti|ophth|ointment|lock|gel')
                           THEN 'excluded_preparation'
                       WHEN UPPER(TRIM(COALESCE(prescriptions.route, ''))) NOT IN ({routes})
                           THEN 'non_iv_or_missing_route'
                       WHEN starttime IS NULL THEN 'missing_start'
                       WHEN stoptime < starttime THEN 'invalid_interval'
                       ELSE 'included'
                   END AS prescription_status
            FROM read_csv_auto({paths['prescriptions']}, types={{
                'subject_id': 'BIGINT', 'hadm_id': 'BIGINT',
                'starttime': 'TIMESTAMP', 'stoptime': 'TIMESTAMP',
                'drug': 'VARCHAR', 'route': 'VARCHAR', 'drug_type': 'VARCHAR'
            }}) prescriptions
            INNER JOIN base ON prescriptions.subject_id = base.subject_id
                           AND prescriptions.hadm_id = base.hadm_id
        """)
        connection.execute("""
            CREATE TEMP TABLE antimicrobial_starts AS
            WITH deduplicated AS (
                SELECT DISTINCT stay_id, starttime, stoptime, LOWER(TRIM(drug)) AS drug, route
                FROM prescription_audit WHERE prescription_status = 'included'
            ), ordered AS (
                SELECT *, MAX(COALESCE(stoptime, starttime)) OVER (
                    PARTITION BY stay_id ORDER BY starttime, stoptime NULLS LAST, drug, route
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS covered_until
                FROM deduplicated
            )
                SELECT *, SUM(CASE
                    WHEN covered_until IS NULL OR starttime > covered_until + INTERVAL 24 HOUR
                    THEN 1 ELSE 0 END
                ) OVER (
                    PARTITION BY stay_id ORDER BY starttime, stoptime NULLS LAST, drug, route
                    ROWS UNBOUNDED PRECEDING
                ) AS episode_id,
                GREATEST(0.0, EPOCH(COALESCE(stoptime, starttime) -
                    GREATEST(starttime, COALESCE(covered_until, starttime))) / 3600.0
                ) AS new_coverage_hours
                    , ROW_NUMBER() OVER (
                        PARTITION BY stay_id ORDER BY starttime, stoptime NULLS LAST, drug, route
                    ) AS antimicrobial_start_id
                FROM ordered
        """)
        connection.execute("""
            CREATE TEMP TABLE episodes AS
            SELECT stay_id, episode_id, MIN(starttime) AS abx_start_time,
                   MAX(stoptime) AS abx_stop_time,
                   SUM(new_coverage_hours) AS abx_known_duration_hours,
                   BOOL_AND(stoptime IS NOT NULL) AS abx_duration_complete,
                   CASE WHEN BOOL_AND(stoptime IS NOT NULL)
                        THEN SUM(new_coverage_hours) END AS abx_duration_hours
            FROM antimicrobial_starts GROUP BY stay_id, episode_id
        """)
        connection.execute(f"""
            CREATE TEMP TABLE culture_test_audit AS
            WITH test_results AS (
            SELECT base.stay_id, micro.micro_specimen_id,
                COALESCE(micro.charttime, CAST(micro.chartdate AS TIMESTAMP)) AS culture_time,
                micro.charttime IS NULL AS culture_time_is_date_only,
                micro.spec_itemid, micro.spec_type_desc, micro.test_itemid, micro.test_name,
                REGEXP_REPLACE(UPPER(TRIM(COALESCE(micro.test_name, ''))), '\\s+', ' ', 'g') AS normalized_test_name,
                REGEXP_REPLACE(UPPER(TRIM(COALESCE(micro.spec_type_desc, ''))), '\\s+', ' ', 'g') AS normalized_specimen,
                BOOL_AND(UPPER(TRIM(COALESCE(micro.org_name, ''))) IN ('CANCELLED', 'CANCELED')) AS cancelled_only
            FROM read_csv_auto({paths['cultures']}, types={{
                'subject_id': 'BIGINT', 'hadm_id': 'BIGINT', 'micro_specimen_id': 'BIGINT',
                'charttime': 'TIMESTAMP', 'chartdate': 'DATE', 'spec_type_desc': 'VARCHAR',
                'spec_itemid': 'INTEGER', 'test_itemid': 'INTEGER', 'test_name': 'VARCHAR', 'org_name': 'VARCHAR'
            }}) micro
            INNER JOIN base ON micro.subject_id = base.subject_id AND micro.hadm_id = base.hadm_id
            GROUP BY ALL
            )
            SELECT *, CASE
                WHEN micro_specimen_id IS NULL OR spec_itemid IS NULL THEN 'missing_specimen_id'
                WHEN test_itemid IS NULL THEN 'missing_test_id'
                WHEN culture_time IS NULL THEN 'missing_culture_time'
                WHEN cancelled_only THEN 'cancelled_test'
                WHEN normalized_specimen IN ({excluded_specimens}) THEN 'surveillance_specimen'
                WHEN normalized_test_name IN ({allowed_tests}) THEN 'diagnostic_culture'
                WHEN REGEXP_MATCHES(normalized_test_name, {sql_string(culture_policy['excluded_test_pattern'])})
                    THEN 'surveillance_or_nonculture_test'
                ELSE 'unmapped_test'
            END AS culture_test_status FROM test_results
        """)
        connection.execute("""
            CREATE TEMP TABLE culture_audit AS
            WITH deduplicated AS (
                SELECT stay_id, micro_specimen_id, culture_time, culture_time_is_date_only,
                    spec_itemid, spec_type_desc,
                    LIST(DISTINCT test_itemid ORDER BY test_itemid) AS diagnostic_test_itemids,
                    LIST(DISTINCT normalized_test_name ORDER BY normalized_test_name) AS diagnostic_test_names,
                    'diagnostic_culture' AS culture_evidence_status
                FROM culture_test_audit WHERE culture_test_status = 'diagnostic_culture'
                GROUP BY stay_id, micro_specimen_id, culture_time, culture_time_is_date_only, spec_itemid, spec_type_desc
            )
            SELECT *, CASE WHEN culture_time_is_date_only THEN culture_time + INTERVAL 1 DAY
                ELSE culture_time END AS culture_time_upper_bound,
                culture_time_is_date_only AS culture_time_upper_exclusive,
                ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY culture_time NULLS LAST,
                    culture_time_is_date_only, micro_specimen_id NULLS LAST, spec_type_desc NULLS LAST) AS culture_event_id
            FROM deduplicated
        """)
        connection.execute("""
            CREATE TEMP TABLE coupled AS
            WITH matched AS (
            SELECT base.*, cultures.* EXCLUDE (stay_id),
                   starts.antimicrobial_start_id, starts.starttime AS abx_start_time,
                   starts.stoptime AS abx_stop_time, starts.drug AS abx_drug, starts.route AS abx_route,
                   episodes.episode_id, episodes.abx_start_time AS abx_episode_start_time,
                   episodes.abx_stop_time AS abx_episode_stop_time,
                   episodes.abx_known_duration_hours, episodes.abx_duration_complete, episodes.abx_duration_hours,
                   GREATEST(cultures.culture_time, starts.starttime - INTERVAL 72 HOUR) AS paired_culture_lower_bound,
                   LEAST(cultures.culture_time_upper_bound, starts.starttime + INTERVAL 24 HOUR) AS paired_culture_upper_bound,
                   cultures.culture_time_upper_exclusive
                       AND cultures.culture_time_upper_bound <= starts.starttime + INTERVAL 24 HOUR
                       AS paired_culture_upper_exclusive,
                   cultures.culture_time >= starts.starttime - INTERVAL 72 HOUR
                       AND cultures.culture_time_upper_bound <= starts.starttime + INTERVAL 24 HOUR
                       AS pairing_is_definite
            FROM base
            INNER JOIN antimicrobial_starts starts ON base.stay_id = starts.stay_id
            INNER JOIN culture_audit cultures ON base.stay_id = cultures.stay_id
                AND cultures.culture_time <= starts.starttime + INTERVAL 24 HOUR
                AND cultures.culture_time_upper_bound >= starts.starttime - INTERVAL 72 HOUR
            INNER JOIN episodes ON starts.stay_id = episodes.stay_id AND starts.episode_id = episodes.episode_id
            WHERE cultures.culture_time IS NOT NULL
            )
            SELECT *, LEAST(paired_culture_lower_bound, abx_start_time) AS suspected_infection_time,
                LEAST(paired_culture_upper_bound, abx_start_time) AS suspected_infection_time_upper_bound,
                paired_culture_upper_exclusive AND paired_culture_upper_bound <= abx_start_time
                    AS suspected_infection_time_upper_exclusive
            FROM matched WHERE paired_culture_lower_bound < paired_culture_upper_bound
                OR (paired_culture_lower_bound = paired_culture_upper_bound AND NOT paired_culture_upper_exclusive)
        """)
        connection.execute("""
            CREATE TEMP TABLE candidates AS
            WITH overlapping AS (
                SELECT *, suspected_infection_time >= icu_intime - INTERVAL 24 HOUR
                    AND suspected_infection_time_upper_bound <= icu_intime + INTERVAL 24 HOUR
                    AS presentation_is_definite,
                    GREATEST(suspected_infection_time, icu_intime - INTERVAL 24 HOUR) AS eligible_sit_lower_bound,
                    LEAST(suspected_infection_time_upper_bound, icu_intime + INTERVAL 24 HOUR) AS eligible_sit_upper_bound,
                    suspected_infection_time_upper_exclusive
                        AND suspected_infection_time_upper_bound <= icu_intime + INTERVAL 24 HOUR
                        AS eligible_sit_upper_exclusive
                FROM coupled
                WHERE (suspected_infection_time_upper_bound > icu_intime - INTERVAL 24 HOUR
                    OR (suspected_infection_time_upper_bound = icu_intime - INTERVAL 24 HOUR
                        AND NOT suspected_infection_time_upper_exclusive))
                    AND suspected_infection_time <= icu_intime + INTERVAL 24 HOUR
            )
            SELECT *, CASE WHEN pairing_is_definite AND presentation_is_definite
                THEN 'definite' ELSE 'possible' END AS candidate_timing_status,
                NOT (pairing_is_definite AND presentation_is_definite) AS requires_timing_adjudication,
                '2.1.0' AS infection_schema_version, ROW_NUMBER() OVER (
                PARTITION BY stay_id ORDER BY suspected_infection_time,
                    culture_time_is_date_only, culture_time, abx_start_time,
                    abx_stop_time NULLS LAST, episode_id, micro_specimen_id NULLS LAST,
                    spec_type_desc NULLS LAST, antimicrobial_start_id, culture_event_id
            ) AS infection_candidate_rank
            FROM overlapping
        """)
        invalid = connection.execute("""
            SELECT COUNT(*) FROM candidates WHERE abx_start_time < abx_episode_start_time
                OR paired_culture_lower_bound < abx_start_time - INTERVAL 72 HOUR
                OR paired_culture_upper_bound > abx_start_time + INTERVAL 24 HOUR
                OR suspected_infection_time > suspected_infection_time_upper_bound
                OR eligible_sit_lower_bound > eligible_sit_upper_bound
                OR (eligible_sit_lower_bound = eligible_sit_upper_bound AND eligible_sit_upper_exclusive)
                OR (suspected_infection_time = suspected_infection_time_upper_bound AND suspected_infection_time_upper_exclusive)
                OR (NOT culture_time_is_date_only AND requires_timing_adjudication)
        """).fetchone()[0]
        if invalid:
            raise AssertionError(f"Invalid infection timing intervals: {invalid}")
        stage_queries = (
            ("base_cohort", "SELECT COUNT(*) FROM base"),
            ("with_iv_antimicrobial", "SELECT COUNT(DISTINCT stay_id) FROM episodes"),
            ("also_with_dated_culture", "SELECT COUNT(DISTINCT episodes.stay_id) FROM episodes JOIN culture_audit USING (stay_id) WHERE culture_time IS NOT NULL"),
            ("with_temporal_pair", "SELECT COUNT(DISTINCT stay_id) FROM coupled"),
            ("within_icu_presentation_window", "SELECT COUNT(DISTINCT stay_id) FROM candidates"),
        )
        attrition = []
        previous = base_count
        for name, query in stage_queries:
            remaining = connection.execute(query).fetchone()[0]
            if remaining > previous:
                raise ValueError("Infection attrition is not monotonic")
            attrition.append({"step": name, "remaining": remaining, "removed": previous - remaining})
            previous = remaining
        summary = {
            "qualifying_prescription_starts": connection.execute("SELECT COUNT(*) FROM antimicrobial_starts").fetchone()[0],
            "antimicrobial_episodes": connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "candidate_pairs": connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            "candidates_using_noninitial_prescription_start": connection.execute("SELECT COUNT(*) FROM candidates WHERE abx_start_time > abx_episode_start_time").fetchone()[0],
            "dated_date_only_culture_records": connection.execute("SELECT COUNT(*) FROM culture_audit WHERE culture_time_is_date_only AND culture_time IS NOT NULL").fetchone()[0],
            "date_only_temporal_pairs": connection.execute("SELECT COUNT(*) FROM coupled WHERE culture_time_is_date_only").fetchone()[0],
            "date_only_candidate_pairs": connection.execute("SELECT COUNT(*) FROM candidates WHERE culture_time_is_date_only").fetchone()[0],
            "possible_pairing_candidates": connection.execute("SELECT COUNT(*) FROM candidates WHERE NOT pairing_is_definite").fetchone()[0],
            "possible_presentation_candidates": connection.execute("SELECT COUNT(*) FROM candidates WHERE NOT presentation_is_definite").fetchone()[0],
            "possible_timing_only_stays": connection.execute("SELECT COUNT(*) FROM (SELECT stay_id FROM candidates GROUP BY stay_id HAVING NOT BOOL_OR(candidate_timing_status = 'definite'))").fetchone()[0],
            "stays_with_multiple_candidates": connection.execute("SELECT COUNT(*) FROM (SELECT stay_id FROM candidates GROUP BY stay_id HAVING COUNT(*) > 1)").fetchone()[0],
            "selected_date_only_cultures": connection.execute("SELECT COUNT(*) FROM candidates WHERE infection_candidate_rank = 1 AND culture_time_is_date_only").fetchone()[0],
            "selected_incomplete_abx_duration": connection.execute("SELECT COUNT(*) FROM candidates WHERE infection_candidate_rank = 1 AND NOT abx_duration_complete").fetchone()[0],
            "selected_known_abx_under_72h": connection.execute("SELECT COUNT(*) FROM candidates WHERE infection_candidate_rank = 1 AND abx_duration_hours < 72").fetchone()[0],
            "cultures_without_date_or_time": connection.execute("SELECT COUNT(*) FROM culture_audit WHERE culture_time IS NULL").fetchone()[0],
        }
        prescription_counts = dict(connection.execute(
            "SELECT prescription_status, COUNT(*) FROM prescription_audit GROUP BY prescription_status"
        ).fetchall())
        culture_counts = dict(connection.execute(
            "SELECT culture_test_status, COUNT(*) FROM culture_test_audit GROUP BY culture_test_status"
        ).fetchall())
        if not previous:
            raise ValueError("No eligible infection stays; inspect the diagnostic culture mapping")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "infection_cohort.parquet": "SELECT * FROM candidates WHERE infection_candidate_rank = 1 ORDER BY stay_id",
            "infection_candidates.parquet": "SELECT * FROM candidates ORDER BY stay_id, infection_candidate_rank",
            "culture_test_evidence.parquet": "SELECT * FROM culture_test_audit ORDER BY stay_id, micro_specimen_id, test_itemid",
        }
        for filename, query in outputs.items():
            connection.execute(f"COPY ({query}) TO {sql_string((processed_dir / filename).as_posix())} (FORMAT PARQUET)")
        connection.execute(f"""
            COPY (
                SELECT test_itemid, test_name, spec_type_desc, culture_test_status,
                    COUNT(*) AS tests, COUNT(DISTINCT stay_id) AS stays
                FROM culture_test_audit GROUP BY ALL ORDER BY culture_test_status, tests DESC
            ) TO {sql_string((metrics_dir / 'culture_qc.csv').as_posix())} (HEADER, DELIMITER ',')
        """)
        connection.execute(f"""
            COPY (
                SELECT drug, route, prescription_status, COUNT(*) AS prescription_rows
                FROM prescription_audit GROUP BY drug, route, prescription_status
                ORDER BY prescription_status, prescription_rows DESC, drug, route
            ) TO {sql_string((metrics_dir / 'antimicrobial_qc.csv').as_posix())} (HEADER, DELIMITER ',')
        """)
    report = {
        "dataset_version": DATASET_VERSION,
        "infection_schema_version": INFECTION_SCHEMA_VERSION,
        "source_database": "MIMIC-IV", "source_version": "3.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "culture_policy": culture_policy,
        "culture_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "culture_test_status_counts": culture_counts,
        "policy": {
            "evidence": "IV antibacterial and antifungal prescriptions; administration not verified",
            "antimicrobial_map_version": ANTIMICROBIAL_MAP_VERSION,
            "antimicrobial_names": list(ANTIMICROBIAL_NAMES), "iv_routes": list(IV_ROUTES),
            "episode_gap_hours": 24, "minimum_abx_duration_hours": None,
            "pairing_anchor": "individual qualifying deduplicated prescription start; episodes are metadata only",
            "duration_fields": "abx_start_time/abx_stop_time describe the matched prescription; abx_duration* and abx_known_duration_hours describe its merged episode",
            "culture_before_abx_hours": 72, "culture_after_abx_hours": 24,
            "icu_presentation_window_hours": [-24, 24],
            "date_only_cultures": "[midnight,next midnight) intersected with the closed [prescription start-72h,start+24h] culture window; preserve endpoint exclusivity",
            "presentation_retention": "nonempty SIT interval intersection with closed ICU admission +/-24h; possible timing is not definite eligibility",
            "sit_bounds": "suspected_infection_time* condition on valid pairing only; eligible_sit_* additionally condition on presentation-window overlap",
            "timing_certainty": "pairing_is_definite covers the original culture interval; presentation_is_definite covers the pairing-conditioned SIT interval; both required for definite candidate timing",
            "selection": "earliest candidate provisional; all candidates retained for Sepsis-3 adjudication",
        },
        "attrition": attrition, "summary": summary, "prescription_status_counts": prescription_counts,
    }
    report_path = metrics_dir / "02_infection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[02] Complete v{DATASET_VERSION}; {time.perf_counter() - started:.1f}s; report: {metrics_dir}/02_infection.json", flush=True)
    return report


if __name__ == "__main__":
    print(f"[02] Identify suspected infection from IV prescriptions and diagnostic cultures. v{DATASET_VERSION}", flush=True)
    build_infection_cohort()
