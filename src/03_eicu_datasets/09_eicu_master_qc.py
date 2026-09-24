"""Audit the eICU processing outputs and replay the selected imputation recipe."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

ROOT = BASE_DIR = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/eicu"
PROCESSED_DIR = DATA / "work"
OUT = METRICS_DIR = ROOT / "outputs/eicu"
COMMON_DIR = ROOT / "src/common"
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
AUDIT_VERSION = "1.0.0"
ORGANS = ("resp", "coag", "liver", "cv", "cns", "renal")
OBSERVED_ONLY = set(TENSOR_SCHEMA["observed_only_features"])
SCENARIOS = ("point", "block6h", "whole_channel")
RTOL, ATOL = 1e-5, 1e-6


def artifact_checks(qc, state):
    manifest = read(DATA / "manifest.json")
    observed = manifest["observed"]
    imputation = manifest["imputation"]
    qc.check('source_identity', manifest.get('source_database') == CONFIG['source_database']
             and manifest.get('source_version') == CONFIG['source_version'])
    qc.check("shared_schema_version", manifest.get("shared_schema_version") == TENSOR_SCHEMA["schema_version"])
    qc.check("dataset_version", manifest["dataset_version"] == DATASET_VERSION and observed["dataset_version"] == DATASET_VERSION
             and imputation["dataset_version"] == DATASET_VERSION)
    qc.check("completed_imputation", imputation.get("status") == "COMPLETE")
    for name, expected in observed["output_manifest"].items():
        path = (DATA / name).resolve()
        qc.check(f"observed_artifact/{name}", path.is_relative_to(DATA) and path.is_file() and
                 path.stat().st_size == expected["size_bytes"] and digest(path) == expected["sha256"])
    for group, entries in (("processing", observed["provenance"]), ("imputation", imputation["artifacts"])):
        for name, expected in entries.items():
            path = (ROOT / name).resolve()
            qc.check(f"{group}/{name}", path.is_relative_to(ROOT) and path.is_file() and
                     path.stat().st_size == expected["size_bytes"] and digest(path) == expected["sha256"])
    for name, expected in observed["input_manifest"].items():
        path = (ROOT / expected["path"]).resolve()
        qc.check(f"observed_input/{name}", path.is_relative_to(ROOT) and path.is_file() and digest(path) == expected["sha256"])
    script = Path(__file__).with_name("08_eicu_saits_imputation.py")
    qc.check("imputation_code_binding", digest(script) == imputation["script_sha256"])
    if any(not row["passed"] for row in qc.checks):
        raise ValueError("Missing or changed processing artifacts; rerun the affected stage")
    core = module(script, "eicu_imputation")
    arrays, policy, bounds, inputs = core.load_inputs(DATA, COMMON_DIR)
    report = read(OUT / "saits.json")
    with np.load(DATA / "imputation_support.npz", allow_pickle=False) as support:
        codes, rejected, partition = support["method_codes"], support["saits_out_of_bounds_fallback_mask"], support["partition"]
    with np.load(OUT / "holdouts.npz", allow_pickle=False) as saved:
        holdouts = {name: saved[name] for name in saved.files}
    state.update(manifest=manifest, tensor=observed, source=DATA, data=PROCESSED_DIR, common=COMMON_DIR, metrics=OUT,
        core=core, recipe=core, arrays=arrays, raw=arrays["raw"], bounds=bounds, inputs=inputs,
        output=array(DATA / "tensor_imputed.npy"), codes=codes, rejected=rejected, partition=partition,
        config=report["configuration"], methods=report["feature_methods"], saits_report=report, saved_holdouts=holdouts)
    qc.check("configuration_input_hashes", report["configuration"]["input_hashes"] == inputs)
    qc.check("configuration_policy", report["configuration"]["policy"] == policy)
    qc.check("recipe_binding", hashlib.sha256(json.dumps(state["methods"], sort_keys=True).encode()).hexdigest() == imputation["feature_recipe_sha256"])
    qc.check("imputation_acceptance", bool(report["acceptance_checks"]) and all(report["acceptance_checks"].values()))
    qc.notes.append(dict(patients=len(arrays["raw"]), temporal_features=arrays["raw"].shape[2],
        patients_without_observed_values=int((~np.isfinite(arrays["raw"]).any(axis=(1, 2))).sum()),
        incomplete_followup_patients=int((arrays["exposure_seconds"].sum(axis=1) < 86400).sum())))


def observed_replay(qc, state):
    builder = module(Path(__file__).with_name("07_eicu_tensor_builder.py"), "eicu_tensor_builder")
    with tempfile.TemporaryDirectory(prefix=".tensor_qc_", dir=OUT) as temporary:
        temporary = Path(temporary)
        replay = builder.build_tensor(processed_dir=PROCESSED_DIR, common_dir=COMMON_DIR,
            metrics_dir=temporary / "metrics", source_metrics_dir=OUT, output_dir=temporary / "data")
        left, right = array(temporary / "data/tensor_observed.npy"), state["raw"]
        qc.check("replayed_observed_tensor", left.shape == right.shape and left.dtype == right.dtype and
                 np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True))
        del left
        with np.load(temporary / "data/tensor_support.npz", allow_pickle=False) as left, np.load(DATA / "tensor_support.npz", allow_pickle=False) as right:
            qc.check("replayed_support_keys", set(left.files) == set(right.files))
            for name in left.files:
                a, b = left[name], right[name]
                matched = a.shape == b.shape and a.dtype == b.dtype and (np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
                    if a.dtype.kind in "fc" else equal(a, b))
                qc.check(f"replayed_support/{name}", matched)
        qc.check("replayed_summary", replay["summary"] == state["tensor"]["summary"])
        qc.check("replayed_acceptance", bool(replay["acceptance_checks"]) and all(replay["acceptance_checks"].values()))

COHORT_INPUTS = {'cleaned': 'events_clean.parquet', 'phenotype': 'phenotypes.parquet', 'candidates': 'infection_candidates.parquet', 'windows': 'extraction_windows.parquet', 'final': 'sepsis_cohort.parquet', 'candidate_audit': 'candidate_decisions.parquet', 'stay_audit': 'stay_decisions.parquet', 'timeline': 'sofa_timeline.parquet', 'respiratory': 'respiratory_evidence.parquet', 'gcs': 'gcs_evidence.parquet', 'pressors': 'pressor_intervals.parquet', 'urine': 'urine_evidence.parquet', 'eligibility': 'base_eligibility.parquet', 'base': 'base_cohort.parquet', 'prescriptions': 'prescription_evidence.parquet', 'cultures': 'culture_evidence.parquet'}


COHORT_INPUTS.update(infection_diagnoses='infection_diagnosis_evidence.parquet',
                     strict_candidates='strict_infection_candidates.parquet',
                     strict_decisions='strict_culture_decisions.parquet')


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalized_label_sql(column):
    return f"TRIM(REGEXP_REPLACE(UPPER(COALESCE({column}, '')), '\\s+', ' ', 'g'))"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_check(checks, name, violations, detail=None):
    checks.append({
        "name": name, "passed": violations == 0,
        "violations": int(violations), "detail": detail,
    })


def structural_checks(connection, checks):
    def check(name, query):
        add_check(checks, name, connection.execute(query).fetchone()[0])

    for table, keys in (
        ("phenotype", "stay_id"), ("phenotype", "subject_id"),
        ("final", "stay_id"), ("final", "subject_id"), ("windows", "stay_id"),
        ("stay_audit", "stay_id"), ("candidates", "stay_id, infection_candidate_rank"),
        ("candidate_audit", "stay_id, infection_candidate_rank"),
        ("timeline", "stay_id, assessment_time"), ("gcs", "stay_id, event_time"),
        ("urine", "stay_id, event_time"),
    ):
        nulls = " OR ".join(f"{key.strip()} IS NULL" for key in keys.split(","))
        check(f"{table}_{keys.replace(', ', '_')}_keys", f"""
            SELECT COUNT(*) FROM (
                SELECT {keys} FROM {table} GROUP BY {keys}
                HAVING COUNT(*) <> 1 OR {nulls}
            )
        """)
    check("candidate_membership", """
        SELECT COUNT(*) FROM candidates AS source FULL JOIN candidate_audit AS audited
            USING (stay_id, subject_id, hadm_id, infection_candidate_rank)
        WHERE source.stay_id IS NULL OR audited.stay_id IS NULL
            OR source.suspected_infection_time IS DISTINCT FROM audited.suspected_infection_time
            OR source.suspected_infection_time_upper_bound IS DISTINCT FROM audited.suspected_infection_time_upper_bound
            OR source.suspected_infection_time_upper_exclusive IS DISTINCT FROM audited.suspected_infection_time_upper_exclusive
            OR source.eligible_sit_lower_bound IS DISTINCT FROM audited.eligible_sit_lower_bound
            OR source.eligible_sit_upper_bound IS DISTINCT FROM audited.eligible_sit_upper_bound
            OR source.eligible_sit_upper_exclusive IS DISTINCT FROM audited.eligible_sit_upper_exclusive
            OR source.pairing_is_definite IS DISTINCT FROM audited.pairing_is_definite
            OR source.presentation_is_definite IS DISTINCT FROM audited.presentation_is_definite
            OR source.requires_timing_adjudication IS DISTINCT FROM audited.requires_timing_adjudication
            OR source.candidate_timing_status IS DISTINCT FROM audited.candidate_timing_status
            OR source.infection_schema_version IS DISTINCT FROM audited.infection_schema_version
    """)
    check("infection_timing_contract", """
        SELECT COUNT(*) FROM candidates WHERE infection_schema_version IS DISTINCT FROM '2.1.0'
            OR infection_evidence_type IS DISTINCT FROM 'documented_infection_proxy'
            OR culture_evidence_status IS DISTINCT FROM 'not_required_for_documented_infection_proxy'
            OR diagnostic_evidence_names IS NULL OR LEN(diagnostic_evidence_names) = 0
            OR pairing_is_definite IS NULL OR presentation_is_definite IS NULL
            OR suspected_infection_time_upper_exclusive IS NULL OR eligible_sit_upper_exclusive IS NULL
            OR requires_timing_adjudication IS DISTINCT FROM NOT (pairing_is_definite AND presentation_is_definite)
            OR candidate_timing_status IS DISTINCT FROM
                CASE WHEN pairing_is_definite AND presentation_is_definite THEN 'definite' ELSE 'possible' END
            OR presentation_is_definite IS DISTINCT FROM
                (suspected_infection_time >= icu_intime - INTERVAL 24 HOUR
                    AND suspected_infection_time_upper_bound <= icu_intime + INTERVAL 24 HOUR)
            OR eligible_sit_lower_bound IS DISTINCT FROM GREATEST(suspected_infection_time, icu_intime - INTERVAL 24 HOUR)
            OR eligible_sit_upper_bound IS DISTINCT FROM LEAST(suspected_infection_time_upper_bound, icu_intime + INTERVAL 24 HOUR)
            OR eligible_sit_upper_exclusive IS DISTINCT FROM
                (suspected_infection_time_upper_exclusive AND suspected_infection_time_upper_bound <= icu_intime + INTERVAL 24 HOUR)
            OR eligible_sit_lower_bound > eligible_sit_upper_bound
            OR (eligible_sit_lower_bound = eligible_sit_upper_bound AND eligible_sit_upper_exclusive)
            OR (suspected_infection_time = suspected_infection_time_upper_bound AND suspected_infection_time_upper_exclusive)
    """)
    check("stay_audit_membership", """
        SELECT COUNT(*) FROM phenotype AS source FULL JOIN stay_audit AS audited USING (stay_id)
        WHERE source.stay_id IS NULL OR audited.stay_id IS NULL
    """)
    check("cohort_linkage", """
        SELECT COUNT(*) FROM final AS selected LEFT JOIN phenotype AS source
            USING (stay_id, subject_id, hadm_id)
        WHERE source.stay_id IS NULL
    """)
    check("cleaned_contract", """
        SELECT COUNT(*) FROM cleaned AS event LEFT JOIN feature_rules USING (feature)
        LEFT JOIN phenotype AS patient USING (stay_id, subject_id, hadm_id)
        WHERE patient.stay_id IS NULL OR feature_rules.feature IS NULL
            OR cleaning_schema_version IS DISTINCT FROM '2.0.0'
            OR evidence_usable IS NULL
            OR numeric_usable IS DISTINCT FROM (valuenum IS NOT NULL)
            OR (numeric_usable AND (
                unit IS NULL OR feature_rules.bound_min IS NULL OR feature_rules.bound_max IS NULL
                OR NOT ISFINITE(valuenum) OR valuenum < feature_rules.bound_min OR valuenum > feature_rules.bound_max
                OR valueuom IS DISTINCT FROM unit OR canonical_unit IS DISTINCT FROM unit
                OR qc_status IS DISTINCT FROM 'accepted'))
            OR (evidence_usable AND (valuenum IS NOT NULL OR qc_status IS DISTINCT FROM 'evidence_only'))
    """)
    score_range = " OR ".join(
        f"(sofa_{organ} IS NOT NULL AND (sofa_{organ} NOT BETWEEN 0 AND 4 "
        f"OR sofa_{organ} <> FLOOR(sofa_{organ})))" for organ in ORGANS
    )
    total = " + ".join(f"COALESCE(sofa_{organ}, 0)" for organ in ORGANS)
    observed = " + ".join(f"CAST(sofa_{organ} IS NOT NULL AS INTEGER)" for organ in ORGANS)
    for table in ("timeline", "final"):
        check(f"{table}_score_arithmetic", f"""
            SELECT COUNT(*) FROM {table} WHERE {score_range}
                OR total_sofa IS DISTINCT FROM ({total})
                OR observed_components IS DISTINCT FROM ({observed})
        """)
    check("timeline_temporal_contract", """
        SELECT COUNT(*) FROM timeline AS assessment LEFT JOIN windows AS envelope USING (stay_id)
        WHERE envelope.stay_id IS NULL
            OR assessment.assessment_time NOT BETWEEN envelope.window_start AND envelope.window_end
            OR assessment.icu_intime IS DISTINCT FROM envelope.icu_intime
            OR assessment.icu_outtime IS DISTINCT FROM envelope.icu_outtime
            OR assessment.care_end IS DISTINCT FROM LEAST(envelope.icu_outtime, envelope.hospital_deathtime, envelope.window_end)
            OR assessment.in_icu_assessment IS DISTINCT FROM
                (assessment.assessment_time BETWEEN envelope.icu_intime AND assessment.care_end)
            OR assessment.lookback_start_exclusive IS DISTINCT FROM assessment.assessment_time - INTERVAL 24 HOUR
            OR assessment.full_icu_lookback IS DISTINCT FROM
                (assessment.assessment_time >= envelope.icu_intime + INTERVAL 24 HOUR)
    """)


def evidence_checks(connection, checks, policy):
    def check(name, query):
        add_check(checks, name, connection.execute(query).fetchone()[0])

    check("respiratory_pairing", f"""
        SELECT COUNT(*) FROM respiratory WHERE
            (pf_ratio IS NOT NULL AND (
                NOT ISFINITE(pf_ratio) OR pao2 IS NULL OR fio2_fraction IS NULL
                OR fio2_fraction NOT BETWEEN 0.2 AND 1
                OR ABS(pf_ratio - pao2 / fio2_fraction) > 0.000001
                OR fio2_pair_source NOT IN ('same_timestamp_group', 'preceding_chart')))
            OR (fio2_pair_source = 'same_timestamp_group' AND (
                specimen_id IS NULL OR specimen_fio2_rows <= 0
                OR fio2_fraction IS DISTINCT FROM specimen_fio2))
            OR (fio2_pair_source = 'preceding_chart' AND (
                chart_fio2_time IS NULL OR chart_fio2_time > event_time
                OR chart_fio2_time < event_time - INTERVAL {policy["fio2_lookback_hours"]} HOUR
                OR fio2_fraction IS DISTINCT FROM chart_fio2))
            OR (fio2_pair_source = 'unpaired' AND (fio2_fraction IS NOT NULL OR pf_ratio IS NOT NULL))
    """)
    check("respiratory_thresholds", """
        SELECT COUNT(*) FROM respiratory WHERE sofa_resp IS DISTINCT FROM
            CASE WHEN pf_ratio IS NULL THEN NULL
                WHEN support_status IN ('invasive_procedure', 'invasive_airway_and_mode') AND pf_ratio < 100 THEN 4
                WHEN support_status IN ('invasive_procedure', 'invasive_airway_and_mode') AND pf_ratio < 200 THEN 3
                WHEN pf_ratio < 300 THEN 2 WHEN pf_ratio < 400 THEN 1 ELSE 0 END
    """)
    check("gcs_assessments", """
        WITH components AS (
            SELECT stay_id, event_time, feature,
                CASE WHEN COUNT(*) = COUNT(valuenum) AND MIN(valuenum) = MAX(valuenum)
                    THEN MIN(valuenum) END AS value
            FROM cleaned WHERE feature IN ('gcs_eye', 'gcs_verbal', 'gcs_motor')
                AND record_qc_status = 'accepted'
            GROUP BY stay_id, event_time, feature
        ), expected AS (
            SELECT stay_id, event_time, COUNT(value) AS observed_components,
                CASE WHEN COUNT(value) = 3 THEN SUM(value) END AS gcs_total
            FROM components GROUP BY stay_id, event_time
        )
        SELECT COUNT(*) FROM expected FULL JOIN gcs USING (stay_id, event_time)
        WHERE expected.stay_id IS NULL OR gcs.stay_id IS NULL
            OR expected.gcs_total IS DISTINCT FROM gcs.gcs_total
            OR expected.observed_components IS DISTINCT FROM gcs.observed_components
    """)
    check("pressor_duration", f"""
        SELECT COUNT(*) FROM pressors WHERE start_time IS NULL OR end_time IS NULL OR qualifying_time IS NULL
            OR feature NOT IN ('dopamine', 'dobutamine', 'epinephrine', 'norepinephrine')
            OR sofa_cv NOT IN (2, 3, 4)
            OR qualifying_time IS DISTINCT FROM start_time + INTERVAL {policy["minimum_vasoactive_duration_minutes"]} MINUTE
            OR qualifying_time > end_time
    """)
    check("urine_evidence", """
        SELECT COUNT(*) FROM urine WHERE urine_ml IS DISTINCT FROM
            CASE WHEN invalid_items > 0 THEN NULL
                WHEN irrigant_in_items = 0 AND irrigant_out_items = 0 THEN ordinary_volume
                WHEN irrigant_in_items > 0 AND irrigant_out_items > 0 AND ordinary_items = 0
                    AND net_irrigant_volume >= 0 THEN net_irrigant_volume
                ELSE NULL END
    """)
    check("urine_coverage_gate", f"""
        SELECT COUNT(*) FROM timeline WHERE urine_coverage_adequate IS DISTINCT FROM COALESCE(
            assessment_time >= icu_intime + INTERVAL 24 HOUR AND invalid_urine_times = 0
            AND last_urine_time - first_urine_time >= INTERVAL {policy["urine_minimum_span_hours"]} HOUR
            AND first_urine_time <= assessment_time - INTERVAL 24 HOUR
                + INTERVAL {policy["urine_boundary_tolerance_hours"]} HOUR
            AND last_urine_time >= assessment_time - INTERVAL {policy["urine_boundary_tolerance_hours"]} HOUR
            AND maximum_internal_gap_hours <= {policy["urine_maximum_gap_hours"]}, FALSE)
            OR urine_24h_ml IS DISTINCT FROM CASE WHEN urine_coverage_adequate THEN observed_urine_ml END
            OR sofa_renal_urine IS DISTINCT FROM CASE WHEN NOT urine_coverage_adequate THEN NULL
                WHEN observed_urine_ml < 200 THEN 4 WHEN observed_urine_ml < 500 THEN 3 ELSE 0 END
            OR sofa_renal IS DISTINCT FROM GREATEST(sofa_renal_creatinine, sofa_renal_urine)
            OR sofa_cv IS DISTINCT FROM GREATEST(sofa_cv_map, sofa_cv_drug)
    """)


def rolling_checks(connection, checks):
    connection.execute("""
        CREATE TEMP TABLE expected_points AS
        SELECT stay_id, event_time,
            MAX(score) FILTER (WHERE organ = 'coag') AS coag,
            MAX(score) FILTER (WHERE organ = 'liver') AS liver,
            MAX(score) FILTER (WHERE organ = 'renal_creatinine') AS renal_creatinine,
            MAX(score) FILTER (WHERE organ = 'cv_map') AS cv_map,
            MAX(score) FILTER (WHERE organ = 'cns') AS cns,
            MAX(score) FILTER (WHERE organ = 'resp') AS resp
        FROM (
            SELECT stay_id, event_time,
                CASE feature WHEN 'platelets' THEN 'coag' WHEN 'bilirubin' THEN 'liver'
                    WHEN 'creatinine' THEN 'renal_creatinine' ELSE 'cv_map' END AS organ,
                CASE feature
                    WHEN 'platelets' THEN CASE WHEN valuenum < 20 THEN 4 WHEN valuenum < 50 THEN 3
                        WHEN valuenum < 100 THEN 2 WHEN valuenum < 150 THEN 1 ELSE 0 END
                    WHEN 'bilirubin' THEN CASE WHEN valuenum >= 12 THEN 4 WHEN valuenum >= 6 THEN 3
                        WHEN valuenum >= 2 THEN 2 WHEN valuenum >= 1.2 THEN 1 ELSE 0 END
                    WHEN 'creatinine' THEN CASE WHEN valuenum >= 5 THEN 4 WHEN valuenum >= 3.5 THEN 3
                        WHEN valuenum >= 2 THEN 2 WHEN valuenum >= 1.2 THEN 1 ELSE 0 END
                    ELSE CASE WHEN valuenum < 70 THEN 1 ELSE 0 END END AS score
            FROM cleaned WHERE numeric_usable AND feature IN ('platelets', 'bilirubin', 'creatinine', 'map')
            UNION ALL
            SELECT stay_id, event_time, 'resp', sofa_resp FROM respiratory WHERE sofa_resp IS NOT NULL
            UNION ALL
            SELECT stay_id, event_time, 'cns',
                CASE WHEN gcs_total < 6 THEN 4 WHEN gcs_total < 10 THEN 3
                    WHEN gcs_total < 13 THEN 2 WHEN gcs_total < 15 THEN 1 ELSE 0 END
            FROM gcs WHERE gcs_total IS NOT NULL
        ) AS evidence GROUP BY stay_id, event_time
    """)
    add_check(checks, "scoring_evidence_assessment_times", connection.execute("""
        SELECT COUNT(*) FROM expected_points AS point LEFT JOIN timeline AS assessment
            ON point.stay_id = assessment.stay_id AND point.event_time = assessment.assessment_time
        WHERE assessment.stay_id IS NULL
    """).fetchone()[0])
    names = ("coag", "liver", "renal_creatinine", "cv_map", "cns", "resp")
    expected = ", ".join(f"MAX(point.{name}) OVER lookback AS expected_{name}" for name in names)
    observed = ", ".join(f"assessment.sofa_{name}" for name in names)
    connection.execute(f"""
        CREATE TEMP TABLE expected_rolling AS
        SELECT assessment.stay_id, assessment.assessment_time, {observed}, {expected}
        FROM timeline AS assessment LEFT JOIN expected_points AS point
            ON point.stay_id = assessment.stay_id AND point.event_time = assessment.assessment_time
        WINDOW lookback AS (PARTITION BY assessment.stay_id ORDER BY assessment.assessment_time
            RANGE BETWEEN INTERVAL '86399999999 MICROSECONDS' PRECEDING AND CURRENT ROW)
    """)
    aggregates = ", ".join(
        f"COUNT(*) FILTER (WHERE sofa_{name} IS DISTINCT FROM expected_{name})" for name in names
    )
    for name, count in zip(names, connection.execute(f"SELECT {aggregates} FROM expected_rolling").fetchone()):
        add_check(checks, f"rolling_{name}", count)
    add_check(checks, "rolling_pressor_scores", connection.execute("""
        SELECT COUNT(*) FROM (
            SELECT assessment.stay_id, assessment.assessment_time, assessment.sofa_cv_drug,
                MAX(pressors.sofa_cv) AS expected
            FROM timeline AS assessment LEFT JOIN pressors
                ON pressors.stay_id = assessment.stay_id
                AND pressors.qualifying_time <= assessment.assessment_time
                AND pressors.end_time > assessment.assessment_time - INTERVAL 24 HOUR
            GROUP BY assessment.stay_id, assessment.assessment_time, assessment.sofa_cv_drug
        ) AS compared WHERE sofa_cv_drug IS DISTINCT FROM expected
    """).fetchone()[0])


def selection_checks(connection, checks):
    timing_keys = ("stay_id, suspected_infection_time, suspected_infection_time_upper_bound, "
                   "suspected_infection_time_upper_exclusive, eligible_sit_lower_bound, "
                   "eligible_sit_upper_bound, eligible_sit_upper_exclusive")
    group_keys = ", ".join(f"source.{name.strip()}" for name in timing_keys.split(","))
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE expected_timing_scores AS
            SELECT {group_keys},
                COUNT(assessment.assessment_time) FILTER (WHERE assessment.in_icu_assessment) AS assessments,
                COUNT(assessment.assessment_time) FILTER (
                    WHERE assessment.in_icu_assessment AND assessment.observed_components > 0) AS observed_assessments,
                MIN(assessment.assessment_time) FILTER (
                    WHERE assessment.in_icu_assessment AND assessment.total_sofa >= 2
                        AND assessment.assessment_time >= source.eligible_sit_lower_bound - INTERVAL 48 HOUR
                        AND (assessment.assessment_time < source.eligible_sit_upper_bound + INTERVAL 24 HOUR
                            OR (assessment.assessment_time = source.eligible_sit_upper_bound + INTERVAL 24 HOUR
                                AND NOT source.eligible_sit_upper_exclusive))) AS possible_time,
                MIN(assessment.assessment_time) FILTER (
                    WHERE assessment.in_icu_assessment AND assessment.total_sofa >= 2
                        AND assessment.assessment_time >= source.suspected_infection_time_upper_bound - INTERVAL 48 HOUR
                        AND assessment.assessment_time <= source.suspected_infection_time + INTERVAL 24 HOUR) AS qualifying_time
            FROM (SELECT DISTINCT {timing_keys} FROM candidates) AS source LEFT JOIN timeline AS assessment
                ON source.stay_id = assessment.stay_id
                AND assessment.assessment_time BETWEEN source.suspected_infection_time - INTERVAL 48 HOUR
                    AND source.suspected_infection_time_upper_bound + INTERVAL 24 HOUR
                AND (NOT source.suspected_infection_time_upper_exclusive
                    OR assessment.assessment_time < source.suspected_infection_time_upper_bound + INTERVAL 24 HOUR)
                AND assessment.assessment_time <= assessment.care_end
            GROUP BY {group_keys}
    """)
    connection.execute(f"""
        CREATE TEMP TABLE expected_candidates AS
        SELECT source.stay_id, source.infection_candidate_rank, scores.* EXCLUDE ({timing_keys}), CASE
            WHEN assessments = 0 THEN 'no_in_icu_assessment'
            WHEN observed_assessments = 0 THEN 'no_usable_sofa_evidence'
            WHEN source.requires_timing_adjudication THEN 'infection_timing_uncertain'
            WHEN qualifying_time IS NULL AND possible_time IS NOT NULL THEN 'date_uncertainty_only'
            WHEN qualifying_time IS NULL THEN 'sofa_below_two'
            WHEN GREATEST(qualifying_time, source.suspected_infection_time_upper_bound) >
                LEAST(source.icu_outtime, source.hospital_deathtime) THEN 'onset_outside_followup'
            ELSE 'qualifies' END AS expected_status
        FROM expected_timing_scores AS scores JOIN candidates AS source USING ({timing_keys})
    """)
    add_check(checks, "candidate_decisions", connection.execute("""
        SELECT COUNT(*) FROM expected_candidates AS expected
        JOIN candidate_audit AS audited USING (stay_id, infection_candidate_rank)
        WHERE expected.expected_status IS DISTINCT FROM audited.adjudication_status
            OR expected.qualifying_time IS DISTINCT FROM audited.first_qualifying_sofa_time
            OR expected.possible_time IS DISTINCT FROM audited.first_possible_sofa_time
    """).fetchone()[0])
    add_check(checks, "earliest_qualifying_candidate_selection", connection.execute("""
        WITH expected AS (
            SELECT stay_id, MIN(infection_candidate_rank) AS candidate_rank
            FROM expected_candidates WHERE expected_status = 'qualifies' GROUP BY stay_id
        )
        SELECT COUNT(*) FROM expected FULL JOIN final USING (stay_id)
        WHERE expected.stay_id IS NULL OR final.stay_id IS NULL
            OR expected.candidate_rank IS DISTINCT FROM final.infection_candidate_rank
    """).fetchone()[0])
    match_components = " OR ".join(
        f"final.sofa_{organ} IS DISTINCT FROM snapshot.sofa_{organ}" for organ in ORGANS
    )
    add_check(checks, "selected_score_and_onset", connection.execute(f"""
        SELECT COUNT(*) FROM final LEFT JOIN candidate_audit AS candidate
            USING (stay_id, subject_id, hadm_id, infection_candidate_rank)
        LEFT JOIN timeline AS snapshot ON final.stay_id = snapshot.stay_id
            AND final.sofa_time = snapshot.assessment_time
        WHERE candidate.stay_id IS NULL OR snapshot.stay_id IS NULL
            OR final.sofa_time IS DISTINCT FROM candidate.first_qualifying_sofa_time
            OR final.suspected_infection_time IS DISTINCT FROM candidate.suspected_infection_time
            OR final.cohort_definition IS DISTINCT FROM candidate.cohort_definition
            OR final.infection_time_is_proxy IS DISTINCT FROM TRUE
            OR final.clinical_infection_onset_known IS DISTINCT FROM FALSE
            OR final.suspected_infection_time_upper_bound IS DISTINCT FROM candidate.suspected_infection_time_upper_bound
            OR final.total_sofa IS DISTINCT FROM snapshot.total_sofa
            OR final.sofa_delta IS DISTINCT FROM snapshot.total_sofa
            OR final.observed_components IS DISTINCT FROM snapshot.observed_components
            OR final.baseline_sofa IS DISTINCT FROM 0
            OR final.baseline_sofa_assumed_zero IS DISTINCT FROM TRUE
            OR final.sepsis3 IS DISTINCT FROM TRUE OR final.total_sofa < 2
            OR final.requires_timing_adjudication IS DISTINCT FROM FALSE
            OR final.candidate_timing_status IS DISTINCT FROM 'definite'
            OR final.suspected_infection_time_upper_exclusive IS DISTINCT FROM candidate.suspected_infection_time_upper_exclusive
            OR final.sepsis_onset_time_upper_exclusive IS DISTINCT FROM
                (final.suspected_infection_time_upper_exclusive AND final.suspected_infection_time_upper_bound > final.sofa_time)
            OR final.sepsis_onset_time IS DISTINCT FROM GREATEST(final.suspected_infection_time, final.sofa_time)
            OR final.sepsis_onset_time_upper_bound IS DISTINCT FROM GREATEST(final.suspected_infection_time_upper_bound, final.sofa_time)
            OR final.followup_end IS DISTINCT FROM LEAST(final.icu_outtime, final.hospital_deathtime)
            OR final.sepsis_onset_time < final.icu_intime OR final.sepsis_onset_time_upper_bound > final.followup_end
            OR final.complete_24h_after_onset IS DISTINCT FROM
                (final.followup_end >= final.sepsis_onset_time_upper_bound + INTERVAL 24 HOUR)
            OR ABS(final.hours_available_after_onset - EPOCH(final.followup_end - final.sepsis_onset_time)/3600.0) > 0.000001
            OR {match_components}
    """).fetchone()[0])
    add_check(checks, "stay_dispositions", connection.execute("""
        SELECT COUNT(*) FROM stay_audit AS audited LEFT JOIN final USING (stay_id)
        WHERE audited.included IS DISTINCT FROM (final.stay_id IS NOT NULL)
            OR audited.selected_candidate_rank IS DISTINCT FROM final.infection_candidate_rank
            OR audited.sepsis_onset_time IS DISTINCT FROM final.sepsis_onset_time
            OR ((audited.reason = 'included') IS DISTINCT FROM audited.included)
    """).fetchone()[0])


def audit_cohort(processed_dir=PROCESSED_DIR, metrics_dir=METRICS_DIR, common_dir=COMMON_DIR,
                 adjudication_report=None):
    processed_dir, metrics_dir, common_dir = Path(processed_dir), Path(metrics_dir), Path(common_dir)
    report_input = Path(adjudication_report) if adjudication_report else metrics_dir / "06_sepsis.json"
    paths = {table: processed_dir / filename for table, filename in COHORT_INPUTS.items()}
    paths.update({
        "sepsis_report": report_input,
        "registry": common_dir / "feature_registry.json",
        "measurement_rules": common_dir / "measurement_rules.json",
        "sofa_policy": CONFIG_PATH,
        "infection_policy": CONFIG_PATH,
    })
    checks, manifest, summary, review = [], {}, {}, {}
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cohort_audit_", dir=metrics_dir) as temporary:
        staging = Path(temporary)
        try:
            for name, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Missing required input: {path}")
                manifest[name] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_hash(path)}
            previous = json.loads(report_input.read_text(encoding="utf-8"))
            policy = json.loads(paths["sofa_policy"].read_text(encoding="utf-8"))["sofa_policy"]
            registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
            rules = json.loads(paths["measurement_rules"].read_text(encoding="utf-8"))
            infection_policy = json.loads(paths["infection_policy"].read_text(encoding="utf-8"))["infection_policy"]
            add_check(checks, "sepsis_schema", int(previous.get("schema_version") != "2.1.0"))
            add_check(checks, "saved_policy_matches_current", int(previous.get("policy") != policy))
            if policy["schema_version"] != "1.1.0" or policy["rolling_hours"] != 24 or policy["association_hours"] != [-48, 24]:
                raise ValueError("Unsupported SOFA policy; audit must be reviewed before using changed definitions")
            with duckdb.connect() as connection:
                connection.execute("SET threads = 4")
                connection.execute("SET preserve_insertion_order = false")
                connection.execute(f"SET temp_directory = {sql_string((staging / 'spill').as_posix())}")
                for table, filename in COHORT_INPUTS.items():
                    connection.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet({sql_string((processed_dir / filename).as_posix())})")
                units = {entry["name"]: entry["canonical_unit"] for entry in registry["temporal_features"]}
                units.update(registry["evidence_features"])
                units.update(CONFIG["source_mapping"]["evidence_features"])
                connection.execute("CREATE TEMP TABLE feature_rules (feature VARCHAR, unit VARCHAR, bound_min DOUBLE, bound_max DOUBLE)")
                connection.executemany("INSERT INTO feature_rules VALUES (?, ?, ?, ?)", [
                    (name, unit, *rules["bounds"].get(name, [None, None])) for name, unit in units.items()
                ])
                structural_checks(connection, checks)
                connection.execute("CREATE TEMP TABLE allowed_cultures(spec_itemid INTEGER, name VARCHAR)")
                connection.executemany("INSERT INTO allowed_cultures VALUES (?,?)",[
                    (int(i),name) for i,name in infection_policy['diagnostic_specimens'].items()])
                add_check(checks,'diagnostic_specimen_mapping',connection.execute(f"""
                    SELECT COUNT(*) FROM strict_candidates c LEFT JOIN allowed_cultures a USING(spec_itemid)
                    WHERE a.spec_itemid IS NULL OR {normalized_label_sql('c.spec_type_desc')} = ''
                        OR {normalized_label_sql('c.spec_type_desc')} IS DISTINCT FROM {normalized_label_sql('a.name')}
                        OR c.diagnostic_evidence_basis IS DISTINCT FROM 'diagnostic_site_proxy_no_test_field'
                """).fetchone()[0])
                eicu_checks(connection, checks)
                provenance_checks(checks, report_input.parent, common_dir)
                add_check(checks, "base_eligibility", connection.execute("""
                    SELECT COUNT(*) FROM final WHERE icu_seq IS DISTINCT FROM 1
                        OR age IS NULL OR age < 18 OR included_in_base_cohort IS DISTINCT FROM TRUE
                        OR UPPER(TRIM(admission_type)) = 'ELECTIVE'
                """).fetchone()[0])
                evidence_checks(connection, checks, policy)
                rolling_checks(connection, checks)
                selection_checks(connection, checks)
                summary = {
                    "suspected_infection_stays": connection.execute("SELECT COUNT(*) FROM phenotype").fetchone()[0],
                    "infection_candidates": connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
                    "assessment_rows": connection.execute("SELECT COUNT(*) FROM timeline").fetchone()[0],
                    "sepsis3_stays": connection.execute("SELECT COUNT(*) FROM final").fetchone()[0],
                    "qualifying_candidates": connection.execute("SELECT COUNT(*) FROM expected_candidates WHERE expected_status = 'qualifies'").fetchone()[0],
                }
                summary_queries = {
                    "strict_culture_sepsis3_subgroup": "SELECT COUNT(*) FROM final WHERE strict_culture_sepsis3",
                    "selected_later_candidate": "SELECT COUNT(*) FROM final WHERE infection_candidate_rank > 1",
                    "selected_date_only_culture": "SELECT COUNT(*) FROM final WHERE culture_time_is_date_only",
                    "selected_pre_sit_sofa": "SELECT COUNT(*) FROM final WHERE sofa_time < suspected_infection_time",
                    "selected_incomplete_24h_followup": "SELECT COUNT(*) FROM final WHERE NOT complete_24h_after_onset",
                    "selected_with_retrospective_chronic_organ_flag": "SELECT COUNT(*) FROM final WHERE retrospective_chronic_organ_disease_flag",
                    "paired_arterial_gases": "SELECT COUNT(*) FROM respiratory WHERE pf_ratio IS NOT NULL",
                    "respiratory_support_conflicts": "SELECT COUNT(*) FROM respiratory WHERE support_status = 'conflict'",
                    "complete_gcs_assessments": "SELECT COUNT(*) FROM gcs WHERE gcs_total IS NOT NULL",
                    "incomplete_gcs_assessments": "SELECT COUNT(*) FROM gcs WHERE gcs_total IS NULL",
                    "urine_coverage_adequate_assessments": "SELECT COUNT(*) FROM timeline WHERE in_icu_assessment AND urine_coverage_adequate",
                    "selected_urine_changes_renal_score": "SELECT COUNT(*) FROM final WHERE sofa_renal_urine > COALESCE(sofa_renal_creatinine, 0)",
                }
                summary.update({name: connection.execute(query).fetchone()[0] for name, query in summary_queries.items()})
                for name, count in summary.items():
                    add_check(checks, f"sepsis_report_{name}", int(previous["summary"].get(name) != count),
                              {"reported": previous["summary"].get(name), "observed": count})
                attrition = dict(connection.execute("SELECT reason, COUNT(*) FROM stay_audit GROUP BY reason").fetchall())
                add_check(checks, "sepsis_report_attrition", int(previous.get("stay_attrition") != attrition))
                missing_components = {
                    organ: connection.execute(f"SELECT COUNT(*) FROM final WHERE sofa_{organ} IS NULL").fetchone()[0]
                    for organ in ORGANS
                }
                add_check(checks, "sepsis_report_missing_components", int(previous.get("selected_missing_components") != missing_components))
                review = {
                    "baseline_assumed_zero_stays": connection.execute("SELECT COUNT(*) FROM final WHERE baseline_sofa_assumed_zero").fetchone()[0],
                    "retrospective_chronic_organ_flags": connection.execute("SELECT COUNT(*) FROM final WHERE retrospective_chronic_organ_disease_flag").fetchone()[0],
                    "incomplete_24h_followup_stays": connection.execute("SELECT COUNT(*) FROM final WHERE NOT complete_24h_after_onset").fetchone()[0],
                    "incomplete_sofa_panel_at_selection": connection.execute("SELECT COUNT(*) FROM final WHERE observed_components < 6").fetchone()[0],
                    "uncertain_onset_stays": connection.execute("SELECT COUNT(*) FROM final WHERE onset_time_uncertain").fetchone()[0],
                    "respiratory_support_conflicts": connection.execute("SELECT COUNT(*) FROM respiratory WHERE support_status = 'conflict'").fetchone()[0],
                    "empty_final_cohort": summary["sepsis3_stays"] == 0,
                    "selected_missing_components": missing_components,
                }
                connection.execute("CREATE TEMP TABLE target_features (position INTEGER, feature VARCHAR, canonical_unit VARCHAR, derived BOOLEAN)")
                connection.executemany("INSERT INTO target_features VALUES (?, ?, ?, ?)", [
                    (position, entry["name"], entry["canonical_unit"], entry["derived"])
                    for position, entry in enumerate(registry["temporal_features"])
                ])
                connection.execute("""
                    CREATE TEMP TABLE post_onset_coverage AS
                    SELECT event.feature, COUNT(*) AS accepted_rows, COUNT(DISTINCT event.stay_id) AS stays_with_measurements
                    FROM cleaned AS event JOIN final AS patient USING (stay_id)
                    WHERE event.numeric_usable AND event.event_time >= patient.sepsis_onset_time
                        AND event.event_time < LEAST(patient.sepsis_onset_time + INTERVAL 24 HOUR, patient.followup_end)
                    GROUP BY event.feature
                """)
                review["direct_features_without_post_onset_measurements"] = [
                    row[0] for row in connection.execute("""
                        SELECT target.feature FROM target_features AS target
                        LEFT JOIN post_onset_coverage AS coverage USING (feature)
                        WHERE NOT target.derived AND COALESCE(coverage.accepted_rows, 0) = 0 ORDER BY target.position
                    """).fetchall()
                ]
        except (OSError, ValueError, KeyError, TypeError, duckdb.Error) as error:
            add_check(checks, "audit_execution", 1, f"{type(error).__name__}: {error}")
        passed = bool(checks) and all(check["passed"] for check in checks)
        report = {
            "dataset_version": DATASET_VERSION,
            "source_specific_limits": CONFIG['eicu'],
            "audit_version": AUDIT_VERSION, "duckdb_version": duckdb.__version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": passed, "checks": checks, "summary": summary, "review_items": review,
            "input_manifest": manifest, "audit_script_sha256": file_hash(__file__),
            "scope": "Internal consistency only; does not prove clinical Sepsis-3 validity, source-data correctness, or agreement with an independently executed reference pipeline.",
            "limitations": [
                "Zero baseline, retrospective chronic-disease flags and unadjudicated sedation require review.",
                "Respiratory and urine evidence audits use saved evidence fields; not every source-level derivation is independently reconstructed.",
                "Post-onset coverage counts accepted point observations in [onset, min(onset+24h, followup_end)); it is not hourly tensor coverage.",
                "PF ratio, NEQ and ventilation channels remain derived during tensor construction; sparse data and shortened follow-up must not be filled as structural absence.",
                "Use the tensor/imputation scripts with matching audited inputs; a passing audit is not clinical validation.",
            ],
        }
        report_file = staging / "cohort_audit.json"
        report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report_file.replace(metrics_dir / report_file.name)
    return report


def provenance_checks(checks, metrics_dir, common_dir):
    stage_reports = ['01_base.json','02_infection.json','03_phenotypes.json',
                     '04_extraction.json','05_cleaning.json','06_sepsis.json']
    scripts = Path(__file__).parent
    shared = {p.name:file_hash(p) for p in common_dir.glob('*.json')}
    for stage, filename in enumerate(stage_reports,1):
        path = metrics_dir / filename
        if not path.is_file():
            add_check(checks,f'stage_{stage:02d}_report',1,'Missing report; rerun this stage')
            continue
        report = json.loads(path.read_text())
        script = next(scripts.glob(f'{stage:02d}_*.py'))
        add_check(checks,f'stage_{stage:02d}_dataset_version',int(report.get('dataset_version') != DATASET_VERSION))
        add_check(checks,f'stage_{stage:02d}_script_hash',int(report.get('script_sha256') != file_hash(script)))
        add_check(checks,f'stage_{stage:02d}_config_hash',int(report.get('dataset_config_sha256') != file_hash(CONFIG_PATH)))
        add_check(checks,f'stage_{stage:02d}_shared_hashes',int(report.get('common_sha256') != shared))
    schema = json.loads((common_dir/'tensor_schema.json').read_text())
    add_check(checks,'shared_schema_version',int(schema['schema_version'] != CONFIG['shared_schema_version']))


def eicu_checks(connection, checks):
    def check(name, query):
        add_check(checks,name,connection.execute(query).fetchone()[0])

    check('required_lab_channels_survive_cleaning', """
        SELECT COUNT(*) FROM (VALUES ('platelets'), ('wbc'), ('rbc'), ('inr'), ('alt'), ('ast'), ('alp')) AS needed(feature)
        WHERE NOT EXISTS (SELECT 1 FROM cleaned c WHERE c.feature=needed.feature AND c.numeric_usable)
    """)

    proxy=CONFIG['infection_policy']['documented_infection_proxy']
    expected_rule='CASE '+ ' '.join(f'WHEN REGEXP_MATCHES(LOWER(COALESCE(d.diagnosisstring,\'\')),{sql_string(pattern)}) THEN {sql_string(name)}'
        for name,pattern in proxy['diagnosis_patterns'].items())+' END'
    check('documented_infection_source_binding',f"""
        SELECT COUNT(*) FROM candidates c LEFT JOIN infection_diagnoses d
            ON c.stay_id=d.stay_id AND c.infection_diagnosis_id=d.diagnosisid
        WHERE d.diagnosisid IS NULL OR d.diagnosis_evidence_status IS DISTINCT FROM 'included'
            OR c.diagnosis_time IS DISTINCT FROM TIMESTAMP '2000-01-01'+d.diagnosisoffset*INTERVAL 1 MINUTE
            OR c.diagnosisstring IS DISTINCT FROM d.diagnosisstring
            OR c.infection_rule IS DISTINCT FROM ({expected_rule})
            OR ({expected_rule}) IS NULL
            OR REGEXP_MATCHES(LOWER(COALESCE(d.diagnosisstring,'')),{sql_string(proxy['excluded_text_pattern'])})
            OR d.diagnosisoffset NOT BETWEEN -1440 AND 1440
            OR d.diagnosisoffset NOT BETWEEN c.hospitaladmitoffset AND LEAST(c.unitdischargeoffset,c.hospitaldischargeoffset)
            OR c.abx_start_time NOT BETWEEN c.diagnosis_time-INTERVAL 24 HOUR AND c.diagnosis_time+INTERVAL 24 HOUR
    """)
    check('proxy_timing_and_identity',"""
        SELECT COUNT(*) FROM candidates WHERE infection_time_is_proxy IS DISTINCT FROM TRUE
            OR clinical_infection_onset_known IS DISTINCT FROM FALSE
            OR cohort_definition IS DISTINCT FROM 'eicu_documented_infection_proxy_sofa'
            OR diagnostic_evidence_basis IS DISTINCT FROM 'documented_infection_plus_iv_order'
            OR infection_time_basis IS DISTINCT FROM 'recorded_iv_order_start_with_nearby_infection_documentation'
            OR suspected_infection_time IS DISTINCT FROM abx_start_time
            OR suspected_infection_time_upper_bound IS DISTINCT FROM abx_start_time
            OR abx_start_time_upper_bound IS DISTINCT FROM abx_start_time
            OR abx_start_time_is_date_only IS DISTINCT FROM FALSE OR abx_start_time_upper_exclusive IS DISTINCT FROM FALSE
            OR culture_time IS NOT NULL OR culture_event_id IS NOT NULL OR spec_itemid IS NOT NULL
            OR abx_duration_hours IS NOT NULL OR abx_duration_complete IS DISTINCT FROM FALSE
            OR has_strict_culture_infection_pair IS DISTINCT FROM
                EXISTS(SELECT 1 FROM strict_candidates s WHERE s.stay_id=candidates.stay_id)
    """)
    check('strict_sensitivity_decision_uniqueness',"""
        SELECT COUNT(*) FROM (SELECT stay_id,infection_candidate_rank,COUNT(*) AS n FROM strict_decisions
            GROUP BY stay_id,infection_candidate_rank HAVING COUNT(*)<>1)
    """)
    check('strict_sensitivity_independent_sofa_window',"""
        WITH expected AS (
            SELECT s.stay_id,s.infection_candidate_rank,s.suspected_infection_time,
                MIN(t.assessment_time) AS expected_time
            FROM strict_candidates s JOIN phenotype p USING(stay_id)
            LEFT JOIN timeline t ON s.stay_id=t.stay_id AND t.in_icu_assessment
                AND t.total_sofa>=2 AND t.observed_components>0
                AND t.assessment_time BETWEEN s.suspected_infection_time-INTERVAL 48 HOUR AND s.suspected_infection_time+INTERVAL 24 HOUR
                AND GREATEST(s.suspected_infection_time,t.assessment_time)<=LEAST(s.icu_outtime,s.hospital_deathtime)
            GROUP BY s.stay_id,s.infection_candidate_rank,s.suspected_infection_time
        ) SELECT COUNT(*) FROM expected e FULL JOIN strict_decisions d USING(stay_id,infection_candidate_rank)
            LEFT JOIN timeline t ON e.stay_id=t.stay_id AND t.assessment_time=e.expected_time
        WHERE e.stay_id IS NULL OR d.stay_id IS NULL
            OR d.first_qualifying_sofa_time IS DISTINCT FROM e.expected_time
            OR d.strict_qualifies IS DISTINCT FROM (e.expected_time IS NOT NULL)
            OR d.strict_sepsis_onset_time IS DISTINCT FROM CASE WHEN e.expected_time IS NOT NULL
                THEN GREATEST(e.suspected_infection_time,e.expected_time) END
            OR d.strict_sofa IS DISTINCT FROM t.total_sofa
    """)
    check('strict_sensitivity_final_flags',"""
        WITH selected AS (SELECT * FROM strict_decisions WHERE strict_qualifies
            QUALIFY ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY infection_candidate_rank,first_qualifying_sofa_time)=1)
        SELECT COUNT(*) FROM final f LEFT JOIN selected s USING(stay_id)
        WHERE f.strict_culture_sepsis3 IS DISTINCT FROM (s.stay_id IS NOT NULL)
            OR f.strict_culture_sit IS DISTINCT FROM s.suspected_infection_time
            OR f.strict_culture_onset_time IS DISTINCT FROM s.strict_sepsis_onset_time
            OR f.strict_culture_sofa IS DISTINCT FROM s.strict_sofa
            OR f.strict_culture_candidate_rank IS DISTINCT FROM s.infection_candidate_rank
            OR f.strict_onset_matches_primary IS DISTINCT FROM CASE WHEN s.stay_id IS NOT NULL
                THEN s.strict_sepsis_onset_time=f.sepsis_onset_time END
    """)

    check('eicu_source',"""
        SELECT COUNT(*) FROM cleaned WHERE source_db IS DISTINCT FROM 'eICU-CRD'
            OR source_table NOT IN ('chartevents','labevents','inputevents','outputevents','procedureevents')
            OR ((numeric_usable OR evidence_usable) AND
                (source_error OR infusion_rate_conflict OR source_unit_conflict OR source_value_conflict))
    """)
    check('first_stay_before_filtering',"""
        SELECT COUNT(*) FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY subject_id
            ORDER BY hospitaladmitoffset DESC NULLS LAST,stay_id) AS expected FROM eligibility)
        WHERE icu_seq IS DISTINCT FROM expected
    """)
    check('eicu_base_membership',"""
        SELECT COUNT(*) FROM eligibility e FULL JOIN base b USING(stay_id)
        WHERE COALESCE(e.included_in_base_cohort,FALSE) IS DISTINCT FROM (b.stay_id IS NOT NULL)
    """)
    check('eicu_first_stay_known',"""
        SELECT COUNT(*) FROM final WHERE first_stay_order_known IS DISTINCT FROM TRUE
            OR hospital_encounters<>1 OR order_ties<>1 OR icu_seq IS DISTINCT FROM 1
            OR age NOT BETWEEN 18 AND 120 OR (age_is_deidentified AND age<>91)
            OR hospital_id IS NULL OR source_subject_id IS NULL
            OR hospital_expire_flag NOT IN (0,1) OR hospital_expire_flag IS NULL
    """)
    check('patient_identifier_mapping',"""
        SELECT COUNT(*) FROM (SELECT subject_id,COUNT(DISTINCT source_subject_id) AS n
            FROM eligibility GROUP BY subject_id) WHERE n<>1
    """)
    check('patient_identifier_reverse_mapping',"""
        SELECT COUNT(*) FROM (SELECT source_subject_id,COUNT(DISTINCT subject_id) AS n
            FROM eligibility GROUP BY source_subject_id) WHERE n<>1
    """)
    check('synthetic_timestamp_contract',"""
        SELECT COUNT(*) FROM final WHERE icu_intime IS DISTINCT FROM TIMESTAMP '2000-01-01'
            OR hospital_admittime IS DISTINCT FROM icu_intime+hospitaladmitoffset*INTERVAL 1 MINUTE
            OR hospital_dischtime IS DISTINCT FROM icu_intime+hospitaldischargeoffset*INTERVAL 1 MINUTE
            OR icu_outtime IS DISTINCT FROM icu_intime+unitdischargeoffset*INTERVAL 1 MINUTE
            OR hospital_admittime>icu_intime OR icu_outtime>hospital_dischtime
    """)
    check('terminal_time_proxy_disclosed',"""
        SELECT COUNT(*) FROM final WHERE exact_death_time_available IS DISTINCT FROM FALSE
            OR death_time_is_proxy IS DISTINCT FROM (hospital_expire_flag=1)
            OR hospital_deathtime IS DISTINCT FROM CASE
                WHEN UPPER(TRIM(unitdischargestatus))='EXPIRED' THEN icu_outtime
                WHEN hospital_expire_flag=1 THEN hospital_dischtime END
            OR death_time_basis IS DISTINCT FROM CASE
                WHEN UPPER(TRIM(unitdischargestatus))='EXPIRED' THEN 'expired_unit_discharge_offset'
                WHEN hospital_expire_flag=1 THEN 'expired_hospital_discharge_offset' ELSE 'not_reported_dead' END
    """)
    check('exact_offset_infection_pairing',"""
        SELECT COUNT(*) FROM strict_candidates WHERE abx_start_time IS NULL OR culture_time IS NULL
            OR abx_start_time_upper_bound IS DISTINCT FROM abx_start_time
            OR abx_start_time_is_date_only IS DISTINCT FROM FALSE
            OR abx_start_time_upper_exclusive IS DISTINCT FROM FALSE
            OR culture_time_upper_bound IS DISTINCT FROM culture_time
            OR culture_time_is_date_only IS DISTINCT FROM FALSE
            OR culture_time_upper_exclusive IS DISTINCT FROM FALSE
            OR culture_time NOT BETWEEN abx_start_time-INTERVAL 72 HOUR AND abx_start_time+INTERVAL 24 HOUR
            OR suspected_infection_time IS DISTINCT FROM LEAST(abx_start_time,culture_time)
            OR suspected_infection_time_upper_bound IS DISTINCT FROM suspected_infection_time
            OR suspected_infection_time NOT BETWEEN icu_intime-INTERVAL 24 HOUR AND icu_intime+INTERVAL 24 HOUR
            OR pairing_is_definite IS DISTINCT FROM TRUE OR presentation_is_definite IS DISTINCT FROM TRUE
            OR abx_duration_hours IS NOT NULL OR abx_duration_complete IS DISTINCT FROM FALSE
    """)
    check('candidate_medication_binding',"""
        WITH source AS (SELECT stay_id,antimicrobial_start_id,abx_start_time,abx_drug,abx_route FROM candidates
            UNION ALL SELECT stay_id,antimicrobial_start_id,abx_start_time,abx_drug,abx_route FROM strict_candidates)
        SELECT COUNT(*) FROM source c LEFT JOIN prescriptions p
            ON c.stay_id=p.stay_id AND c.antimicrobial_start_id=p.medicationid
        WHERE p.medicationid IS NULL OR p.prescription_status IS DISTINCT FROM 'included'
            OR p.abx_start_time IS DISTINCT FROM c.abx_start_time
            OR LOWER(TRIM(p.drug)) IS DISTINCT FROM c.abx_drug
            OR p.route IS DISTINCT FROM c.abx_route
            OR LOWER(TRIM(COALESCE(p.drugordercancelled,''))) NOT IN ('no','false','0')
    """)
    check('candidate_culture_binding',"""
        SELECT COUNT(*) FROM strict_candidates c LEFT JOIN cultures p USING(stay_id,culture_event_id)
        WHERE p.culture_event_id IS NULL OR p.culture_evidence_status IS DISTINCT FROM 'diagnostic_culture'
            OR p.culture_time IS DISTINCT FROM c.culture_time OR p.spec_type_desc IS DISTINCT FROM c.spec_type_desc
            OR c.culture_time IS DISTINCT FROM TIMESTAMP '2000-01-01'+c.culturetakenoffset*INTERVAL 1 MINUTE
    """)
    check('event_offset_binding',"""
        SELECT COUNT(*) FROM cleaned WHERE event_time IS DISTINCT FROM
            TIMESTAMP '2000-01-01'+source_offset_minutes*INTERVAL 1 MINUTE
    """)
    check('lab_revisions_and_conflicts_withheld',"""
        SELECT COUNT(*) FROM cleaned WHERE source_table='labevents' AND numeric_usable
            AND (is_rewritten_or_cancelled OR source_value_conflict OR source_unit_conflict)
    """)
    cleaner = module(Path(__file__).with_name('05_eicu_temporal_clean.py'), 'eicu_unit_qc')
    signatures = cleaner.lab_unit_signatures(read(COMMON_DIR/'feature_registry.json'), read(COMMON_DIR/'measurement_rules.json'))
    connection.execute('CREATE TEMP TABLE qc_lab_units(feature VARCHAR, unit VARCHAR, signature VARCHAR)')
    connection.executemany('INSERT INTO qc_lab_units VALUES (?,?,?)', signatures)
    norm = lambda col: f"REGEXP_REPLACE(LOWER(REPLACE(REPLACE(TRIM(COALESCE({col},'')),'µ','u'),'μ','u')),'\\s+','','g')"
    check('lab_source_unit_conflict_replay', f"""
        SELECT COUNT(*) FROM cleaned e
        LEFT JOIN qc_lab_units s ON s.feature=e.feature AND s.unit={norm('e.source_system_unit')}
        LEFT JOIN qc_lab_units i ON i.feature=e.feature AND i.unit={norm('e.source_interface_unit')}
        WHERE e.source_table='labevents' AND e.source_unit_conflict IS DISTINCT FROM (
            NULLIF(TRIM(e.source_system_unit),'') IS NOT NULL AND NULLIF(TRIM(e.source_interface_unit),'') IS NOT NULL
            AND COALESCE(s.signature,{norm('e.source_system_unit')})<>COALESCE(i.signature,{norm('e.source_interface_unit')}))
    """)
    lookup = {(feature, unit): signature for feature, unit, signature in signatures}
    fixtures = [('platelets','k/mcl','k/cmm',True), ('platelets','k/mcl','1000/cmm',True),
                ('rbc','m/mcl','mil/mm3',True), ('rbc','m/mcl','*10^6/ul',True),
                ('rbc','m/mcl','x(10)6/ul',True), ('rbc','m/mcl','10e6/mcl',True),
                ('rbc','m/mcl','mill/cmm',True), ('rbc','m/mcl','x10e6/ul',True),
                ('sodium','mmol/l','meq/l',True),
                ('calcium_total','mg/dl','mmol/l',False), ('platelets','k/mcl','cells/ul',False),
                ('platelets','k/mcl','10',False), ('hemoglobin','g/dl','%',False)]
    violations = sum((lookup.get((f,a)) is not None and lookup.get((f,a))==lookup.get((f,b))) != expected
                     for f,a,b,expected in fixtures)
    add_check(checks, 'lab_unit_equivalence_regressions', violations,
              'Equivalent spellings and monovalent electrolyte units agree; unknown and different scales remain distinct.')
    check('inferred_lab_groups_disclosed',"""
        SELECT COUNT(*) FROM cleaned WHERE source_table='labevents'
            AND (specimen_id_is_inferred IS DISTINCT FROM TRUE OR specimen_id IS NULL)
    """)
    check('arterial_category_evidence',"""
        SELECT COUNT(*) FROM cleaned WHERE numeric_usable AND feature IN ('pao2','paco2')
            AND (source_table_original IS DISTINCT FROM 'lab' OR source_lab_type IS DISTINCT FROM 7
                OR specimen_type IS DISTINCT FROM 'ARTERIAL' OR specimen_type_conflict)
    """)
    check('infusion_endpoint_and_weight_evidence',"""
        SELECT COUNT(*) FROM cleaned WHERE source_table='inputevents' AND numeric_usable
            AND (source_table_original IS DISTINCT FROM 'infusionDrug'
                OR event_end_time IS NULL OR event_end_time<=event_time OR event_end_time>event_time+INTERVAL 4 HOUR
                OR NULLIF(TRIM(raw_unit),'') IS NULL
                OR (unit_rule='documented_weight_conversion' AND weight_observation_time IS DISTINCT FROM event_time))
    """)
    check('urine_measured_volume_only',"""
        SELECT COUNT(*) FROM cleaned WHERE numeric_usable AND feature='urine_output'
            AND (source_table_original IS DISTINCT FROM 'intakeOutput'
                OR LOWER(TRIM(source_label)) NOT LIKE 'flowsheet|flowsheet cell labels|i&o|output (ml)|%'
                OR source_value_conflict OR valueuom IS DISTINCT FROM 'mL')
    """)
    airway_labels=', '.join(sql_string(s) for s in CONFIG['source_mapping']['invasive_airways'])
    check('invasive_ventilation_source',f"""
        SELECT COUNT(*) FROM cleaned WHERE feature='vent_invasive' AND evidence_usable
            AND (source_table_original IS DISTINCT FROM 'respiratoryCare'
                OR LOWER(TRIM(raw_value)) NOT IN ({airway_labels})
                OR event_end_time IS NULL OR event_end_time<=event_time)
    """)



def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def array(path):
    return np.load(path, allow_pickle=False, mmap_mode="r")


def equal(left, right, tolerant=False):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if tolerant and left.dtype.kind in "fc":
        return bool(np.allclose(left, right, rtol=RTOL, atol=ATOL, equal_nan=True))
    return bool(np.array_equal(left, right, equal_nan=left.dtype.kind in "fc"))


def near(left, right):
    """Nested metric comparison: identities exact; floating arithmetic tolerant."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(near(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(near(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return isinstance(left, (float, int)) and isinstance(right, (float, int)) and bool(
            np.isclose(left, right, rtol=RTOL, atol=ATOL))
    return left == right


class QC:
    def __init__(self):
        self.checks, self.notes, self.feature_rows, self.quality_rows = [], [], [], []
        self.section = "startup"

    def check(self, name, passed, detail=None, required=True):
        row = dict(section=self.section, name=name, passed=bool(passed), required=required, detail=detail)
        self.checks.append(row)
        if not passed:
            print(f"    {'FAIL' if required else 'NOTE'} {name}: {detail}", flush=True)
        return bool(passed)

    def phase(self, name, action):
        self.section = name
        print(f"[MASTER QC] {name}", flush=True)
        try:
            action()
        except Exception as error:
            self.check("execution_completed", False, f"{type(error).__name__}: {error}")
            with (OUT / "qc_errors.log").open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)

    def report(self):
        failures = [item for item in self.checks if item["required"] and not item["passed"]]
        passed = bool(self.checks) and not failures
        report = dict(schema_version="1.0.0", qc_script_sha256=digest(__file__),
            source_database=CONFIG['source_database'], source_version=CONFIG['source_version'],
            source_specific_limits=CONFIG['eicu'],
            imputation_fitting_protocol=CONFIG['imputation']['fitting_protocol'],
            generated_at_utc=datetime.now(timezone.utc).isoformat(), dataset_version=DATASET_VERSION,
            dataset_path=str(DATA), manifest_sha256=digest(DATA / "manifest.json") if (DATA / "manifest.json").exists() else None,
            status="PASS" if passed else "FAIL",
            eligibility="ELIGIBLE_FOR_RETROSPECTIVE_RESEARCH_UNDER_DATASET_CONTRACT" if passed else "NOT_ELIGIBLE_UNTIL_FAILED_CHECKS_RESOLVED",
            required_checks=sum(item["required"] for item in self.checks), failed_required_checks=len(failures),
            checks=self.checks, diagnostics=self.notes,
            quality_interpretation="Feature-specific held-out baseline regressions are reported, not hidden or used for post-test tuning. PASS is not a claim that SAITS wins every feature/scenario.",
            scope_limits=["No clinical adjudication or external reference cohort is performed.",
                "Accuracy at naturally missing values is unobservable; artificial holdouts are a proxy.",
                "Cohort/observed replay uses the processing implementation plus independent array and linkage checks, not an independently authored Sepsis-3 implementation.",
                "The first 24 hours and bidirectional imputation are retrospective, not online forecasting inputs.",
                "Seven semantic/derived channels and unavailable follow-up intentionally retain missingness.",
                "Downstream evaluation must use the saved subject partitions or refit imputation within its own folds."],
            replay_tolerances=dict(relative=RTOL, absolute=ATOL))
        write(OUT / "qc.json", report)
        for filename, rows in (("features.csv", self.feature_rows), ("imputation_metrics.csv", self.quality_rows)):
            if rows:
                with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                (OUT / filename).write_text("", encoding="utf-8")
        print(f"\n[MASTER QC {'PASS' if passed else 'FAIL'}] {report['eligibility']}", flush=True)
        print(f"    Required checks: {report['required_checks']}; failures: {len(failures)}", flush=True)
        print(f"    Report: {OUT / 'qc.json'}", flush=True)
        print(f"    Feature coverage/distributions: {OUT / 'features.csv'}", flush=True)
        print(f"    Per-feature test comparisons: {OUT / 'imputation_metrics.csv'}", flush=True)
        return passed


def cohort_checks(qc, state):
    import duckdb
    audit = sys.modules[__name__]
    with tempfile.TemporaryDirectory(prefix=".cohort_qc_", dir=OUT) as temporary:
        result = audit.audit_cohort(processed_dir=state["data"], metrics_dir=Path(temporary),
            common_dir=state["common"], adjudication_report=state["metrics"] / "06_sepsis.json")
    for check in result["checks"]:
        qc.check(check["name"], check["passed"], dict(violations=check["violations"], detail=check["detail"]))
    qc.check("nonempty_cohort", result["summary"].get("sepsis3_stays", 0) > 0)
    qc.notes.append(dict(cohort_limitations=result["review_items"]))
    with duckdb.connect() as con:
        for name, filename in (("base", "base_cohort.parquet"), ("eligibility", "base_eligibility.parquet"),
                               ("final", "sepsis_cohort.parquet"),
                               ("metadata", "cohort.parquet")):
            path_sql = str((DATA if name == "metadata" else state["data"]) / filename).replace("\\", "/").replace("'", "''")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path_sql}')")
        queries = dict(
            first_icu_rank="""SELECT COUNT(*) FROM (SELECT *, ROW_NUMBER() OVER
                (PARTITION BY subject_id ORDER BY hospitaladmitoffset DESC NULLS LAST, stay_id) AS expected FROM eligibility)
                WHERE icu_seq IS DISTINCT FROM expected""",
            base_selected_from_eligibility="""SELECT COUNT(*) FROM eligibility e FULL JOIN base b USING(stay_id)
                WHERE COALESCE(e.included_in_base_cohort, FALSE) IS DISTINCT FROM (b.stay_id IS NOT NULL)""",
            final_subset_base="""SELECT COUNT(*) FROM final f LEFT JOIN base b USING(stay_id, subject_id, hadm_id)
                WHERE b.stay_id IS NULL""",
            metadata_exact_cohort="""SELECT COUNT(*) FROM final f FULL JOIN metadata m USING(stay_id, subject_id, hadm_id)
                WHERE f.stay_id IS NULL OR m.stay_id IS NULL OR f.sepsis_onset_time IS DISTINCT FROM m.sepsis_onset_time
                OR f.followup_end IS DISTINCT FROM m.followup_end
                OR f.cohort_definition IS DISTINCT FROM m.cohort_definition
                OR f.infection_time_is_proxy IS DISTINCT FROM m.infection_time_is_proxy
                OR f.strict_culture_sepsis3 IS DISTINCT FROM m.strict_culture_sepsis3
                OR f.strict_culture_onset_time IS DISTINCT FROM m.strict_culture_onset_time
                OR f.hospital_expire_flag IS DISTINCT FROM m.hospital_expire_flag""",
            age_and_admission="""SELECT COUNT(*) FROM final WHERE age IS NULL OR age < 18 OR admission_type IS NULL
                OR UPPER(TRIM(admission_type)) = 'ELECTIVE' OR icu_seq IS DISTINCT FROM 1""",
            age_formula="""SELECT COUNT(*) FROM base WHERE age IS DISTINCT FROM
                CASE WHEN REGEXP_REPLACE(TRIM(source_age),'[[:space:]]+','','g')='>89' THEN 91
                ELSE TRY_CAST(source_age AS INTEGER) END""",
            mortality_labels="SELECT COUNT(*) FROM final WHERE hospital_expire_flag IS NULL OR hospital_expire_flag NOT IN (0,1)")
        for name, query in queries.items():
            violations = con.execute(query).fetchone()[0]
            qc.check(name, violations == 0, dict(violations=violations))
        hospital_rows = con.execute("""SELECT hospital_id, COUNT(*) AS patients,
            SUM(hospital_expire_flag) AS hospital_deaths,
            COUNT(*) FILTER (WHERE NOT complete_24h_after_onset) AS incomplete_followup
            FROM final GROUP BY hospital_id ORDER BY hospital_id""").fetchall()
        qc.notes.append(dict(hospital_summary_columns=['hospital_id','patients','hospital_deaths','incomplete_followup'],
            hospital_summary=hospital_rows,
            hospital_split_note='SAITS partitions are patient-based; they do not establish held-out-hospital generalization.'))
        metadata = con.execute("SELECT patient_index, stay_id, subject_id, hadm_id, hospital_expire_flag FROM metadata ORDER BY patient_index").fetchnumpy()
        qc.check("metadata_patient_indices", np.array_equal(metadata["patient_index"], np.arange(len(state["raw"]))))
        for key, column in (("stay_ids", "stay_id"), ("subject_ids", "subject_id"), ("hadm_ids", "hadm_id"), ("labels", "hospital_expire_flag")):
            qc.check(f"metadata_{key}_alignment", np.array_equal(metadata[column], state["arrays"][key]))
        seconds = con.execute("""SELECT GREATEST(0, EPOCH(LEAST(m.sepsis_onset_time + (h+1) * INTERVAL 1 HOUR,
            CASE WHEN hospital_expire_flag=1 AND hospital_deathtime IS NULL THEN sepsis_onset_time
                 ELSE LEAST(sepsis_onset_time + INTERVAL 24 HOUR, followup_end) END)
            - (sepsis_onset_time + h * INTERVAL 1 HOUR))) AS duration
            FROM metadata m CROSS JOIN RANGE(24) t(h) ORDER BY patient_index,h""").fetchnumpy()["duration"]
        qc.check("exposure_from_timestamps", np.allclose(seconds.reshape(-1, 24), state["arrays"]["exposure_seconds"], rtol=0, atol=1e-6))


def tensor_checks(qc, state):
    raw, output, arrays, codes = state["raw"], state["output"], state["arrays"], state["codes"]
    source, names = state["source"], arrays["features"].tolist()
    for feature in ('platelets', 'wbc', 'rbc', 'inr', 'alt', 'ast', 'alp'):
        qc.check(f"required_channel_observed/{feature}", np.isfinite(raw[:, :, names.index(feature)]).any())
    qc.check("tensor_dimensions", raw.shape == output.shape == codes.shape and raw.shape[1:] == (24, 50))
    qc.check("tensor_dtype", raw.dtype == output.dtype == np.float64 and codes.dtype == np.uint8)
    qc.check("no_infinity", not np.isinf(raw).any() and not np.isinf(output).any())
    counts = arrays["observation_counts"]
    qc.check("observation_counts", counts.dtype == np.uint32 and np.array_equal(counts > 0, np.isfinite(raw)))
    qc.check("exact_observation_preservation", np.array_equal(output[np.isfinite(raw)], raw[np.isfinite(raw)]))
    qc.check("structural_missingness", np.isnan(output[arrays["structural_mask"]]).all())
    qc.check("known_method_codes", np.isin(codes, [0, 1, 2, 3, 4]).all())
    qc.check("observed_code_mask", np.array_equal(codes == 1, np.isfinite(raw)))
    qc.check("unfilled_code_mask", np.array_equal(codes == 0, np.isnan(output)))
    qc.check("generated_code_mask", np.array_equal(codes >= 2, np.isnan(raw) & np.isfinite(output)))
    rejected = state["rejected"]
    qc.check("rejected_predictions_are_baselines", rejected.dtype == bool and rejected.shape == raw.shape and not np.any(rejected & ~np.isin(codes, [3, 4])))
    policy = dict(CONFIG["tensor"], release_version=DATASET_VERSION)
    bounds = dict(state["bounds"])
    bounds.update(policy["derived_bounds"])
    bounds["urine_output"] = [0, None]  # Per-event mL limits do not bound hourly sums.
    for index, name in enumerate(names):
        values = output[:, :, index]
        lower, upper = bounds[name]
        finite = np.isfinite(values)
        qc.check(f"bounds/{name}", not np.any(finite & ((values < lower - 1e-10) | (values > (upper if upper is not None else np.inf) + 1e-10))),
                 dict(unit=str(arrays["units"][index]), minimum=lower, maximum=upper))
        if name in OBSERVED_ONLY:
            qc.check(f"observed_only/{name}", equal(values, raw[:, :, index]) and not np.any(codes[:, :, index] >= 2))
        if name in {"vent", "gcs_eye", "gcs_verbal", "gcs_motor"}:
            qc.check(f"integer_domain/{name}", np.equal(values[finite], np.floor(values[finite])).all())
        measured = raw[:, :, index][np.isfinite(raw[:, :, index])]
        generated = values[codes[:, :, index] >= 2]
        row = dict(feature=name, unit=str(arrays["units"][index]), observed_cells=len(measured), generated_cells=len(generated),
            unfilled_cells=int(np.isnan(values).sum()), patients_with_observations=int(np.isfinite(raw[:, :, index]).any(axis=1).sum()),
            remaining_within_followup_missing=int((np.isnan(values) & ~arrays["structural_mask"]).sum()),
            saits_cells=int((codes[:, :, index] == 2).sum()), forward_fill_cells=int((codes[:, :, index] == 3).sum()),
            median_cells=int((codes[:, :, index] == 4).sum()), out_of_bounds_fallback_cells=int(rejected[:, :, index].sum()))
        attempted = row["saits_cells"] + row["out_of_bounds_fallback_cells"]
        row["saits_out_of_bounds_fraction"] = row["out_of_bounds_fallback_cells"] / attempted if attempted else None
        for label, sample in (("observed", measured), ("generated", generated)):
            for q in (0, 50, 95, 99, 100):
                row[f"{label}_p{q}"] = float(np.percentile(sample, q)) if len(sample) else None
            row[f"{label}_zero_fraction"] = float((sample == 0).mean()) if len(sample) else None
        qc.feature_rows.append(row)
    static = arrays["static"]
    static_names = arrays["static_features"].tolist()
    qc.check("static_shape_and_names", static.shape == (len(raw), len(policy["static_features"])) and static_names == [entry["name"] for entry in policy["static_features"]])
    qc.check("static_no_infinity", not np.isinf(static).any())
    expected_predictors = np.asarray([entry["predictor_default"] for entry in policy["static_features"]], dtype=bool)
    qc.check("static_predictor_roles", equal(arrays["static_predictor_mask"], expected_predictors))
    for name, labels in policy["category_vocabularies"].items():
        values = static[:, static_names.index(name)]
        qc.check(f"static_category/{name}", np.isin(values[np.isfinite(values)], np.arange(len(labels))).all())
    qc.check("retrospective_context_excluded_from_default_predictors", not any(
        expected_predictors[static_names.index(name)] for name in ("charlson_comorbidity_index", "baseline_sofa", "baseline_pf_ratio")))


def preprocessing_checks(qc, state):
    core, config, arrays = state["core"], state["config"], state["arrays"]
    prepared = core.prepare(arrays, config["policy"], state["bounds"])
    state["prepared"] = prepared
    qc.check("train_only_statistics", near(prepared["statistics"], config["statistics"]))
    qc.check("model_feature_order", prepared["model_indices"] == config["model_feature_indices"] and
             arrays["features"][prepared["model_indices"]].tolist() == config["model_features"])
    qc.check("model_unit_order", arrays["units"][prepared["model_indices"]].tolist() == config["model_units"])
    qc.check("observed_only_policy", set(config["policy"]["observed_only_features"]) == OBSERVED_ONLY)
    qc.check("method_feature_inventory", set(state["methods"]) == set(config["model_features"]))
    qc.check("unsupported_features", config["unsupported_features"] == prepared["unsupported"])
    splits = prepared["splits"]
    ordering = np.asarray(sorted(range(len(arrays["raw"])), key=lambda index: hashlib.sha256(
        f"{config['policy']['seed']}:{int(arrays['subject_ids'][index])}".encode("ascii")).digest()), dtype=np.int64)
    qc.check("independent_label_free_split_order", np.array_equal(np.concatenate(list(splits.values())), ordering))
    for entry in config["statistics"]:
        training = arrays["raw"][splits["train"], :, entry["index"]]
        patients = int(np.isfinite(training).any(axis=1).sum())
        values = training[np.isfinite(training)]
        transformed = np.log1p(values) if entry["log1p"] else values
        std = float(np.std(transformed))
        qc.check(f"independent_scaler/{entry['feature']}", entry["count"] == len(values) and
            patients >= config["policy"].get("minimum_training_patients", 1) and
            np.isclose(entry["mean"], np.mean(transformed), rtol=1e-12, atol=1e-12) and
            np.isclose(entry["scale"], std if std else 1., rtol=1e-12, atol=1e-12) and
            entry["median"] == float(np.median(values)))
    partition = state["partition"]
    qc.check("partition_shape_domain", partition.shape == (len(arrays["raw"]),) and partition.dtype == np.uint8 and np.isin(partition, [0, 1, 2]).all())
    for code, (role, indices) in enumerate(splits.items()):
        qc.check(f"split/{role}", equal(state["saved_holdouts"][role + "_indices"], indices) and np.all(partition[indices] == code))
        subjects = arrays["subject_ids"][indices]
        qc.check(f"split_subject_unique/{role}", len(np.unique(subjects)) == len(subjects))
        qc.check(f"both_outcomes/{role}", set(arrays["labels"][indices].tolist()) == {0., 1.})
    qc.check("exhaustive_disjoint_patient_split", np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(arrays["raw"]))))
    mid = len(splits["validation"]) // 2
    state["stopping"], state["selection"] = splits["validation"][:mid], splits["validation"][mid:]
    for name, indices in (("early_stopping", state["stopping"]), ("method_selection", state["selection"])):
        qc.check(f"validation_subset/{name}", equal(state["saved_holdouts"][name + "_indices"], indices))
    expected = np.zeros(arrays["raw"].shape, dtype=bool)
    for index in prepared["model_indices"]:
        expected[:, :, index] = np.isnan(arrays["raw"][:, :, index]) & ~arrays["structural_mask"] & prepared["context"][:, None]
    qc.check("exact_imputation_eligibility", np.array_equal(expected, state["codes"] >= 2))
    qc.check("no_context_not_imputed", not np.any(state["codes"][~prepared["context"]] >= 2))
    qc.check("finite_checkpoint_selection", np.isfinite(config["best_loss"]) and 1 <= config["best_epoch"] <= config["model_options"]["epochs"])


def holdout_checks(qc, state):
    recipe, prepared, arrays = state["recipe"], state["prepared"], state["arrays"]
    policy = state["config"]["policy"]
    rule = policy["final_recipe"]["rules"]
    qc.check("recorded_recipe_rules", recipe.RULES == rule)
    state["holdouts"] = {}
    for role, indices in (("early_stopping", state["stopping"]), ("validation", state["selection"]), ("test", prepared["splits"]["test"])):
        for scenario in SCENARIOS:
            selected = indices
            if role == "early_stopping" and scenario != "whole_channel":
                selected = selected[:rule["stopping_patients_per_scenario"]]
            cap = rule["stopping_channel_patients_per_feature"] if role == "early_stopping" else rule["whole_channel_patients_per_feature"]
            offset = {"early_stopping": 100, "validation": 200, "test": 300}[role]
            evidence = recipe.evidence_for(prepared, arrays, selected, scenario,
                policy["seed"] + offset + SCENARIOS.index(scenario), policy["holdout_fraction"], cap)
            prefix = role + "_" + scenario
            saved = state["saved_holdouts"]
            qc.check(f"fixed_holdouts/{role}/{scenario}", equal(saved[prefix + "_indices"], evidence["indices"]) and
                equal(saved[prefix + "_mask"], evidence["mask"]) and
                np.array_equal(saved["model_feature_indices"], prepared["model_indices"]))
            observed = np.isfinite(evidence["original"])
            qc.check(f"holdout_semantics/{role}/{scenario}", not np.any(evidence["mask"] & ~observed) and
                np.isfinite(evidence["masked"]).any(axis=(1, 2)).all() and
                not np.isfinite(evidence["masked"][evidence["mask"]]).any() and
                np.isin(evidence["indices"], indices).all())
            if role != "early_stopping":
                state["holdouts"][(role, scenario)] = evidence
    validation = state["saits_report"]["metrics"]["validation"]
    selected = recipe.select_methods(validation, prepared["statistics"])
    qc.check("methods_selected_only_from_validation", near(selected, state["methods"]))
    for name, item in state["methods"].items():
        qc.check(f"method_policy/{name}", item["method"] in {"saits", "median", "forward_fill"} and
            item["baseline"] in {"median", "forward_fill"} and (item["method"] == "saits" or item["method"] == item["baseline"]))
        if item["method"] == "saits":
            qc.check(f"saits_quality_gates/{name}", item["adequate_validation"] and bool(item["saits_gates"]) and all(item["saits_gates"].values()))


def checkpoint_checks(qc, state, device):
    recipe, prepared, config = state["recipe"], state["prepared"], state["config"]
    options = dict(config["model_options"])
    if device is not None:
        options["device"] = device
    model, runtime = recipe.make_gap_model(config["policy"], options, prepared["scaled"][prepared["fit_indices"]])
    model.load(str(OUT / "saits.pypots"))
    qc.notes.append(dict(replay_runtime=runtime, original_runtime=config["runtime"]))
    replayed_metrics = {}
    for role in ("validation", "test"):
        rows = []
        for scenario in SCENARIOS:
            print(f"    Checkpoint metrics: {role}/{scenario}", flush=True)
            evidence = state["holdouts"].pop((role, scenario))
            rows.extend(recipe.score_scenario(model, prepared, state["arrays"], evidence, role, scenario,
                state["methods"] if role == "test" else None))
        replayed_metrics[role] = rows
        qc.check(f"replayed_metrics/{role}", near(rows, state["saits_report"]["metrics"][role]))
    qc.check("replayed_method_selection", near(recipe.select_methods(replayed_metrics["validation"], prepared["statistics"]), state["methods"]))
    print("    Checkpoint reconstruction: all eligible patients/cells", flush=True)
    output, codes, rejected = recipe.reconstruct(model, prepared, state["arrays"], state["methods"])
    qc.check("checkpoint_reproduces_tensor", equal(output, state["output"], tolerant=True))
    qc.check("checkpoint_reproduces_methods", equal(codes, state["codes"]))
    qc.check("checkpoint_reproduces_fallback_reasons", equal(rejected, state["rejected"]))


def baseline_and_quality_checks(qc, state):
    raw, output, codes = state["raw"], state["output"], state["codes"]
    for entry in state["config"]["statistics"]:
        index, median = entry["index"], entry["median"]
        previous = np.full(len(raw), median, dtype=np.float64)
        seen = np.zeros(len(raw), dtype=bool)
        failures = 0
        for hour in range(24):
            measured = np.isfinite(raw[:, hour, index])
            previous[measured] = raw[measured, hour, index]
            seen |= measured
            forward = codes[:, hour, index] == 3
            med = codes[:, hour, index] == 4
            failures += int(np.count_nonzero(forward & (~seen | (output[:, hour, index] != previous))))
            failures += int(np.count_nonzero(med & (output[:, hour, index] != median)))
        qc.check(f"independent_baseline_values/{entry['feature']}", failures == 0, dict(violations=failures))
    saved = state["saits_report"]
    qc.check("reported_method_counts", saved["feature_method_counts"] == dict(Counter(item["method"] for item in state["methods"].values())))
    qc.check("reported_cell_counts", saved["cell_method_counts"] == {name: int((codes == code).sum()) for name, code in
             {"unfilled": 0, "observed": 1, "saits": 2, "forward_fill": 3, "median": 4}.items()})
    qc.check("reported_rejection_count", saved["saits_out_of_bounds_fallback_cells"] == int(state["rejected"].sum()))
    metrics = state["saits_report"]["metrics"]["test"]
    qc.check("complete_test_metric_inventory", len(metrics) == len(state["methods"]) * 3 and
             {(row["feature"], row["scenario"]) for row in metrics} == {(feature, scenario) for feature in state["methods"] for scenario in SCENARIOS})
    for row in metrics:
        enough = row["targets"] > 0
        qc.check(f"test_scores/{row['scenario']}/{row['feature']}", not enough or all(
            row[key] is not None and np.isfinite(row[key]) and row[key] >= 0
            for key in ("final_mae", "final_rmse", "median_mae", "forward_fill_mae")))
        if not enough:
            continue
        qc.quality_rows.append(dict(feature=row["feature"], scenario=row["scenario"], selected_method=row["selected_method"],
            target_patients=row["target_patients"], targets=row["targets"], final_mae=row["final_mae"], final_rmse=row["final_rmse"],
            median_mae=row["median_mae"], forward_fill_mae=row["forward_fill_mae"],
            worse_than_median=row["final_mae"] > row["median_mae"] + 1e-12,
            worse_than_forward_fill=row["final_mae"] > row["forward_fill_mae"] + 1e-12,
            relative_change_vs_median=(row["final_mae"] / row["median_mae"] - 1) if row["median_mae"] > 0 else None,
            relative_change_vs_forward_fill=(row["final_mae"] / row["forward_fill_mae"] - 1) if row["forward_fill_mae"] > 0 else None))
    summary = {}
    for scenario in SCENARIOS:
        rows = [row for row in qc.quality_rows if row["scenario"] == scenario]
        summary[scenario] = dict(scored_features=len(rows), targets=sum(row["targets"] for row in rows),
            final_worse_than_median=sum(row["worse_than_median"] for row in rows),
            final_worse_than_forward_fill=sum(row["worse_than_forward_fill"] for row in rows))
    qc.check("reported_test_summary", summary == saved["test_summary"])
    regressions = [row for row in qc.quality_rows if row["worse_than_median"] or row["worse_than_forward_fill"]]
    qc.notes.append(dict(heldout_regressions=regressions, interpretation="Reported honestly; no new pass threshold was chosen after viewing this test set."))
    for row in regressions:
        print(f"    TEST COMPARISON {row['feature']}/{row['scenario']}: final MAE={row['final_mae']:.6g}, "
              f"median={row['median_mae']:.6g}, forward-fill={row['forward_fill_mae']:.6g}", flush=True)
    # Diagnostic only: these channels may originate at different measurement times.
    differential = ["lymphocytes_pct", "monocytes_pct", "neutrophils_pct", "basophils_pct", "eosinophils_pct"]
    indices = [state["arrays"]["features"].tolist().index(name) for name in differential]
    for label, tensor in (("observed", raw), ("imputed", output)):
        panel = tensor[:, :, indices]
        complete = np.isfinite(panel).all(axis=2)
        sums = panel.sum(axis=2)[complete]
        qc.notes.append(dict(differential_panel=label, complete_hours=int(complete.sum()),
            sum_percentiles=np.percentile(sums, [0, 1, 50, 99, 100]).tolist() if len(sums) else None,
            interpretation="Distribution diagnostic; independent hourly aggregation does not guarantee a simultaneous complete differential summing to 100."))
    qc.notes.append(dict(feature_distribution_note="Compare observed/generated quantiles and zero fractions in features.csv, especially eosinophils. Different missingness populations do not imply identical distributions."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Checkpoint replay device; defaults to the saved model option")
    parser.add_argument('--cohort-only', action='store_true', help='Audit stages 01-06 without requiring tensors or model weights')
    args = parser.parse_args()
    if args.cohort_only:
        result = audit_cohort()
        print(f"[COHORT QC] {'PASS' if result['passed'] else 'FAIL'}: {len(result['checks'])} checks; report: {OUT / 'cohort_audit.json'}", flush=True)
        raise SystemExit(0 if result['passed'] else 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qc_errors.log").write_text("", encoding="utf-8")
    qc, state = QC(), {}
    phases = [
        ("1/8 Artifacts, versions and provenance", lambda: artifact_checks(qc, state)),
        ("2/8 Cohort, clinical evidence, labels and timing", lambda: cohort_checks(qc, state)),
        ("3/8 Observed tensor replay", lambda: observed_replay(qc, state)),
        ("4/8 Units, bounds, masks and static roles", lambda: tensor_checks(qc, state)),
        ("5/8 Training-only preprocessing and patient splits", lambda: preprocessing_checks(qc, state)),
        ("6/8 Holdouts and validation-selected methods", lambda: holdout_checks(qc, state)),
        ("7/8 Checkpoint and imputation replay", lambda: checkpoint_checks(qc, state, args.device)),
        ("8/8 Baselines and feature quality", lambda: baseline_and_quality_checks(qc, state)),
    ]
    for name, action in phases:
        qc.phase(name, action)
        if name.startswith("1/") and any(not row["passed"] for row in qc.checks):
            qc.check("remaining_phases_completed", False, "Required inputs are missing or changed")
            break
    raise SystemExit(0 if qc.report() else 1)


if __name__ == "__main__":
    main()
