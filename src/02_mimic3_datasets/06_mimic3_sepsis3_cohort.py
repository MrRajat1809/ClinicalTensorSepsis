"""Adjudicate Sepsis-3 using rolling SOFA and definite infection timing."""

import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_VERSION = json.loads(Path(__file__).with_name("dataset_config.json").read_text(encoding="utf-8"))["dataset_version"]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimic3-carevue" / "work"
METRICS_DIR = BASE_DIR / "outputs" / "mimic3-carevue"
POLICY_FILE = Path(__file__).with_name("dataset_config.json")
REGISTRY_FILE = BASE_DIR / "src" / "common" / "feature_registry.json"
SCHEMA_VERSION = "2.1.0"
ORGANS = ("resp", "coag", "liver", "cv", "cns", "renal")
SOFA_FEATURES = (
    "map", "platelets", "bilirubin", "creatinine", "gcs_eye", "gcs_verbal", "gcs_motor",
    "pao2", "fio2", "norepinephrine", "epinephrine", "dopamine", "dobutamine",
    "urine_output", "urine_irrigant_in", "urine_irrigant_out",
    "vent_invasive", "vent_noninvasive", "oxygen_device", "ventilator_mode",
    "airway_status", "extubation_status",
)


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_list(values):
    return ", ".join(sql_string(value) for value in values)


def columns(connection, table):
    return [row[0] for row in connection.execute(f"DESCRIBE {table}").fetchall()]


def require_zero(connection, query, message):
    count = connection.execute(query).fetchone()[0]
    if count:
        raise ValueError(f"{message}: {count:,}")


def validate_inputs(connection):
    require_zero(connection, """
        SELECT COUNT(*) FROM candidates WHERE infection_schema_version IS DISTINCT FROM '2.1.0'
            OR culture_evidence_status IS DISTINCT FROM 'diagnostic_culture'
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
    """, "Invalid diagnostic culture/timing contract; rerun infection selection")
    require_zero(connection, "SELECT COUNT(*) - COUNT(DISTINCT subject_id) FROM cohort",
                 "Duplicate/missing patients in first-ICU cohort")
    for table in ("cohort", "windows"):
        require_zero(connection, f"""
            SELECT COUNT(*) - COUNT(DISTINCT stay_id) FROM {table}
        """, f"Duplicate/missing stay IDs in {table}")
    require_zero(connection, """
        SELECT COUNT(*) FROM (
            SELECT stay_id, infection_candidate_rank FROM candidates
            GROUP BY stay_id, infection_candidate_rank HAVING COUNT(*) <> 1
        )
    """, "Duplicate infection candidates")
    require_zero(connection, """
        SELECT COUNT(*) FROM candidates AS candidate
        LEFT JOIN cohort AS patient USING (stay_id, subject_id, hadm_id)
        LEFT JOIN windows AS envelope USING (stay_id, subject_id, hadm_id)
        WHERE patient.stay_id IS NULL OR envelope.stay_id IS NULL
            OR candidate.infection_candidate_rank IS NULL OR candidate.infection_candidate_rank < 1
            OR candidate.suspected_infection_time IS NULL
            OR candidate.suspected_infection_time_upper_bound IS NULL
            OR candidate.suspected_infection_time_upper_bound < candidate.suspected_infection_time
            OR envelope.window_start > candidate.suspected_infection_time - INTERVAL 72 HOUR
            OR envelope.window_end < candidate.suspected_infection_time_upper_bound + INTERVAL 48 HOUR
            OR candidate.icu_intime IS DISTINCT FROM envelope.icu_intime
            OR candidate.icu_outtime IS DISTINCT FROM envelope.icu_outtime
            OR candidate.hospital_deathtime IS DISTINCT FROM envelope.hospital_deathtime
            OR patient.icu_intime IS DISTINCT FROM envelope.icu_intime
            OR patient.icu_outtime IS DISTINCT FROM envelope.icu_outtime
            OR patient.hospital_deathtime IS DISTINCT FROM envelope.hospital_deathtime
    """, "Candidate linkage/timestamps disagree with extraction")
    require_zero(connection, """
        SELECT COUNT(*) FROM cohort AS patient
        LEFT JOIN (SELECT DISTINCT stay_id FROM candidates) AS candidate USING (stay_id)
        WHERE candidate.stay_id IS NULL
    """, "Phenotype stays without candidates")
    require_zero(connection, """
        SELECT COUNT(*) FROM windows WHERE subject_id IS NULL OR hadm_id IS NULL
            OR icu_intime IS NULL OR icu_outtime IS NULL OR icu_intime >= icu_outtime
            OR window_start IS NULL OR window_end IS NULL OR window_start >= window_end
    """, "Invalid extraction windows")
    require_zero(connection, """
        SELECT COUNT(*) FROM cleaned AS event
        LEFT JOIN cohort AS patient USING (stay_id, subject_id, hadm_id)
        LEFT JOIN unit_map USING (feature)
        WHERE patient.stay_id IS NULL OR unit_map.feature IS NULL
            OR event.cleaning_schema_version IS DISTINCT FROM '2.0.0'
            OR event.numeric_usable IS NULL OR event.evidence_usable IS NULL
            OR event.numeric_usable IS DISTINCT FROM (event.valuenum IS NOT NULL)
            OR (event.numeric_usable AND (
                NOT ISFINITE(event.valuenum) OR event.valueuom IS DISTINCT FROM unit_map.unit
                OR event.canonical_unit IS DISTINCT FROM unit_map.unit
                OR event.record_qc_status IS DISTINCT FROM 'accepted'
                OR event.value_qc_status IS DISTINCT FROM 'accepted'))
            OR (event.evidence_usable AND event.record_qc_status IS DISTINCT FROM 'accepted')
            OR (event.evidence_usable AND event.value_qc_status IS DISTINCT FROM 'evidence_only')
            OR (event.numeric_usable AND event.evidence_usable)
            OR ((event.numeric_usable OR event.evidence_usable) AND event.event_time IS NULL)
            OR ((event.numeric_usable OR event.evidence_usable)
                AND event.source_table IN ('inputevents', 'procedureevents') AND (
                    event.effective_start_time IS NULL OR event.effective_end_time IS NULL
                    OR event.effective_start_time >= event.effective_end_time))
    """, "Incompatible or invalid cleaned measurements")


def build_respiratory(connection, policy):
    connection.execute("""
        CREATE TEMP TABLE gases AS
        SELECT stay_id, event_time, specimen_id,
            CASE WHEN COUNT(*) FILTER (WHERE feature = 'pao2') =
                    COUNT(valuenum) FILTER (WHERE feature = 'pao2')
                AND MIN(valuenum) FILTER (WHERE feature = 'pao2') =
                    MAX(valuenum) FILTER (WHERE feature = 'pao2')
                THEN MIN(valuenum) FILTER (WHERE feature = 'pao2') END AS pao2,
            COUNT(*) FILTER (WHERE feature = 'fio2') AS specimen_fio2_rows,
            CASE WHEN COUNT(*) FILTER (WHERE feature = 'fio2') =
                    COUNT(valuenum) FILTER (WHERE feature = 'fio2')
                AND MIN(valuenum) FILTER (WHERE feature = 'fio2') =
                    MAX(valuenum) FILTER (WHERE feature = 'fio2')
                THEN MIN(valuenum) FILTER (WHERE feature = 'fio2') END AS specimen_fio2
        FROM events WHERE source_table = 'labevents' AND feature IN ('pao2', 'fio2')
            AND record_qc_status = 'accepted'
        GROUP BY stay_id, event_time, specimen_id
        HAVING COUNT(*) FILTER (WHERE feature = 'pao2') > 0
    """)
    connection.execute("""
        CREATE TEMP TABLE chart_fio2 AS
        SELECT stay_id, event_time,
            CASE WHEN COUNT(*) = COUNT(valuenum) AND MIN(valuenum) = MAX(valuenum)
                THEN MIN(valuenum) END AS fio2
        FROM events WHERE source_table = 'chartevents' AND feature = 'fio2'
            AND record_qc_status = 'accepted'
        GROUP BY stay_id, event_time
    """)
    for feature, table in (("oxygen_device", "devices"), ("ventilator_mode", "modes"),
                           ("airway_status", "airways"), ("extubation_status", "extubations")):
        connection.execute(f"""
            CREATE TEMP TABLE {table} AS
            SELECT stay_id, event_time,
                CASE WHEN COUNT(DISTINCT LOWER(TRIM(raw_value))) = 1
                    THEN MIN(LOWER(TRIM(raw_value))) END AS label
            FROM events WHERE feature = {sql_string(feature)} AND evidence_usable
            GROUP BY stay_id, event_time
        """)
    connection.execute("DELETE FROM extubations WHERE label NOT IN ('extubated','self extubation') AND label IS NOT NULL")
    connection.execute("""
        CREATE TEMP TABLE respiratory_matched AS
        SELECT gases.*, chart.fio2 AS chart_fio2, chart.event_time AS chart_fio2_time,
            device.label AS device, device.event_time AS device_time,
            mode.label AS mode, mode.event_time AS mode_time,
            airway.label AS airway, airway.event_time AS airway_time,
            extubation.label AS extubation, extubation.event_time AS extubation_time
        FROM gases
        ASOF LEFT JOIN chart_fio2 AS chart
            ON gases.stay_id = chart.stay_id AND gases.event_time >= chart.event_time
        ASOF LEFT JOIN devices AS device
            ON gases.stay_id = device.stay_id AND gases.event_time >= device.event_time
        ASOF LEFT JOIN modes AS mode
            ON gases.stay_id = mode.stay_id AND gases.event_time >= mode.event_time
        ASOF LEFT JOIN airways AS airway
            ON gases.stay_id = airway.stay_id AND gases.event_time >= airway.event_time
        ASOF LEFT JOIN extubations AS extubation
            ON gases.stay_id = extubation.stay_id AND gases.event_time >= extubation.event_time
    """)
    connection.execute(f"""
        CREATE TEMP TABLE respiratory AS
        WITH paired AS (
            SELECT matched.*,
                CASE WHEN specimen_id IS NOT NULL AND specimen_fio2_rows > 0 THEN specimen_fio2
                    WHEN chart_fio2_time >= event_time - INTERVAL {policy["fio2_lookback_hours"]} HOUR
                        THEN chart_fio2 END AS fio2_fraction,
                CASE WHEN specimen_id IS NOT NULL AND specimen_fio2_rows > 0 THEN 'same_timestamp_group'
                    WHEN chart_fio2_time >= event_time - INTERVAL {policy["fio2_lookback_hours"]} HOUR
                        THEN 'preceding_chart' ELSE 'unpaired' END AS fio2_pair_source,
                device_time >= event_time - INTERVAL {policy["support_lookback_hours"]} HOUR AS device_recent,
                mode_time >= event_time - INTERVAL {policy["support_lookback_hours"]} HOUR AS mode_recent,
                EXISTS (SELECT 1 FROM events AS support WHERE support.stay_id = matched.stay_id
                    AND support.feature = 'vent_invasive' AND support.evidence_usable
                    AND support.effective_start_time <= matched.event_time
                    AND support.effective_end_time > matched.event_time) AS invasive_procedure,
                EXISTS (SELECT 1 FROM events AS support WHERE support.stay_id = matched.stay_id
                    AND support.feature = 'vent_noninvasive' AND support.evidence_usable
                    AND support.effective_start_time <= matched.event_time
                    AND support.effective_end_time > matched.event_time) AS noninvasive_procedure
            FROM respiratory_matched AS matched
        ), supported AS (
            SELECT *,
                COALESCE(device_recent AND device = 'ventilator'
                    AND airway = 'intubated/trach'
                    AND airway_time >= event_time - INTERVAL {policy["support_lookback_hours"]} HOUR
                    AND (extubation_time IS NULL OR extubation_time < airway_time)
                    AND mode_recent AND mode IN ({sql_list(policy["active_ventilator_modes"])}), FALSE)
                    AS airway_and_active_mode,
                COALESCE(device_recent AND device IN ({sql_list(policy["noninvasive_devices"])}), FALSE)
                    OR COALESCE(mode_recent AND mode IN ({sql_list(policy["noninvasive_modes"])}), FALSE)
                    OR noninvasive_procedure AS noninvasive_evidence,
                COALESCE(device_recent AND device IN ({sql_list(policy["nonventilated_devices"])}), FALSE)
                    OR COALESCE(mode_recent AND mode IN ('standby', 'ambient'), FALSE) AS nonventilated_evidence
            FROM paired
        ), classified AS (
            SELECT *,
                CASE WHEN (invasive_procedure OR airway_and_active_mode)
                        AND (noninvasive_evidence OR nonventilated_evidence
                            OR (device_recent AND device IS NULL)
                            OR (mode_recent AND mode IS NULL)) THEN 'conflict'
                    WHEN invasive_procedure THEN 'invasive_procedure'
                    WHEN airway_and_active_mode THEN 'invasive_airway_and_mode'
                    WHEN noninvasive_evidence THEN 'noninvasive'
                    WHEN nonventilated_evidence THEN 'no_invasive_support_documented'
                    ELSE 'unknown' END AS support_status,
                pao2 / NULLIF(fio2_fraction, 0) AS pf_ratio
            FROM supported
        )
        SELECT *,
            CASE WHEN pf_ratio IS NULL THEN NULL
                WHEN support_status IN ('invasive_procedure', 'invasive_airway_and_mode')
                    AND pf_ratio < 100 THEN 4
                WHEN support_status IN ('invasive_procedure', 'invasive_airway_and_mode')
                    AND pf_ratio < 200 THEN 3
                WHEN pf_ratio < 300 THEN 2 WHEN pf_ratio < 400 THEN 1 ELSE 0 END AS sofa_resp,
            support_status IN ('unknown', 'conflict') AS support_uncertain
        FROM classified
    """)


def build_gcs_and_pressors(connection, policy):
    connection.execute("""
        CREATE TEMP TABLE gcs AS
        WITH components AS (
            SELECT stay_id, event_time, feature,
                CASE WHEN COUNT(*) = COUNT(valuenum) AND MIN(valuenum) = MAX(valuenum)
                    THEN MIN(valuenum) END AS value
            FROM events WHERE feature IN ('gcs_eye', 'gcs_verbal', 'gcs_motor')
                AND record_qc_status = 'accepted'
            GROUP BY stay_id, event_time, feature
        )
        SELECT stay_id, event_time, COUNT(value) AS observed_components,
            CASE WHEN COUNT(value) = 3 THEN SUM(value) END AS gcs_total
        FROM components GROUP BY stay_id, event_time
    """)
    connection.execute(f"""
        CREATE TEMP TABLE pressor_intervals AS
        WITH scored AS (
            SELECT stay_id, feature, effective_start_time AS start_time, effective_end_time AS end_time,
                CASE WHEN (feature = 'dopamine' AND valuenum > 15)
                        OR (feature IN ('epinephrine', 'norepinephrine') AND valuenum > 0.1) THEN 4
                    WHEN (feature = 'dopamine' AND valuenum > 5)
                        OR feature IN ('epinephrine', 'norepinephrine') THEN 3 ELSE 2 END AS score
            FROM events WHERE feature IN ('dopamine', 'dobutamine', 'epinephrine', 'norepinephrine')
                AND numeric_usable AND valuenum > 0
        ), thresholds AS (
            SELECT DISTINCT scored.stay_id, scored.feature, scored.start_time, scored.end_time, threshold
            FROM scored
            CROSS JOIN (VALUES (2), (3), (4)) AS levels(threshold) WHERE threshold <= score
        ), ordered AS (
            SELECT *, MAX(end_time) OVER (
                PARTITION BY stay_id, feature, threshold ORDER BY start_time, end_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS previous_end FROM thresholds
        ), grouped AS (
            SELECT *, SUM(CASE WHEN previous_end IS NULL OR start_time > previous_end THEN 1 ELSE 0 END)
                OVER (PARTITION BY stay_id, feature, threshold ORDER BY start_time, end_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS episode
            FROM ordered
        )
        SELECT stay_id, feature, threshold AS sofa_cv, MIN(start_time) AS start_time,
            MIN(start_time) + INTERVAL {policy["minimum_vasoactive_duration_minutes"]} MINUTE AS qualifying_time,
            MAX(end_time) AS end_time
        FROM grouped GROUP BY stay_id, feature, threshold, episode
        HAVING MAX(end_time) >= MIN(start_time) + INTERVAL {policy["minimum_vasoactive_duration_minutes"]} MINUTE
    """)


def build_urine(connection):
    connection.execute("""
        CREATE TEMP TABLE urine AS
        WITH items AS (
            SELECT stay_id, event_time, feature, itemid,
                CASE WHEN COUNT(*) = COUNT(valuenum) AND MIN(valuenum) = MAX(valuenum)
                    THEN MIN(valuenum) END AS volume
            FROM events WHERE feature IN ('urine_output', 'urine_irrigant_in', 'urine_irrigant_out')
                AND record_qc_status = 'accepted'
            GROUP BY stay_id, event_time, feature, itemid
        ), totals AS (
            SELECT stay_id, event_time,
                COUNT(*) FILTER (WHERE volume IS NULL) AS invalid_items,
                COUNT(*) FILTER (WHERE feature = 'urine_irrigant_in') AS irrigant_in_items,
                COUNT(*) FILTER (WHERE feature = 'urine_irrigant_out') AS irrigant_out_items,
                COUNT(*) FILTER (WHERE feature = 'urine_output') AS ordinary_items,
                SUM(volume) FILTER (WHERE feature = 'urine_output') AS ordinary_volume,
                SUM(volume) FILTER (WHERE feature = 'urine_irrigant_out') -
                    SUM(volume) FILTER (WHERE feature = 'urine_irrigant_in') AS net_irrigant_volume
            FROM items GROUP BY stay_id, event_time
        ), adjudicated AS (
            SELECT *,
                CASE WHEN invalid_items > 0 THEN NULL
                    WHEN irrigant_in_items = 0 AND irrigant_out_items = 0 THEN ordinary_volume
                    WHEN irrigant_in_items > 0 AND irrigant_out_items > 0
                        AND ordinary_items = 0 AND net_irrigant_volume >= 0 THEN net_irrigant_volume
                    ELSE NULL END AS urine_ml
            FROM totals
        )
        SELECT *, LAG(event_time) OVER (PARTITION BY stay_id ORDER BY event_time) AS previous_time
        FROM adjudicated
    """)


def build_timeline(connection, policy):
    connection.execute("""
        CREATE TEMP TABLE point_scores AS
        SELECT stay_id, event_time,
            MAX(score) FILTER (WHERE organ = 'resp') AS sofa_resp,
            MAX(score) FILTER (WHERE organ = 'coag') AS sofa_coag,
            MAX(score) FILTER (WHERE organ = 'liver') AS sofa_liver,
            MAX(score) FILTER (WHERE organ = 'cv') AS sofa_cv_map,
            MAX(score) FILTER (WHERE organ = 'cns') AS sofa_cns,
            MAX(score) FILTER (WHERE organ = 'renal') AS sofa_renal_creatinine
        FROM (
            SELECT stay_id, event_time,
                CASE feature WHEN 'map' THEN 'cv' WHEN 'platelets' THEN 'coag'
                    WHEN 'bilirubin' THEN 'liver' ELSE 'renal' END AS organ,
                CASE feature
                    WHEN 'map' THEN CASE WHEN valuenum < 70 THEN 1 ELSE 0 END
                    WHEN 'platelets' THEN CASE WHEN valuenum < 20 THEN 4 WHEN valuenum < 50 THEN 3
                        WHEN valuenum < 100 THEN 2 WHEN valuenum < 150 THEN 1 ELSE 0 END
                    WHEN 'bilirubin' THEN CASE WHEN valuenum >= 12 THEN 4 WHEN valuenum >= 6 THEN 3
                        WHEN valuenum >= 2 THEN 2 WHEN valuenum >= 1.2 THEN 1 ELSE 0 END
                    ELSE CASE WHEN valuenum >= 5 THEN 4 WHEN valuenum >= 3.5 THEN 3
                        WHEN valuenum >= 2 THEN 2 WHEN valuenum >= 1.2 THEN 1 ELSE 0 END END AS score
            FROM events WHERE numeric_usable AND feature IN ('map', 'platelets', 'bilirubin', 'creatinine')
            UNION ALL
            SELECT stay_id, event_time, 'resp', sofa_resp FROM respiratory WHERE sofa_resp IS NOT NULL
            UNION ALL
            SELECT stay_id, event_time, 'cns',
                CASE WHEN gcs_total < 6 THEN 4 WHEN gcs_total <= 9 THEN 3
                    WHEN gcs_total <= 12 THEN 2 WHEN gcs_total <= 14 THEN 1 ELSE 0 END
            FROM gcs WHERE gcs_total IS NOT NULL
        ) AS scores GROUP BY stay_id, event_time
    """)
    connection.execute("""
        CREATE TEMP TABLE assessment_times AS
        WITH times AS (
            SELECT stay_id, event_time AS assessment_time FROM point_scores
            UNION SELECT stay_id, event_time FROM urine
            UNION SELECT stay_id, event_time FROM gcs
            UNION SELECT stay_id, event_time FROM respiratory
            UNION SELECT stay_id, qualifying_time FROM pressor_intervals
            UNION SELECT stay_id, icu_intime FROM windows
            UNION SELECT stay_id, LEAST(icu_outtime, hospital_deathtime, window_end) FROM windows
            UNION SELECT stay_id, suspected_infection_time FROM candidates
            UNION SELECT stay_id, suspected_infection_time - INTERVAL 48 HOUR FROM candidates
            UNION SELECT stay_id, suspected_infection_time_upper_bound - INTERVAL 48 HOUR FROM candidates
            UNION SELECT stay_id, suspected_infection_time + INTERVAL 24 HOUR FROM candidates
            UNION SELECT stay_id, suspected_infection_time_upper_bound + INTERVAL 24 HOUR FROM candidates
            UNION SELECT stay_id, eligible_sit_lower_bound - INTERVAL 48 HOUR FROM candidates
            UNION SELECT stay_id, eligible_sit_upper_bound + INTERVAL 24 HOUR FROM candidates
            UNION
            SELECT stay_id, UNNEST(GENERATE_SERIES(window_start, window_end, INTERVAL 1 HOUR))
            FROM windows
        )
        SELECT times.*, envelope.icu_intime, envelope.icu_outtime,
            LEAST(envelope.icu_outtime, envelope.hospital_deathtime, envelope.window_end) AS care_end
        FROM times JOIN windows AS envelope USING (stay_id)
        WHERE assessment_time BETWEEN envelope.window_start AND envelope.window_end
    """)
    point_organs = ("resp", "coag", "liver", "cv_map", "cns", "renal_creatinine")
    expressions = ", ".join(
        f"MAX(points.sofa_{organ}) OVER rolling AS sofa_{organ}" for organ in point_organs
    )
    connection.execute(f"""
        CREATE TEMP TABLE rolling_points AS
        SELECT times.*, {expressions}
        FROM assessment_times AS times LEFT JOIN point_scores AS points
            ON times.stay_id = points.stay_id AND times.assessment_time = points.event_time
        WINDOW rolling AS (PARTITION BY times.stay_id ORDER BY assessment_time
            RANGE BETWEEN INTERVAL '86399999999 MICROSECONDS' PRECEDING AND CURRENT ROW)
    """)
    connection.execute("""
        CREATE TEMP TABLE rolling_pressors AS
        SELECT times.stay_id, times.assessment_time, MAX(pressor.sofa_cv) AS sofa_cv_drug
        FROM assessment_times AS times LEFT JOIN pressor_intervals AS pressor
            ON times.stay_id = pressor.stay_id AND pressor.qualifying_time <= times.assessment_time
            AND pressor.end_time > times.assessment_time - INTERVAL 24 HOUR
        GROUP BY times.stay_id, times.assessment_time
    """)
    connection.execute(f"""
        CREATE TEMP TABLE rolling_urine AS
        WITH coverage AS (
            SELECT times.stay_id, times.assessment_time, times.icu_intime,
                ROUND(SUM(urine.urine_ml), 6) AS observed_urine_ml,
                COUNT(urine.event_time) AS urine_observations,
                COUNT(*) FILTER (WHERE urine.event_time IS NOT NULL AND urine.urine_ml IS NULL) AS invalid_urine_times,
                MIN(urine.event_time) AS first_urine_time, MAX(urine.event_time) AS last_urine_time,
                MAX(EPOCH(urine.event_time - urine.previous_time) / 3600.0)
                    FILTER (WHERE urine.previous_time > times.assessment_time - INTERVAL 24 HOUR)
                    AS maximum_internal_gap_hours
            FROM assessment_times AS times LEFT JOIN urine
                ON times.stay_id = urine.stay_id
                AND urine.event_time > times.assessment_time - INTERVAL 24 HOUR
                AND urine.event_time <= times.assessment_time
            GROUP BY times.stay_id, times.assessment_time, times.icu_intime
        ), eligible AS (
            SELECT *, COALESCE(
                assessment_time >= icu_intime + INTERVAL 24 HOUR
                AND invalid_urine_times = 0
                AND last_urine_time - first_urine_time >= INTERVAL {policy["urine_minimum_span_hours"]} HOUR
                AND first_urine_time <= assessment_time - INTERVAL 24 HOUR
                    + INTERVAL {policy["urine_boundary_tolerance_hours"]} HOUR
                AND last_urine_time >= assessment_time - INTERVAL {policy["urine_boundary_tolerance_hours"]} HOUR
                AND maximum_internal_gap_hours <= {policy["urine_maximum_gap_hours"]}, FALSE)
                AS urine_coverage_adequate
            FROM coverage
        )
        SELECT * EXCLUDE (icu_intime),
            CASE WHEN urine_coverage_adequate THEN observed_urine_ml END AS urine_24h_ml,
            CASE WHEN NOT urine_coverage_adequate THEN NULL
                WHEN observed_urine_ml < 200 THEN 4 WHEN observed_urine_ml < 500 THEN 3
                ELSE 0 END AS sofa_renal_urine
        FROM eligible
    """)
    observed = " + ".join(f"CAST(sofa_{organ} IS NOT NULL AS INTEGER)" for organ in ORGANS)
    total = " + ".join(f"COALESCE(sofa_{organ}, 0)" for organ in ORGANS)
    connection.execute(f"""
        CREATE TEMP TABLE timeline AS
        WITH components AS (
            SELECT points.* EXCLUDE (sofa_cv_map, sofa_renal_creatinine),
                points.sofa_cv_map, pressors.sofa_cv_drug, points.sofa_renal_creatinine,
                urine.* EXCLUDE (stay_id, assessment_time),
                GREATEST(points.sofa_cv_map, pressors.sofa_cv_drug) AS sofa_cv,
                GREATEST(points.sofa_renal_creatinine, urine.sofa_renal_urine) AS sofa_renal
            FROM rolling_points AS points
            JOIN rolling_pressors AS pressors USING (stay_id, assessment_time)
            JOIN rolling_urine AS urine USING (stay_id, assessment_time)
        )
        SELECT *, {total} AS total_sofa, {observed} AS observed_components,
            assessment_time BETWEEN icu_intime AND care_end AS in_icu_assessment,
            assessment_time >= icu_intime + INTERVAL 24 HOUR AS full_icu_lookback,
            assessment_time - INTERVAL 24 HOUR AS lookback_start_exclusive
        FROM components
    """)


def adjudicate_candidates(connection):
    timing_keys = ("stay_id, suspected_infection_time, suspected_infection_time_upper_bound, "
                   "suspected_infection_time_upper_exclusive, eligible_sit_lower_bound, "
                   "eligible_sit_upper_bound, eligible_sit_upper_exclusive")
    connection.execute(f"CREATE TEMP TABLE candidate_timings AS SELECT DISTINCT {timing_keys} FROM candidates")
    group_keys = ", ".join(f"candidate.{name.strip()}" for name in timing_keys.split(","))
    connection.execute(f"""
        CREATE TEMP TABLE candidate_timing_scores AS
        SELECT {group_keys},
            COUNT(timeline.assessment_time) FILTER (WHERE timeline.in_icu_assessment) AS icu_assessments,
            COUNT(timeline.assessment_time) FILTER (
                WHERE timeline.in_icu_assessment AND timeline.observed_components > 0
            ) AS icu_assessments_with_evidence,
            MAX(timeline.total_sofa) FILTER (WHERE timeline.in_icu_assessment) AS maximum_associated_sofa,
            MIN(timeline.assessment_time) FILTER (
                WHERE timeline.in_icu_assessment AND timeline.total_sofa >= 2
                    AND timeline.assessment_time >= candidate.eligible_sit_lower_bound - INTERVAL 48 HOUR
                    AND (timeline.assessment_time < candidate.eligible_sit_upper_bound + INTERVAL 24 HOUR
                        OR (timeline.assessment_time = candidate.eligible_sit_upper_bound + INTERVAL 24 HOUR
                            AND NOT candidate.eligible_sit_upper_exclusive))) AS first_possible_sofa_time,
            MIN(timeline.assessment_time) FILTER (
                WHERE timeline.in_icu_assessment AND timeline.total_sofa >= 2
                    AND timeline.assessment_time >= candidate.suspected_infection_time_upper_bound - INTERVAL 48 HOUR
                    AND timeline.assessment_time <= candidate.suspected_infection_time + INTERVAL 24 HOUR
            ) AS first_qualifying_sofa_time,
            MAX(timeline.total_sofa) FILTER (
                WHERE timeline.observed_components > 0
                    AND timeline.assessment_time < candidate.suspected_infection_time
            ) AS sensitivity_pre_sit_peak_sofa,
            MAX(timeline.total_sofa) FILTER (
                WHERE timeline.observed_components > 0
                    AND timeline.assessment_time >= candidate.suspected_infection_time
                    AND timeline.assessment_time <= candidate.suspected_infection_time + INTERVAL 24 HOUR
            ) AS sensitivity_post_sit_peak_sofa
        FROM candidate_timings AS candidate LEFT JOIN timeline
            ON candidate.stay_id = timeline.stay_id
            AND timeline.assessment_time >= candidate.suspected_infection_time - INTERVAL 48 HOUR
            AND timeline.assessment_time <= candidate.suspected_infection_time_upper_bound + INTERVAL 24 HOUR
            AND (NOT candidate.suspected_infection_time_upper_exclusive
                OR timeline.assessment_time < candidate.suspected_infection_time_upper_bound + INTERVAL 24 HOUR)
            AND timeline.assessment_time <= timeline.care_end
        GROUP BY {group_keys}
    """)
    connection.execute(f"""
        CREATE TEMP TABLE candidate_scores AS
        SELECT candidate.stay_id, candidate.infection_candidate_rank, scores.* EXCLUDE ({timing_keys})
        FROM candidates AS candidate JOIN candidate_timing_scores AS scores USING ({timing_keys})
    """)
    connection.execute("""
        CREATE TEMP TABLE candidate_audit AS
        WITH assessed AS (
            SELECT candidate.*, scores.* EXCLUDE (stay_id, infection_candidate_rank),
                snapshot.* EXCLUDE (stay_id, assessment_time, icu_intime, icu_outtime, care_end),
                CASE WHEN scores.first_qualifying_sofa_time IS NOT NULL
                    THEN GREATEST(candidate.suspected_infection_time, scores.first_qualifying_sofa_time)
                    END AS sepsis_onset_time,
                CASE WHEN scores.first_qualifying_sofa_time IS NOT NULL
                    THEN GREATEST(candidate.suspected_infection_time_upper_bound, scores.first_qualifying_sofa_time)
                    END AS sepsis_onset_time_upper_bound,
                CASE WHEN scores.first_qualifying_sofa_time IS NOT NULL
                    THEN candidate.suspected_infection_time_upper_exclusive
                        AND candidate.suspected_infection_time_upper_bound > scores.first_qualifying_sofa_time
                    END AS sepsis_onset_time_upper_exclusive,
                scores.sensitivity_post_sit_peak_sofa - scores.sensitivity_pre_sit_peak_sofa
                    AS sensitivity_peak_delta,
                LEAST(candidate.icu_outtime, candidate.hospital_deathtime) AS followup_end
            FROM candidates AS candidate
            JOIN candidate_scores AS scores USING (stay_id, infection_candidate_rank)
            LEFT JOIN timeline AS snapshot ON candidate.stay_id = snapshot.stay_id
                AND scores.first_qualifying_sofa_time = snapshot.assessment_time
        )
        SELECT *, CASE
            WHEN icu_assessments = 0 THEN 'no_in_icu_assessment'
            WHEN icu_assessments_with_evidence = 0 THEN 'no_usable_sofa_evidence'
            WHEN requires_timing_adjudication THEN 'infection_timing_uncertain'
            WHEN first_qualifying_sofa_time IS NULL AND first_possible_sofa_time IS NOT NULL
                THEN 'date_uncertainty_only'
            WHEN first_qualifying_sofa_time IS NULL THEN 'sofa_below_two'
            WHEN sepsis_onset_time_upper_bound > followup_end THEN 'onset_outside_followup'
            ELSE 'qualifies' END AS adjudication_status,
            0 AS baseline_sofa, TRUE AS baseline_sofa_assumed_zero,
            total_sofa AS sofa_delta,
            sensitivity_peak_delta >= 2 AS sensitivity_measured_baseline_sepsis,
            sepsis_onset_time_upper_bound > sepsis_onset_time AS onset_time_uncertain
        FROM assessed
    """)
    connection.execute("""
        CREATE TEMP TABLE selected AS
        SELECT * FROM candidate_audit WHERE adjudication_status = 'qualifies'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY infection_candidate_rank,
            first_qualifying_sofa_time) = 1
    """)
    candidate_columns = set(columns(connection, "selected"))
    annotation_columns = [
        name for name in columns(connection, "cohort")
        if name not in candidate_columns and name not in (
            "icu_hours_after_sit", "icu_ge_24h_after_sit", "restricted_phenotype_eligible",
        )
    ]
    annotations = ", ".join(f'patient."{name}"' for name in annotation_columns)
    connection.execute(f"""
        CREATE TEMP TABLE final_cohort AS
        SELECT selected.*, {annotations},
            selected.first_qualifying_sofa_time AS sofa_time,
            selected.first_qualifying_sofa_time AS sofa_deterioration_time,
            TRUE AS sepsis3,
            EPOCH(selected.icu_outtime - selected.suspected_infection_time) / 3600.0 AS icu_hours_after_sit,
            selected.icu_outtime >= selected.suspected_infection_time + INTERVAL 24 HOUR AS icu_ge_24h_after_sit,
            NOT patient.has_alternative_diagnosis
                AND selected.icu_outtime >= selected.suspected_infection_time + INTERVAL 24 HOUR
                AS restricted_phenotype_eligible,
            EPOCH(selected.followup_end - selected.sepsis_onset_time) / 3600.0 AS hours_available_after_onset,
            selected.followup_end >= selected.sepsis_onset_time_upper_bound + INTERVAL 24 HOUR
                AS complete_24h_after_onset,
            patient.cci_renal > 0 OR patient.cci_mild_liver > 0 OR patient.cci_sev_liver > 0
                OR patient.cci_chf > 0 AS retrospective_chronic_organ_disease_flag,
            (SELECT MIN(respiratory.pf_ratio) FROM respiratory WHERE respiratory.stay_id = selected.stay_id
                AND respiratory.event_time >= selected.sepsis_onset_time - INTERVAL 24 HOUR
                AND respiratory.event_time < selected.sepsis_onset_time) AS baseline_pf_ratio
        FROM selected JOIN cohort AS patient USING (stay_id, subject_id, hadm_id)
    """)
    connection.execute("""
        CREATE TEMP TABLE stay_audit AS
        SELECT patient.stay_id, COUNT(candidate.infection_candidate_rank) AS candidates,
            COUNT(*) FILTER (WHERE candidate.adjudication_status = 'qualifies') AS qualifying_candidates,
            MAX(candidate.maximum_associated_sofa) AS maximum_associated_sofa,
            BOOL_OR(candidate.first_possible_sofa_time IS NOT NULL) AS possible_sepsis,
            CASE WHEN BOOL_OR(candidate.sensitivity_measured_baseline_sepsis) THEN TRUE
                WHEN COUNT(*) FILTER (WHERE candidate.sensitivity_measured_baseline_sepsis IS NULL) > 0
                    THEN NULL ELSE FALSE END AS measured_baseline_sensitivity,
            final.stay_id IS NOT NULL AS included,
            final.infection_candidate_rank AS selected_candidate_rank,
            final.sepsis_onset_time,
            CASE WHEN final.stay_id IS NOT NULL THEN 'included'
                WHEN BOOL_OR(candidate.adjudication_status = 'infection_timing_uncertain') THEN 'infection_timing_uncertain'
                WHEN BOOL_OR(candidate.adjudication_status = 'date_uncertainty_only') THEN 'date_uncertainty_only'
                WHEN BOOL_OR(candidate.adjudication_status = 'onset_outside_followup') THEN 'onset_outside_followup'
                WHEN MAX(candidate.icu_assessments) = 0 THEN 'no_in_icu_assessment'
                WHEN MAX(candidate.icu_assessments_with_evidence) = 0 THEN 'no_usable_sofa_evidence'
                ELSE 'sofa_below_two' END AS reason
        FROM cohort AS patient JOIN candidate_audit AS candidate USING (stay_id)
        LEFT JOIN final_cohort AS final ON patient.stay_id = final.stay_id
        GROUP BY patient.stay_id, final.stay_id, final.infection_candidate_rank, final.sepsis_onset_time
    """)


def build_sepsis_cohort(processed_dir=PROCESSED_DIR, metrics_dir=METRICS_DIR,
                       policy_file=POLICY_FILE, registry_file=REGISTRY_FILE):
    started = time.perf_counter()
    processed_dir, metrics_dir = Path(processed_dir), Path(metrics_dir)
    policy = json.loads(Path(policy_file).read_text(encoding="utf-8"))["sofa_policy"]
    if policy["schema_version"] != "1.1.0" or policy["rolling_hours"] != 24 or policy["association_hours"] != [-48, 24]:
        raise ValueError("Unsupported SOFA policy")
    for key in ("fio2_lookback_hours", "support_lookback_hours", "minimum_vasoactive_duration_minutes",
                "urine_minimum_span_hours", "urine_maximum_gap_hours", "urine_boundary_tolerance_hours"):
        if not isinstance(policy[key], int) or not 0 < policy[key] <= 1440:
            raise ValueError(f"Invalid SOFA policy parameter: {key}")
    inputs = {
        "cleaned": processed_dir / "events_clean.parquet",
        "cohort": processed_dir / "phenotypes.parquet",
        "candidates": processed_dir / "infection_candidates.parquet",
        "windows": processed_dir / "extraction_windows.parquet",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required adjudication input not found: {path}")
    registry = json.loads(Path(registry_file).read_text(encoding="utf-8"))
    units = {entry["name"]: entry["canonical_unit"] for entry in registry["temporal_features"]}
    units.update(registry["evidence_features"])
    units.update(json.loads(Path(policy_file).read_text())["source_mapping"]["evidence_features"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".05_", dir=processed_dir) as temporary:
        staging = Path(temporary)
        with duckdb.connect() as connection:
            connection.execute("SET threads = 4")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(f"SET temp_directory = {sql_string((staging / 'spill').as_posix())}")
            for name, path in inputs.items():
                connection.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_string(path.as_posix())})")
            connection.execute("CREATE TEMP TABLE unit_map (feature VARCHAR, unit VARCHAR)")
            connection.executemany("INSERT INTO unit_map VALUES (?, ?)", list(units.items()))
            validate_inputs(connection)
            connection.execute(f"""
                CREATE TEMP TABLE events AS SELECT * FROM cleaned WHERE feature IN ({sql_list(SOFA_FEATURES)})
            """)
            build_respiratory(connection, policy)
            build_gcs_and_pressors(connection, policy)
            build_urine(connection)
            build_timeline(connection, policy)
            adjudicate_candidates(connection)
            require_zero(connection, """
                SELECT COUNT(*) FROM timeline WHERE total_sofa < 0 OR total_sofa > 24
                    OR observed_components < 0 OR observed_components > 6
            """, "Invalid SOFA totals")
            require_zero(connection, """
                SELECT COUNT(*) FROM final_cohort WHERE sofa_delta < 2
                    OR requires_timing_adjudication OR candidate_timing_status IS DISTINCT FROM 'definite'
                    OR sepsis_onset_time < icu_intime OR sepsis_onset_time_upper_bound > followup_end
                    OR first_qualifying_sofa_time < suspected_infection_time_upper_bound - INTERVAL 48 HOUR
                    OR first_qualifying_sofa_time > suspected_infection_time + INTERVAL 24 HOUR
            """, "Invalid final sepsis decisions")
            require_zero(connection, """
                SELECT COUNT(*) - COUNT(DISTINCT stay_id) FROM final_cohort
            """, "Duplicate final stays")
            summary_queries = {
                "suspected_infection_stays": "SELECT COUNT(*) FROM cohort",
                "infection_candidates": "SELECT COUNT(*) FROM candidates",
                "assessment_rows": "SELECT COUNT(*) FROM timeline",
                "qualifying_candidates": "SELECT COUNT(*) FROM candidate_audit WHERE adjudication_status = 'qualifies'",
                "sepsis3_stays": "SELECT COUNT(*) FROM final_cohort",
                "selected_later_candidate": "SELECT COUNT(*) FROM final_cohort WHERE infection_candidate_rank > 1",
                "selected_date_only_culture": "SELECT COUNT(*) FROM final_cohort WHERE culture_time_is_date_only",
                "selected_pre_sit_sofa": "SELECT COUNT(*) FROM final_cohort WHERE sofa_time < suspected_infection_time",
                "selected_incomplete_24h_followup": "SELECT COUNT(*) FROM final_cohort WHERE NOT complete_24h_after_onset",
                "selected_with_retrospective_chronic_organ_flag": "SELECT COUNT(*) FROM final_cohort WHERE retrospective_chronic_organ_disease_flag",
                "paired_arterial_gases": "SELECT COUNT(*) FROM respiratory WHERE pf_ratio IS NOT NULL",
                "respiratory_support_conflicts": "SELECT COUNT(*) FROM respiratory WHERE support_status = 'conflict'",
                "complete_gcs_assessments": "SELECT COUNT(*) FROM gcs WHERE gcs_total IS NOT NULL",
                "incomplete_gcs_assessments": "SELECT COUNT(*) FROM gcs WHERE gcs_total IS NULL",
                "urine_coverage_adequate_assessments": "SELECT COUNT(*) FROM timeline WHERE in_icu_assessment AND urine_coverage_adequate",
                "selected_urine_changes_renal_score": "SELECT COUNT(*) FROM final_cohort WHERE sofa_renal_urine > COALESCE(sofa_renal_creatinine, 0)",
            }
            summary = {name: connection.execute(query).fetchone()[0] for name, query in summary_queries.items()}
            if not summary["sepsis3_stays"]:
                raise ValueError("No final sepsis stays; review infection/adjudication logs before proceeding")
            reasons = dict(connection.execute("SELECT reason, COUNT(*) FROM stay_audit GROUP BY reason ORDER BY reason").fetchall())
            missingness = {
                organ: connection.execute(f"SELECT COUNT(*) FROM final_cohort WHERE sofa_{organ} IS NULL").fetchone()[0]
                for organ in ORGANS
            }
            comparison = [
                {"primary": primary, "measured_baseline_sensitivity": sensitivity, "stays": count}
                for primary, sensitivity, count in connection.execute("""
                    SELECT included, measured_baseline_sensitivity, COUNT(*)
                    FROM stay_audit GROUP BY included, measured_baseline_sensitivity
                """).fetchall()
            ]
            outputs = {
                "sepsis_cohort.parquet": "final_cohort",
                "candidate_decisions.parquet": "candidate_audit",
                "stay_decisions.parquet": "stay_audit",
                "sofa_timeline.parquet": "timeline",
                "respiratory_evidence.parquet": "respiratory",
                "gcs_evidence.parquet": "gcs",
                "pressor_intervals.parquet": "pressor_intervals",
                "urine_evidence.parquet": "urine",
            }
            for filename, table in outputs.items():
                connection.execute(f"COPY {table} TO {sql_string((staging / filename).as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
        report = {
            "common_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE_DIR / "src/common").glob("*.json")},
            "dataset_config_sha256": hashlib.sha256(Path(__file__).with_name("dataset_config.json").read_bytes()).hexdigest(),
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION, "duckdb_version": duckdb.__version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": policy, "policy_sha256": hashlib.sha256(Path(policy_file).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {name: {"path": str(path), "size_bytes": path.stat().st_size} for name, path in inputs.items()},
            "summary": summary, "stay_attrition": reasons,
            "selected_missing_components": missingness, "baseline_sensitivity_comparison": comparison,
            "limitations": [
                "Research phenotype, not proven infection causality or patient-specific acute SOFA increase.",
                "Zero baseline may misclassify preexisting dysfunction; CCI is retrospective, not a baseline SOFA.",
                "GCS sedation confounding is not adjudicated; ETT and incomplete assessments are withheld.",
                "Urine coverage is a recording-density proxy; unpaired irrigation prevents urine-based scoring.",
                "Event timestamps are retrospective; storetime availability is not an online detection guarantee.",
                "Support freshness, pressor duration, missingness and onset conventions differ from reference SQL.",
                "baseline_pf_ratio is the worst paired P/F in 24h strictly before released onset, not a chronic baseline.",
            ],
        }
        report_file = staging / "06_sepsis.json"
        report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for filename in outputs:
            (staging / filename).replace(processed_dir / filename)
        report_file.replace(metrics_dir / report_file.name)
    print(f"[06] Complete v{DATASET_VERSION}; {time.perf_counter() - started:.1f}s; report: {metrics_dir}/06_sepsis.json", flush=True)
    return report


if __name__ == "__main__":
    print(f"[06] Adjudicate Sepsis-3 using rolling SOFA and definite infection timing. v{DATASET_VERSION}", flush=True)
    build_sepsis_cohort()
