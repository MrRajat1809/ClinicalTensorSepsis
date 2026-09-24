"""Select CareVue infection candidates, preserving uncertain prescription dates.

Pairing uses c-a in [-72h,24h]. Project the feasible antibiotic/culture
rectangle onto each axis before computing the range of min(a,c).
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parents[2]
MIMIC_DIR = BASE_DIR / "data/raw/mimic3-carevue/1.4"
PROCESSED_DIR = BASE_DIR / "data/processed/mimic3-carevue/work"
METRICS_DIR = BASE_DIR / "outputs/mimic3-carevue"
CONFIG_PATH = Path(__file__).with_name("dataset_config.json")
CONFIG = json.loads(CONFIG_PATH.read_text())


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalized_label_sql(column):
    return f"TRIM(REGEXP_REPLACE(UPPER(COALESCE({column}, '')), '\\s+', ' ', 'g'))"


def pair_candidates(connection):
    """Consume base, antimicrobial_starts and culture_audit; produce candidates."""
    connection.execute("""
        CREATE TEMP TABLE coupled AS
        WITH projected AS (
            SELECT base.*, a.* EXCLUDE(stay_id), c.* EXCLUDE(stay_id),
                GREATEST(c.culture_time, a.abx_start_time - INTERVAL 72 HOUR) AS c_lo,
                LEAST(c.culture_time_upper_bound, a.abx_start_time_upper_bound + INTERVAL 24 HOUR) AS c_hi,
                (c.culture_time_upper_exclusive AND c.culture_time_upper_bound <= a.abx_start_time_upper_bound + INTERVAL 24 HOUR)
                    OR a.abx_start_time_upper_bound + INTERVAL 24 HOUR <= c.culture_time_upper_bound AS c_open,
                GREATEST(a.abx_start_time, c.culture_time - INTERVAL 24 HOUR) AS a_lo,
                LEAST(a.abx_start_time_upper_bound, c.culture_time_upper_bound + INTERVAL 72 HOUR) AS a_hi,
                a.abx_start_time_upper_bound <= c.culture_time_upper_bound + INTERVAL 72 HOUR
                    OR (c.culture_time_upper_exclusive AND c.culture_time_upper_bound + INTERVAL 72 HOUR <= a.abx_start_time_upper_bound) AS a_open,
                c.culture_time >= a.abx_start_time_upper_bound - INTERVAL 72 HOUR
                    AND c.culture_time_upper_bound <= a.abx_start_time + INTERVAL 24 HOUR AS pairing_is_definite
            FROM base JOIN antimicrobial_starts a USING(stay_id)
            JOIN culture_audit c USING(stay_id)
            WHERE c.culture_time < a.abx_start_time_upper_bound + INTERVAL 24 HOUR
                AND (c.culture_time_upper_bound > a.abx_start_time - INTERVAL 72 HOUR
                    OR (c.culture_time_upper_bound = a.abx_start_time - INTERVAL 72 HOUR AND NOT c.culture_time_upper_exclusive))
        )
        SELECT *, LEAST(a_lo,c_lo) AS suspected_infection_time,
            LEAST(a_hi,c_hi) AS suspected_infection_time_upper_bound,
            (a_open AND a_hi <= c_hi) OR (c_open AND c_hi <= a_hi) AS suspected_infection_time_upper_exclusive
        FROM projected
        WHERE (a_lo < a_hi OR (a_lo = a_hi AND NOT a_open))
            AND (c_lo < c_hi OR (c_lo = c_hi AND NOT c_open))
    """)
    connection.execute("""
        CREATE TEMP TABLE candidates AS
        WITH presentation AS (
            SELECT *, suspected_infection_time >= icu_intime - INTERVAL 24 HOUR
                AND suspected_infection_time_upper_bound <= icu_intime + INTERVAL 24 HOUR AS presentation_is_definite,
                GREATEST(suspected_infection_time, icu_intime - INTERVAL 24 HOUR) AS eligible_sit_lower_bound,
                LEAST(suspected_infection_time_upper_bound, icu_intime + INTERVAL 24 HOUR) AS eligible_sit_upper_bound,
                suspected_infection_time_upper_exclusive
                    AND suspected_infection_time_upper_bound <= icu_intime + INTERVAL 24 HOUR AS eligible_sit_upper_exclusive
            FROM coupled
            WHERE suspected_infection_time <= icu_intime + INTERVAL 24 HOUR
                AND (suspected_infection_time_upper_bound > icu_intime - INTERVAL 24 HOUR
                    OR (suspected_infection_time_upper_bound = icu_intime - INTERVAL 24 HOUR
                        AND NOT suspected_infection_time_upper_exclusive))
        )
        SELECT *, CASE WHEN pairing_is_definite AND presentation_is_definite THEN 'definite' ELSE 'possible' END AS candidate_timing_status,
            NOT(pairing_is_definite AND presentation_is_definite) AS requires_timing_adjudication,
            '2.1.0' AS infection_schema_version,
            ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY suspected_infection_time,
                culture_time_is_date_only, culture_time, abx_start_time, abx_stop_time NULLS LAST,
                culture_event_id, antimicrobial_start_id) AS infection_candidate_rank
        FROM presentation
    """)


def build_infection_cohort(mimic_dir=MIMIC_DIR, processed_dir=PROCESSED_DIR, metrics_dir=METRICS_DIR):
    mimic_dir, processed_dir, metrics_dir = map(Path, (mimic_dir, processed_dir, metrics_dir))
    policy = CONFIG['infection_policy']
    if any(not isinstance(name, str) or not name.strip() for name in policy['diagnostic_specimens'].values()):
        raise ValueError('Diagnostic specimen mappings require nonempty expected names')
    pattern = '(^|[^a-z])(' + '|'.join(re.escape(n) for n in CONFIG['antimicrobial_names']) + ')([^a-z]|$)'
    routes = ','.join(sql_string(r) for r in CONFIG['iv_routes'])
    with duckdb.connect() as connection:
        connection.execute(f"CREATE TABLE base AS SELECT * FROM read_parquet({sql_string(processed_dir / 'base_cohort.parquet')})")
        connection.execute(f"""
            CREATE TABLE prescription_audit AS
            SELECT base.stay_id, p.row_id, p.startdate, p.enddate, p.drug,
                UPPER(TRIM(p.route)) AS route,
                DATE_TRUNC('day',p.startdate) AS abx_start_time,
                DATE_TRUNC('day',p.startdate) + INTERVAL 1 DAY AS abx_start_time_upper_bound,
                DATE_TRUNC('day',p.enddate) AS abx_stop_time,
                CASE WHEN NOT REGEXP_MATCHES(LOWER(COALESCE(drug,'')),{sql_string(pattern)}) THEN 'unmapped_name'
                    WHEN UPPER(TRIM(COALESCE(drug_type,''))) = 'BASE'
                        OR REGEXP_MATCHES(LOWER(drug),'cream|desensiti|ophth|ointment|lock|gel') THEN 'excluded_preparation'
                    WHEN UPPER(TRIM(COALESCE(p.route,''))) NOT IN ({routes}) THEN 'non_iv_or_missing_route'
                    WHEN startdate IS NULL THEN 'missing_start'
                    WHEN enddate < startdate THEN 'invalid_interval'
                    ELSE 'included' END AS prescription_status
            FROM read_csv_auto({sql_string(mimic_dir / 'PRESCRIPTIONS.csv.gz')},
                types={{'startdate':'TIMESTAMP','enddate':'TIMESTAMP','drug':'VARCHAR','route':'VARCHAR'}}) p
            JOIN base USING(subject_id,hadm_id)
        """)
        connection.execute("""
            CREATE TABLE antimicrobial_starts AS
            SELECT stay_id, abx_start_time, abx_start_time_upper_bound, abx_stop_time,
                LOWER(TRIM(drug)) AS abx_drug, route AS abx_route,
                TRUE AS abx_start_time_is_date_only, TRUE AS abx_start_time_upper_exclusive,
                MIN(row_id) AS antimicrobial_start_id,
                NULL::DOUBLE AS abx_duration_hours, FALSE AS abx_duration_complete
            FROM prescription_audit WHERE prescription_status = 'included' GROUP BY ALL
        """)
        connection.execute('CREATE TABLE specimen_map(spec_itemid INTEGER, mapped_name VARCHAR)')
        connection.executemany('INSERT INTO specimen_map VALUES (?,?)',[(int(k),v) for k,v in policy['diagnostic_specimens'].items()])
        connection.execute(f"""
            CREATE TABLE culture_evidence AS
            WITH grouped AS (
                SELECT base.stay_id, m.spec_itemid, m.spec_type_desc,
                    COALESCE(m.charttime,DATE_TRUNC('day',m.chartdate)) AS culture_time,
                    m.charttime IS NULL AS culture_time_is_date_only,
                    MIN(m.row_id) AS culture_event_id, COUNT(*) AS source_result_rows,
                    BOOL_AND(UPPER(TRIM(COALESCE(m.org_name,''))) IN ('CANCELLED','CANCELED')) AS cancelled_only
                FROM read_csv_auto({sql_string(mimic_dir / 'MICROBIOLOGYEVENTS.csv.gz')},
                    types={{'charttime':'TIMESTAMP','chartdate':'TIMESTAMP','spec_itemid':'INTEGER','org_name':'VARCHAR'}}) m
                JOIN base USING(subject_id,hadm_id) GROUP BY ALL
            )
            SELECT grouped.*, CASE WHEN culture_time IS NULL THEN 'missing_culture_time'
                WHEN cancelled_only THEN 'cancelled_test'
                WHEN {normalized_label_sql('spec_type_desc')} = '' THEN 'missing_specimen_name'
                WHEN mapped_name IS NULL THEN 'unmapped_or_excluded_specimen'
                WHEN {normalized_label_sql('spec_type_desc')} IS DISTINCT FROM
                    {normalized_label_sql('mapped_name')} THEN 'specimen_label_mismatch'
                ELSE 'diagnostic_culture' END AS culture_evidence_status
            FROM grouped LEFT JOIN specimen_map USING(spec_itemid)
        """)
        connection.execute("""
            CREATE TABLE culture_audit AS SELECT *,
                CASE WHEN culture_time_is_date_only THEN culture_time + INTERVAL 1 DAY ELSE culture_time END AS culture_time_upper_bound,
                culture_time_is_date_only AS culture_time_upper_exclusive,
                [spec_type_desc] AS diagnostic_evidence_names,
                'specimen_proxy_no_test_field' AS diagnostic_evidence_basis
            FROM culture_evidence WHERE culture_evidence_status = 'diagnostic_culture'
        """)
        pair_candidates(connection)
        counts = {table:connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                  for table in ['base','antimicrobial_starts','culture_evidence','coupled','candidates']}
        if not counts['candidates']:
            raise ValueError('No infection candidates; inspect source mappings')
        metrics_dir.mkdir(parents=True,exist_ok=True)
        for filename, query in {
            'infection_cohort.parquet':'SELECT * FROM candidates WHERE infection_candidate_rank=1 ORDER BY stay_id',
            'infection_candidates.parquet':'SELECT * FROM candidates ORDER BY stay_id,infection_candidate_rank',
            'culture_evidence.parquet':'SELECT * FROM culture_evidence',
            'prescription_evidence.parquet':'SELECT * FROM prescription_audit',
        }.items():
            connection.execute(f'COPY ({query}) TO {sql_string(processed_dir/filename)} (FORMAT PARQUET)')
        for table, status, filename in [('culture_evidence','culture_evidence_status','culture_qc.csv'),('prescription_audit','prescription_status','antimicrobial_qc.csv')]:
            group = 'spec_itemid,spec_type_desc' if table == 'culture_evidence' else 'drug,route'
            connection.execute(f'COPY (SELECT {group},{status},COUNT(*) AS rows FROM {table} GROUP BY ALL ORDER BY rows DESC) TO {sql_string(metrics_dir/filename)} (HEADER)')
        counts['infection_stays'] = connection.execute('SELECT COUNT(DISTINCT stay_id) FROM candidates').fetchone()[0]
        counts['definite_infection_stays'] = connection.execute("SELECT COUNT(DISTINCT stay_id) FROM candidates WHERE candidate_timing_status='definite'").fetchone()[0]
        report = {
            'common_sha256': {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (BASE_DIR / 'src/common').glob('*.json')
            },
            'source_database': CONFIG['source_database'],
            'source_version': '1.4',
            'dataset_version': CONFIG['dataset_version'],
            'infection_schema_version': '2.1.0',
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dataset_config_sha256': hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            'policy': policy,
            'prescription_time_policy': CONFIG['carevue']['prescription_time'],
            'summary': counts,
        }
        (metrics_dir/'02_infection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"[02] {counts['infection_stays']:,} candidate stays; {counts['definite_infection_stays']:,} with definite timing",flush=True)
    return report


if __name__ == '__main__':
    build_infection_cohort()
