"""Select an adult first ICU stay using source offsets before eligibility filtering."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name('dataset_config.json')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
RAW = BASE_DIR / CONFIG['eicu']['raw_directory']
WORK = BASE_DIR / 'data/processed/eicu/work'
OUT = BASE_DIR / 'outputs/eicu'
ORIGIN = "TIMESTAMP '2000-01-01'"


def sql_string(value):
    return "'" + str(value).replace('\\', '/').replace("'", "''") + "'"


def source_path(directory, table):
    matches = [p for p in Path(directory).iterdir()
               if p.name.lower() in (table.lower()+'.csv.gz', table.lower()+'.csv')]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one {table}.csv[.gz] in {directory}; found {len(matches)}')
    return matches[0]


def csv_query(directory, table, columns):
    """Read only declared columns; source text is never stripped into a numeric value."""
    path = source_path(directory, table)
    fields = ', '.join(f'TRY_CAST("{name}" AS {dtype}) AS "{name}"' for name, dtype in columns.items())
    return f'SELECT {fields} FROM read_csv_auto({sql_string(path)}, all_varchar=true)'


def at_offset(expression):
    return f'({ORIGIN} + ({expression}) * INTERVAL 1 MINUTE)'


def write_report(filename, script, summary, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(dataset_version=CONFIG['dataset_version'], duckdb_version=duckdb.__version__, source_database=CONFIG['source_database'],
        source_version=CONFIG['source_version'], generated_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(script).read_bytes()).hexdigest(),
        dataset_config_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        common_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE_DIR/'src/common').glob('*.json')},
        summary=summary, **details)
    (OUT/filename).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def save_table(con, table, filename, directory=WORK):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    pending = path.with_suffix(path.suffix+'.pending')
    con.execute(f'COPY (SELECT * FROM {table}) TO {sql_string(pending)} (FORMAT PARQUET, COMPRESSION ZSTD)')
    pending.replace(path)


def build_base_cohort():
    WORK.mkdir(parents=True, exist_ok=True)
    patient = {k:'BIGINT' for k in ['patientunitstayid','patienthealthsystemstayid','hospitalid',
        'hospitaladmitoffset','hospitaldischargeoffset','unitdischargeoffset','unitvisitnumber','hospitaldischargeyear']}
    patient.update({k:'VARCHAR' for k in ['uniquepid','age','gender','ethnicity','unittype',
        'hospitaldischargestatus','unitdischargestatus','unitstaytype']})
    with duckdb.connect() as con:
        con.execute(f'CREATE TABLE patient AS {csv_query(RAW,"patient",patient)}')
        count, unique, nulls = con.execute('''SELECT COUNT(*), COUNT(DISTINCT patientunitstayid),
            COUNT(*) FILTER (WHERE patientunitstayid IS NULL OR patienthealthsystemstayid IS NULL
                OR NULLIF(TRIM(uniquepid),'') IS NULL) FROM patient''').fetchone()
        if count == 0 or count != unique or nulls:
            raise ValueError('Invalid patient stay IDs or missing patient/encounter identifiers')
        con.execute(f'''CREATE TABLE surgery AS
            SELECT patientunitstayid,
                CASE WHEN COUNT(DISTINCT electivesurgery) FILTER (WHERE electivesurgery IN (0,1))=1
                    THEN MIN(electivesurgery) FILTER (WHERE electivesurgery IN (0,1)) END AS elective_surgery,
                COUNT(DISTINCT electivesurgery) FILTER (WHERE electivesurgery IN (0,1))>1 AS surgery_conflict
            FROM ({csv_query(RAW,'apachePredVar',dict(patientunitstayid='BIGINT',electivesurgery='INTEGER'))}) GROUP BY patientunitstayid''')
        accepted = CONFIG['tensor']['category_vocabularies']['first_careunit']
        excluded = CONFIG['eicu']['excluded_unit_types']
        unit_sql = ','.join(sql_string(s) for s in accepted)
        all_units = ','.join(sql_string(s) for s in accepted+excluded)
        unknown_units = con.execute(f"SELECT unittype, COUNT(*) FROM patient WHERE UPPER(TRIM(COALESCE(unittype,''))) NOT IN ({all_units}) GROUP BY unittype").fetchall()
        if unknown_units:
            write_report('01_base.json', __file__, dict(unknown_unit_types=unknown_units), status='FAILED_UNIT_VOCABULARY')
            raise ValueError(f'Unmapped unit types; classify explicitly in dataset_config.json: {unknown_units}')
        con.execute(f'''CREATE TABLE eligibility AS
            WITH ranked AS (
                SELECT *, DENSE_RANK() OVER (ORDER BY uniquepid) AS subject_id,
                    COUNT(DISTINCT patienthealthsystemstayid) OVER (PARTITION BY uniquepid) AS hospital_encounters,
                    COUNT(*) FILTER (WHERE hospitaladmitoffset IS NULL) OVER (PARTITION BY uniquepid)>0 AS missing_order_offset,
                    ROW_NUMBER() OVER (PARTITION BY uniquepid ORDER BY hospitaladmitoffset DESC NULLS LAST,patientunitstayid) AS icu_seq,
                    COUNT(*) OVER (PARTITION BY uniquepid,hospitaladmitoffset) AS order_ties
                FROM patient
            ), joined AS (
                SELECT r.*, patientunitstayid AS stay_id, patienthealthsystemstayid AS hadm_id,
                    uniquepid AS source_subject_id, hospitalid AS hospital_id, age AS source_age,
                    REGEXP_REPLACE(TRIM(age),'\\s+','','g')='>89' AS age_is_deidentified,
                    CASE WHEN REGEXP_REPLACE(TRIM(age),'\\s+','','g')='>89' THEN 91 ELSE TRY_CAST(age AS INTEGER) END AS canonical_age,
                    UPPER(TRIM(ethnicity)) AS race, UPPER(TRIM(unittype)) AS first_careunit,
                    CASE elective_surgery WHEN 1 THEN 'ELECTIVE' WHEN 0 THEN 'NON_ELECTIVE' ELSE 'UNKNOWN_SURGERY_STATUS' END AS admission_type,
                    elective_surgery, COALESCE(surgery_conflict,FALSE) AS surgery_conflict,
                    {ORIGIN} AS icu_intime, {at_offset('unitdischargeoffset')} AS icu_outtime,
                    {at_offset('hospitaladmitoffset')} AS hospital_admittime,
                    {at_offset('hospitaldischargeoffset')} AS hospital_dischtime,
                    CASE UPPER(TRIM(hospitaldischargestatus)) WHEN 'EXPIRED' THEN 1 WHEN 'ALIVE' THEN 0 END AS hospital_expire_flag,
                    CASE WHEN UPPER(TRIM(unitdischargestatus))='EXPIRED' THEN {at_offset('unitdischargeoffset')}
                         WHEN UPPER(TRIM(hospitaldischargestatus))='EXPIRED' THEN {at_offset('hospitaldischargeoffset')} END AS hospital_deathtime,
                    CASE WHEN UPPER(TRIM(unitdischargestatus))='EXPIRED' THEN 'expired_unit_discharge_offset'
                         WHEN UPPER(TRIM(hospitaldischargestatus))='EXPIRED' THEN 'expired_hospital_discharge_offset'
                         ELSE 'not_reported_dead' END AS death_time_basis,
                    UPPER(TRIM(hospitaldischargestatus))='EXPIRED' AS death_time_is_proxy,
                    FALSE AS exact_death_time_available,
                    unitdischargeoffset/1440.0 AS icu_los_days,
                    unitdischargeoffset/60.0 AS icu_los_hours,
                    unitdischargeoffset>=1440 AS icu_los_ge_24h,
                    hospital_encounters=1 AND NOT missing_order_offset AND order_ties=1 AS first_stay_order_known
                FROM ranked r LEFT JOIN surgery USING(patientunitstayid)
            ), reasoned AS (
                SELECT * EXCLUDE(age,canonical_age,gender), canonical_age AS age, UPPER(TRIM(gender)) AS gender,
                    hospital_expire_flag=1 AND hospital_deathtime IS NULL AS death_time_unavailable,
                    CASE WHEN hospital_encounters<>1 THEN 'ambiguous_hospital_order'
                         WHEN missing_order_offset THEN 'missing_first_stay_offset'
                         WHEN icu_seq<>1 THEN 'not_first_icu_stay'
                         WHEN order_ties<>1 THEN 'ambiguous_first_stay_offset'
                         WHEN first_careunit NOT IN ({unit_sql}) THEN 'non_icu_unit'
                         WHEN unitdischargeoffset IS NULL OR unitdischargeoffset<=0 THEN 'invalid_icu_interval'
                         WHEN hospitaladmitoffset>0 OR hospitaldischargeoffset IS NULL
                             OR hospitaldischargeoffset<unitdischargeoffset THEN 'invalid_hospital_interval'
                         WHEN canonical_age IS NULL THEN 'missing_age'
                         WHEN canonical_age<18 THEN 'age_under_18'
                         WHEN canonical_age>120 THEN 'invalid_age'
                         WHEN elective_surgery=1 THEN 'elective_surgery'
                         WHEN surgery_conflict THEN 'conflicting_elective_surgery'
                         WHEN hospital_expire_flag IS NULL THEN 'missing_mortality_label'
                         WHEN UPPER(TRIM(unitdischargestatus))='EXPIRED' AND hospital_expire_flag<>1 THEN 'conflicting_mortality'
                         ELSE 'included' END AS eligibility_reason
                FROM joined
            ) SELECT *, eligibility_reason='included' AS included_in_base_cohort FROM reasoned''')
        con.execute('CREATE TABLE base AS SELECT * FROM eligibility WHERE included_in_base_cohort ORDER BY stay_id')
        n, subjects = con.execute('SELECT COUNT(*), COUNT(DISTINCT subject_id) FROM base').fetchone()
        reasons = dict(con.execute('SELECT eligibility_reason,COUNT(*) FROM eligibility GROUP BY eligibility_reason').fetchall())
        save_table(con,'eligibility','base_eligibility.parquet')
        save_table(con,'base','base_cohort.parquet')
        write_report('01_base.json',__file__,dict(source_stays=count,base_stays=n,attrition=reasons),source_specific_limits=CONFIG['eicu'])
        if not n or n != subjects:
            raise ValueError('Empty or nonunique base cohort; inspect 01_base.json')
    print(f'[01] Complete v{CONFIG["dataset_version"]}; {n:,} adult first ICU stays',flush=True)


if __name__ == '__main__':
    build_base_cohort()
