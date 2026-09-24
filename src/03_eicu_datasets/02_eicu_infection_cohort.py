"""Select documented-infection/IV-order proxies and retain strict culture sensitivity evidence."""
import importlib
import re

import duckdb

io = importlib.import_module('01_eicu_base_cohort')
CONFIG, RAW, WORK, OUT = io.CONFIG, io.RAW, io.WORK, io.OUT
sql_string, at_offset = io.sql_string, io.at_offset


def build_infection_cohort():
    pattern = '(^|[^a-z])('+'|'.join(re.escape(n) for n in CONFIG['antimicrobial_names'])+')([^a-z]|$)'
    routes = ','.join(sql_string(r) for r in CONFIG['iv_routes'])
    medication = dict(patientunitstayid='BIGINT',medicationid='BIGINT',drugstartoffset='BIGINT',
        drugstopoffset='BIGINT',drugname='VARCHAR',routeadmin='VARCHAR',drugordercancelled='VARCHAR')
    micro = dict(patientunitstayid='BIGINT',microlabid='BIGINT',culturetakenoffset='BIGINT',
        culturesite='VARCHAR',organism='VARCHAR')
    with duckdb.connect() as con:
        con.execute(f'CREATE TABLE base AS SELECT * FROM read_parquet({sql_string(WORK/"base_cohort.parquet")})')
        con.execute(f'''CREATE TABLE prescription_audit AS
            SELECT b.stay_id, m.*, drugname AS drug, UPPER(TRIM(routeadmin)) AS route,
                {at_offset('drugstartoffset')} AS abx_start_time,
                {at_offset('drugstopoffset')} AS abx_stop_time,
                CASE WHEN NOT REGEXP_MATCHES(LOWER(COALESCE(drugname,'')),{sql_string(pattern)}) THEN 'unmapped_name'
                    WHEN REGEXP_MATCHES(LOWER(drugname),'cream|desensiti|ophth|ointment|lock|gel') THEN 'excluded_preparation'
                    WHEN UPPER(TRIM(COALESCE(routeadmin,''))) NOT IN ({routes}) THEN 'non_iv_or_missing_route'
                    WHEN LOWER(TRIM(COALESCE(m.drugordercancelled,''))) NOT IN ('no','false','0') THEN 'cancelled_or_unknown'
                    WHEN drugstartoffset IS NULL THEN 'missing_start'
                    WHEN drugstopoffset < drugstartoffset THEN 'invalid_interval'
                    WHEN drugstartoffset < b.hospitaladmitoffset OR drugstartoffset > b.hospitaldischargeoffset THEN 'outside_hospital_interval'
                    ELSE 'included' END AS prescription_status
            FROM ({io.csv_query(RAW,'medication',medication)}) m
            JOIN base b ON b.stay_id=m.patientunitstayid''')
        con.execute('''CREATE TABLE antimicrobial_starts AS
            SELECT stay_id,abx_start_time,abx_stop_time,LOWER(TRIM(drug)) AS abx_drug,route AS abx_route,
                MIN(medicationid) AS antimicrobial_start_id,
                abx_start_time AS abx_start_time_upper_bound,
                FALSE AS abx_start_time_is_date_only, FALSE AS abx_start_time_upper_exclusive,
                NULL::DOUBLE AS abx_duration_hours, FALSE AS abx_duration_complete
            FROM prescription_audit WHERE prescription_status='included' GROUP BY ALL''')
        con.execute('CREATE TABLE site_map(spec_itemid INTEGER,normalized_site VARCHAR)')
        con.executemany('INSERT INTO site_map VALUES (?,?)',[(int(k),v) for k,v in CONFIG['infection_policy']['diagnostic_specimens'].items()])
        con.execute(f'''CREATE TABLE culture_evidence AS
            WITH grouped AS (
                SELECT b.stay_id, culturetakenoffset, culturesite AS spec_type_desc,
                    MIN(microlabid) AS culture_event_id, COUNT(*) AS source_result_rows,
                    BOOL_AND(UPPER(TRIM(COALESCE(organism,''))) IN ('CANCELLED','CANCELED')) AS cancelled_only,
                    MIN(b.hospitaladmitoffset) AS hospital_start_offset, MIN(b.hospitaldischargeoffset) AS hospital_end_offset
                FROM ({io.csv_query(RAW,'microLab',micro)}) m
                JOIN base b ON b.stay_id=m.patientunitstayid GROUP BY b.stay_id,culturetakenoffset,culturesite
            )
            SELECT g.*, s.spec_itemid, {at_offset('culturetakenoffset')} AS culture_time,
                FALSE AS culture_time_is_date_only,
                CASE WHEN culturetakenoffset IS NULL THEN 'missing_culture_time'
                    WHEN cancelled_only THEN 'cancelled_test'
                    WHEN NULLIF(TRIM(spec_type_desc),'') IS NULL THEN 'missing_specimen_name'
                    WHEN spec_itemid IS NULL THEN 'unmapped_or_excluded_specimen'
                    WHEN culturetakenoffset NOT BETWEEN hospital_start_offset AND hospital_end_offset THEN 'outside_hospital_interval'
                    ELSE 'diagnostic_culture' END AS culture_evidence_status
            FROM grouped g LEFT JOIN site_map s
                ON TRIM(REGEXP_REPLACE(UPPER(COALESCE(spec_type_desc,'')),'\\s+',' ','g'))=s.normalized_site''')
        con.execute('''CREATE TABLE paired AS
                SELECT b.*, a.* EXCLUDE(stay_id), c.* EXCLUDE(stay_id),
                    culture_time AS culture_time_upper_bound, FALSE AS culture_time_upper_exclusive,
                    [spec_type_desc] AS diagnostic_evidence_names,
                    'diagnostic_site_proxy_no_test_field' AS diagnostic_evidence_basis,
                    LEAST(abx_start_time,culture_time) AS suspected_infection_time
                FROM base b JOIN antimicrobial_starts a USING(stay_id)
                JOIN culture_evidence c USING(stay_id)
                WHERE c.culture_evidence_status='diagnostic_culture'
                    AND culture_time BETWEEN abx_start_time-INTERVAL 72 HOUR AND abx_start_time+INTERVAL 24 HOUR''')
        con.execute('''CREATE TABLE strict_candidates AS
            SELECT *, suspected_infection_time AS suspected_infection_time_upper_bound,
                FALSE AS suspected_infection_time_upper_exclusive,
                suspected_infection_time AS eligible_sit_lower_bound,
                suspected_infection_time AS eligible_sit_upper_bound,
                FALSE AS eligible_sit_upper_exclusive, TRUE AS pairing_is_definite,
                TRUE AS presentation_is_definite, FALSE AS requires_timing_adjudication,
                'definite' AS candidate_timing_status, '2.1.0' AS infection_schema_version,
                ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY suspected_infection_time,culture_time,
                    abx_start_time,abx_stop_time NULLS LAST,culture_event_id,antimicrobial_start_id) AS infection_candidate_rank
            FROM paired WHERE suspected_infection_time BETWEEN icu_intime-INTERVAL 24 HOUR AND icu_intime+INTERVAL 24 HOUR''')
        proxy = CONFIG['infection_policy']['documented_infection_proxy']
        if not proxy['enabled'] or CONFIG['infection_policy']['main_definition'] != 'documented_infection_plus_iv_order_proxy':
            raise ValueError('This release implements the documented-infection proxy with strict culture sensitivity')
        if proxy['presentation_hours'] != [-24,24] or proxy['maximum_pair_gap_hours'] != 24:
            raise ValueError('Review source selection and QC before changing proxy timing windows')
        diagnosis_columns = dict(patientunitstayid='BIGINT',diagnosisid='BIGINT',diagnosisoffset='BIGINT',
            diagnosisstring='VARCHAR',icd9code='VARCHAR',diagnosispriority='VARCHAR')
        rule_case = 'CASE '+ ' '.join(f'WHEN REGEXP_MATCHES(LOWER(COALESCE(diagnosisstring,\'\')),{sql_string(pattern)}) THEN {sql_string(name)}'
            for name,pattern in proxy['diagnosis_patterns'].items())+' END'
        con.execute(f'''CREATE TABLE infection_diagnoses AS
            WITH mapped AS (
                SELECT b.stay_id,d.*,{at_offset('diagnosisoffset')} AS diagnosis_time,
                    {rule_case} AS infection_rule,
                    REGEXP_MATCHES(LOWER(COALESCE(diagnosisstring,'')),{sql_string(proxy['excluded_text_pattern'])}) AS excluded_text,
                    b.hospitaladmitoffset,b.hospitaldischargeoffset,b.unitdischargeoffset
                FROM ({io.csv_query(RAW,'diagnosis',diagnosis_columns)}) d
                JOIN base b ON b.stay_id=d.patientunitstayid
            ) SELECT *,CASE WHEN infection_rule IS NULL THEN 'unmapped_diagnosis'
                WHEN excluded_text THEN 'excluded_uncertain_or_noninfectious_text'
                WHEN diagnosisoffset IS NULL THEN 'missing_documentation_time'
                WHEN diagnosisoffset NOT BETWEEN hospitaladmitoffset AND LEAST(unitdischargeoffset,hospitaldischargeoffset) THEN 'outside_care_interval'
                WHEN diagnosisoffset NOT BETWEEN -1440 AND 1440 THEN 'outside_presentation_window'
                ELSE 'included' END AS diagnosis_evidence_status FROM mapped''')
        con.execute('''CREATE TABLE proxy_pairs AS
            SELECT b.*,a.* EXCLUDE(stay_id),d.diagnosisid AS infection_diagnosis_id,
                d.diagnosis_time,d.diagnosisoffset,d.diagnosisstring,d.infection_rule,
                [d.diagnosisstring] AS diagnostic_evidence_names,
                'documented_infection_plus_iv_order' AS diagnostic_evidence_basis,
                'documented_infection_proxy' AS infection_evidence_type,
                'eicu_documented_infection_proxy_sofa' AS cohort_definition,
                TRUE AS infection_time_is_proxy, FALSE AS clinical_infection_onset_known,
                'recorded_iv_order_start_with_nearby_infection_documentation' AS infection_time_basis,
                a.abx_start_time AS suspected_infection_time
            FROM base b JOIN antimicrobial_starts a USING(stay_id)
            JOIN infection_diagnoses d USING(stay_id)
            WHERE d.diagnosis_evidence_status='included'
                AND a.abx_start_time BETWEEN b.icu_intime-INTERVAL 24 HOUR AND b.icu_intime+INTERVAL 24 HOUR
                AND a.abx_start_time BETWEEN d.diagnosis_time-INTERVAL 24 HOUR AND d.diagnosis_time+INTERVAL 24 HOUR
            QUALIFY ROW_NUMBER() OVER(PARTITION BY b.stay_id,a.antimicrobial_start_id
                ORDER BY ABS(EPOCH(d.diagnosis_time-a.abx_start_time)),d.diagnosis_time,d.diagnosisid)=1''')
        con.execute('''CREATE TABLE candidates AS SELECT *,
            suspected_infection_time AS suspected_infection_time_upper_bound,
            FALSE AS suspected_infection_time_upper_exclusive,
            suspected_infection_time AS eligible_sit_lower_bound,suspected_infection_time AS eligible_sit_upper_bound,
            FALSE AS eligible_sit_upper_exclusive,TRUE AS pairing_is_definite,TRUE AS presentation_is_definite,
            FALSE AS requires_timing_adjudication,'definite' AS candidate_timing_status,'2.1.0' AS infection_schema_version,
            NULL::TIMESTAMP AS culture_time,NULL::TIMESTAMP AS culture_time_upper_bound,
            FALSE AS culture_time_is_date_only,FALSE AS culture_time_upper_exclusive,
            NULL::BIGINT AS culture_event_id,NULL::INTEGER AS spec_itemid,NULL::VARCHAR AS spec_type_desc,
            'not_required_for_documented_infection_proxy' AS culture_evidence_status,
            ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY suspected_infection_time,
                antimicrobial_start_id,infection_diagnosis_id) AS infection_candidate_rank,
            EXISTS(SELECT 1 FROM strict_candidates s WHERE s.stay_id=proxy_pairs.stay_id) AS has_strict_culture_infection_pair
            FROM proxy_pairs''')
        con.execute('CREATE TABLE infection AS SELECT * FROM candidates WHERE infection_candidate_rank=1 ORDER BY stay_id')
        for table, filename in [('candidates','infection_candidates.parquet'),('infection','infection_cohort.parquet'),
                                ('culture_evidence','culture_evidence.parquet'),('prescription_audit','prescription_evidence.parquet'),
                                ('infection_diagnoses','infection_diagnosis_evidence.parquet'),('strict_candidates','strict_infection_candidates.parquet')]:
            io.save_table(con,table,filename)
        for table,fields,filename in [('culture_evidence','spec_type_desc,culture_evidence_status','culture_qc.csv'),
                                     ('prescription_audit','drug,route,prescription_status','antimicrobial_qc.csv')]:
            con.execute(f'COPY (SELECT {fields},COUNT(*) AS rows, COUNT(DISTINCT stay_id) AS stays FROM {table} GROUP BY ALL ORDER BY rows DESC) TO {sql_string(OUT/filename)} (HEADER)')
        summary = {t:con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                   for t in ['base','prescription_audit','antimicrobial_starts','culture_evidence','candidates','infection']}
        funnel_queries = {
            'base_stays': 'SELECT COUNT(*) FROM base',
            'any_medication_record': 'SELECT COUNT(DISTINCT stay_id) FROM prescription_audit',
            'mapped_antimicrobial_name': "SELECT COUNT(DISTINCT stay_id) FROM prescription_audit WHERE prescription_status<>'unmapped_name'",
            'eligible_iv_antimicrobial': 'SELECT COUNT(DISTINCT stay_id) FROM antimicrobial_starts',
            'any_culture_record': 'SELECT COUNT(DISTINCT stay_id) FROM culture_evidence',
            'mapped_diagnostic_culture_site': 'SELECT COUNT(DISTINCT stay_id) FROM culture_evidence WHERE spec_itemid IS NOT NULL',
            'eligible_diagnostic_culture': "SELECT COUNT(DISTINCT stay_id) FROM culture_evidence WHERE culture_evidence_status='diagnostic_culture'",
            'both_eligible_sources_any_time': "SELECT COUNT(DISTINCT a.stay_id) FROM antimicrobial_starts a JOIN culture_evidence c USING(stay_id) WHERE c.culture_evidence_status='diagnostic_culture'",
            'paired_within_72h_24h': 'SELECT COUNT(DISTINCT stay_id) FROM paired',
            'strict_sit_within_icu_plus_minus_24h': 'SELECT COUNT(DISTINCT stay_id) FROM strict_candidates',
            'documented_infection_in_presentation_window': "SELECT COUNT(DISTINCT stay_id) FROM infection_diagnoses WHERE diagnosis_evidence_status='included'",
            'main_documented_infection_plus_iv_proxy': 'SELECT COUNT(*) FROM infection',
            'main_proxy_with_strict_culture_pair': 'SELECT COUNT(*) FROM infection WHERE has_strict_culture_infection_pair',
        }
        funnel = {name:con.execute(query).fetchone()[0] for name,query in funnel_queries.items()}
        status_counts = {table:dict(con.execute(f'SELECT {column}, COUNT(*) FROM {table} GROUP BY {column}').fetchall())
            for table,column in [('prescription_audit','prescription_status'),('culture_evidence','culture_evidence_status'),
                                 ('infection_diagnoses','diagnosis_evidence_status')]}
        coverage = con.execute('''SELECT b.hospital_id,COUNT(*) AS base_stays,
            COUNT(*) FILTER (WHERE EXISTS(SELECT 1 FROM prescription_audit p WHERE p.stay_id=b.stay_id)) AS medication_record_stays,
            COUNT(*) FILTER (WHERE EXISTS(SELECT 1 FROM antimicrobial_starts p WHERE p.stay_id=b.stay_id)) AS iv_antimicrobial_stays,
            COUNT(*) FILTER (WHERE EXISTS(SELECT 1 FROM culture_evidence c WHERE c.stay_id=b.stay_id)) AS culture_record_stays,
            COUNT(*) FILTER (WHERE EXISTS(SELECT 1 FROM infection i WHERE i.stay_id=b.stay_id)) AS infection_stays
            FROM base b GROUP BY b.hospital_id ORDER BY b.hospital_id''').fetchall()
        io.write_report('02_infection.json',__file__,summary,policy=CONFIG['infection_policy'],
            stay_funnel=funnel, row_status_counts=status_counts,
            funnel_note='Medication, culture and diagnosis branches are parallel; counts are not one sequential attrition chain. Strict culture counts are sensitivity results; the main cohort requires documented infection plus IV orders.',
            hospital_coverage_columns=['hospital_id','base_stays','medication_record_stays','iv_antimicrobial_stays','culture_record_stays','infection_stays'],
            hospital_coverage=coverage)
        print('[02] Infection evidence (distinct stays):',flush=True)
        for name,count in funnel.items():
            print(f'    {name}: {count:,}',flush=True)
        print(f'    hospitals_with_culture_records: {sum(row[4]>0 for row in coverage)}/{len(coverage)}',flush=True)
        if not summary['infection']:
            raise ValueError('No documented-infection/IV-order proxy stays; inspect diagnosis and medication rejection counts in 02_infection.json.')
    print(f'[02] Complete v{CONFIG["dataset_version"]}; {summary["infection"]:,} documented-infection proxy stays; '
          f'{funnel["main_proxy_with_strict_culture_pair"]:,} also have strict culture pairing (SOFA pending)',flush=True)


if __name__ == '__main__':
    build_infection_cohort()
