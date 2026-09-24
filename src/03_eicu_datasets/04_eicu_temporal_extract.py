"""Translate eICU offset records into the shared measurement-evidence contract."""
import hashlib
import importlib
import json
import re
import tempfile
from pathlib import Path

import duckdb

io = importlib.import_module('01_eicu_base_cohort')
CONFIG, RAW, WORK, OUT = io.CONFIG, io.RAW, io.WORK, io.OUT
sql_string, at_offset = io.sql_string, io.at_offset
REGISTRY_FILE = io.BASE_DIR/'src/common/feature_registry.json'
REGISTRY = json.loads(REGISTRY_FILE.read_text(encoding='utf-8'))
UNITS = {f['name']:f['canonical_unit'] for f in REGISTRY['temporal_features']}
UNITS.update(REGISTRY['evidence_features'])
cleaning = importlib.import_module('05_eicu_temporal_clean')
LAB_UNIT_ALIASES = cleaning.LAB_UNIT_ALIASES


def numeric(expression):
    return f"CASE WHEN REGEXP_FULL_MATCH(TRIM({expression}), '[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?') THEN TRY_CAST({expression} AS DOUBLE) END"


def labels(values):
    return ','.join(sql_string(v.lower()) for v in values)


def project(query, physical, source, offset, record_id, feature, value, unit,
            label, end='NULL::BIGINT', weight='NULL::DOUBLE', conflict='FALSE',
            unit_conflict='FALSE', error='FALSE', lab_type='NULL::INTEGER', store_offset=None,
            value_conflict='FALSE', source_value=None, rewritten='FALSE', source_numeric='NULL::DOUBLE',
            system_unit='NULL::VARCHAR', interface_unit='NULL::VARCHAR'):
    event_time = at_offset(offset)
    event_end = at_offset(end)
    interval = source in ('inputevents','procedureevents')
    fields = f'''{sql_string(CONFIG['source_database'])} AS source_db,
        {sql_string(source)} AS source_table, {sql_string(physical)} AS source_table_original,
        w.subject_id,w.hadm_id,w.stay_id,w.hospital_id,
        m.itemid,m.feature,m.canonical_unit, m.feature IN ('pao2','paco2') AS arterial_specimen_required,
        {offset} AS source_offset_minutes, {label} AS source_label,
        {event_time} AS event_time, {at_offset(store_offset) if store_offset else 'NULL::TIMESTAMP'} AS storetime,
        CAST({value} AS VARCHAR) AS raw_value, {numeric(f'CAST({value} AS VARCHAR)')} AS raw_valuenum,
        CAST({source_value or value} AS VARCHAR) AS source_value_text,
        {source_numeric} AS source_numeric_value,
        {unit} AS raw_unit, NULL::VARCHAR AS item_unit,
        {system_unit} AS source_system_unit, {interface_unit} AS source_interface_unit,
        {event_end} AS event_end_time, {record_id} AS source_event_id,
        {'DENSE_RANK() OVER (ORDER BY w.stay_id,'+offset+')' if source=='labevents' else 'NULL::BIGINT'} AS specimen_id,
        {'TRUE' if source=='labevents' else 'FALSE'} AS specimen_id_is_inferred,
        {lab_type} AS source_lab_type,
        CASE WHEN {lab_type}=7 THEN 'ARTERIAL' END AS specimen_type,
        FALSE AS specimen_type_conflict,
        NULL::BIGINT AS order_id,NULL::BIGINT AS link_order_id,
        {weight} AS patient_weight_kg,
        CASE WHEN {weight} IS NOT NULL THEN {event_time} END AS weight_observation_time,
        NULL::DOUBLE AS raw_amount, NULL::VARCHAR AS raw_amount_unit,
        NULL::VARCHAR AS source_status,NULL::VARCHAR AS order_category,NULL::VARCHAR AS source_warning,
        {conflict} AS infusion_rate_conflict, {unit_conflict} AS source_unit_conflict,
        {value_conflict} AS source_value_conflict,
        {error} AS source_error, {rewritten} AS is_rewritten_or_cancelled,
        {'CASE WHEN '+event_end+'>'+event_time+" THEN 'valid_interval' ELSE 'unresolved_interval' END" if interval else "'point'"} AS interval_status,
        'explicit_stay_offset' AS linkage_status,
        {event_time} BETWEEN w.icu_intime AND w.icu_outtime AS within_icu_interval,
        {event_time} BETWEEN w.hospital_admittime AND w.hospital_dischtime AS within_hospital_interval,
        {event_time}>w.hospital_deathtime AS after_recorded_hospital_death'''
    # Include unresolved rates near the window; no invented end time or duration.
    window = (f"(({event_time}<w.window_end AND {event_end}>w.window_start) OR "
              f"({event_time} BETWEEN w.window_start-INTERVAL 4 HOUR AND w.window_end))"
              if interval else f'{event_time} BETWEEN w.window_start AND w.window_end')
    return f'''SELECT {fields} FROM ({query}) e JOIN windows w ON e.patientunitstayid=w.stay_id
        JOIN item_mapping m ON m.source_table={sql_string(source)} AND m.feature=({feature})
        WHERE {window}'''


def extract_temporal_data():
    OUT.mkdir(parents=True,exist_ok=True)
    all_features = set().union(*(set(v) for v in CONFIG['source_mapping']['sources'].values()))
    required = {f['name'] for f in REGISTRY['temporal_features'] if not f['derived']}
    if not required <= all_features or not all_features <= UNITS.keys():
        raise ValueError(f'Invalid feature mapping: missing={required-all_features}, unknown={all_features-UNITS.keys()}')
    with tempfile.TemporaryDirectory(prefix='.extract_',dir=WORK) as temporary, duckdb.connect() as con:
        staging=Path(temporary)
        con.execute('SET threads=4')
        con.execute('SET preserve_insertion_order=false')
        con.execute(f"SET temp_directory={sql_string(staging/'spill')}")
        con.execute(f'''CREATE TABLE windows AS
            WITH limits AS (SELECT stay_id,MIN(suspected_infection_time)-INTERVAL 72 HOUR AS window_start,
                MAX(suspected_infection_time_upper_bound)+INTERVAL 48 HOUR AS window_end,COUNT(*) AS infection_candidate_count
                FROM (SELECT stay_id,suspected_infection_time,suspected_infection_time_upper_bound
                      FROM read_parquet({sql_string(WORK/'infection_candidates.parquet')})
                      UNION ALL SELECT stay_id,suspected_infection_time,suspected_infection_time_upper_bound
                      FROM read_parquet({sql_string(WORK/'strict_infection_candidates.parquet')})) GROUP BY stay_id)
            SELECT p.subject_id,p.hadm_id,p.stay_id,p.hospital_id,p.icu_intime,p.icu_outtime,
                p.hospital_admittime,p.hospital_dischtime,p.hospital_deathtime,limits.* EXCLUDE(stay_id)
            FROM read_parquet({sql_string(WORK/'phenotypes.parquet')}) p JOIN limits USING(stay_id)''')
        con.execute('CREATE TABLE item_mapping(source_table VARCHAR,itemid INTEGER,feature VARCHAR,canonical_unit VARCHAR)')
        con.executemany('INSERT INTO item_mapping VALUES (?,?,?,?)',[(s,i,f,UNITS[f])
            for s,features in CONFIG['source_mapping']['sources'].items() for f,ids in features.items() for i in ids])
        con.execute('CREATE TABLE mapping_audit(source_table VARCHAR,source_label VARCHAR,feature VARCHAR,rows BIGINT,stays BIGINT)')
        parts=[]
        source_files={}

        def source(table, columns):
            path=io.source_path(RAW,table)
            source_files[table]=dict(path=str(path),size_bytes=path.stat().st_size)
            return io.csv_query(RAW,table,dict(patientunitstayid='BIGINT',**columns))

        def emit(name, query):
            path=staging/(name+'.parquet')
            con.execute(f'COPY ({query}) TO {sql_string(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')
            parts.append(path)

        print('[04] Extracting vitals, laboratories and nursing evidence',flush=True)
        for table, fields in [
            ('vitalPeriodic',{'heartrate':('hr','bpm'),'systemicmean':('map','mmHg'),
                              'respiration':('rr','breaths/min'),'temperature':('temp_c','C'),'sao2':('spo2','%')}),
            ('vitalAperiodic',{'noninvasivemean':('map','mmHg')})]:
            source_id='vitalperiodicid' if table=='vitalPeriodic' else 'vitalaperiodicid'
            con.execute(f'''CREATE TEMP TABLE vital AS
                SELECT v.* FROM ({source(table,dict(observationoffset="BIGINT",**{source_id:"BIGINT"},**{k:"VARCHAR" for k in fields}))}) v
                JOIN windows w ON v.patientunitstayid=w.stay_id
                WHERE {at_offset('observationoffset')} BETWEEN w.window_start AND w.window_end''')
            queries=[]
            for column,(feature,unit) in fields.items():
                queries.append(project(f'SELECT * FROM vital WHERE {column} IS NOT NULL',table,'chartevents','observationoffset',source_id,
                    sql_string(feature),column,sql_string(unit),sql_string(column)))
            emit(table,' UNION ALL '.join(queries))
            con.execute('DROP TABLE vital')

        con.execute('CREATE TABLE lab_map(label VARCHAR,feature VARCHAR)')
        con.executemany('INSERT INTO lab_map VALUES (?,?)',[(label.lower(),feature)
            for feature,names in CONFIG['source_mapping']['lab_labels'].items() for label in names])
        lab_query=source('lab',dict(labid='BIGINT',labresultoffset='BIGINT',labresultrevisedoffset='BIGINT',
            labtypeid='INTEGER',labname='VARCHAR',labresult='DOUBLE',labresulttext='VARCHAR',
            labmeasurenamesystem='VARCHAR',labmeasurenameinterface='VARCHAR'))
        con.execute(f'''CREATE TABLE labs AS SELECT l.*,m.feature AS mapped_feature
            FROM ({lab_query}) l JOIN windows w ON l.patientunitstayid=w.stay_id
            LEFT JOIN lab_map m ON LOWER(TRIM(labname))=m.label
            WHERE {at_offset('labresultoffset')} BETWEEN w.window_start AND w.window_end''')
        con.execute("INSERT INTO mapping_audit SELECT 'lab',labname,mapped_feature,COUNT(*),COUNT(DISTINCT patientunitstayid) FROM labs GROUP BY ALL")
        rules=json.loads((io.BASE_DIR/'src/common/measurement_rules.json').read_text(encoding='utf-8'))
        con.execute('CREATE TABLE unit_aliases(feature VARCHAR,unit VARCHAR,signature VARCHAR)')
        con.executemany('INSERT INTO unit_aliases VALUES (?,?,?)',cleaning.lab_unit_signatures(REGISTRY,rules))
        norm_sql=lambda col:f"REGEXP_REPLACE(LOWER(REPLACE(REPLACE(TRIM(COALESCE({col},'')),'µ','u'),'μ','u')),'\\s+','','g')"
        lab_query=f'''SELECT l.*,
            DENSE_RANK() OVER(PARTITION BY patientunitstayid,labname,labresultoffset ORDER BY labresultrevisedoffset DESC NULLS LAST) AS revision_rank,
            ROW_NUMBER() OVER(PARTITION BY patientunitstayid,labname,labresultoffset,labresultrevisedoffset,
                labresult,labresulttext,labmeasurenamesystem,labmeasurenameinterface ORDER BY labid) AS duplicate_rank,
            COUNT(DISTINCT COALESCE(CAST(labresult AS VARCHAR),'__NULL__')) OVER
                (PARTITION BY patientunitstayid,labname,labresultoffset,labresultrevisedoffset)>1 AS result_conflict,
            COALESCE(NULLIF(TRIM(labmeasurenamesystem),''),NULLIF(TRIM(labmeasurenameinterface),'')) AS documented_unit,
            NULLIF(TRIM(labmeasurenamesystem),'') IS NOT NULL AND NULLIF(TRIM(labmeasurenameinterface),'') IS NOT NULL
                AND COALESCE(s.signature,{norm_sql('labmeasurenamesystem')})<>COALESCE(i.signature,{norm_sql('labmeasurenameinterface')}) AS units_conflict,
            CASE WHEN REGEXP_FULL_MATCH(TRIM(COALESCE(labresulttext,'')),'[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?')
                OR NULLIF(TRIM(labresulttext),'') IS NULL THEN CAST(labresult AS VARCHAR) ELSE labresulttext END AS result_value
            FROM labs l LEFT JOIN unit_aliases s ON s.feature=l.mapped_feature AND s.unit={norm_sql('labmeasurenamesystem')}
                LEFT JOIN unit_aliases i ON i.feature=l.mapped_feature AND i.unit={norm_sql('labmeasurenameinterface')}'''
        emit('lab',project(lab_query,'lab','labevents','labresultoffset','labid','mapped_feature','result_value',
            'documented_unit','labname',unit_conflict='units_conflict',lab_type='labtypeid',store_offset='labresultrevisedoffset',
            value_conflict=f"result_conflict OR ({numeric('labresulttext')} IS NOT NULL AND {numeric('labresulttext')} IS DISTINCT FROM labresult)",
            source_value='labresulttext',rewritten='revision_rank>1 OR duplicate_rank>1',source_numeric='labresult',
            system_unit='labmeasurenamesystem',interface_unit='labmeasurenameinterface'))
        nurse=source('nurseCharting',dict(nursingchartid='BIGINT',nursingchartoffset='BIGINT',nursingchartentryoffset='BIGINT',
            nursingchartcelltypevallabel='VARCHAR',nursingchartcelltypevalname='VARCHAR',nursingchartvalue='VARCHAR'))
        nurse_query=f'''SELECT *,CASE LOWER(TRIM(nursingchartcelltypevalname))
                WHEN 'eyes' THEN 'gcs_eye' WHEN 'verbal' THEN 'gcs_verbal' WHEN 'motor' THEN 'gcs_motor' END AS mapped_feature
            FROM ({nurse}) WHERE LOWER(TRIM(nursingchartcelltypevallabel))='glasgow coma score' '''
        emit('gcs',project(nurse_query,'nurseCharting','chartevents','nursingchartoffset','nursingchartid',
            'mapped_feature','nursingchartvalue',"'score'",'nursingchartcelltypevalname',store_offset='nursingchartentryoffset'))

        print('[04] Extracting respiratory, urine and infusion evidence',flush=True)
        con.execute('CREATE TABLE resp_map(label VARCHAR,feature VARCHAR)')
        con.executemany('INSERT INTO resp_map VALUES (?,?)',[(label.lower(),feature)
            for feature,names in CONFIG['source_mapping']['respiratory_labels'].items() for label in names])
        resp=source('respiratoryCharting',dict(respchartid='BIGINT',respchartoffset='BIGINT',respchartentryoffset='BIGINT',
            respchartvaluelabel='VARCHAR',respchartvalue='VARCHAR'))
        con.execute(f'''CREATE TABLE resp AS SELECT r.*,m.feature AS mapped_feature
            FROM ({resp}) r JOIN windows w ON r.patientunitstayid=w.stay_id
            LEFT JOIN resp_map m ON LOWER(TRIM(respchartvaluelabel))=m.label
            WHERE {at_offset('respchartoffset')} BETWEEN w.window_start AND w.window_end''')
        con.execute("INSERT INTO mapping_audit SELECT 'respiratoryCharting',respchartvaluelabel,mapped_feature,COUNT(*),COUNT(DISTINCT patientunitstayid) FROM resp GROUP BY ALL")
        device_case='CASE LOWER(TRIM(respchartvalue)) '+ ' '.join(f'WHEN {sql_string(k)} THEN {sql_string(v)}' for k,v in CONFIG['source_mapping']['device_aliases'].items())+' ELSE LOWER(TRIM(respchartvalue)) END'
        resp_query=f'''SELECT *,CASE WHEN mapped_feature='oxygen_device' THEN {device_case}
            WHEN mapped_feature='fio2' AND REGEXP_FULL_MATCH(TRIM(respchartvalue),'[0-9]+([.][0-9]+)?%')
                THEN REPLACE(TRIM(respchartvalue),'%','') ELSE respchartvalue END AS translated_value,
            CASE WHEN mapped_feature='fio2' AND ENDS_WITH(TRIM(respchartvalue),'%') THEN '%' ELSE '' END AS documented_unit
            FROM resp'''
        emit('resp',project(resp_query,'respiratoryCharting','chartevents','respchartoffset','respchartid',
            'mapped_feature','translated_value','documented_unit','respchartvaluelabel',store_offset='respchartentryoffset',source_value='respchartvalue'))
        care=source('respiratoryCare',dict(respcareid='BIGINT',respcarestatusoffset='BIGINT',airwaytype='VARCHAR',
            ventstartoffset='BIGINT',ventendoffset='BIGINT'))
        invasive=labels(CONFIG['source_mapping']['invasive_airways'])
        # Only a documented interval plus an invasive airway establishes invasive ventilation.
        care_query=f'''SELECT * FROM ({care}) WHERE LOWER(TRIM(airwaytype)) IN ({invasive})'''
        emit('vent',project(care_query,'respiratoryCare','procedureevents','ventstartoffset','respcareid',
            "'vent_invasive'",'airwaytype','NULL::VARCHAR','airwaytype',end='ventendoffset',store_offset='respcarestatusoffset'))

        urine=source('intakeOutput',dict(intakeoutputid='BIGINT',intakeoutputoffset='BIGINT',intakeoutputentryoffset='BIGINT',
            cellpath='VARCHAR',cellvaluenumeric='DOUBLE',cellvaluetext='VARCHAR'))
        urine_leaves=labels(CONFIG['source_mapping']['urine_leaves'])
        con.execute(f'''CREATE TABLE urine AS WITH tagged AS (
            SELECT *,LOWER(TRIM(cellpath)) AS path,
                LOWER(TRIM(LIST_EXTRACT(STRING_SPLIT(cellpath,'|'),-1))) AS leaf
            FROM ({urine}) u JOIN windows w ON u.patientunitstayid=w.stay_id
            WHERE {at_offset('intakeoutputoffset')} BETWEEN w.window_start AND w.window_end
        ) SELECT *,CASE WHEN path LIKE 'flowsheet|flowsheet cell labels|i&o|output (ml)|%'
                    AND leaf IN ({urine_leaves}) THEN 'urine_output'
                WHEN REGEXP_MATCHES(path,'irrig|3 way foley') AND path LIKE '%|output (ml)|%' THEN 'urine_irrigant_out'
                WHEN REGEXP_MATCHES(path,'irrig') AND path LIKE '%|intake (ml)|%' THEN 'urine_irrigant_in' END AS mapped_feature
            FROM tagged''')
        con.execute("INSERT INTO mapping_audit SELECT 'intakeOutput',cellpath,mapped_feature,COUNT(*),COUNT(DISTINCT patientunitstayid) FROM urine GROUP BY ALL")
        # Multiple labels can be overlapping summaries, not independent volumes. Withhold ambiguous timestamps.
        urine_query='''WITH dedup AS (
            SELECT patientunitstayid,intakeoutputoffset,mapped_feature,cellpath,cellvaluenumeric,cellvaluetext,
                MIN(intakeoutputid) AS intakeoutputid,MAX(intakeoutputentryoffset) AS intakeoutputentryoffset
            FROM urine WHERE mapped_feature IS NOT NULL GROUP BY ALL
        ) SELECT *,COUNT(*) OVER(PARTITION BY patientunitstayid,intakeoutputoffset,mapped_feature)>1 AS volume_conflict,
            COALESCE(NULLIF(TRIM(cellvaluetext),''),CAST(cellvaluenumeric AS VARCHAR)) AS documented_value
            FROM dedup'''
        emit('urine',project(urine_query,'intakeOutput','outputevents','intakeoutputoffset','intakeoutputid',
            'mapped_feature','documented_value',"'mL'",'cellpath',
            value_conflict=f"volume_conflict OR mapped_feature<>'urine_output' OR ({numeric('cellvaluetext')} IS NOT NULL AND {numeric('cellvaluetext')} IS DISTINCT FROM cellvaluenumeric)",
            store_offset='intakeoutputentryoffset'))

        inf=source('infusionDrug',dict(infusiondrugid='BIGINT',infusionoffset='BIGINT',drugname='VARCHAR',drugrate='VARCHAR',patientweight='VARCHAR'))
        drug_case='CASE '+' '.join('WHEN REGEXP_MATCHES(LOWER(drugname),'+sql_string('(^|[^a-z])('+'|'.join(re.escape(n) for n in names)+')([^a-z]|$)')+') THEN '+sql_string(feature)
            for feature,names in CONFIG['source_mapping']['infusion_names'].items())+' END'
        con.execute(f'''CREATE TABLE infusions AS SELECT d.*, {drug_case} AS mapped_feature,
            LOWER(TRIM(REGEXP_EXTRACT(drugname,'\\(([^()]*)\\)\\s*$',1))) AS documented_unit
            FROM ({inf}) d JOIN windows w ON d.patientunitstayid=w.stay_id''')
        con.execute("INSERT INTO mapping_audit SELECT 'infusionDrug',drugname,mapped_feature,COUNT(*),COUNT(DISTINCT patientunitstayid) FROM infusions GROUP BY ALL")
        inf_query=f'''WITH grouped AS (
            SELECT patientunitstayid,infusionoffset,mapped_feature,MIN(infusiondrugid) AS infusiondrugid,
                MIN(drugname) AS drugname,MIN(drugrate) AS drugrate,MIN(documented_unit) AS documented_unit,
                CASE WHEN COUNT(*)=COUNT({numeric('patientweight')}) AND COUNT(DISTINCT {numeric('patientweight')})=1
                    THEN MIN({numeric('patientweight')}) END AS documented_weight,
                COUNT(DISTINCT COALESCE(drugrate,'__NULL__'))>1 OR COUNT(DISTINCT documented_unit)>1 AS rate_conflict
            FROM infusions WHERE mapped_feature IS NOT NULL GROUP BY patientunitstayid,infusionoffset,mapped_feature
        ), timed AS (
            SELECT *,LEAD(infusionoffset) OVER(PARTITION BY patientunitstayid,mapped_feature ORDER BY infusionoffset) AS next_offset
            FROM grouped
        ) SELECT *,CASE WHEN next_offset>infusionoffset AND next_offset<=infusionoffset+240 THEN next_offset END AS end_offset FROM timed'''
        emit('infusion',project(inf_query,'infusionDrug','inputevents','infusionoffset','infusiondrugid',
            'mapped_feature','drugrate','documented_unit','drugname',end='end_offset',weight='documented_weight',conflict='rate_conflict'))

        paths=','.join(sql_string(p) for p in parts)
        con.execute(f'CREATE VIEW extracted AS SELECT * FROM read_parquet([{paths}])')
        summary=dict(zip(['raw_rows','stays_with_events','unit_conflicts','infusion_conflicts'],con.execute('''SELECT COUNT(*),COUNT(DISTINCT stay_id),
            COUNT(*) FILTER (WHERE source_unit_conflict),COUNT(*) FILTER (WHERE infusion_rate_conflict) FROM extracted''').fetchone()))
        io.save_table(con,'extracted','events_raw.parquet')
        io.save_table(con,'windows','extraction_windows.parquet')
        con.execute(f'COPY (SELECT * FROM mapping_audit ORDER BY source_table,rows DESC,source_label) TO {sql_string(OUT/"source_mapping_qc.csv")} (HEADER)')
        io.write_report('04_extraction.json',__file__,summary,schema_version='2.0.0',
            registry_sha256=hashlib.sha256(REGISTRY_FILE.read_bytes()).hexdigest(),source_files=source_files,
            source_specific_limits=CONFIG['eicu'],lab_unit_aliases=LAB_UNIT_ALIASES,
            unit_comparison='Feature-specific conversion signatures; equal numeric scales only. Unknown or different scales are not reconciled.')
    print(f'[04] Complete v{CONFIG["dataset_version"]}; {summary["raw_rows"]:,} source evidence rows',flush=True)


if __name__ == '__main__':
    extract_temporal_data()
