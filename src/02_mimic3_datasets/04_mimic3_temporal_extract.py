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
MIMIC_DIR = BASE_DIR / "data" / "raw" / "mimic3-carevue" / "1.4"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimic3-carevue" / "work"
METRICS_DIR = BASE_DIR / "outputs" / "mimic3-carevue"
REGISTRY_FILE = BASE_DIR / "src" / "common" / "feature_registry.json"
SCHEMA_VERSION = "2.0.0"
SOURCE_MODULES = {"chartevents":"CHARTEVENTS", "labevents":"LABEVENTS",
                  "inputevents":"INPUTEVENTS_CV", "outputevents":"OUTPUTEVENTS"}
OPTIONAL_FIELDS = {
    "weight_observation_time": "TIMESTAMP",
    "event_end_time": "TIMESTAMP", "source_event_id": "BIGINT", "specimen_id": "BIGINT",
    "order_id": "BIGINT", "link_order_id": "BIGINT", "patient_weight_kg": "DOUBLE",
    "raw_amount": "DOUBLE", "raw_amount_unit": "VARCHAR",
    "source_status": "VARCHAR", "order_category": "VARCHAR", "source_warning": "VARCHAR",
}


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalized_label_sql(column):
    return f"TRIM(REGEXP_REPLACE(UPPER(COALESCE({column}, '')), '\\s+', ' ', 'g'))"


def load_registry(path):
    registry_bytes = Path(path).read_bytes()
    registry = json.loads(registry_bytes)
    registry["sources"] = CONFIG["source_mapping"]["sources"]
    registry["evidence_features"].update(CONFIG["source_mapping"]["evidence_features"])
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
        for feature, itemids in registry["sources"][source_table].items():
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
    """Translate CareVue records into the frozen measurement evidence contract."""
    lab = source_table == 'labevents'
    infusion = source_table == 'inputevents'
    types = {'subject_id':'BIGINT','hadm_id':'BIGINT','itemid':'INTEGER','row_id':'BIGINT','charttime':'TIMESTAMP'}
    if not lab:
        types.update(icustay_id='BIGINT',storetime='TIMESTAMP')
    fields = {name:f'NULL::{dtype}' for name,dtype in OPTIONAL_FIELDS.items()}
    fields['source_event_id'] = 'events.row_id'
    fields['source_warning'] = 'events.flag' if lab else 'NULL::VARCHAR'
    source_error = 'FALSE'
    if source_table in ('chartevents','labevents'):
        types.update(value='VARCHAR',valuenum='DOUBLE',valueuom='VARCHAR')
        raw_value, raw_number, raw_unit = 'events.value','events.valuenum','events.valueuom'
        if not lab:
            types.update(error='VARCHAR',warning='VARCHAR')
            fields['source_warning'] = 'events.warning'
            source_error = "COALESCE(TRY_CAST(events.error AS INTEGER)=1,FALSE)"
    elif source_table == 'outputevents':
        types.update(value='DOUBLE',valueuom='VARCHAR',iserror='VARCHAR')
        raw_value,raw_number,raw_unit = 'CAST(events.value AS VARCHAR)','events.value','events.valueuom'
        source_error = 'COALESCE(TRY_CAST(events.iserror AS INTEGER)=1,FALSE)'
    else:
        types.update(rate='DOUBLE',rateuom='VARCHAR',amount='DOUBLE',amountuom='VARCHAR',
                     stopped='VARCHAR',linkorderid='BIGINT',orderid='BIGINT')
        raw_value,raw_number,raw_unit = 'CAST(events.rate AS VARCHAR)','events.rate','events.rateuom'
        fields.update(event_end_time='events.endtime',order_id='events.orderid',link_order_id='events.linkorderid',
                      raw_amount='events.amount',raw_amount_unit='events.amountuom',source_status='events.stopped',
                      patient_weight_kg="CASE WHEN weight.event_time >= events.charttime - INTERVAL 24 HOUR THEN weight.weight END",
                      weight_observation_time='weight.event_time')
    type_sql = ','.join(f'{sql_string(k)}:{sql_string(v)}' for k,v in types.items())
    csv = f'read_csv_auto({sql_string(path.as_posix())}, types={{{type_sql}}})'
    prefix = ''
    conflict = 'FALSE'
    if infusion:
        # Retain all rate changes/stops before windowing so the next event is real.
        prefix = f"""WITH grouped AS (
            SELECT p.subject_id,p.hadm_id,p.icustay_id,MIN(p.itemid) AS itemid,p.charttime,p.linkorderid,
                MIN(p.row_id) AS row_id, MIN(p.orderid) AS orderid, MAX(p.storetime) AS storetime,
                MIN(p.amount) AS amount,MIN(p.amountuom) AS amountuom,
                MIN(p.rateuom) FILTER (WHERE p.rate IS NOT NULL) AS rateuom,
                BOOL_OR(LOWER(COALESCE(p.stopped,''))='stopped' OR LOWER(COALESCE(p.stopped,'')) LIKE 'd/c%') AS is_stop,
                MIN(p.rate) AS rate,
                CASE WHEN is_stop THEN 'Stopped' ELSE MIN(p.stopped) END AS stopped,
                COUNT(DISTINCT p.rate)>1 OR COUNT(DISTINCT p.rateuom) FILTER (WHERE p.rate IS NOT NULL)>1 AS rate_conflict,
                mapping.feature
            FROM {csv} p JOIN windows w ON p.subject_id=w.subject_id AND p.hadm_id=w.hadm_id AND p.icustay_id=w.stay_id
            JOIN item_mapping mapping ON mapping.source_table='inputevents' AND mapping.itemid=p.itemid
            WHERE p.rate IS NOT NULL OR LOWER(COALESCE(p.stopped,''))='stopped' OR LOWER(COALESCE(p.stopped,'')) LIKE 'd/c%'
            GROUP BY p.subject_id,p.hadm_id,p.icustay_id,p.charttime,p.linkorderid,mapping.feature
        ), timed AS (
            SELECT *, LEAD(charttime) OVER (PARTITION BY icustay_id,feature,linkorderid ORDER BY charttime,itemid) AS next_time
            FROM grouped
        ), intervals AS (
            SELECT *, CASE WHEN next_time <= charttime + INTERVAL 4 HOUR THEN next_time END AS endtime
            FROM timed
        ), source AS (
            SELECT i.*,
                rate_conflict OR EXISTS (SELECT 1 FROM intervals j
                    WHERE i.icustay_id=j.icustay_id AND i.feature=j.feature AND i.row_id<>j.row_id
                        AND i.charttime < j.endtime AND j.charttime < i.endtime
                        AND (i.rate IS DISTINCT FROM j.rate OR i.rateuom IS DISTINCT FROM j.rateuom)) AS infusion_rate_conflict
            FROM intervals i
        )"""
        csv = 'source'
        conflict = 'events.infusion_rate_conflict'
    elif lab:
        # This is an adapter grouping key, never a source specimen identifier.
        prefix = f"""WITH source AS (SELECT *,
            DENSE_RANK() OVER (ORDER BY subject_id,hadm_id,charttime)::BIGINT AS timestamp_group_id FROM {csv})"""
        csv = 'source'
        fields['specimen_id'] = 'events.timestamp_group_id'
    linkage = ("events.subject_id=windows.subject_id AND (events.hadm_id=windows.hadm_id OR "
               "(events.hadm_id IS NULL AND events.charttime BETWEEN windows.hospital_admittime AND windows.hospital_dischtime))") if lab else (
               'events.subject_id=windows.subject_id AND events.hadm_id=windows.hadm_id AND events.icustay_id=windows.stay_id')
    filter_sql = 'events.charttime BETWEEN windows.window_start AND windows.window_end'
    if source_table == 'chartevents':
        filter_sql = '(events.charttime BETWEEN windows.window_start - INTERVAL 24 HOUR AND windows.window_end)'
    if infusion:
        filter_sql = '(events.charttime BETWEEN windows.window_start AND windows.window_end OR (events.charttime < windows.window_start AND events.endtime > windows.window_start))'
    unit_case = 'CASE events.itemid ' + ' '.join(f'WHEN {int(i)} THEN {sql_string(u)}' for i,u in CONFIG['carevue']['item_units'].items()) + ' END'
    if source_table == 'outputevents':
        unit_case = "'mL'"
    elif source_table != 'chartevents':
        unit_case = 'NULL::VARCHAR'
    elif source_table == 'chartevents':
        unit_case = f"CASE WHEN events.itemid IN (763,3580) THEN 'kg' ELSE {unit_case} END"
    interval_status = "CASE WHEN events.endtime > events.charttime THEN 'valid_interval' ELSE 'missing_end' END" if infusion else "'point'"
    link_status = "CASE WHEN events.hadm_id IS NULL THEN 'inferred_hospital_time' ELSE 'exact_admission' END" if lab else "'exact_stay'"
    weight_join = 'ASOF LEFT JOIN weights weight ON events.icustay_id=weight.stay_id AND events.charttime>=weight.event_time' if infusion else ''
    field_sql = ','.join(f'{v} AS {k}' for k,v in fields.items())
    storetime = 'NULL::TIMESTAMP' if lab else 'events.storetime'
    return f"""{prefix}
        SELECT 'MIMIC-III CareVue' AS source_db,{sql_string(source_table)} AS source_table,
            windows.subject_id,windows.hadm_id,windows.stay_id,events.hadm_id AS source_hadm_id,
            events.itemid,mapping.feature,mapping.canonical_unit,mapping.arterial_specimen_required,
            events.charttime AS event_time,{storetime} AS storetime,
            {raw_value} AS raw_value,{raw_number} AS raw_valuenum,{raw_unit} AS raw_unit,
            {field_sql},{interval_status} AS interval_status,{link_status} AS linkage_status,
            FALSE AS is_rewritten_or_cancelled,{source_error} AS source_error,
            {conflict} AS infusion_rate_conflict,{unit_case} AS item_unit,
            {'TRUE' if lab else 'FALSE'} AS specimen_id_is_inferred,
            events.charttime BETWEEN windows.icu_intime AND windows.icu_outtime AS within_icu_interval,
            events.charttime BETWEEN windows.hospital_admittime AND windows.hospital_dischtime AS within_hospital_interval,
            events.charttime > windows.hospital_deathtime AS after_recorded_hospital_death
        FROM {csv} events JOIN windows ON {linkage}
        JOIN item_mapping mapping ON mapping.itemid=events.itemid AND mapping.source_table={sql_string(source_table)}
        {weight_join} WHERE {filter_sql}
    """


def validate_dictionary(connection, mimic_dir, metrics_dir):
    """Validate source tables and configured lab labels; retain the mapping audit."""
    paths = [mimic_dir/'D_ITEMS.csv.gz', mimic_dir/'D_LABITEMS.csv.gz']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    connection.execute(f"""
        CREATE TEMP TABLE dictionary AS
        SELECT CASE WHEN LOWER(linksto)='inputevents_cv' THEN 'inputevents' ELSE LOWER(linksto) END AS source_table,
            itemid,label,unitname AS source_unit
        FROM read_csv_auto({sql_string(paths[0].as_posix())},types={{'itemid':'INTEGER','label':'VARCHAR','unitname':'VARCHAR'}})
        WHERE LOWER(dbsource)='carevue'
        UNION ALL
        SELECT 'labevents',itemid,label,fluid FROM read_csv_auto({sql_string(paths[1].as_posix())},
            types={{'itemid':'INTEGER','label':'VARCHAR','fluid':'VARCHAR'}})
    """)
    missing = connection.execute("""
        SELECT mapping.source_table,mapping.itemid,mapping.feature
        FROM item_mapping mapping LEFT JOIN dictionary USING(source_table,itemid)
        WHERE dictionary.itemid IS NULL
    """).fetchall()
    if missing:
        raise ValueError(f'Missing/wrong-table source item mappings: {missing}')
    expected = CONFIG['source_mapping']['lab_dictionary_labels']
    mapped_lab_ids = {int(itemid) for items in CONFIG['source_mapping']['sources']['labevents'].values()
                      for itemid in items}
    expected_ids = {int(itemid) for itemid in expected}
    if expected_ids != mapped_lab_ids:
        raise ValueError(f'Lab label inventory differs from extraction mappings: '
                         f'missing={sorted(mapped_lab_ids - expected_ids)}, '
                         f'unused={sorted(expected_ids - mapped_lab_ids)}')
    if any(not isinstance(label, str) or not label.strip() for label in expected.values()):
        raise ValueError('Every mapped lab item requires a nonempty expected dictionary label')
    connection.execute('CREATE TEMP TABLE expected_lab_labels(itemid INTEGER, expected_label VARCHAR)')
    connection.executemany('INSERT INTO expected_lab_labels VALUES (?, ?)',
                          [(int(itemid), label) for itemid, label in expected.items()])
    connection.execute(f"""
        CREATE TEMP TABLE mapping_audit AS
        SELECT mapping.*, dictionary.label, dictionary.source_unit, expected.expected_label,
            CASE WHEN mapping.source_table <> 'labevents' THEN 'table_and_item_only'
                WHEN {normalized_label_sql('dictionary.label')} = '' THEN 'missing_label'
                WHEN {normalized_label_sql('dictionary.label')} IS DISTINCT FROM
                    {normalized_label_sql('expected.expected_label')} THEN 'label_mismatch'
                ELSE 'label_match' END AS label_validation
        FROM item_mapping mapping JOIN dictionary USING(source_table,itemid)
        LEFT JOIN expected_lab_labels expected
            ON mapping.source_table = 'labevents' AND mapping.itemid = expected.itemid
    """)
    connection.execute(f"""
        COPY (SELECT * FROM mapping_audit ORDER BY source_table,feature,itemid)
        TO {sql_string((metrics_dir/'source_mapping_qc.csv').as_posix())} (HEADER)
    """)
    mismatches = connection.execute("""
        SELECT itemid,feature,expected_label,label FROM mapping_audit
        WHERE label_validation IN ('missing_label','label_mismatch') ORDER BY itemid
    """).fetchall()
    if mismatches:
        raise ValueError(f'Lab dictionary labels disagree with configured features: {mismatches}; '
                         f'see {metrics_dir / "source_mapping_qc.csv"}')


def extract_temporal_data(mimic_dir=MIMIC_DIR, processed_dir=PROCESSED_DIR,
                          metrics_dir=METRICS_DIR, registry_file=REGISTRY_FILE):
    started = time.perf_counter()
    mimic_dir, processed_dir, metrics_dir = map(Path, (mimic_dir, processed_dir, metrics_dir))
    registry, mapping_rows, registry_hash = load_registry(registry_file)
    cohort_file = processed_dir / "phenotypes.parquet"
    candidate_file = processed_dir / "infection_candidates.parquet"
    source_paths = {
        name: mimic_dir / f"{module}.csv.gz" for name, module in SOURCE_MODULES.items()
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
            connection.executemany("INSERT INTO item_mapping VALUES (?, ?, ?, ?, ?)",
                [('chartevents',763,'__weight_kg','kg',False),('chartevents',3580,'__weight_kg','kg',False)])
            validate_dictionary(connection, mimic_dir, metrics_dir)
            parts = []
            for source_table, path in source_paths.items():
                print(f"    Scanning {source_table}...", flush=True)
                part = staging / f"{source_table}.parquet"
                query = extraction_query(source_table, path)
                connection.execute(f"COPY ({query}) TO {sql_string(part.as_posix())} (FORMAT PARQUET)")
                parts.append(sql_string(part.as_posix()))
                if source_table == 'chartevents':
                    connection.execute(f"""CREATE TEMP TABLE weights AS
                        SELECT stay_id, event_time,
                            CASE WHEN COUNT(*) = COUNT(raw_valuenum) AND MIN(raw_valuenum)=MAX(raw_valuenum)
                                AND NOT BOOL_OR(source_error)
                                AND BOOL_AND(LOWER(TRIM(COALESCE(raw_unit,''))) IN ('','kg','kgs','kilogram','kilograms'))
                                AND NOT BOOL_OR(REGEXP_MATCHES(TRIM(COALESCE(raw_value,'')), '^[<>]'))
                                THEN MIN(raw_valuenum) END AS weight
                        FROM read_parquet({sql_string(part.as_posix())}) WHERE feature='__weight_kg'
                            AND within_icu_interval AND NOT COALESCE(after_recorded_hospital_death,FALSE)
                        GROUP BY stay_id,event_time""")
            connection.execute(
                f"CREATE TEMP VIEW raw_events AS SELECT * FROM read_parquet([{', '.join(parts)}]) WHERE feature <> '__weight_kg'"
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
            "common_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE_DIR / "src/common").glob("*.json")},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION, "source_database": "MIMIC-III CareVue", "source_version": "1.4",
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
                "specimen": "inferred exact subject/admission/charttime group with explicit specimen-type item; conflicts left unknown",
                "ventilation": "invasive/noninvasive procedures and text evidence remain separate; no binary inference",
                "quality_control": "invalid units, intervals, rewritten records and non-arterial blood gases must be adjudicated in cleaning/adjudication",
                "weight_fallback": None, "carevue": CONFIG["carevue"],
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
