"""Package completed v1.0.0 datasets and atlas as QC-bound CSV.gz tables in four ZIPs."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'
DATASETS = ('mimiciv', 'mimic3-carevue', 'eicu')
PARTITIONS = ('train', 'validation', 'test')
METHODS = '0=unfilled; 1=observed; 2=SAITS; 3=forward_fill; 4=training_median'
COHORT_DEFINITIONS = {
    'mimiciv': 'mimiciv_culture_iv_antimicrobial_sofa',
    'mimic3-carevue': 'mimic3_carevue_culture_iv_antimicrobial_sofa',
    'eicu': 'eicu_documented_infection_proxy_sofa',
}
FEATURE_NAMES = dict(zip(
    'hr map rr temp_c spo2 gcs_eye gcs_verbal gcs_motor pao2 fio2 pf_ratio paco2 lactate '
    'creatinine bun bilirubin platelets wbc hemoglobin ph pt aptt albumin potassium sodium '
    'glucose chloride urine_output neq vent anion_gap bicarbonate calcium_total hematocrit '
    'magnesium phosphate inr lymphocytes_pct monocytes_pct neutrophils_pct basophils_pct '
    'eosinophils_pct alt alp ast mch mchc mcv rdw_cv rbc'.split(),
    ['Heart rate', 'Mean arterial pressure', 'Respiratory rate', 'Body temperature',
     'Peripheral oxygen saturation', 'GCS eye response', 'GCS verbal response', 'GCS motor response',
     'Arterial oxygen partial pressure', 'Inspired oxygen fraction', 'Paired PaO2/FiO2 ratio',
     'Arterial carbon dioxide partial pressure', 'Lactate', 'Creatinine', 'Blood urea nitrogen',
     'Total bilirubin', 'Platelet count', 'White blood cell count', 'Hemoglobin', 'Blood pH',
     'Prothrombin time', 'Activated partial thromboplastin time', 'Albumin', 'Potassium',
     'Sodium', 'Glucose', 'Chloride', 'Validated recorded urine volume',
     'Concurrent norepinephrine-equivalent rate, five-drug subset', 'Invasive ventilation evidence',
     'Anion gap', 'Bicarbonate', 'Total calcium', 'Hematocrit', 'Magnesium', 'Phosphate',
     'International normalized ratio', 'Lymphocyte percentage', 'Monocyte percentage',
     'Neutrophil percentage', 'Basophil percentage', 'Eosinophil percentage',
     'Alanine aminotransferase', 'Alkaline phosphatase', 'Aspartate aminotransferase',
     'Mean corpuscular hemoglobin', 'Mean corpuscular hemoglobin concentration',
     'Mean corpuscular volume', 'Red cell distribution width, coefficient of variation',
     'Red blood cell count']))
PATIENT_DOC = {
    'patient_key': ('string', '<dataset>:<stay_id>, e.g. mimiciv:12345678 (format example); unique across sources. '
                    'Join patient/embedding tables on patient_key and hourly tables on patient_key plus hour.'),
    'dataset': ('string', 'Source dataset; source definitions are not interchangeable'),
    'tensor_row': ('integer', 'Zero-based row in this source tensor; repeats across datasets. '
                   'Use patient_key for joins; every patient has 24 rows in each hourly table.'),
    'stay_id': ('integer', 'Source ICU stay identifier'),
    'subject_id': ('integer', 'Within-source patient identifier; eICU uses the adapter surrogate'),
    'hadm_id': ('integer', 'Within-source hospital encounter identifier'),
    'source_subject_id': ('string', 'Original patient key when supplied, including eICU uniquepid'),
    'hospital_id': ('integer', 'Source hospital identifier when available; not a hospital-held-out split'),
    'partition': ('string', 'Within-source subject split: 70% train, 15% validation, remainder test, using '
                  'SHA-256(42:subject_id) ordering. Retained in the atlas; not hospital-held-out. '
                  'Imputation uses each source training/validation set; the encoder uses MIMIC-IV training patients.'),
    'hospital_mortality': ('integer', 'Recorded hospital mortality: 0=alive, 1=dead; retrospective endpoint'),
    'age': ('number', 'Age in years under source deidentification; 91 can represent a protected older-age group'),
    'gender': ('string', 'Recorded source category, not an ordinal model code'),
    'race': ('string', 'Recorded race/ethnicity; vocabularies differ across sources'),
    'admission_type': ('string', 'Source admission category; eICU uses an elective-surgery proxy'),
    'first_careunit': ('string', 'Recorded ICU unit category'),
    'charlson_comorbidity_index': ('number', 'Retrospective diagnosis-derived context, not verified pre-admission disease'),
    'baseline_sofa': ('number', 'Operational assumed-zero SOFA baseline, NOT a measured baseline'),
    'baseline_pf_ratio': ('number', 'Lowest paired P/F in the 24 hours strictly before released onset; '
                          'retrospective context, not a chronic baseline or admission-time predictor'),
    'cohort_definition': ('string', 'Source operational cohort definition'),
    'infection_time_is_proxy': ('boolean', 'eICU documentation/order infection proxy flag; '
                               'blank in MIMIC is not a claim of exact biological onset'),
    'strict_culture_sepsis3': ('boolean', 'eICU strict-culture sensitivity membership, not a main-cohort requirement'),
    'strict_onset_matches_primary': ('boolean', 'Whether strict sensitivity onset equals main proxy onset'),
    'death_time_is_proxy': ('boolean', 'eICU death/end-of-care proxy flag; '
                           '0 includes patients not reported dead, not proof of an exact death time'),
    'death_time_basis': ('string', 'Source evidence supporting death/end-of-care time'),
    'age_is_deidentified': ('boolean', 'Explicit age-deidentification flag where supplied'),
    'onset_offset_minutes': ('number', 'Operational sepsis onset minus ICU admission, in minutes: the later of '
                            'selected suspected-infection time and the first qualifying SOFA assessment. '
                            'Date-only evidence uses the lower-bound onset; its upper bound is not in this CSV. '
                            'In eICU the infection anchor is the IV-order-start proxy, not exact biological onset.'),
    'strict_onset_offset_minutes': ('number', 'eICU strict-culture sensitivity onset relative to ICU admission'),
    'followup_hours': ('number', 'Sum of exposure_seconds/3600 across hours 0..23, from onset to the earliest of '
                       'onset+24h, ICU discharge and recorded death. Range 0..24; partial hours retained. '
                       'Recorded death without a usable death time makes all hours structural (0 exposure).'),
}
ATLAS_DOC = {
    'atlas_row': ('integer', 'Zero-based combined row across all three sources, including ineligible patients. '
                  'Distinct from the within-source tensor_row; patient_key is the stable join key.'),
    'eligible': ('boolean', '1 if sufficient observed shared cells and channels exist for representation'),
    'eligibility_reason': ('string', 'Disjoint eligibility classification, never a patient deletion'),
    'selected_map': ('string', 'Dataset-level validation-locked map, or reference_identity for MIMIC-IV; '
                             'a map name does not imply this patient has an embedding or received correction'),
    'transport_supported': ('boolean', '1 if selected target map supports correction; '
                            '0 for the reference, ineligible patients, identity maps or unsupported targets'),
    'original_x': ('number', 'Original embedding projected on reference-training principal component 1'),
    'original_y': ('number', 'Original embedding projected on reference-training principal component 2'),
    'adapted_x': ('number', 'Adapted embedding projected on the same principal component 1'),
    'adapted_y': ('number', 'Adapted embedding projected on the same principal component 2'),
    'original_cluster': ('integer', 'Reference-training descriptive group before adaptation, not a validated phenotype'),
    'adapted_cluster': ('integer', 'Reference-training descriptive group after adaptation, not a validated phenotype'),
}
HOUR_DOC = {
    'exposure_seconds': 'Seconds in [onset+hour h, onset+(hour+1) h), truncated at ICU discharge or recorded death '
                        'and limited to the 24h window: 0=structural, 0<x<3600=partial, 3600=full. '
                        'A recorded death without a usable death time makes every hour structural. '
                        'Hours reflect retrospective event time, not when information became available.',
    'neq_known_seconds': 'Seconds with supported norepinephrine-equivalent infusion evidence',
    'neq_invalid_seconds': 'Seconds with unresolved or conflicting vasoactive evidence',
    'vent_known_seconds': 'Seconds with classified respiratory-support evidence',
    'vent_conflict_seconds': 'Seconds with conflicting respiratory-support evidence',
    'urine_invalid_times': 'Count of invalid/ambiguous urine timestamps withholding this hourly value',
}

DICTIONARY_COLUMNS = ('archive', 'file', 'file_description', 'row_unit', 'key_columns', 'position', 'column',
    'type', 'unit', 'description', 'missing_value', 'allowed_values', 'aggregation', 'screening_min',
    'screening_max', 'role', 'predictor_default')
FLOW_DOC = {
    'dataset': 'Source dataset; counts are source ICU stays, with one stay per retained patient after first-stay selection.',
    'stage': '01=base selection; 02=infection evidence; 06=SOFA/onset adjudication; '
             '07-08=tensor construction/imputation; atlas=representation eligibility and transport support.',
    'kind': 'selection=nested retention; exclusion_reason=disjoint exclusions; parallel_evidence and '
            'sensitivity_or_subgroup can overlap. Other kinds describe retained patients, not additional exclusions.',
    'step': 'Machine-readable selection, exclusion, evidence or support category; defined by step_description.',
    'step_description': 'Operational definition of this row; read with denominator_population and kind.',
    'denominator_population': 'Named population counted by denominator; each row is source-specific.',
    'denominator': 'Number of source stays in this row\'s parent population; do not sum parallel branches.',
    'count': 'Number belonging to this category; exclusion_reason counts excluded stays, not survivors.',
    'count_meaning': 'Meaning of count: retained_after_filter, excluded_for_reason, retained_in_release, '
                     'eligible_for_representation, ineligible_but_retained, reference_unchanged, '
                     'eligible_target_subgroup, evidence_branch_members or subgroup_members.',
    'excluded': 'denominator minus count for selection rows only; blank for other kinds, including exclusion_reason.',
    'note': 'Additional qualifications about source definitions, sensitivity membership and overlapping branches.'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_record(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return dict(sha256=h.hexdigest(), size_bytes=Path(path).stat().st_size)


def project_path(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT), f'Path outside project: {name}')
    return path


def parquet_rows(path):
    import duckdb
    with duckdb.connect() as con:
        cursor = con.execute('SELECT * FROM read_parquet(?)', [str(path)])
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def validate_inputs():
    """Verify the completed QC chain without training or modifying processing outputs."""
    expected = {}
    def bind(records):
        for name, record in records.items():
            require(name not in expected or expected[name] == record, f'Conflicting provenance: {name}')
            expected[name] = record
    qc_path = ROOT / 'outputs/atlas/qc.json'
    require(qc_path.is_file(),
            'No completed atlas QC found. Build from raw data with bash pipeline/01_build_dataset.sh first.')
    qc = read_json(qc_path)
    require(qc['version'] == VERSION and qc['status'] == 'PASS' and qc['failures'] == 0,
            'Atlas QC must pass on current artifacts; run atlas stage 06')
    bind(qc['code'])
    bind(qc['dependencies'])
    stages = {}
    for stage in range(1, 6):
        report = read_json(ROOT / f'outputs/atlas/{stage:02d}.json')
        require(report['version'] == VERSION and report['status'] == 'COMPLETE', f'Incomplete atlas stage {stage}')
        for section in ('code', 'dependencies', 'outputs'):
            bind(report[section])
        stages[stage] = report
    bind(stages[1]['external'])
    require(qc['selected'] == stages[4]['selected'] == stages[5]['selected'], 'Atlas map selection mismatch')
    require(qc['test_exposure'] == stages[4]['test_exposure'], 'Evaluation history mismatch')
    manifests, reports = {}, {}
    for name in DATASETS:
        folder = ROOT / 'data/processed' / name
        manifest = read_json(folder / 'manifest.json')
        source_qc = read_json(ROOT / 'outputs' / name / 'qc.json')
        require(source_qc['status'] == 'PASS' and source_qc['failed_required_checks'] == 0,
                f'{name}: full dataset QC must pass')
        require(manifest['dataset_version'] == source_qc['dataset_version'] == VERSION, f'{name}: version mismatch')
        require(source_qc['manifest_sha256'] == file_record(folder / 'manifest.json')['sha256'], f'{name}: stale QC')
        require(manifest['imputation']['status'] == 'COMPLETE', f'{name}: imputation incomplete')
        bind(manifest['observed']['provenance'])
        bind(manifest['imputation']['artifacts'])
        bind({(folder / filename).relative_to(ROOT).as_posix(): record
              for filename, record in manifest['observed']['output_manifest'].items()})
        manifests[name] = manifest
        reports[name] = {stage: read_json(ROOT / 'outputs' / name / filename) for stage, filename in
                         ((1, '01_base.json'), (2, '02_infection.json'), (6, '06_sepsis.json'))}
    for name, record in expected.items():
        path = project_path(name)
        require(path.is_file() and file_record(path) == record, f'Missing or changed certified input: {name}')
    paths = [qc_path, ROOT / 'src/common/feature_registry.json', ROOT / 'src/common/tensor_schema.json',
             ROOT / 'src/common/measurement_rules.json', ROOT / 'src/04_atlas_datasets/atlas_config.json']
    for name in DATASETS:
        paths.extend((ROOT / 'data/processed' / name).glob('*.*'))
        paths.extend(ROOT / 'outputs' / name / f for f in
                     ('01_base.json', '02_infection.json', '06_sepsis.json', 'qc.json'))
    paths.extend((ROOT / 'data/processed/atlas').glob('*.*'))
    paths.extend(ROOT / f'outputs/atlas/{i:02d}.json' for i in range(1, 6))
    paths.extend((ROOT / 'pipeline').glob('*.*'))
    inputs = {p.relative_to(ROOT).as_posix(): file_record(p) for p in paths if p.is_file()}
    return manifests, reports, stages, qc, inputs


def field(name, dtype, description, **extra):
    return dict(column=name, type=dtype, description=description, missing_value='empty field', **extra)


def table_structure(filename):
    """Document the row unit and composite key without adding another release file."""
    if filename == 'data_dictionary.csv':
        return dict(file_description='Column definitions for every released CSV, including this dictionary.',
                    row_unit='one column definition per released table', key_columns='archive; file; column')
    if filename == 'cohort_flow.csv':
        return dict(file_description='Source-specific cohort selection and atlas eligibility/support counts; '
                                    'parallel evidence and sensitivity branches are not additional filters.',
                    row_unit='one source population count at one processing step',
                    key_columns='dataset; stage; kind; step')
    if filename.endswith('trajectory_profiles.csv.gz'):
        return dict(file_description='Descriptive quantiles of original observed values grouped by adapted atlas '
                                    'cluster, pooling all partitions; neither imputed nor transported clinical values.',
                    row_unit='one dataset/adapted-cluster/clinical-feature/hour summary; all partitions pooled',
                    key_columns='dataset; cluster; feature; hour')
    if '_hourly_' in filename:
        purpose = ('Accepted observations in canonical units; missing cells remain empty.' if '_observed.' in filename
                   else 'Observed values preserved; eligible gaps filled by within-source, validation-selected '
                        'SAITS or baselines. Still contains missing cells; consult the matching hourly_support table.'
                   if '_imputed.' in filename else
                   'Exposure, accepted observation counts and per-cell imputation method codes for the matching '
                   'observed/imputed tables; join on patient_key and hour.')
        return dict(file_description=purpose,
                    row_unit='one retained patient and onset-relative hour; exactly 24 rows per patient',
                    key_columns='patient_key; hour')
    if 'embeddings_original' in filename:
        purpose = ('Shared MIMIC-IV-trained encoder coordinates before transport, standardized on reference training '
                   'patients. Latent dimensions are not clinical variables; all dimensions are empty when ineligible.')
    elif 'embeddings_adapted' in filename:
        purpose = ('The same latent coordinates after the selected target map. Reference and unsupported patients '
                   'retain original coordinates; ineligible patients have empty coordinates. Clinical tensors are unchanged.')
    elif filename.startswith('atlas_'):
        purpose = ('Union of the three source patient tables plus atlas eligibility, selected map, support, '
                   'PCA coordinates and clusters; no clinical-cohort patient is removed.')
    else:
        purpose = ('Source cohort identifiers, patient split, retrospective outcome/context and follow-up; '
                   'join to this source hourly tables and the combined atlas using patient_key.')
    return dict(file_description=purpose, row_unit='one retained patient, including representation-ineligible patients',
                key_columns='patient_key')


def complete_dictionary(rows):
    metadata = {
        'archive': 'Containing ZIP filename; blank identifies a top-level CSV outside an archive.',
        'file': 'Exact CSV filename; .csv.gz members must be decompressed after opening their containing ZIP.',
        'file_description': 'Purpose of the file and its relation to the other released tables.',
        'row_unit': 'What one row of the documented table represents.',
        'key_columns': 'Semicolon-separated columns jointly identifying a row in that table. '
                       'A blank archive is valid for top-level files; use explicit keys rather than row order.',
        'position': 'Zero-based column position in the documented CSV header.',
        'column': 'Exact case-sensitive column name in the documented table.',
        'type': 'CSV interpretation: string, integer, number or boolean. Booleans are serialized as 0/1.',
        'unit': 'Physical unit or count unit; blank means dimensionless, categorical or not applicable, never zero.',
        'description': 'Meaning and construction of this column, including source-specific qualifications.',
        'missing_value': 'Meaning of an empty CSV field in the documented column; not permitted means no blanks allowed.',
        'allowed_values': 'Permitted codes/ranges; JSON arrays list categories observed in this release. '
                          'Blank means no enumeration is supplied, not that all values are valid.',
        'aggregation': 'Observed hourly reducer: max/min/mean over accepted evidence; coherent_gcs uses one chosen '
                       'assessment; paired_min uses paired P/F; validated_volume_sum sums accepted urine; '
                       'concurrent_peak is peak concurrent NEQ; any_invasive_with_coverage classifies support. '
                       'For imputed tables this describes preserved observations, not how gaps are filled.',
        'screening_min': 'Inclusive lower screening limit in unit; not a clinical reference limit. '
                         'Blank means no numeric limit is supplied, not zero.',
        'screening_max': 'Inclusive upper screening limit in unit; not a clinical reference limit. '
                         'Urine has no finite hourly upper bound because accepted volumes are summed.',
        'role': 'Construction role: observed_only channels are never imputed; imputation_eligible does not guarantee '
                'a filled value. Other labels distinguish recorded admission context from retrospective context.',
        'predictor_default': 'Static-feature policy only: 1=default candidate admission context; 0=exclude by default '
                             '(retrospective or assumed baseline). Blank=not assigned, not permission to use as a predictor. '
                             'This flag does not establish availability at prediction time.'}
    rows = [dict(row, **table_structure(row['file'])) for row in rows if row['file'] != 'data_dictionary.csv']
    for position, name in enumerate(DICTIONARY_COLUMNS):
        dtype = ('integer' if name == 'position' else 'number' if name in {'screening_min', 'screening_max'} else
                 'boolean' if name == 'predictor_default' else 'string')
        entry = field(name, dtype, metadata[name])
        entry['missing_value'] = ('empty=top-level CSV' if name == 'archive' else
                                  'empty=not applicable or not specified; see description' if name in
                                  {'unit', 'allowed_values', 'aggregation', 'screening_min', 'screening_max',
                                   'role', 'predictor_default'} else 'not permitted')
        if name == 'position':
            entry.update(unit='column index', allowed_values='nonnegative integer')
        elif name == 'type':
            entry['allowed_values'] = 'string; integer; number; boolean'
        elif name == 'predictor_default':
            entry['allowed_values'] = '0=exclude by default; 1=candidate admission context; empty=not assigned'
        rows.append(dict(archive='', file='data_dictionary.csv', position=position,
                         **table_structure('data_dictionary.csv'), **entry))
    return rows


def documented(name):
    dtype, description = {**PATIENT_DOC, **ATLAS_DOC}[name]
    allowed = '0=false; 1=true' if dtype == 'boolean' else ''
    if name == 'partition':
        allowed = '; '.join(PARTITIONS)
    if name == 'hospital_mortality':
        allowed = '0=alive; 1=dead'
    if name == 'dataset':
        allowed = '; '.join(DATASETS)
    if name == 'eligibility_reason':
        allowed = 'eligible; insufficient_observed_cells; insufficient_observed_features'
    unit = {'age': 'years', 'baseline_sofa': 'score', 'baseline_pf_ratio': 'mmHg',
            'charlson_comorbidity_index': 'score', 'onset_offset_minutes': 'min',
            'strict_onset_offset_minutes': 'min', 'followup_hours': 'h'}.get(name, '')
    result = field(name, dtype, description, unit=unit, allowed_values=allowed)
    required = {'patient_key', 'dataset', 'tensor_row', 'stay_id', 'subject_id', 'hadm_id',
                'partition', 'hospital_mortality', 'age', 'cohort_definition', 'onset_offset_minutes',
                'followup_hours', 'atlas_row', 'eligible', 'eligibility_reason', 'selected_map', 'transport_supported'}
    if name in required:
        result['missing_value'] = 'not permitted'
    elif name.startswith(('original_', 'adapted_')):
        result['missing_value'] = 'empty when eligible=0; no -1 cluster is released'
    elif name in {'hospital_id', 'strict_culture_sepsis3', 'strict_onset_matches_primary',
                  'infection_time_is_proxy', 'death_time_is_proxy', 'death_time_basis',
                  'strict_onset_offset_minutes'}:
        result['missing_value'] = 'empty=not supplied/not applicable in MIMIC, or unavailable in eICU; not false'
    return result


def patient_fields(names, rows, schema, dataset, config):
    fields = [documented(name) for name in names]
    roles = {item['name']: item for item in schema['policy']['static_features']}
    for entry in fields:
        name = entry['column']
        identifiers = {
            'stay_id': 'MIMIC-IV stay_id; MIMIC-III CareVue icustay_id; eICU patientunitstayid.',
            'subject_id': 'MIMIC subject_id; eICU project surrogate from dense rank of uniquepid in source data. '
                          'Use source_subject_id to join eICU back to uniquepid; numeric IDs are not cross-dataset keys.',
            'hadm_id': 'MIMIC hadm_id; eICU patienthealthsystemstayid. Unique only within its source.',
            'source_subject_id': 'Original eICU uniquepid, or the MIMIC subject_id serialized as text. '
                                 'Read as a string to preserve source identifiers.',
            'hospital_id': 'eICU hospitalid; blank in both MIMIC sources. Patient partitions may share hospitals.'}
        if name in identifiers:
            entry['description'] = identifiers[name]
        if name == 'source_subject_id':
            entry['missing_value'] = 'not permitted'
        if name in roles:
            entry.update(role=roles[name]['role'], predictor_default=roles[name]['predictor_default'])
        if name in {'gender', 'race', 'admission_type', 'first_careunit', 'cohort_definition',
                    'death_time_basis', 'selected_map'}:
            values = sorted({str(row[name]) for row in rows if row.get(name) not in (None, '')})
            entry['allowed_values'] = json.dumps(values, ensure_ascii=False)
        if name in {'original_cluster', 'adapted_cluster'}:
            entry['allowed_values'] = f"0..{config['atlas']['clusters'] - 1}; descriptive labels, no ordinal meaning"
        if name == 'eligible':
            p = config['eligibility']
            entry['description'] = (f"1 requires >= {p['minimum_patient_observed_cells']} observed cells and "
                f">= {p['minimum_patient_observed_features']} observed channels across the 24-hour window among "
                'the shared encoder features listed in release_manifest.json:atlas.encoder_features; '
                'this is representation eligibility, not clinical-cohort inclusion.')
        if name == 'eligibility_reason':
            entry['description'] = ('Representation eligibility, evaluated before transport: insufficient cells takes '
                'precedence over insufficient distinct channels; ineligible patients retain clinical tables but have '
                'blank embeddings, coordinates and clusters. This is not an OT solver or support failure.')
        if name == 'age':
            policies = {
                'mimiciv': 'anchor_age + admission_year - anchor_year; protected anchor_age=91 can yield other released ages',
                'mimic3-carevue': 'Completed years at admission; shifted ages >=300 mapped to 91',
                'eicu': 'Recorded age; source >89 mapped to 91',
            }
            entry['description'] = policies.get(dataset, 'Source-specific age; MIMIC-IV anchor-year arithmetic, '
                'CareVue shifted ages >=300 mapped to 91, eICU >89 mapped to 91; see age_is_deidentified')
    return fields


def temporal_fields(registry, schema, bounds, imputed=False):
    """Document hourly cell domains, which differ from per-event limits for summed urine."""
    domains = {**bounds, **schema['policy']['derived_bounds'], 'urine_output': [0, None]}
    detail = {
        'pf_ratio': 'Minimum paired source P/F; never the ratio of separately aggregated hourly PaO2 and FiO2.',
        'urine_output': 'Sum of accepted recorded volumes; no hourly upper cap and no extrapolation of partial hours.',
        'neq': 'Peak sum of concurrent recorded rates; absent records are unknown, not zero. Five-drug subset only.',
        'vent': '1=any classified invasive segment; 0=all segments classified noninvasive/off support; '
                'missing evidence is not zero.',
    }
    fields = []
    for feature in registry['temporal_features']:
        name = feature['name']
        require(name in domains and name in FEATURE_NAMES, f'Missing feature documentation: {name}')
        low, high = domains[name]
        description = FEATURE_NAMES[name]
        if name.startswith('gcs_'):
            description += '. Same chosen assessment across components: lowest complete total, latest on ties; '
            description += 'latest partial assessment if no complete one exists.'
        elif name in detail:
            description += '. ' + detail[name]
        observed_only = name in schema['observed_only_features']
        if imputed and not observed_only:
            description += '. Observations preserved; gaps use the matching hourly_support method. '
            description += 'SAITS is retrospective/bidirectional. Forward fill uses earlier observed values, '
            description += 'with training median before the first measurement; an out-of-bounds SAITS prediction '
            description += 'uses the preselected baseline. Models/statistics are fitted separately within each source.'
        elif imputed:
            description += '. Never imputed; exactly the same accepted observations and missing cells as hourly_observed.'
        if name == 'neq':
            weights = schema['policy']['neq']['coefficients']
            description += '. Concurrent-rate sum: ' + ' + '.join(f'{value:g}*{drug}' for drug, value in weights.items())
            description += '; vasopressin input is units/min, other drug inputs are mcg/kg/min. '
            description += 'Take the hourly maximum; excludes dobutamine and unextracted agents.'
        entry = field(name, 'number', description, unit=feature['canonical_unit'],
                      aggregation=schema['policy']['aggregation'][name], screening_min=low, screening_max=high,
                      role='observed_only' if observed_only else 'imputation_eligible')
        entry['missing_value'] = ('empty=unfilled; exposure_seconds=0 means structural; see per-cell method' if imputed
                                  else 'empty=no accepted observation; exposure_seconds=0 means structural; zero is a value')
        entry['allowed_values'] = 'Inclusive hourly screening domain; not a normal range'
        if high is None:
            entry['allowed_values'] += '; no finite upper bound'
        fields.append(entry)
    return fields


def csv_value(value):
    if value is None:
        return ''
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ''
        require(math.isfinite(value), 'Infinite value cannot be released')
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


class Archive:
    """Stream deterministic gzip members into a ZIP without compressing them twice."""
    def __init__(self, folder, name, dictionary, records):
        self.folder, self.name = folder, name
        self.dictionary, self.records = dictionary, records
        self.zip = zipfile.ZipFile(folder / f'{name}.zip', 'w', compression=zipfile.ZIP_STORED, allowZip64=True)

    def table(self, filename, fields, rows):
        require(len({f['column'] for f in fields}) == len(fields), f'Duplicate fields in {filename}')
        filename = f'{self.name}_{filename}'
        path = self.folder / filename
        count = 0
        with path.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as text:
                    writer = csv.writer(text, lineterminator='\n')
                    writer.writerow([f['column'] for f in fields])
                    for row in rows:
                        require(len(row) == len(fields), f'Row width mismatch: {filename}')
                        writer.writerow([csv_value(value) for value in row])
                        count += 1
        info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        with path.open('rb') as source, self.zip.open(info, 'w', force_zip64=True) as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        self.records[f'{self.name}.zip/{filename}'] = dict(
            **file_record(path), rows=count, columns=[f['column'] for f in fields])
        self.dictionary.extend(dict(archive=f'{self.name}.zip', file=filename, position=i,
                                    **table_structure(filename), **f)
                               for i, f in enumerate(fields))
        path.unlink()
        print(f'[EXPORT] {self.name}/{filename}: {count:,} rows', flush=True)

    def close(self):
        self.zip.close()


def offset_minutes(row, column):
    value, origin = row.get(column), row.get('icu_intime')
    return None if value is None or origin is None else (value - origin).total_seconds() / 60


def hourly_rows(keys, array):
    for i, key in enumerate(keys):
        for hour in range(24):
            yield [key, hour, *array[i, hour].tolist()]


def eligibility_reasons(counts, indices, policy):
    observed = counts[:, :, indices] > 0
    enough_cells = observed.sum(axis=(1, 2)) >= policy['minimum_patient_observed_cells']
    enough_features = observed.any(axis=1).sum(axis=1) >= policy['minimum_patient_observed_features']
    return np.where(~enough_cells, 'insufficient_observed_cells',
                    np.where(~enough_features, 'insufficient_observed_features', 'eligible'))


def export_source(name, archive, registry, schema, bounds, atlas_inputs, domain, config):
    folder = ROOT / 'data/processed' / name
    cohort = parquet_rows(folder / 'cohort.parquet')
    required = {'stay_id', 'icu_intime', 'sepsis_onset_time', 'gender', 'race', 'admission_type', 'first_careunit'}
    if name == 'eicu':
        required |= {'source_subject_id', 'hospital_id', 'strict_culture_sepsis3',
                     'infection_time_is_proxy', 'death_time_is_proxy'}
    require(cohort and required.issubset(cohort[0]), f'{name}: missing release patient metadata')
    with np.load(folder / 'tensor_support.npz', allow_pickle=False) as support, \
         np.load(folder / 'imputation_support.npz', allow_pickle=False) as imp:
        ids, subjects, admissions = (support[k] for k in ('stay_ids', 'subject_ids', 'hadm_ids'))
        features = support['features'].tolist()
        require(features == [f['name'] for f in registry['temporal_features']], f'{name}: feature order')
        require(support['units'].tolist() == [f['canonical_unit'] for f in registry['temporal_features']],
                f'{name}: feature units')
        require(np.array_equal(ids, [row['stay_id'] for row in cohort]), f'{name}: cohort row order')
        require(len(np.unique(ids)) == len(ids), f'{name}: repeated stays')
        keys = [f'{name}:{int(stay)}' for stay in ids]
        counts, exposure, methods = support['observation_counts'], support['exposure_seconds'], imp['method_codes']
        require(counts.shape == methods.shape == (len(ids), 24, len(features)), f'{name}: support shape')
        require(exposure.shape == (len(ids), 24), f'{name}: exposure shape')
        require(np.isfinite(exposure).all() and ((exposure >= 0) & (exposure <= 3600)).all(),
                f'{name}: invalid follow-up exposure')
        require(np.isin(methods, range(5)).all(), f'{name}: unknown imputation method code')
        static, static_names = support['static'], support['static_features'].tolist()
        partitions, labels = imp['partition'], support['labels']
        require(np.isin(partitions, range(3)).all(), f'{name}: partitions')
        require(np.isin(labels, [0, 1]).all(), f'{name}: mortality labels')
        atlas_rows = np.flatnonzero(atlas_inputs['dataset'] == domain)
        require(np.array_equal(atlas_inputs['stay_id'][atlas_rows], ids), f'{name}: atlas row binding')
        require(np.array_equal(atlas_inputs['partition'][atlas_rows], partitions), f'{name}: atlas partitions')
        require(np.array_equal(atlas_inputs['subject_id'][atlas_rows], subjects), f'{name}: atlas subject binding')
        reasons = eligibility_reasons(counts, atlas_inputs['common_indices'], config['eligibility'])
        require(np.array_equal(reasons == 'eligible', atlas_inputs['eligible'][atlas_rows]),
                f'{name}: representation eligibility differs')
        patient_rows = []
        for i, row in enumerate(cohort):
            result = {column: row.get(column) for column in PATIENT_DOC}
            result.update(patient_key=keys[i], dataset=name, tensor_row=i, stay_id=int(ids[i]),
                          subject_id=int(subjects[i]), hadm_id=int(admissions[i]),
                          partition=PARTITIONS[int(partitions[i])], hospital_mortality=int(labels[i]),
                          onset_offset_minutes=offset_minutes(row, 'sepsis_onset_time'),
                          strict_onset_offset_minutes=offset_minutes(row, 'strict_culture_onset_time'),
                          followup_hours=float(exposure[i].sum() / 3600))
            result['cohort_definition'] = result['cohort_definition'] or COHORT_DEFINITIONS[name]
            result['source_subject_id'] = result['source_subject_id'] or str(int(subjects[i]))
            require(result['onset_offset_minutes'] is not None, f'{name}: missing onset/ICU origin')
            for column in ('age', 'charlson_comorbidity_index', 'baseline_sofa', 'baseline_pf_ratio'):
                result[column] = float(static[i, static_names.index(column)])
            patient_rows.append(result)
        archive.table('patients.csv.gz', patient_fields(PATIENT_DOC, patient_rows, schema, name, config),
                      ([row[k] for k in PATIENT_DOC] for row in patient_rows))
        key_fields = [documented('patient_key'), field('hour', 'integer',
            'Zero-based hour 0..23; [onset+hour, onset+hour+1h), truncated by follow-up',
            unit='h', allowed_values='0..23')]
        key_fields[1]['missing_value'] = 'not permitted'
        observed = np.load(folder / 'tensor_observed.npy', mmap_mode='r', allow_pickle=False)
        imputed = np.load(folder / 'tensor_imputed.npy', mmap_mode='r', allow_pickle=False)
        require(observed.shape == imputed.shape == counts.shape, f'{name}: tensor shape')
        for start in range(0, len(ids), 256):
            sl = slice(start, start + 256)
            measured = np.isfinite(observed[sl])
            require(np.array_equal(measured, counts[sl] > 0), f'{name}: observed mask mismatch')
            require(np.array_equal(measured, methods[sl] == 1), f'{name}: observed method mismatch')
            require(np.array_equal(np.isfinite(imputed[sl]), methods[sl] != 0), f'{name}: imputed method mismatch')
            require(np.array_equal(imputed[sl][measured], observed[sl][measured]), f'{name}: observations changed')
            require(not np.isfinite(imputed[sl][exposure[sl] == 0]).any(), f'{name}: structural imputation')
        archive.table('hourly_observed.csv.gz', key_fields + temporal_fields(registry, schema, bounds),
                      hourly_rows(keys, observed))
        archive.table('hourly_imputed.csv.gz', key_fields + temporal_fields(registry, schema, bounds, imputed=True),
                      hourly_rows(keys, imputed))
        hourly = {key: support[key] for key in HOUR_DOC}
        fields = key_fields + [field(k, 'number', v, unit='count' if k == 'urine_invalid_times' else 's')
                               for k, v in HOUR_DOC.items()]
        fields += [field(f'{f}_observations', 'integer', f'Accepted evidence count for {f}; >0 means observed')
                   for f in features]
        fields += [field(f'{f}_method', 'integer', f'Final value provenance for {f}', allowed_values=METHODS)
                   for f in features]
        for entry in fields:
            entry['missing_value'] = ('not permitted' if entry['column'] == 'patient_key' else
                                      'not permitted; zero is explicitly coded')
            if entry['column'].endswith('_observations'):
                entry['unit'] = 'count'
                entry['allowed_values'] = 'nonnegative integer; 0=no accepted hourly value'
                feature = entry['column'].removesuffix('_observations')
                if feature in ('neq', 'vent'):
                    entry['description'] = f'Count of classified contributing segments for {feature}, not drug orders or measurements'
                elif feature.startswith('gcs_'):
                    entry['description'] = '1 if this component is accepted from the chosen GCS assessment, otherwise 0'
                elif feature == 'pf_ratio':
                    entry['description'] = 'Number of accepted paired P/F records contributing to the hourly minimum'
        def support_rows():
            for i, key in enumerate(keys):
                for hour in range(24):
                    yield [key, hour, *(hourly[k][i, hour] for k in HOUR_DOC),
                           *counts[i, hour].tolist(), *methods[i, hour].tolist()]
        archive.table('hourly_support.csv.gz', fields, support_rows())
    return patient_rows, reasons


def export_atlas(archive, inputs, patients, reasons, selected, schema, config):
    folder = ROOT / 'data/processed/atlas'
    coords = parquet_rows(folder / 'coordinates.parquet')
    require(len(coords) == len(patients) == len(inputs['eligible']), 'Atlas length mismatch')
    with np.load(folder / 'representations.npz', allow_pickle=False) as rep, \
         np.load(folder / 'transport.npz', allow_pickle=False) as transport:
        original, adapted = rep['z'], transport['selected']
        supported = transport['selected_supported']
        require(original.shape == adapted.shape and len(original) == len(patients), 'Embedding shape mismatch')
        require(np.isfinite(original).all() and np.isfinite(adapted).all(), 'Nonfinite working embeddings')
        require(not supported[~inputs['eligible']].any(), 'Ineligible patients cannot be transported')
        require(np.array_equal(adapted[~supported], original[~supported]), 'Unsupported embeddings changed')
        metadata = []
        names = list(PATIENT_DOC) + list(ATLAS_DOC)
        for i, (row, coordinate) in enumerate(zip(patients, coords)):
            require(row['patient_key'] == coordinate['patient_key'], 'Atlas patient key mismatch')
            eligible = bool(inputs['eligible'][i])
            require(bool(coordinate['eligible']) == eligible
                    and bool(coordinate['transport_supported']) == bool(supported[i]), 'Atlas support mismatch')
            result = dict(row, atlas_row=i, eligible=eligible, eligibility_reason=str(reasons[i]),
                          selected_map=selected.get(row['dataset'], 'reference_identity'),
                          transport_supported=bool(supported[i]))
            for column in ATLAS_DOC:
                if column.startswith(('original_', 'adapted_')):
                    result[column] = coordinate[column] if eligible else None
            metadata.append(result)
        archive.table('patients.csv.gz', patient_fields(names, metadata, schema, 'atlas', config),
                      ([row[k] for k in names] for row in metadata))
        fields = [documented('patient_key')] + [field(f'z_{j:02d}', 'number',
                  'Reference-standardized temporal embedding coordinate, not a clinical measurement', unit='latent')
                  for j in range(original.shape[1])]
        for entry in fields[1:]:
            entry['missing_value'] = 'empty when eligible=0; finite otherwise'
        for filename, values in (('embeddings_original.csv.gz', original), ('embeddings_adapted.csv.gz', adapted)):
            archive.table(filename, fields, ([row['patient_key'], *(values[i].tolist()
                if inputs['eligible'][i] else [None] * values.shape[1])] for i, row in enumerate(patients)))
    profiles = parquet_rows(folder / 'trajectories.parquet')
    docs = {
        'dataset': PATIENT_DOC['dataset'], 'cluster': ATLAS_DOC['adapted_cluster'],
        'feature': ('string', 'Clinical feature name; see hourly table definitions'),
        'unit': ('string', 'Canonical clinical unit'), 'hour': ('integer', 'Hour 0..23 after source-specific onset'),
        'patients_in_group': ('integer', 'Eligible patients from this dataset assigned to this adapted cluster; '
                              'all partitions, including eligible unsupported targets with unchanged embeddings'),
        'exposed_patients': ('integer', 'Of patients_in_group, number with exposure_seconds >0 in this hour'),
        'observed_patients': ('integer', 'Of exposed_patients, number with a finite original observed value '
                             'for this feature/hour; denominator for the reported quantiles'),
        'q25': ('number', '25th percentile across original observed hourly patient values; linear interpolation'),
        'median': ('number', '50th percentile across original observed hourly patient values; linear interpolation'),
        'q75': ('number', '75th percentile across original observed hourly patient values; linear interpolation'),
    }
    profile_fields = [field(k, *v) for k, v in docs.items()]
    for entry in profile_fields:
        name = entry['column']
        entry['missing_value'] = 'empty iff observed_patients=0' if name in {'q25', 'median', 'q75'} else 'not permitted'
        if name in {'q25', 'median', 'q75'}:
            entry['unit'] = 'given by unit in the same row'
            entry['description'] += '; no imputed values or OT-modified clinical values'
        elif name in {'patients_in_group', 'exposed_patients', 'observed_patients'}:
            entry['unit'] = 'patients'
            entry['allowed_values'] = 'nonnegative integer; observed_patients <= exposed_patients <= patients_in_group'
        elif name == 'hour':
            entry.update(unit='h', allowed_values='0..23')
        elif name == 'cluster':
            entry['allowed_values'] = f"0..{config['atlas']['clusters'] - 1}; descriptive, not ordinal"
    archive.table('trajectory_profiles.csv.gz', profile_fields,
                  ([row[k] for k in docs] for row in profiles))
    return metadata


def describe_flow(rows, reports, config):
    pairing = 'A diagnostic culture from 72h before to 24h after an eligible IV-antimicrobial prescription start; '
    pairing += 'suspected infection is anchored at the earlier culture/prescription time.'
    infection = {
        'mimiciv': pairing + ' MIMIC-IV uses diagnostic test evidence; prescription does not prove administration.',
        'mimic3-carevue': pairing + ' CareVue uses a diagnostic specimen-ID/name proxy; test-level evidence is unavailable.',
        'eicu': 'Configured infection diagnosis and IV-antimicrobial order, both within ICU admission +/-24h and '
                'within 24h of each other. Infection time is IV-order start; cultures are not required for the main cohort.'}
    first_stay = {
        'mimiciv': 'not the first ICU stay per subject in the supplied MIMIC-IV data, selected before eligibility filters',
        'mimic3-carevue': 'not the first ICU stay per subject within the supplied CareVue subset, before eligibility filters; '
                          'not necessarily the first across other MIMIC-III systems',
        'eicu': 'not the first ICU stay by descending hospitaladmitoffset within a single recorded hospital encounter; '
                'not a lifetime-first-ICU claim'}
    reasons = {
        'missing_linked_records': 'missing linked patient or hospital-admission records',
        'invalid_icu_interval': 'missing or nonpositive ICU duration',
        'invalid_hospital_interval': 'missing or inconsistent hospital times/offsets relative to the ICU stay',
        'missing_age': 'missing or unparseable age', 'age_under_18': 'age below 18 years',
        'invalid_age': 'eICU age above 120 after deidentification handling',
        'missing_admission_type': 'missing admission type', 'elective_admission': 'ELECTIVE admission type',
        'ambiguous_hospital_order': 'more than one recorded eICU hospital encounter for the same uniquepid; '
                                    'encounters cannot be reliably ordered',
        'missing_first_stay_offset': 'missing hospitaladmitoffset needed to rank eICU ICU stays',
        'ambiguous_first_stay_offset': 'tied offsets at the candidate first eICU ICU stay',
        'non_icu_unit': 'an explicitly excluded non-ICU/stepdown/test unit category',
        'elective_surgery': 'APACHE elective-surgery indicator equal to 1; missing status alone is retained',
        'conflicting_elective_surgery': 'conflicting recorded elective-surgery indicators',
        'missing_mortality_label': 'unrecognized or missing hospital discharge survival status',
        'conflicting_mortality': 'ICU discharge marked expired but hospital discharge not marked expired',
        'infection_timing_uncertain': 'no retained candidate and at least one candidate with only possible, '
                                      'rather than definite, infection pairing/presentation timing',
        'date_uncertainty_only': 'SOFA qualifies for some possible infection times but not robustly across '
                                 'the date-only infection interval',
        'no_in_icu_assessment': 'no in-ICU SOFA assessment in the candidate association windows',
        'no_usable_sofa_evidence': 'no usable observed SOFA component evidence in candidate in-ICU assessments',
        'sofa_below_two': 'no candidate qualifies at SOFA >=2 under the primary timing rules after other exclusions',
        'onset_outside_followup': 'the conservative onset upper bound exceeds documented follow-up'}
    descriptions = {
        'all_icu_stays': 'All ICU stays in this supplied source release, before first-stay or clinical filters.',
        'base_stays': 'Stays retained by base-cohort selection; denominator for the parallel evidence branches.',
        'with_iv_antimicrobial': 'At least one qualifying IV antibacterial/antifungal prescription; not verified administration.',
        'also_with_dated_culture': 'Among stays with eligible IV prescriptions, at least one eligible culture with a date/time.',
        'with_temporal_pair': pairing,
        'within_icu_presentation_window': 'The possible suspected-infection interval overlaps ICU admission +/-24h; '
                                          'definite timing is subsequently required for primary Sepsis-3 membership.',
        'definite_infection_timing': 'At least one candidate with definite culture/prescription pairing and '
                                     'definite ICU-presentation eligibility; this is not yet SOFA adjudication.',
        'any_medication_record': 'At least one source medication record, regardless of antimicrobial eligibility.',
        'mapped_antimicrobial_name': 'At least one medication name mapped to the configured antimicrobial vocabulary.',
        'eligible_iv_antimicrobial': 'At least one eligible antimicrobial with an accepted IV route and usable order timing.',
        'any_culture_record': 'At least one source microLab record, regardless of diagnostic-site eligibility.',
        'mapped_diagnostic_culture_site': 'At least one culture site matching the configured diagnostic-site vocabulary.',
        'eligible_diagnostic_culture': 'At least one diagnostic-site culture meeting source evidence and timing checks.',
        'both_eligible_sources_any_time': 'Both an eligible IV order and diagnostic culture, without requiring their temporal pairing.',
        'paired_within_72h_24h': pairing,
        'strict_sit_within_icu_plus_minus_24h': 'Culture-paired suspected-infection time within ICU admission +/-24h; '
                                               'strict SOFA membership is evaluated separately.',
        'documented_infection_in_presentation_window': 'Accepted source infection-diagnosis entry within '
                                                       'ICU admission +/-24h and documented care.',
        'main_documented_infection_plus_iv_proxy': infection['eicu'],
        'main_proxy_with_strict_culture_pair': 'Main proxy-infection stays also having a strict culture/prescription pair; '
                                               'this does not by itself establish strict Sepsis-3 membership.',
        'strict_culture_sepsis3': 'Main-cohort patients qualifying under separate culture-paired infection and '
                                  'SOFA adjudication. Released tensors remain aligned to the main proxy onset.',
        'released_observed_and_imputed_patients': 'Final clinical cohort retained in both tensor versions: '
                                                  '24 rows per patient, including structural and unfilled hours.',
        'all_cohort_patients_retained': 'Every final clinical-cohort patient has an atlas patient row, even if ineligible.',
        'reference_identity': 'Eligible MIMIC-IV patients; reference embeddings remain unchanged by design.',
        'selected_map_supported': 'Eligible target patients supported by the validation-selected map; '
                                    'only their latent representations are eligible for correction.',
        'eligible_target_unchanged': 'Eligible target patients with an identity map or insufficient selected-map support; '
                                      'their original latent representations are retained.'}
    eligibility = config['eligibility']
    cells, features = eligibility['minimum_patient_observed_cells'], eligibility['minimum_patient_observed_features']
    descriptions.update(
        eligible_representation=f'At least {cells} observed cells and {features} distinct observed channels across '
                                '24h among the shared encoder features listed in release_manifest.json.',
        insufficient_observed_cells=f'Fewer than {cells} observed cells among shared encoder features; clinical patient retained.',
        insufficient_observed_features=f'At least {cells} observed cells but fewer than {features} distinct shared '
                                         'observed channels; clinical patient retained.')
    for row in rows:
        dataset, step = row['dataset'], row['step']
        reason = step.removeprefix('after_excluding_')
        if reason == 'not_first_icu_stay':
            description = first_stay[dataset]
        else:
            description = reasons.get(reason)
        if description is not None:
            prefix = ('Retained after excluding stays that are ' if step.startswith('after_excluding_') else
                      'Excluded stays that are ') if reason == 'not_first_icu_stay' else (
                      'Retained after excluding stays with ' if step.startswith('after_excluding_') else 'Excluded stays with ')
            description = prefix + description + '.'
        elif step == 'base_cohort':
            description = 'Adult first-stay base cohort after source-specific eligibility filters; no minimum 24h ICU stay.'
            if dataset == 'eicu':
                description += ' Requires one recorded hospital encounter, unambiguous first-stay order and '
                description += 'ICU type, valid care offsets and mortality label; excludes elective surgery.'
        elif step == 'suspected_infection':
            description = infection[dataset]
            if dataset != 'eicu':
                description += ' Suspected-infection interval must overlap ICU admission +/-24h; uncertain candidates '
                description += 'remain here but do not automatically enter the final Sepsis-3 cohort.'
        elif step == 'operational_sepsis_cohort':
            low, high = reports[dataset][6]['policy']['association_hours']
            description = (f'Primary source-specific infection definition plus an in-ICU SOFA >=2 assessment within '
                           f'[{low}, +{high}] h of selected suspected infection, using a rolling 24h SOFA and assumed-zero '
                           'baseline. Missing components contribute zero only to the total; some observed evidence is required. '
                           'Onset is the later infection/qualifying-assessment time and must fit follow-up. ' + infection[dataset])
        else:
            require(step in descriptions, f'Missing release flow definition: {dataset}/{step}')
            description = descriptions[step]
        row['step_description'] = description
    return rows


def cohort_flow(reports, patients, atlas):
    """Keep nested selection, disjoint exclusions, and overlapping subgroups distinct."""
    rows = []
    def add(dataset, stage, kind, step, denominator, count, note='', denominator_population=None):
        require(0 <= count <= denominator, f'Invalid cohort flow: {dataset}/{step}')
        meanings = {'selection': 'retained_after_filter', 'exclusion_reason': 'excluded_for_reason',
                    'retention': 'retained_in_release', 'representation_eligibility': 'eligible_for_representation',
                    'eligibility_reason': 'ineligible_but_retained', 'reference': 'reference_unchanged',
                    'transport_support': 'eligible_target_subgroup',
                    'parallel_evidence': 'evidence_branch_members', 'sensitivity_or_subgroup': 'subgroup_members'}
        if denominator_population is None:
            if stage == '01':
                denominator_population = 'all source stays before base-cohort filters'
            elif stage == '02':
                denominator_population = ('stage-02 suspected-infection stays' if kind == 'sensitivity_or_subgroup'
                                          else 'stage-01 retained base-cohort stays')
            elif stage == '06' and kind != 'sensitivity_or_subgroup':
                denominator_population = 'stage-02 retained suspected-infection stays'
            elif stage == 'atlas' and kind in {'reference', 'transport_support'}:
                denominator_population = 'representation-eligible patients from this dataset'
            else:
                denominator_population = 'final clinical cohort from this dataset; all retained in release'
        rows.append(dict(dataset=dataset, stage=stage, kind=kind, step=step,
                         denominator_population=denominator_population,
                         denominator=int(denominator), count=int(count),
                         count_meaning=meanings[kind],
                         excluded=int(denominator - count) if kind == 'selection' else None, note=note))
    for name in DATASETS:
        base, infection, sepsis = (reports[name][i] for i in (1, 2, 6))
        if 'attrition' in base:
            previous = base['attrition'][0]['remaining']
            parent = 'all source stays before base-cohort filters'
            for item in base['attrition']:
                require(previous - item['remaining'] == item['removed'], f'{name}: base attrition mismatch')
                step = item['step'] if item['step'] == 'all_icu_stays' else 'after_excluding_' + item['step']
                note = ('CareVue-only source ICU stays' if name == 'mimic3-carevue' else 'MIMIC-IV source ICU stays')
                add(name, '01', 'selection', step, previous, item['remaining'],
                    note + '; count is retained, excluded is removed at this filter', denominator_population=parent)
                parent = 'stage-01 stays retained at ' + step
                previous = item['remaining']
            require(previous == base['base_cohort_stays'], f'{name}: base total mismatch')
        else:
            summary = base['summary']
            require(sum(summary['attrition'].values()) == summary['source_stays'], 'eICU base exclusions do not sum')
            previous = summary['base_stays']
            require(summary['attrition']['included'] == previous, 'eICU included base mismatch')
            add(name, '01', 'selection', 'base_cohort', summary['source_stays'], previous)
            for reason, count in summary['attrition'].items():
                if reason != 'included':
                    add(name, '01', 'exclusion_reason', reason, summary['source_stays'], count,
                        'Mutually exclusive recorded first-failure reasons, not a sequential funnel')
        if 'attrition' in infection:
            parent = 'stage-01 retained base-cohort stays'
            for item in infection['attrition']:
                require(previous - item['remaining'] == item['removed'], f'{name}: infection attrition mismatch')
                add(name, '02', 'selection', item['step'], previous, item['remaining'], denominator_population=parent)
                parent = 'stage-02 stays retained at ' + item['step']
                previous = item['remaining']
        else:
            require(infection['summary']['base'] == previous, f'{name}: infection denominator mismatch')
            included = infection['summary']['infection_stays' if name == 'mimic3-carevue' else 'infection']
            add(name, '02', 'selection', 'suspected_infection', previous, included,
                'Documented-infection plus IV-order proxy' if name == 'eicu' else 'Includes uncertain-timing candidates')
            if name == 'eicu':
                for step, count in infection['stay_funnel'].items():
                    add(name, '02', 'parallel_evidence', step, previous, count, infection['funnel_note'])
            elif 'definite_infection_stays' in infection['summary']:
                add(name, '02', 'sensitivity_or_subgroup', 'definite_infection_timing', included,
                    infection['summary']['definite_infection_stays'],
                    'At least one stage-02 candidate with definite pairing/presentation timing; '
                    'not a sequential filter or the later SOFA/onset timing adjudication')
            previous = included
        require(sepsis['summary']['suspected_infection_stays'] == previous, f'{name}: sepsis denominator mismatch')
        attrition = sepsis['stay_attrition']
        require(sum(attrition.values()) == previous, f'{name}: sepsis exclusion reasons do not sum')
        final = sepsis['summary']['sepsis3_stays']
        require(attrition['included'] == final == len(patients[name]), f'{name}: released cohort count mismatch')
        add(name, '06', 'selection', 'operational_sepsis_cohort', previous, final)
        for reason, count in attrition.items():
            if reason != 'included':
                add(name, '06', 'exclusion_reason', reason, previous, count,
                    'Mutually exclusive stay decisions; counts sum to stage-06 exclusions')
        if name == 'eicu':
            add(name, '06', 'sensitivity_or_subgroup', 'strict_culture_sepsis3', final,
                sepsis['summary']['strict_culture_sepsis3_subgroup'], 'Main tensors retain proxy-onset alignment')
        add(name, '07-08', 'selection', 'released_observed_and_imputed_patients', final, final,
            'All patients and all 24 hourly rows retained, including structural hours')
        subset = [row for row in atlas if row['dataset'] == name]
        require(len(subset) == final, f'{name}: atlas retention mismatch')
        add(name, 'atlas', 'retention', 'all_cohort_patients_retained', final, final,
            'Representation ineligibility never removes a patient from the released atlas table')
        eligible = sum(row['eligible'] for row in subset)
        add(name, 'atlas', 'representation_eligibility', 'eligible_representation', final, eligible,
            'Representation eligibility is assessed before OT; ineligible patients retain clinical data '
            'with blank released embeddings/coordinates/clusters, not failed transport matches')
        require(all(row['eligible'] == (row['eligibility_reason'] == 'eligible') for row in subset),
                f'{name}: inconsistent eligibility reasons')
        require(all(row['eligibility_reason'] in {'eligible', 'insufficient_observed_cells',
                    'insufficient_observed_features'} for row in subset), f'{name}: unknown eligibility reason')
        for reason in ('insufficient_observed_cells', 'insufficient_observed_features'):
            add(name, 'atlas', 'eligibility_reason', reason, final,
                sum(row['eligibility_reason'] == reason for row in subset),
                'Disjoint: insufficient cells first, then insufficient features among patients with enough cells')
        supported = sum(row['transport_supported'] for row in subset)
        require(not any(row['transport_supported'] and not row['eligible'] for row in subset),
                f'{name}: ineligible patient marked supported')
        if name == 'mimiciv':
            require(supported == 0, 'Reference patients must not be transported')
            add(name, 'atlas', 'reference', 'reference_identity', eligible, eligible,
                'All eligible reference embeddings unchanged by design; target support is not applicable')
        else:
            add(name, 'atlas', 'transport_support', 'selected_map_supported', eligible, supported,
                'Eligible target patients supported by the locked map; not an exclusion count')
            add(name, 'atlas', 'transport_support', 'eligible_target_unchanged', eligible, eligible - supported,
                'Retain original embedding because selected map is identity or patient lacks its support')
    return describe_flow(rows, reports, read_json(ROOT / 'src/04_atlas_datasets/atlas_config.json'))


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fields})


RELEASE_FILES = ('mimiciv.zip', 'mimic3-carevue.zip', 'eicu.zip', 'atlas.zip',
                 'data_dictionary.csv', 'cohort_flow.csv', 'release_manifest.json')


def publish_release(content, outdir):
    """Replace only managed release files after validation; restore them on publication errors."""
    require(set(p.name for p in content.iterdir()) == set(RELEASE_FILES), 'Incomplete release package')
    outdir.mkdir(parents=True, exist_ok=True)
    for name in RELEASE_FILES:
        destination = outdir/name
        require(not destination.is_symlink() and (not destination.exists() or destination.is_file()),
                f'Release member is not a regular file: {destination}')
    backup = outdir/('.previous-release-'+uuid.uuid4().hex)
    backup.mkdir()
    previous, published = [], []
    try:
        # Remove the old completion marker first, and publish the new one last.
        for name in ('release_manifest.json', *RELEASE_FILES[:-1]):
            if (outdir/name).exists():
                os.replace(outdir/name, backup/name)
                previous.append(name)
        for name in RELEASE_FILES:
            os.replace(content/name, outdir/name)
            published.append(name)
    except BaseException:
        try:
            for name in published:
                (outdir/name).unlink()
            for name in reversed(previous):
                os.replace(backup/name, outdir/name)
            backup.rmdir()
        except OSError as recovery_error:
            raise RuntimeError(f'Release recovery incomplete; original files retained at {backup}') from recovery_error
        raise
    for name in previous:
        (backup/name).unlink()
    backup.rmdir()


def validate_destination(outdir):
    require(not ROOT.is_relative_to(outdir) and (not outdir.exists() or outdir.is_dir()),
            'Choose a release directory, not the project root, a parent directory or a file')
    require(not any(outdir == ROOT / part or (ROOT / part) in outdir.parents
                    for part in ('src', 'data', 'outputs', 'pipeline', 'docker', '.git')),
            'Choose a separate release destination')


def build_release(outdir):
    validate_destination(outdir)
    print('[EXPORT] Verifying completed dataset and atlas provenance', flush=True)
    manifests, reports, stages, qc, inputs = validate_inputs()
    registry = read_json(ROOT / 'src/common/feature_registry.json')
    schema = read_json(ROOT / 'src/common/tensor_schema.json')
    rules = read_json(ROOT / 'src/common/measurement_rules.json')
    config = read_json(ROOT / 'src/04_atlas_datasets/atlas_config.json')
    require(registry['schema_version'] == schema['schema_version'] == config['version'] == VERSION,
            'Release schema version mismatch')
    with np.load(ROOT / 'data/processed/atlas/inputs.npz', allow_pickle=False) as archive:
        atlas_inputs = {key: archive[key] for key in archive.files}
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.release-', dir=outdir.parent) as temporary:
        staging = Path(temporary).resolve()
        require(staging.parent == outdir.parent, 'Unexpected staging path')
        content = staging / 'package'
        content.mkdir()
        dictionary, members, patients, all_reasons = [], {}, {}, []
        for domain, name in enumerate(DATASETS):
            archive = Archive(content, name, dictionary, members)
            try:
                patients[name], reasons = export_source(
                    name, archive, registry, schema, rules['bounds'], atlas_inputs, domain, config)
                all_reasons.extend(reasons.tolist())
            finally:
                archive.close()
        archive = Archive(content, 'atlas', dictionary, members)
        try:
            atlas = export_atlas(archive, atlas_inputs, [r for name in DATASETS for r in patients[name]],
                                 all_reasons, stages[4]['selected'], schema, config)
        finally:
            archive.close()
        flow = cohort_flow(reports, patients, atlas)
        flow_docs = FLOW_DOC
        for i, (key, description) in enumerate(flow_docs.items()):
            numeric = key in ('denominator', 'count', 'excluded')
            entry = field(key, 'integer' if numeric else 'string', description,
                          unit='source stays' if numeric else '')
            entry['missing_value'] = 'empty for non-selection rows; not an unrecorded exclusion' if key == 'excluded' else (
                'empty=no additional qualification' if key == 'note' else 'not permitted')
            dictionary.append(dict(archive='', file='cohort_flow.csv', position=i,
                                   **table_structure('cohort_flow.csv'), **entry))
        dictionary = complete_dictionary(dictionary)
        write_csv(content / 'data_dictionary.csv', DICTIONARY_COLUMNS, dictionary)
        write_csv(content / 'cohort_flow.csv',
                  list(flow_docs), flow)
        for path in content.glob('*.zip'):
            with zipfile.ZipFile(path) as zipped:
                require(zipped.testzip() is None, f'Corrupt ZIP: {path.name}')
        for name, expected in inputs.items():
            require(file_record(project_path(name)) == expected, f'Input changed during export: {name}')
        manifest = dict(
            dataset_version=VERSION, generated_at_utc=datetime.now(timezone.utc).isoformat(), status='COMPLETE',
            encoding='UTF-8 CSV; gzip members inside ZIP_STORED containers',
            numeric_serialization='Round-trip Python float text; empty=missing, zero remains zero; booleans=0/1',
            identifiers='Join on patient_key; hourly tables additionally use hour. tensor_row is zero-based.',
            dictionary_conventions=dict(
                file_description='Purpose and relationships of each file; data_dictionary.csv also defines its own columns.',
                position='Zero-based CSV column position within file.',
                key_columns='Semicolon-separated columns that jointly identify a row within file.',
                row_unit='What one table row represents; patient rows are retained even without an atlas embedding.',
                allowed_values='JSON arrays enumerate source categories observed in this release; other entries '
                               'describe permitted codes or screening domains.',
                screening_bounds='Inclusive project screening limits, not clinical reference ranges; empty bounds '
                                 'are not zero. A missing finite limit is stated in allowed_values.',
                aggregation='Construction of observed hourly values; imputed gaps instead follow hourly_support method codes.',
                predictor_default='Static-feature flag: 1=candidate admission context; 0=exclude by default; '
                                  'empty=not assigned, not approval for prediction use.'),
            tensor_shape='For each source: patients x 24 hours x 50 registry-ordered clinical features',
            feature_order=[f['name'] for f in registry['temporal_features']],
            time_contract=schema['policy']['window'], time_semantics=schema['policy']['time_semantics'],
            missingness=schema['missingness'], method_codes=METHODS, static_roles=schema['policy']['static_features'],
            screening_limits_interpretation=rules['bounds_policy'],
            atlas=dict(patients=len(atlas), eligible=sum(r['eligible'] for r in atlas),
                       ineligible_retained=sum(not r['eligible'] for r in atlas),
                       selected=stages[4]['selected'], mapping_policy=stages[3]['mapping_policy'],
                       eligibility=config['eligibility'], encoder_features=stages[1]['common_features'],
                       test_exposure=stages[4]['test_exposure'],
                       interpretation='Adapted latent representations, not OT-modified clinical tensors; '
                                      'blank embeddings for ineligible patients; original embeddings for unsupported targets.'),
            source_versions={name: m['source_version'] for name, m in manifests.items()},
            cohort_definitions={name: reports[name][6]['policy']['definition'] for name in DATASETS},
            cohort_definition_codes=COHORT_DEFINITIONS,
            source_limitations={name: m.get('source_specific_limits', reports[name][6].get('limitations', []))
                                for name, m in manifests.items()},
            protocol=config['protocol'],
            quality=dict(atlas_status=qc['status'], atlas_checks=qc['required_checks'],
                         source_qc={name: {k: read_json(ROOT / 'outputs' / name / 'qc.json')[k]
                                           for k in ('status', 'required_checks', 'failed_required_checks')}
                                    for name in DATASETS}),
            files={p.name: file_record(p) for p in sorted(content.iterdir())}, members=members,
            input_provenance=inputs)
        (content / 'release_manifest.json').write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        publish_release(content, outdir)
    print(f'[RELEASE] Complete v{VERSION}: {outdir}; 4 archives, dictionary, cohort flow and manifest', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'For a raw-data build, run bash pipeline/01_build_dataset.sh. This exporter requires completed '
        'source stages 01-09 and atlas stages 01-06; it does not train models or create figures. '
        'Relative destinations are resolved from the project root. Only the four ZIPs, data dictionary, '
        'cohort flow and release manifest are replaced, after validation; other destination files are retained.'))
    parser.add_argument('--outdir', default='release', help='Destination; existing managed release files are replaced after validation')
    parser.add_argument('--check-only', action='store_true', help='Verify export inputs and destination without writing archives')
    args = parser.parse_args()
    path = Path(args.outdir).expanduser()
    outdir = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        validate_destination(outdir)
        if args.check_only:
            validate_inputs()
            print('[EXPORT CHECK PASS] Certified datasets and atlas are ready; no files written.', flush=True)
            return
        build_release(outdir)
    except (ValueError, OSError, KeyError, ModuleNotFoundError) as error:
        print(f'[EXPORT FAILED] {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
