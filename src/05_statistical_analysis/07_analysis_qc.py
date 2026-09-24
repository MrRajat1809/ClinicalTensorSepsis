"""Validate statistical export integrity; scientific benefit is never a QC requirement."""
import argparse
import csv
import math
from pathlib import Path
import analysis_common as A

STATUSES={'estimated','descriptive','insufficient_precision','not_estimable'}


def validate_rows(path,expected,checks,config=None):
    with path.open('r',encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        A.require(reader.fieldnames==expected['columns'],f'{path.name}: headers changed')
        ids=set(); count=0
        test_rows=[]
        for row in reader:
            count+=1
            A.require(row['result_id'] not in ids,f'{path.name}: duplicated result ID')
            ids.add(row['result_id'])
            A.require(row['status'] in STATUSES and row['metric'],f'{path.name}: missing status or metric')
            for key in ('estimate','ci_low','ci_high','n_patients','n_observations','n_reference','n_hospitals','bootstrap_valid','bootstrap_requested'):
                if row[key]:
                    value=float(row[key])
                    A.require(math.isfinite(value),f'{path.name}: nonfinite serialized {key}')
                    if key.startswith(('n_','bootstrap_')):
                        A.require(value>=0 and value.is_integer(),f'{path.name}: invalid count {key}')
            A.require(bool(row['ci_low'])==bool(row['ci_high']),f'{path.name}: one-sided missing interval')
            if row['ci_low']:
                A.require(float(row['ci_low'])<=float(row['ci_high']),f'{path.name}: inverted interval')
            if row['bootstrap_valid'] and row['bootstrap_requested']:
                A.require(int(row['bootstrap_valid'])<=int(row['bootstrap_requested']),f'{path.name}: impossible replicate count')
                if config is not None:
                    key='bootstrap_replicates'
                    A.require(int(row['bootstrap_requested'])==config[key],f'{path.name}: bootstrap count differs from configuration')
            if row['status']=='estimated':
                A.require(row['estimate'] and row['ci_low'] and row['ci_high'],f'{path.name}: estimated result lacks value/interval')
            if row.get('test_name'):
                A.require(row.get('test_family') and row.get('test_null'), 'Test lacks a declared family/null')
                tested=row.get('test_status')=='tested'
                A.require(tested==bool(row.get('p_value')), 'Test status and p-value disagree')
                if tested:
                    p,q=float(row['p_value']),float(row['q_value_bh'])
                    A.require(0<=p<=q<=1, 'Invalid p-value or adjusted q-value')
                    A.require(int(row['family_test_count'])>0, 'Missing multiplicity denominator')
                    test_rows.append(dict(row,p_value=p))
            if row['question']=='feature_readiness':
                flags=set(filter(None,row['review_flags'].split(';')))
                A.require(flags <= {'no_observed_values','constant_observed_values','sparse_patient_coverage',
                                    'mostly_generated_cells','absent_at_some_hospitals','heldout_imputation_regret'}, 'Unknown feature-review flag')
                A.require(row['review_status']==('review_required' if flags else 'no_configured_flag'),
                          'Feature review status does not match flags')
                A.require(int(row['observed_cells'])+int(row['generated_cells'])==int(row['filled_cells']),
                          'Feature provenance counts do not reconcile')
                A.require(int(row['hospitals_without_observations'])<=int(row['n_hospitals']),
                          'Impossible hospital availability count')
            if row['question']=='cohort_flow' and row['kind']=='selection':
                A.require(int(row['denominator'])-int(row['count'])==int(row['excluded']),'Flow arithmetic changed')
        A.require(count==expected['rows'] and count>0,f'{path.name}: row count changed')
        replay=[dict(row) for row in test_rows]
        A.adjust_pvalues(replay)
        for saved,actual in zip(test_rows,replay):
            A.require(math.isclose(float(saved['q_value_bh']),actual['q_value_bh'],abs_tol=1e-12)
                      and int(saved['family_test_count'])==actual['family_test_count'], 'BH correction does not replay')
    checks.append(dict(check='table_schema_and_estimates',file=path.name,rows=count,passed=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=A.ROOT/'outputs/statistical_analysis')
    parser.add_argument('--config',type=Path,default=A.HERE/'analysis_config.json')
    args=parser.parse_args()
    out=A.output_directory(args.outdir)
    checks=[]
    try:
        manifest=A.read_json(out/'analysis_manifest.json')
        A.require(manifest['version']=='1.0.0','Wrong analysis version')
        A.require(manifest['config_record']==A.record(args.config),'Analysis configuration changed')
        A.require(set(manifest['stages'])=={str(i) for i in range(1,7)},'Run all six analyses first')
        all_inputs={}
        for stage,names in A.TABLES.items():
            entry=manifest['stages'][str(stage)]
            A.require(entry['status']=='COMPLETE',f'Incomplete analysis {stage}')
            current={p.relative_to(A.ROOT).as_posix():A.record(p) for p in [A.HERE/'analysis_common.py',*A.HERE.glob(f'{stage:02d}_*.py')]}
            A.require(entry['code']==current,f'Analysis {stage}: code changed')
            A.require(set(entry['outputs'])=={n+'.csv' for n in names},f'Analysis {stage}: missing outputs')
            for name,record in entry['inputs'].items():
                A.require(name not in all_inputs or all_inputs[name]==record,f'Analyses used different input versions: {name}')
                all_inputs[name]=record
            for name,record in entry['outputs'].items():
                A.require(A.record(out/name)=={k:record[k] for k in ('sha256','size_bytes')},f'Changed table: {name}')
                validate_rows(out/name,record,checks,manifest['config'])
            for item in entry['audit']:
                if 'overlap' in item:
                    A.require(item['overlap']==0,'Training/evaluation patient overlap')
                if 'passed' in item:
                    A.require(item['passed'],'Failed numerical replay')
                checks.append(dict(item,passed=True))
        for name,record in all_inputs.items():
            path=(A.ROOT/name).resolve()
            A.require(path.is_relative_to(A.ROOT) and A.record(path)==record,f'Changed analysis input: {name}')
        checks.append(dict(check='all_input_hashes',files=len(all_inputs),passed=True))
        # Required replay coverage prevents a partial analysis being presented as complete.
        replays=manifest['stages']['3']['audit']
        actual={(r['dataset'],r['scenario']) for r in replays if r['check']=='frozen_holdout_replay'}
        wanted={(d,s) for d in manifest['config']['datasets'] for s in manifest['config']['imputation']['scenarios']}
        A.require(actual==wanted,'Missing imputation replay scenarios')
        for stage,check,expected in [(4,'selected_transport_replay',set(manifest['config']['datasets'][1:])),
                                     (5,'dataset_integrity_replay',set(manifest['config']['datasets'])),
                                     (6,'site_and_split_patient_units',set(manifest['config']['datasets']))]:
            actual={r['dataset'] for r in manifest['stages'][str(stage)]['audit'] if r['check']==check}
            A.require(actual==expected,f'Missing {check}')
        # Historical outcome benchmarks are not part of the dataset-validation deliverables.
        for name in ('mortality_comparisons.csv.gz', 'calibration_curves.csv.gz',
                     'mortality_comparisons.csv', 'calibration_curves.csv',
                     'atlas_stability.csv.gz', 'atlas_clinical_profiles.csv.gz', 'atlas_stability.csv', 'atlas_clinical_profiles.csv',
                     *(name+'.csv.gz' for names in A.TABLES.values() for name in names)):
            (out/name).unlink(missing_ok=True)
        with (out/'feature_readiness.csv').open('r',encoding='utf-8',newline='') as stream:
            feature_rows=[r for r in csv.DictReader(stream) if r['metric']=='patients_with_any_observation']
        registry=A.read_json(A.ROOT/'src/common/feature_registry.json')
        expected={(d,f['name']) for d in manifest['config']['datasets'] for f in registry['temporal_features']}
        A.require(len(feature_rows)==len(expected) and {(r['dataset'],r['feature']) for r in feature_rows}==expected,
                  'Incomplete feature-readiness inventory')
        review={d:[r['feature'] for r in feature_rows if r['dataset']==d and r['review_status']=='review_required']
                for d in manifest['config']['datasets']}
        manifest['status']='VERIFIED'
        A.write_json(out/'analysis_manifest.json',manifest)
        report=dict(version='1.0.0',status='PASS',checks=checks,failures=0,
            features_requiring_review=review,
            analysis_manifest=A.record(out/'analysis_manifest.json'),qc_code=A.record(Path(__file__)),
            interpretation='Analysis provenance, joins, replay and export integrity verified; feature-review flags remain explicit; not a declaration that every feature is suitable for every model.')
    except (ValueError,OSError,KeyError) as error:
        A.write_json(out/'qc.json',dict(version='1.0.0',status='FAIL',checks=checks,failures=1,error=str(error)))
        raise
    A.write_json(out/'qc.json',report)
    print(f'[ANALYSIS QC PASS] {len(checks)} checks; {len(A.TABLES)} analyses; {out}',flush=True)
    print('    Feature review counts: '+str({name:len(values) for name,values in review.items()}),flush=True)


if __name__=='__main__':
    main()
