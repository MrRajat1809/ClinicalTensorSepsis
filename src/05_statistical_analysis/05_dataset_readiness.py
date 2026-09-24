"""Audit tensor integrity, feature reliability and cohort-definition sensitivity."""
from collections import Counter

import numpy as np
import analysis_common as A


def feature_integrity(raw, filled, codes, exposed):
    observed = np.isfinite(raw)
    available = np.isfinite(filled)
    return {
        'no_infinite_values': not np.isinf(raw).any() and not np.isinf(filled).any(),
        'observations_unchanged': np.array_equal(raw[observed], filled[observed]),
        'structural_cells_unavailable': not available[~exposed].any() and not observed[~exposed].any(),
        'observed_codes_match': np.array_equal(codes == 1, observed),
        'filled_codes_match': np.isin(codes, [0, 1, 2, 3, 4]).all() and np.array_equal(codes > 0, available),
    }


def feature_flags(values, patient_fraction, generated_fraction, zero_hospitals, config):
    flags = []
    if not len(values):
        flags.append('no_observed_values')
    elif values.min() == values.max():
        flags.append('constant_observed_values')
    if patient_fraction < config['minimum_patient_coverage']:
        flags.append('sparse_patient_coverage')
    if generated_fraction > config['generated_fraction_warning']:
        flags.append('mostly_generated_cells')
    if zero_hospitals:
        flags.append('absent_at_some_hospitals')
    return flags


def review_features(run, name, source, filled):
    rows, checks, flagged = [], [], 0
    exposed = source['exposure_seconds'] > 0
    policy = run.config['readiness']
    hospitals = np.array([str(r.get('hospital_id', '')) for r in source['patients']])
    sites = [site for site, count in Counter(hospitals).items()
             if name == 'eicu' and count >= policy['minimum_hospital_patients']]
    for j, feature in enumerate(source['features'].tolist()):
        raw, values, codes = source['raw'][:, :, j], filled[:, :, j], source['method_codes'][:, :, j]
        integrity = feature_integrity(raw, values, codes, exposed)
        for check, passed in integrity.items():
            A.require(passed, f'{name}/{feature}: {check}')
        checks.append(integrity)
        seen = np.isfinite(raw)
        measured = raw[seen]
        any_seen = seen.any(axis=1)
        generated = np.isin(codes, [2, 3, 4])
        n_filled = int(np.isfinite(values).sum())
        generated_fraction = generated.sum() / n_filled if n_filled else np.nan
        absent = [site for site in sites if not seen[hospitals == site].any()]
        flags = feature_flags(measured, any_seen.mean(), generated_fraction, len(absent), policy)
        flagged += bool(flags)
        base = dict(dataset=name, feature=feature, question='feature_readiness',
                    review_status='review_required' if flags else 'no_configured_flag',
                    review_flags=';'.join(flags), observed_cells=int(seen.sum()),
                    generated_cells=int(generated.sum()), filled_cells=n_filled,
                    observed_min=float(measured.min()) if len(measured) else None,
                    observed_max=float(measured.max()) if len(measured) else None,
                    n_hospitals=len(sites), hospitals_without_observations=len(absent),
                    hospital_ids_without_observations=';'.join(sorted(absent)),
                    positive_observed_cells=int((measured > 0).sum()),
                    note='Flags guide later feature selection, not automatic deletion. Unobserved is not normal or untreated. '
                         'Hospital counts include sites meeting the configured cohort-size threshold only.')
        if feature == 'vent':
            base['note'] += (' Positive invasive ventilation requires source airway/support evidence. '
                             'An all-zero observed channel is not evidence that no patient was ventilated.')
        rows.append(dict(**base, metric='patients_with_any_observation', unit='fraction',
                         **A.proportion(int(any_seen.sum()), len(raw), run.config)))
        rows.append(dict(**base, metric='generated_fraction_of_filled_cells', unit='fraction',
                         **A.descriptive(generated_fraction, len(raw))))
        denominator = exposed.sum(axis=1)
        per_patient = np.divide(generated.sum(axis=1), denominator,
                                out=np.full(len(raw), np.nan), where=denominator > 0)
        rows.append(dict(**base, metric='mean_patient_generated_hour_fraction', unit='fraction',
                         **A.bootstrap_mean(per_patient, run.config, (name, feature, 'generated'))))
    run.audit.append(dict(check='dataset_integrity_replay', dataset=name, features=len(checks),
                          checks_per_feature=len(checks[0]), passed=True))
    return rows, flagged


def cohort_sensitivity(run, name, source):
    report = run.json(A.ROOT/'outputs'/name/'06_sepsis.json')
    counts = {True: 0, False: 0, None: 0}
    for item in report['baseline_sensitivity_comparison']:
        if item['primary']:
            counts[item['measured_baseline_sensitivity']] += item['stays']
    A.require(sum(counts.values()) == len(source['stay_ids']), f'{name}: baseline sensitivity denominator')
    rows = []
    for value, label in [(True, 'also_qualifies'), (False, 'does_not_qualify'), (None, 'not_assessable')]:
        rows.append(dict(dataset=name, question='baseline_sofa_sensitivity', stratum=label,
                         metric='primary_cohort_fraction', unit='fraction',
                         **A.proportion(counts[value], len(source['stay_ids']), run.config),
                         note='Measured pre-infection baseline alternative to assumed zero; this is definition sensitivity, not diagnostic accuracy.'))
    return rows


def main():
    run = A.Run(5)
    features, datasets = [], []
    comparisons = run.csv(run.out/'imputation_comparisons.csv')
    for name in run.config['datasets']:
        source = run.source(name)
        filled = np.load(run.track(A.ROOT/'data/processed'/name/'tensor_imputed.npy'), mmap_mode='r', allow_pickle=False)
        A.require(filled.shape == source['raw'].shape == source['method_codes'].shape, f'{name}: tensor shape mismatch')
        rows, flagged = review_features(run, name, source, filled)
        regret = {}
        for item in comparisons:
            if (item['dataset']==name and item['stratum']=='all'
                    and item['metric']=='paired_patient_mae_difference'
                    and item['ci_low'] and float(item['ci_low'])>0):
                regret.setdefault(item['feature'], []).append(item['scenario']+'/'+item['comparator'])
        for row in rows:
            row['imputation_regret_comparisons']=';'.join(regret.get(row['feature'], []))
            if row['feature'] in regret:
                row['review_flags']=';'.join(filter(None,[row['review_flags'],'heldout_imputation_regret']))
                row['review_status']='review_required'
                row['note']+=' Exploratory primary holdout MAE interval favors a baseline; retain the locked recipe and use this evidence for later feature selection.'
        flagged=len({row['feature'] for row in rows if row['review_status']=='review_required'})
        features.extend(rows)
        datasets.extend(cohort_sensitivity(run, name, source))
        datasets.append(dict(dataset=name, question='dataset_readiness', metric='features_with_review_flags',
                             **A.descriptive(flagged, len(source['stay_ids'])), denominator=len(source['features']),
                             note='Tensor integrity passed. Review flags do not make a cohort ineligible; retain the schema and select features for the later modelling task.'))
        print(f'[05] {name}: tensor integrity verified; {flagged}/{len(source["features"])} features flagged for review', flush=True)
    run.write('feature_readiness', features)
    run.write('dataset_readiness', datasets)
    run.finish()


if __name__ == '__main__':
    main()
