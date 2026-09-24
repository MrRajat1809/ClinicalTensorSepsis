"""Quantify within-source partition balance and eICU hospital heterogeneity."""
import numpy as np
from scipy import stats
import analysis_common as A


def welch_anova(groups):
    """Welch one-way ANOVA, including SciPy versions predating equal_var=False."""
    sizes = np.array([len(x) for x in groups], float)
    means = np.array([np.mean(x) for x in groups])
    variances = np.array([np.var(x, ddof=1) for x in groups])
    k = len(groups)
    if k < 2 or (sizes < 2).any() or (variances <= 0).any():
        return np.nan, np.nan, np.nan
    weights = sizes / variances
    center = np.sum(weights * means) / weights.sum()
    term = np.sum((1-weights/weights.sum())**2/(sizes-1))
    statistic = (np.sum(weights*(means-center)**2)/(k-1)) / (1+2*(k-2)*term/(k*k-1))
    df2 = (k*k-1)/(3*term)
    return statistic, stats.f.sf(statistic, k-1, df2), df2


def summaries(values, groups, base):
    rows = []
    for name, mask in groups.items():
        x = values[mask]
        finite = x[np.isfinite(x)]
        quantiles = np.quantile(finite, [.1,.25,.5,.75,.9]) if len(finite) else [np.nan]*5
        rows.append(dict(**base, stratum=name, metric='patient_summary_mean',
            **A.descriptive(A.finite_mean(finite), len(finite)), group_patients=int(mask.sum()),
            missing_patients=int((~np.isfinite(x)).sum()),
            standard_deviation=np.std(finite, ddof=1) if len(finite)>1 else np.nan,
            **dict(zip(('p10','p25','median','p75','p90'), quantiles))))
    return rows


def hospital_tests(values, sites, config):
    counts = [(site, values[mask][np.isfinite(values[mask])]) for site,mask in sites.items()]
    groups = [x for _,x in counts if len(x) >= config['minimum_patients']]
    k, n = len(groups), sum(map(len, groups))
    base = dict(n_hospitals=k, omitted_hospitals=len(sites)-k,
                group_names=';'.join(site for site,x in counts if len(x)>=config['minimum_patients']),
                note='Observed patient means; unadjusted site heterogeneity includes case mix and recording differences. '
                     'It does not identify a technical batch effect. Small sites are reported but excluded from inference.')
    varying = k>=2 and np.ptp(np.concatenate(groups))>0
    eta = np.nan
    if varying:
        center = np.mean(np.concatenate(groups))
        total = sum(np.sum((x-center)**2) for x in groups)
        eta = sum(len(x)*(x.mean()-center)**2 for x in groups)/total
    f,p,df = welch_anova(groups) if k>=2 else (np.nan,)*3
    rows = [dict(**base, metric='between_hospital_eta_squared', **A.descriptive(eta),
                 **A.test_fields('welch_anova','Equal hospital means',p,f,np.isfinite(p)),
                 numerator_df=k-1 if k>=2 else None, denominator_df=df)]
    h,p = stats.kruskal(*groups) if varying else (np.nan,np.nan)
    epsilon = max(0,(h-k+1)/(n-k)) if varying and n>k else np.nan
    rows.append(dict(**base, metric='rank_epsilon_squared', **A.descriptive(epsilon),
                     **A.test_fields('kruskal_wallis','Identical hospital distributions',p,h,np.isfinite(p))))
    deviations = [np.abs(x-np.median(x)) for x in groups]
    # Constant within-site deviations make a variance test unidentifiable.
    if varying and any(np.var(x)>0 for x in deviations):
        f,p = stats.levene(*groups, center='median')
    else:
        f,p = np.nan,np.nan
    sd = np.array([np.std(x,ddof=1) for x in groups])
    rows.append(dict(**base, metric='hospital_sd_spread',
        **A.descriptive(sd.max()-sd.min() if len(sd) else np.nan),
        **A.test_fields('brown_forsythe','Equal hospital variances',p,f,np.isfinite(p))))
    for row in rows:
        row['n_patients']=n
    return rows


def availability_test(observed, sites):
    table = np.array([[int(observed[mask].sum()), int((~observed[mask]).sum())] for mask in sites.values()])
    statistic,p,effect = np.nan,np.nan,np.nan
    expected_min = np.nan
    valid = False
    if len(table)>=2 and (table.sum(axis=0)>0).all():
        statistic,p,_,expected = stats.chi2_contingency(table, correction=False)
        expected_min = expected.min()
        effect = np.sqrt(statistic/table.sum())
        valid = expected_min>=5
    return dict(metric='availability_cramers_v', **A.descriptive(effect, int(table.sum())),
                n_hospitals=len(table), minimum_expected_count=expected_min,
                **A.test_fields('chi_square','Observation availability is independent of hospital',
                                p if valid else np.nan,statistic,valid),
                note='One available/unavailable indicator per patient. Asymptotic p-value withheld when any expected count <5. '
                     'Cramers V is descriptive; association does not separate case mix from interfaces.')


def main():
    run = A.Run(6)
    tests, profiles = [], []
    for name in run.config['datasets']:
        source = run.source(name)
        values = dict(age=source['static'][:,source['static_features'].tolist().index('age')],
                      followup_hours=source['followup'], observed_fraction=source['coverage'])
        units = dict(zip(source['features'].tolist(), source['units'].tolist()))
        units.update(age='years',followup_hours='h',observed_fraction='fraction')
        for feature in run.config['clinical_features']:
            j = source['features'].tolist().index(feature)
            values[feature] = A.finite_mean(source['raw'][:,:,j], axis=1)
        groups = {label:source['partition']==i for i,label in enumerate(('train','validation','test'))}
        for feature,x in values.items():
            base = dict(dataset=name, feature=feature, question='partition_balance', unit=units[feature])
            profiles.extend(summaries(x, groups, base))
            for label in ('validation','test'):
                for result in A.two_sample_tests(x[groups[label]], x[groups['train']], run.config):
                    row = dict(base, population=label+'_minus_train', comparator='train', **result,
                               note='Independent patients within one source; no test-driven repartitioning. Clinical values are observed patient means.')
                    if result['test_name']=='mann_whitney_u': row['unit']='rank_biserial'
                    tests.append(row)
        if name=='eicu':
            hospital = np.array([str(r['hospital_id']) for r in source['patients']])
            sites = {'hospital='+h:hospital==h for h in sorted(set(hospital))
                     if (hospital==h).sum()>=run.config['tests']['minimum_hospital_patients']}
            for feature,x in values.items():
                base = dict(dataset=name, feature=feature, question='hospital_heterogeneity', unit=units[feature])
                profiles.extend(summaries(x, sites, base))
                for result in hospital_tests(x, sites, run.config):
                    row=dict(base, **result)
                    if result['metric']!='hospital_sd_spread': row['unit']='fraction'
                    tests.append(row)
            for j,feature in enumerate(source['features'].tolist()):
                available=np.isfinite(source['raw'][:,:,j]).any(axis=1)
                tests.append(dict(dataset=name, feature=feature, question='hospital_availability', unit='fraction',
                                  **availability_test(available, sites)))
        run.audit.append(dict(check='site_and_split_patient_units', dataset=name,
                              patients=len(source['stay_ids']), unique_subjects=len(np.unique(source['subject_ids'])), passed=True))
        print(f'[06] {name}: patient-level split and site diagnostics complete', flush=True)
    run.write('site_and_split_tests', tests)
    run.write('site_and_split_summaries', profiles)
    run.finish()


if __name__=='__main__':
    main()
