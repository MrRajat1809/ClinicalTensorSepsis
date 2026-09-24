"""Describe cohort selection, representation coverage and source-specific sensitivities."""
import numpy as np
import analysis_common as A


def covariates(source, config):
    values = {'hospital_mortality':source['labels'].astype(float), 'followup_hours':source['followup'],
              'observed_fraction':source['coverage']}
    for feature in ('age','charlson_comorbidity_index'):
        values[feature] = source['static'][:,source['static_features'].tolist().index(feature)].astype(float)
    for field in ('gender','race'):
        categories = [row.get(field) for row in source['patients']]
        for category in sorted({str(v) for v in categories if v not in (None,'')}):
            values[field+'='+category] = np.array([float(str(v)==category) if v not in (None,'') else np.nan for v in categories])
    for feature in config['clinical_features']:
        values[feature] = A.finite_mean(source['raw'][:,:,source['features'].tolist().index(feature)],axis=1)
    return values


def unit_for(feature,units):
    return {'hospital_mortality':'fraction','followup_hours':'h','observed_fraction':'fraction',
            'age':'years','charlson_comorbidity_index':'score'}.get(feature,
            'fraction' if feature.startswith(('gender=','race=')) else units.get(feature,''))


def contrast(rows, values, left, right, name, comparison, config, units):
    for feature,x in values.items():
        binary=feature=='hospital_mortality' or feature.startswith(('gender=','race='))
        for result in A.two_sample_tests(x[left], x[right], config, binary):
            rows.append(dict(question='selection_bias', dataset=name, population=comparison,
                feature=feature, unit='rank_biserial' if result['test_name']=='mann_whitney_u' else unit_for(feature,units),
                **result, n_group=int(left.sum()), n_comparator_group=int(right.sum()),
                note='Patient-level within-source comparison. Effect direction is first group minus second. '
                     'Missing values excluded per variable; U tests distributions, not necessarily medians.'))
        for standardized in (False,True):
            metric = 'standardized_mean_difference' if standardized else 'mean_difference'
            binary=feature=='hospital_mortality' or feature.startswith(('gender=','race='))
            if binary and not standardized:
                metric='risk_difference'
                result=A.risk_difference(x[left],x[right],config)
            else:
                result = A.independent_difference(x[left],x[right],config,(name,comparison,feature,metric),standardized)
            rows.append(dict(question='selection_bias',dataset=name,partition='all',population=comparison,
                feature=feature,metric=metric,unit='SD' if standardized else unit_for(feature,units),
                **result,n_group=int(left.sum()),n_comparator_group=int(right.sum()),
                note='First named group minus second; missing values excluded per variable. '
                     'SMD uses sqrt((sample_variance_1 + sample_variance_0)/2). '
                     'Observed clinical means describe recorded evidence, not a standardized severity score.'))


def main():
    run = A.Run(1)
    config, rows, cohorts, atlas, summaries, reports = run.config, [], {}, [], {}, {}
    transport = run.npz(A.ROOT/'data/processed/atlas/transport.npz')
    atlas_config = run.json(A.ROOT/'src/04_atlas_datasets/atlas_config.json')
    exporter = A.load_module(run.track(A.ROOT/'pipeline/export_release.py'),'analysis_release_metadata')
    for name in config['datasets']:
        print(f'[01] Describing selection and coverage: {name}',flush=True)
        source = run.source(name)
        values = covariates(source,config)
        units=dict(zip(source['features'].tolist(),source['units'].tolist()))
        summaries[name] = values
        cohorts[name] = source['patients']
        eligible = source['eligible']
        supported = transport['selected_supported'][source['atlas_rows']]
        reasons = exporter.eligibility_reasons(source['observation_counts'],run.atlas_inputs['common_indices'],atlas_config['eligibility'])
        A.require(np.array_equal(reasons=='eligible',eligible),'Eligibility does not match saved contract')
        atlas.extend(dict(dataset=name,eligible=bool(e),eligibility_reason=str(r),transport_supported=bool(s))
                     for e,r,s in zip(eligible,reasons,supported))
        for feature,x in values.items():
            finite=x[np.isfinite(x)]
            estimate=(A.proportion(int(finite.sum()),len(finite),config) if feature=='hospital_mortality' or feature.startswith(('gender=','race='))
                      else A.bootstrap_mean(x,config,(name,feature)))
            rows.append(dict(question='cohort_description',dataset=name,feature=feature,metric='patient_mean',
                unit=unit_for(feature,units),**estimate,n_group=len(x),n_missing=int((~np.isfinite(x)).sum())))
            for q in (0.25,0.5,0.75):
                finite=x[np.isfinite(x)]
                rows.append(dict(question='cohort_description',dataset=name,feature=feature,metric='patient_quantile',
                    quantile=q,unit=unit_for(feature,units),**A.descriptive(np.quantile(finite,q) if len(finite) else np.nan,len(finite))))
        contrast(rows,values,eligible,~eligible,name,'eligible_minus_ineligible',config,units)
        if name!='mimiciv':
            contrast(rows,values,eligible & supported,eligible & ~supported,name,'supported_minus_unsupported_eligible',config,units)
        contrast(rows,values,source['followup']>=24,source['followup']<24,name,'full_minus_incomplete_followup',config,units)
        if name=='eicu':
            strict=np.array([bool(r['strict_culture_sepsis3']) for r in source['patients']])
            contrast(rows,values,strict,~strict,name,'strict_culture_minus_proxy_only',config,units)
        base=run.parquet(A.ROOT/'data/processed'/name/'work/base_cohort.parquet')
        final_ids=set(map(int,source['stay_ids']))
        included=np.array([r['stay_id'] in final_ids for r in base])
        A.require(included.sum()==len(final_ids),'Final cohort is not a subset of base cohort')
        basic={k:np.array([r.get(column) if r.get(column) is not None else np.nan for r in base],float)
               for k,column in [('age','age'),('hospital_mortality','hospital_expire_flag')]}
        contrast(rows,basic,included,~included,name,'final_minus_other_base_patients',config,units)
        reports[name]={stage:run.json(A.ROOT/'outputs'/name/file) for stage,file in
                       [(1,'01_base.json'),(2,'02_infection.json'),(6,'06_sepsis.json')]}
        print(f'[01] {name}: {len(eligible):,} clinical patients; {(~eligible).sum()} without representation',flush=True)
    for item in exporter.cohort_flow(reports,cohorts,atlas):
        rows.append(dict(question='cohort_flow',population='cohort_selection_steps',metric='stays',
                         **A.descriptive(item['count'],item['count']),**item))
    for name in config['datasets'][1:]:
        for feature in set(summaries[name]) & set(summaries['mimiciv']):
            x,y=summaries[name][feature],summaries['mimiciv'][feature]
            x,y=x[np.isfinite(x)],y[np.isfinite(y)]
            scale=np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2) if min(len(x),len(y))>1 else np.nan
            rows.append(dict(question='source_composition',dataset=name,comparator='mimiciv',feature=feature,
                metric='standardized_mean_difference',unit='SD',n_reference=len(y),
                **A.descriptive((x.mean()-y.mean())/scale if scale>0 else np.nan,len(x)),
                note='Descriptive source contrast; cross-version MIMIC patient independence is not established.'))
    run.write('cohort_selection',rows)
    run.finish()


if __name__=='__main__':
    main()
