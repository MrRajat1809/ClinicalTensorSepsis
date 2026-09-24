"""Account for unit screening, exposure, observed coverage and imputation provenance."""
from collections import Counter
import numpy as np
import analysis_common as A


def gap_lengths(observed,exposure):
    counts=Counter()
    for valid,followup in zip(observed,exposure):
        run=0
        for seen,available in zip(valid,followup>0):
            if available and not seen:
                run+=1
            elif run:
                counts[run]+=1
                run=0
        if run:
            counts[run]+=1
    return counts


def main():
    run=A.Run(2)
    quality,coverage=[],[]
    bounds=run.json(A.ROOT/'src/common/measurement_rules.json')['bounds']
    schema=run.json(A.ROOT/'src/common/tensor_schema.json')
    bounds={**bounds,**schema['policy']['derived_bounds'],'urine_output':[0,None]}
    for name in run.config['datasets']:
        source=run.source(name)
        audit=run.csv(A.ROOT/'outputs'/name/'measurement_qc.csv')
        denominators=Counter()
        for r in audit:
            denominators[r['feature']]+=int(r['rows'])
        for r in audit:
            count=int(r['rows'])
            quality.append(dict(question='measurement_screening',dataset=name,metric='audited_event_records',population='extracted_infection_candidate_evidence',
                **A.descriptive(count),**{k:v for k,v in r.items() if k!='rows'},
                n_observations=count,denominator=denominators[r['feature']],
                note='Event-row accounting, not independent patients; per-rule stay counts overlap and must not be summed.'))
        converted=run.query_parquet(A.ROOT/'data/processed'/name/'work/events_clean.parquet','''
            SELECT source_table, feature, raw_unit, canonical_unit, unit_rule,
                   COUNT(*) AS records, COUNT(DISTINCT stay_id) AS patients,
                   QUANTILE_CONT(valuenum, [0.01,0.25,0.5,0.75,0.99]) AS quantiles
            FROM read_parquet(?) WHERE numeric_usable AND valuenum IS NOT NULL
            GROUP BY source_table, feature, raw_unit, canonical_unit, unit_rule
            ORDER BY source_table, feature, raw_unit, canonical_unit, unit_rule''')
        for item in converted:
            for q,value in zip((.01,.25,.5,.75,.99),item['quantiles']):
                quality.append(dict(question='unit_rule_distribution',dataset=name,population='accepted_infection_candidate_evidence',
                    feature=item['feature'],source_table=item['source_table'],raw_unit=item['raw_unit'],unit_rule=item['unit_rule'],
                    unit=item['canonical_unit'],metric='canonical_event_quantile',quantile=q,
                    **A.descriptive(value,int(item['patients'])),n_observations=int(item['records']),
                    note='Accepted canonical event values by conversion rule, before final cohort selection/hourly aggregation. '
                         'Differences may reflect case mix or recording practice; not an equality test.'))
        exposure=source['exposure_seconds']
        alive=exposure>0
        groups={'all':np.ones(len(alive),bool),'died':source['labels']==1,'survived':source['labels']==0,
                'full_followup':source['followup']>=24,'incomplete_followup':source['followup']<24}
        if name=='eicu':
            hospitals=np.array([str(r['hospital_id']) for r in source['patients']])
            groups.update({'hospital='+h:hospitals==h for h in sorted(set(hospitals))})
        # Observed clinical strata use training-derived quartiles, separately by source.
        severity=A.finite_mean(source['raw'][:,:,source['features'].tolist().index('lactate')],axis=1)
        training=severity[(source['partition']==0)&np.isfinite(severity)]
        if len(training):
            edge=np.quantile(training,[0.25,0.75])
            groups.update(lactate_observed_low=severity<=edge[0],lactate_observed_high=severity>=edge[1],
                          lactate_unobserved=~np.isfinite(severity))
        for hour in range(24):
            for metric,mask in [('exposed',alive[:,hour]),('structural',~alive[:,hour]),
                                ('partial',(exposure[:,hour]>0)&(exposure[:,hour]<3600))]:
                coverage.append(dict(question='followup',dataset=name,hour=hour,metric=metric,unit='fraction',
                    **A.proportion(int(mask.sum()),len(mask),run.config),denominator_population='all clinical patients'))
        for j,feature in enumerate(source['features'].tolist()):
            raw=source['raw'][:,:,j]
            observed=np.isfinite(raw)
            unit=str(source['units'][j])
            for group,members in groups.items():
                denominators=alive[members].sum(axis=1)
                patient_fraction=np.divide(observed[members].sum(axis=1),denominators,
                    out=np.full(int(members.sum()),np.nan),where=denominators>0)
                # Hospital profiles use Wilson coverage of at least one observation, avoiding event-level inference.
                coverage.append(dict(question='feature_availability',dataset=name,feature=feature,stratum=group,
                    metric='patients_with_any_observation',unit='fraction',
                    **A.proportion(int(observed[members].any(axis=1).sum()),int(members.sum()),run.config),
                    mean_patient_observed_fraction=A.finite_mean(patient_fraction),
                    zero_followup_patients=int((denominators==0).sum())))
                if not group.startswith('hospital='):
                    coverage.append(dict(question='feature_availability',dataset=name,feature=feature,stratum=group,
                        metric='mean_patient_observed_fraction',unit='fraction',
                        **A.bootstrap_mean(patient_fraction,run.config,(name,feature,group)),
                        note='Positive-exposure hours only; each patient has equal weight.'))
            for hour in range(24):
                denominator=int(alive[:,hour].sum())
                coverage.append(dict(question='hourly_coverage',dataset=name,feature=feature,hour=hour,
                    metric='observed_fraction',unit='fraction',
                    **A.proportion(int(observed[:,hour].sum()),denominator,run.config),
                    denominator_population='patients with positive exposure in this hour'))
                for code,label in enumerate(('unfilled','observed','saits','forward_fill','median')):
                    count=int(((source['method_codes'][:,hour,j]==code)&alive[:,hour]).sum())
                    coverage.append(dict(question='cell_provenance',dataset=name,feature=feature,hour=hour,
                        method=label,metric='within_followup_method_fraction',unit='fraction',
                        **A.proportion(count,denominator,run.config)))
            for length,count in sorted(gap_lengths(observed,exposure).items()):
                coverage.append(dict(question='missing_gap_length',dataset=name,feature=feature,metric='gap_runs',
                    gap_hours=length,**A.descriptive(count,len(raw)),unit='runs',
                    note='Run includes only hours with positive exposure; multiple runs per patient, no independent-run CI.'))
            values=raw[observed]
            low,high=bounds[feature]
            for metric,value in [('at_screening_lower',int((values==low).sum())),
                                 ('at_screening_upper',int((values==high).sum()) if high is not None else np.nan)]:
                quality.append(dict(question='screening_boundary',dataset=name,feature=feature,metric=metric,
                    **A.descriptive(value,int(observed.any(axis=1).sum())),n_observations=len(values),unit='cells',
                    note='Exact boundary counts; no finite hourly urine upper bound. Screening ranges are not normal ranges.'))
            for q in (0,.01,.1,.25,.5,.75,.9,.99,1):
                quality.append(dict(question='canonical_values',dataset=name,feature=feature,metric='observed_quantile',
                    quantile=q,unit=unit,**A.descriptive(np.quantile(values,q) if len(values) else np.nan,int(observed.any(axis=1).sum())),
                    n_observations=len(values),note='Observed hourly values; event-density weighted across patients, descriptive only.'))
        print(f'[02] {name}: units, coverage and gap profiles complete',flush=True)
    run.write('measurement_quality',quality)
    run.write('coverage_profiles',coverage)
    run.finish()


if __name__=='__main__':
    main()
