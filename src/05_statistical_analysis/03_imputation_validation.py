"""Replay frozen holdouts and estimate paired patient-level imputation errors."""
from collections import defaultdict
import numpy as np
import analysis_common as A


def patient_means(ids,values):
    unique,inverse=np.unique(ids,return_inverse=True)
    return unique,np.bincount(inverse,weights=values)/np.bincount(inverse)


def temporal_errors(truth,prediction,hidden):
    known=np.isfinite(truth)
    restored=np.where(hidden,prediction,truth)
    pairs=known[:,1:]&known[:,:-1]&(hidden[:,1:]|hidden[:,:-1])
    difference=np.diff(restored,axis=1)-np.diff(truth,axis=1)
    change=A.finite_mean(np.where(pairs,np.abs(difference),np.nan),axis=1)
    enough=(known.sum(axis=1)>=3)&hidden.any(axis=1)
    spread=np.full(len(truth),np.nan)
    for i in np.flatnonzero(enough):
        spread[i]=abs(np.std(restored[i,known[i]])-np.std(truth[i,known[i]]))
    return change,spread


def gap_spans(remaining,exposed):
    spans=np.zeros(remaining.shape,dtype=int)
    for i in range(len(remaining)):
        start=None
        for h in range(remaining.shape[1]+1):
            missing=h<remaining.shape[1] and exposed[i,h] and not np.isfinite(remaining[i,h])
            if missing and start is None:
                start=h
            elif not missing and start is not None:
                spans[i,start:h]=h-start
                start=None
    return spans


def summarize(run,name,feature,unit,scenario,cells,temporal,tail_limits,expected):
    config=run.config
    data={k:np.concatenate([r[k] for r in cells]) for k in cells[0]}
    truth,ids=data['truth'],data['patient']
    comparisons,profiles=[],[]
    for method in ('final','median','forward_fill'):
        mae=float(np.abs(data[method]-truth).mean())
        metric_key='final_mae' if method=='final' else method+'_mae'
        A.require(np.isclose(mae,expected[metric_key],rtol=config['imputation']['replay_rtol'],
                             atol=config['imputation']['replay_atol']),f'{name}/{scenario}/{feature}/{method}: checkpoint metric replay mismatch')
    A.require(len(truth)==expected['targets'] and len(np.unique(ids))==expected['target_patients'], 'Holdout counts do not replay')
    context=data['context']
    strata={'all':np.ones(len(ids),bool),'measured_tail':(truth<=tail_limits[0])|(truth>=tail_limits[1]),
            'context_1_to_23':context<24,'context_24_to_95':(context>=24)&(context<96),'context_96_plus':context>=96,
            'gap_1_to_2h':data['gap']<=2,'gap_3_to_6h':(data['gap']>=3)&(data['gap']<=6),'gap_7_to_24h':data['gap']>=7}
    for stratum,mask in strata.items():
        keys=dict(question='imputation_accuracy',dataset=name,partition='test',feature=feature,scenario=scenario,
                  stratum=stratum,unit=unit,population='fixed_saved_holdout_targets',n_observations=int(mask.sum()))
        errors={method:data[method][mask]-truth[mask] for method in ('final','median','forward_fill')}
        for method,error in errors.items():
            pid,per_patient=patient_means(ids[mask],np.abs(error))
            for metric,values in [('patient_mae',per_patient),('patient_signed_error',patient_means(ids[mask],error)[1])]:
                comparisons.append(dict(**keys,method=method,metric=metric,
                    **A.bootstrap_mean(values,config,(name,feature,scenario,stratum,method,metric))))
            comparisons.append(dict(**keys,method=method,metric='cell_weighted_mae',
                **A.descriptive(A.finite_mean(np.abs(error)),len(pid)),note='Replays saved weighting; patient-MAE is the primary equal-patient estimand.'))
            for q in config['imputation']['error_quantiles']:
                profiles.append(dict(**keys,method=method,metric='absolute_error_quantile',quantile=q,
                    **A.descriptive(np.quantile(np.abs(error),q) if len(error) else np.nan,len(pid))))
        for comparator in config['imputation']['comparators']:
            _,delta=patient_means(ids[mask],np.abs(errors['final'])-np.abs(errors[comparator]))
            comparisons.append(dict(**keys,method='final',comparator=comparator,metric='paired_patient_mae_difference',
                **A.bootstrap_mean(delta,config,(name,feature,scenario,stratum,comparator)),
                note='Final minus comparator; negative favors final. All examples from one patient are averaged before resampling.'))
            for result in A.paired_tests(delta, config):
                context=dict(keys)
                if result['test_name']=='paired_sign':
                    context['unit']='fraction'
                comparisons.append(dict(**context, method='final', comparator=comparator, **result,
                    note='One MAE difference per patient, final minus comparator. Negative favors final. '
                         'Sign test concerns direction among nonties; t test concerns the mean. No recipe reselection.'))
    for metric in ('change_absolute_error','variability_absolute_error'):
        for method in ('final','median','forward_fill'):
            patient=np.concatenate([r['patient'] for r in temporal])
            value=np.concatenate([r[method+'_'+metric] for r in temporal])
            valid=np.isfinite(value)
            pid,means=patient_means(patient[valid],value[valid])
            comparisons.append(dict(question='temporal_distortion',dataset=name,partition='test',feature=feature,
                scenario=scenario,method=method,metric=metric,unit=unit,**A.bootstrap_mean(means,config,(name,feature,scenario,method,metric)),
                note='Only originally measured cells; adjacent changes require at least one hidden endpoint. '
                     'Variability requires >=3 measured hours and a hidden cell; no claim about natural-gap truth.'))
    return comparisons,profiles


def analyze_source(run,name):
    source=run.source(name)
    report=run.json(A.ROOT/'outputs'/name/'saits.json')
    frozen=report['configuration']
    script=next((A.ROOT/'src'/A.SOURCES[name]).glob('08_*.py'))
    recipe=A.load_module(run.track(script),'analysis_saits_'+name.replace('-','_'))
    holdouts=run.npz(A.ROOT/'outputs'/name/'holdouts.npz')
    statistics=frozen['statistics']
    indices=[e['index'] for e in statistics]
    A.require(indices==frozen['model_feature_indices']==holdouts['model_feature_indices'].tolist(),'Model channel binding mismatch')
    scaled=np.empty((len(source['raw']),24,len(indices)),dtype=np.float32)
    for j,entry in enumerate(statistics):
        raw=source['raw'][:,:,entry['index']]
        scaled[:,:,j]=((np.log1p(raw) if entry['log1p'] else raw)-entry['mean'])/entry['scale']
    train=holdouts['train_indices']
    A.require(np.all(source['partition'][train]==0),'Imputation training partition mismatch')
    options=dict(frozen['model_options'])
    if run.args.device is not None:
        options['device']=run.args.device
    model,_=recipe.make_gap_model(frozen['policy'],options,scaled[train[np.isfinite(scaled[train]).any(axis=(1,2))]])
    model.load(str(run.track(A.ROOT/'outputs'/name/'saits.pypots')))
    comparisons,profiles=[],[]
    for scenario in run.config['imputation']['scenarios']:
        print(f'[03] {name}: frozen test/{scenario} replay',flush=True)
        selected=holdouts['test_'+scenario+'_indices']
        masks=holdouts['test_'+scenario+'_mask']
        A.require(np.all(source['partition'][selected]==2),'Holdout contains non-test patients')
        A.require(masks.shape==(len(selected),24,len(indices)),'Holdout mask shape mismatch')
        cell_records,temporal_records=defaultdict(list),defaultdict(list)
        for start in range(0,len(selected),256):
            ids=selected[start:start+256]
            mask=masks[start:start+256]
            values=scaled[ids].copy()
            A.require(np.isfinite(values[mask]).all(),'Hidden target was not measured')
            values[mask]=np.nan
            A.require(np.isfinite(values).any(axis=(1,2)).all(),'Holdout has no remaining model context')
            prediction=recipe.predict(model,values)
            A.require(np.isfinite(prediction[mask]).all(),'Nonfinite holdout prediction')
            physical,rejected=recipe.inverse_predictions(prediction,statistics)
            context=np.isfinite(values).sum(axis=(1,2))
            for j,entry in enumerate(statistics):
                hidden=mask[:,:,j]
                if not hidden.any():
                    continue
                truth=np.asarray(source['raw'][ids,:,entry['index']])
                remaining=truth.copy(); remaining[hidden]=np.nan
                forward,_=recipe.baseline_values(remaining,entry)
                median=np.full_like(forward,entry['median'])
                method=report['feature_methods'][entry['feature']]
                baseline=forward if method['baseline']=='forward_fill' else median
                final=np.where(rejected[:,:,j],baseline,physical[:,:,j]) if method['method']=='saits' else baseline
                row,hour=np.where(hidden)
                spans=gap_spans(remaining,source['exposure_seconds'][ids]>0)
                cell_records[j].append(dict(patient=ids[row],hour=hour,truth=truth[hidden],final=final[hidden],
                    median=median[hidden],forward_fill=forward[hidden],context=context[row],gap=spans[hidden]))
                temporal=dict(patient=ids)
                for key,predicted in [('final',final),('median',median),('forward_fill',forward)]:
                    change,spread=temporal_errors(truth,predicted,hidden)
                    temporal[key+'_change_absolute_error']=change
                    temporal[key+'_variability_absolute_error']=spread
                temporal_records[j].append(temporal)
        expected={r['feature']:r for r in report['metrics']['test'] if r['scenario']==scenario}
        for j,entry in enumerate(statistics):
            if j not in cell_records:
                A.require(expected[entry['feature']]['targets']==0,'Missing scored feature')
                comparisons.append(dict(question='imputation_accuracy',dataset=name,partition='test',feature=entry['feature'],
                    scenario=scenario,metric='patient_mae',status='not_estimable',n_patients=0,n_observations=0,note='No saved test targets.'))
                continue
            training=source['raw'][train,:,entry['index']]
            limits=np.quantile(training[np.isfinite(training)],run.config['imputation']['tail_quantiles'])
            a,b=summarize(run,name,entry['feature'],str(source['units'][entry['index']]),scenario,
                cell_records[j],temporal_records[j],limits,expected[entry['feature']])
            comparisons.extend(a); profiles.extend(b)
            if (j+1)%10==0:
                print(f'[03] {name}/{scenario}: uncertainty estimates {j+1}/{len(statistics)} features',flush=True)
        run.audit.append(dict(check='frozen_holdout_replay',dataset=name,scenario=scenario,examples=len(selected),
                              patients=len(np.unique(selected)),target_cells=int(masks.sum()),passed=True))
    return comparisons,profiles


def main():
    run=A.Run(3)
    comparisons,profiles=[],[]
    for name in run.config['datasets']:
        a,b=analyze_source(run,name)
        comparisons.extend(a); profiles.extend(b)
    run.write('imputation_comparisons',comparisons)
    run.write('imputation_error_profiles',profiles)
    run.finish()


if __name__=='__main__':
    main()
