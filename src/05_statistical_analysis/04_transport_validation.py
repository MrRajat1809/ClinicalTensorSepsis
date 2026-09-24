"""Estimate held-out alignment, recording sensitivity and clinical preservation."""
import sys
import numpy as np
import analysis_common as A


def geometry(original,adapted,config,key):
    rng=A.rng_for(config,key)
    left,right=rng.integers(len(original),size=(2,config['transport']['geometry_pairs']))
    before=np.linalg.norm(original[left]-original[right],axis=1)
    after=np.linalg.norm(adapted[left]-adapted[right],axis=1)
    return after[before>1e-6]/before[before>1e-6]


def main():
    run=A.Run(4)
    import joblib
    inputs=run.atlas_inputs
    rep=run.npz(A.ROOT/'data/processed/atlas/representations.npz')
    transport=run.npz(A.ROOT/'data/processed/atlas/transport.npz')
    atlas_folder=A.ROOT/'src/04_atlas_datasets'
    sys.path.insert(0,str(atlas_folder))
    module=A.load_module(run.track(atlas_folder/'03_fit_domain_transport.py'),'analysis_frozen_transport')
    models=joblib.load(run.track(A.ROOT/'outputs/atlas/transport.joblib'))
    config=run.json(atlas_folder/'atlas_config.json')
    effects,preservation=[],[]
    for domain,name in enumerate(run.config['datasets'][1:],start=1):
        rows=np.flatnonzero((inputs['dataset']==domain)&(inputs['partition']==2)&inputs['eligible'])
        A.require(len(rows)>=2,f'{name}: not enough eligible test patients')
        original,adapted=rep['z'][rows],transport['selected'][rows]
        common,choice=models[name]['common'],models[name]['selected']
        replay,supported=module.apply_map(original,rep['anchors'][rows],common,choice)
        A.require(np.allclose(replay,adapted,atol=1e-5,rtol=1e-5),'Selected map does not replay')
        A.require(np.array_equal(supported,transport['selected_supported'][rows]),'Transport support does not replay')
        A.require(np.array_equal(adapted[~supported],original[~supported]),'Unsupported patients changed')
        base=dict(dataset=name,partition='test',population='all_eligible_target_patients',method='selected',comparator='unadapted')
        effects.append(dict(**base,question='transport_support',metric='supported_fraction',unit='fraction',
                            **A.proportion(int(supported.sum()),len(rows),run.config)))
        delta=np.sqrt(np.mean((adapted-original)**2,axis=1))
        for group,mask in [('all_eligible',np.ones(len(rows),bool)),('supported',supported),('unsupported',~supported)]:
            for q in (.1,.5,.9,1.):
                effects.append(dict(**base,question='correction_magnitude',stratum=group,metric='correction_rms_quantile',
                    unit='standardized_latent',quantile=q,**A.descriptive(np.quantile(delta[mask],q) if mask.any() else np.nan,int(mask.sum()))))
        before=module.reference_distance(original,rep['anchors'][rows],common)
        after=module.reference_distance(adapted,rep['anchors'][rows],common)
        for group,mask in [('all_eligible',np.ones(len(rows),bool)),('supported',supported)]:
            valid=mask&np.isfinite(before)&np.isfinite(after)
            effects.append(dict(**base,question='clinical_anchor_alignment',stratum=group,metric='reference_distance_difference',
                unit='mean_squared_standardized_latent',**A.bootstrap_mean(after[valid]-before[valid],run.config,(name,group,'alignment')),
                n_unscored=int(mask.sum()-valid.sum()),note='Distance to nearest clinically compatible reference training prototype; negative favors alignment.'))
            for result in A.paired_tests(after[valid]-before[valid], run.config):
                effects.append(dict(**base, question='clinical_anchor_alignment', stratum=group, **result,
                    unit='fraction' if result['test_name']=='paired_sign' else 'mean_squared_standardized_latent',
                    note='Patient-paired adapted minus original reference distance; negative favors alignment.'))
        full=rep['observed_z'][rows]
        adapted_full=module.apply_map(full,rep['anchors'][rows],common,choice)[0]
        for scenario in ('point','block'):
            perturbed=rep[scenario+'_z'][rows]
            corrected=module.apply_map(perturbed,rep[scenario+'_anchors'][rows],common,choice)[0]
            before=np.linalg.norm(full-perturbed,axis=1)
            after=np.linalg.norm(adapted_full-corrected,axis=1)
            effects.append(dict(**base,question='recording_perturbation',scenario=scenario,metric='embedding_instability_difference',
                unit='standardized_latent_L2',**A.bootstrap_mean(after-before,run.config,(name,scenario)),
                note='Paired fixed observed-only views; generated values excluded from both views. Negative favors stability. '
                     'Does not simulate changes in raw event density or hourly extrema.'))
            for result in A.paired_tests(after-before, run.config):
                effects.append(dict(**base, question='recording_perturbation', scenario=scenario, **result,
                    unit='fraction' if result['test_name']=='paired_sign' else 'standardized_latent_L2',
                    note='Patient-paired instability change; negative favors robustness to removed hourly evidence.'))
        prediction0=original@rep['probe_coef'].T+rep['probe_intercept']
        prediction1=adapted@rep['probe_coef'].T+rep['probe_intercept']
        for j,probe in enumerate(rep['probe_names'].tolist()):
            truth=rep['probes'][rows,j]
            low,high=rep['probe_tails'][j]
            for stratum,mask in [('all_measured',np.isfinite(truth)),('measured_tail',np.isfinite(truth)&((truth<=low)|(truth>=high)))]:
                context=dict(**base,question='clinical_information',feature=probe,stratum=stratum,unit='transformed_probe')
                if not rep['probe_valid'][j]:
                    preservation.append(dict(**context,metric='probe_mae_difference',status='not_estimable',n_patients=int(mask.sum()),note='Probe not fitted on sufficient source training data.'))
                    continue
                before=np.abs(prediction0[mask,j]-truth[mask]); after=np.abs(prediction1[mask,j]-truth[mask])
                preservation.append(dict(**context,metric='probe_mae_difference',
                    **A.bootstrap_mean(after-before,run.config,(name,probe,stratum,'difference')),
                    original_mae=A.finite_mean(before),adapted_mae=A.finite_mean(after)))
                policy=config['selection']
                residual=after-policy['maximum_probe_error_ratio']*before-policy['probe_error_slack']
                result=A.bootstrap_mean(residual,run.config,(name,probe,stratum,'constraint'))
                conclusion=('within_engineering_tolerance' if result['ci_high']<=0 else
                    'exceeds_engineering_tolerance' if result['ci_low']>0 else 'inconclusive')
                preservation.append(dict(**context,metric='engineering_constraint_residual',**result,interpretation=conclusion,
                    error_ratio_limit=policy['maximum_probe_error_ratio'],absolute_slack=policy['probe_error_slack'],
                    note='after_error - configured_ratio * before_error - configured_slack; upper interval <=0 supports this constraint only. '
                         'Post-hoc pointwise interval, not clinical equivalence or biological invariance.'))
            if probe.endswith('/change') and rep['probe_valid'][j]:
                known=np.isfinite(truth)&(np.abs(prediction0[:,j])>.05)
                agreement=np.sign(prediction0[known,j])==np.sign(prediction1[known,j])
                preservation.append(dict(**base,question='temporal_direction',feature=probe,metric='predicted_change_sign_retention',
                    unit='fraction',**A.proportion(int(agreement.sum()),len(agreement),run.config),
                    note='Agreement with unadapted predicted direction, not accuracy against biological truth.'))
        ratios=geometry(original,adapted,run.config,('geometry',name))
        for q in (.1,.5,.9):
            preservation.append(dict(**base,question='patient_geometry',metric='pair_distance_ratio_quantile',quantile=q,unit='ratio',
                **A.descriptive(np.quantile(ratios,q) if len(ratios) else np.nan,len(rows)),n_pairs=len(ratios),
                note='Fixed sampled patient pairs; dependent pairs are not treated as independent observations.'))
        run.audit.append(dict(check='selected_transport_replay',dataset=name,patients=len(rows),passed=True))
        print(f'[04] {name}: alignment, perturbation and clinical probes complete',flush=True)
    run.write('transport_effects',effects)
    run.write('clinical_preservation',preservation)
    run.finish()


if __name__=='__main__':
    main()
