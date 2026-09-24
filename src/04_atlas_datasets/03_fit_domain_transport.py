"""Fit training-only local unbalanced OT and select on unlabelled validation, v1.0.0."""
import importlib
import warnings

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf

C = importlib.import_module('01_prepare_atlas_inputs')
P, S = (C.CONFIG['transport'], C.CONFIG['selection'])
LOCAL_NEIGHBOURS = 8
SMALL_STRENGTH_FACTORS = (0.5, 0.75)


def strength_grid():
    """Use the same declared, bounded validation search in both target domains."""
    base = P['strengths']
    return sorted(set(base) | {min(base) * factor for factor in SMALL_STRENGTH_FACTORS})


def mapping_policy():
    return dict(method='clinically_compatible_local_barycentric_residuals',
                neighbours=LOCAL_NEIGHBOURS, strengths=strength_grid(),
                neighbour_gate='clinical compatibility and training support radius',
                ranking='validation_alignment_improvement', legacy_transport_ridge_used=False)


def anchor_cost(a, b):
    """Age contributes to distance; only measured clinical anchors count toward overlap."""
    total = np.zeros((len(a), len(b)), dtype=np.float64)
    count = np.zeros_like(total)
    clinical = np.zeros_like(total)
    for j in range(a.shape[1]):
        known = np.isfinite(a[:, j, None]) & np.isfinite(b[None, :, j])
        delta = np.nan_to_num(a[:, j, None] - b[None, :, j], nan=0)
        total += np.square(delta) * known
        count += known
        if j > 0:
            clinical += known
    cost = total / np.maximum(count, 1)
    allowed = (clinical >= P['minimum_shared_anchors']) & (cost <= P['maximum_anchor_rms'] ** 2)
    return (cost, allowed)


def prototype_rows(rows, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, min(len(rows), P['maximum_prototypes']), replace=False))


def support_radius(values, prototypes):
    nearest = []
    for start in range(0, len(values), 256):
        distances = cdist(values[start:start + 256], prototypes, metric='sqeuclidean') / values.shape[1]
        if distances.shape[1] == 1:
            nearest.extend(distances[:, 0])
        else:
            # Avoid zero self-distance when calibrating support on training prototypes.
            nearest.extend(np.partition(distances, 1, axis=1)[:, 1])
    return max(float(np.quantile(nearest, P['support_quantile'])), 1e-06)


def matrix_power(matrix, power):
    eigenvalues, vectors = np.linalg.eigh(matrix)
    return vectors * np.maximum(eigenvalues, 1e-06) ** power @ vectors.T


def fit_common(rep, source_rows, target_rows, domain):
    reference = prototype_rows(source_rows, C.CONFIG['seed'])
    target = prototype_rows(target_rows, C.CONFIG['seed'] + domain)
    zr, zt = (rep['z'][reference], rep['z'][target])
    reference_cov = LedoitWolf().fit(rep['z'][source_rows]).covariance_
    target_cov = LedoitWolf().fit(rep['z'][target_rows]).covariance_
    covariance = matrix_power(target_cov, -0.5) @ matrix_power(reference_cov, 0.5)
    return dict(
        reference_rows=reference, target_rows=target,
        source_fit_rows=source_rows, target_fit_rows=target_rows,
        reference_z=zr, target_z=zt,
        reference_anchors=rep['anchors'][reference], target_anchors=rep['anchors'][target],
        reference_radius=support_radius(rep['z'][source_rows], zr),
        target_radius=support_radius(rep['z'][target_rows], zt), covariance=covariance,
        source_mean=rep['z'][source_rows].mean(axis=0),
        target_mean=rep['z'][target_rows].mean(axis=0))


def fit_ot(common, entropy):
    """Fit UOT on training prototypes and retain barycentric prototype residuals."""
    import ot
    zt, zr = (common['target_z'], common['reference_z'])
    latent = cdist(zt, zr, metric='sqeuclidean') / zt.shape[1]
    clinical, allowed = anchor_cost(common['target_anchors'], common['reference_anchors'])
    usable = allowed.any(axis=1)
    if usable.sum() < S['minimum_validation_patients']:
        return dict(available=False, reason='insufficient clinically compatible training prototypes', entropy=entropy)
    normalizer = max(float(np.median(latent[allowed])), 1e-06)
    cost = latent / normalizer + P['anchor_cost_weight'] * clinical
    cost[~allowed] = 1000.0
    a = np.full(len(zt), 1 / len(zt))
    b = np.full(len(zr), 1 / len(zr))
    active_target = np.flatnonzero(usable)
    active_reference = np.flatnonzero(allowed.any(axis=0))
    active = np.ix_(active_target, active_reference)
    # Remove fully incompatible rows/columns before solving; audit forbidden mass afterward.
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        reduced_plan, log = ot.unbalanced.sinkhorn_unbalanced(
            a[active_target], b[active_reference], cost[active], reg=entropy,
            reg_m=P['mass_regularization'], method='sinkhorn_stabilized', reg_type='kl',
            numItermax=P['solver_iterations'], stopThr=P['solver_tolerance'], log=True)
    plan = np.zeros_like(cost)
    plan[active] = reduced_plan
    errors = log.get('err', [])
    converged = len(errors) > 0 and np.isfinite(errors[-1]) and (errors[-1] <= P['solver_tolerance'] * 10)
    numerical_warning = any(('numerical' in str(item.message).lower() for item in recorded))
    if not converged or numerical_warning or (not np.isfinite(plan).all()) or np.any(plan < 0):
        return dict(
            available=False, reason='unbalanced solver failed convergence/numerical checks',
            entropy=entropy,
            final_error=float(errors[-1]) if len(errors) and np.isfinite(errors[-1]) else None,
            warnings=[str(item.message) for item in recorded])
    mass = plan.sum(axis=1)
    forbidden_fraction = np.sum(plan * ~allowed, axis=1) / np.maximum(mass, 1e-300)
    mass_ratio = mass / a
    accepted = (usable & (mass_ratio >= P['minimum_mass_ratio'])
                & (forbidden_fraction <= P['maximum_forbidden_mass_fraction']))
    if accepted.sum() < S['minimum_validation_patients']:
        return dict(available=False, reason='insufficient transported mass on admissible pairs', entropy=entropy)
    barycenter = plan @ zr / np.maximum(mass[:, None], 1e-300)
    prototype_delta = barycenter - zt
    prototype_reliability = np.clip(mass_ratio, 0.0, 1.0)
    prototype_reliability[~accepted] = 0.0
    return dict(
        available=True, entropy=entropy, prototype_accepted=accepted,
        prototype_delta=prototype_delta, prototype_reliability=prototype_reliability,
        mass_ratio=mass_ratio, forbidden_fraction=forbidden_fraction,
        cost_normalizer=normalizer, barycenter=barycenter,
        final_error=float(errors[-1]), transported_mass=float(mass.sum()),
        warnings=[str(item.message) for item in recorded], mapping=mapping_policy()['method'])


def ot_target_prototypes(common, ot_model=None):
    """Return target prototypes eligible to support an OT correction."""
    if ot_model is None:
        indices = np.arange(len(common['target_z']))
    else:
        indices = np.flatnonzero(ot_model['prototype_accepted'])
    return (indices, common['target_z'][indices])


def support(z, anchors, common, ot_model=None):
    """Conservative support gate."""
    result = np.zeros(len(z), dtype=bool)
    target_indices, target_prototypes = ot_target_prototypes(common, ot_model)
    if not len(target_indices):
        return result
    for start in range(0, len(z), 256):
        query = z[start:start + 256]
        query_anchors = anchors[start:start + 256]
        _, allowed = anchor_cost(query_anchors, common['reference_anchors'])
        reference_distance = cdist(query, common['reference_z'], metric='sqeuclidean') / z.shape[1]
        reference_distance[~allowed] = np.inf
        target_distance = cdist(query, target_prototypes, metric='sqeuclidean') / z.shape[1]
        if ot_model is not None:
            _, compatible = anchor_cost(query_anchors, common['target_anchors'][target_indices])
            target_distance[~compatible] = np.inf
        nearest_target = target_distance.min(axis=1)
        okay = (np.isfinite(reference_distance.min(axis=1))
                & (reference_distance.min(axis=1) <= common['reference_radius'])
                & (nearest_target <= common['target_radius']))
        result[start:start + len(query)] = okay
    return result


def local_ot_delta(z, anchors, common, model):
    """Interpolate only nearby, clinically compatible training-prototype corrections."""
    accepted = np.flatnonzero(model['prototype_accepted'])
    C.require(len(accepted) > 0, 'OT model has no accepted training prototypes')
    prototypes = common['target_z'][accepted]
    prototype_delta = model['prototype_delta'][accepted]
    reliability = model['prototype_reliability'][accepted]
    neighbours = min(LOCAL_NEIGHBOURS, len(accepted))
    output = np.zeros_like(z, dtype=np.float64)
    confidence = np.zeros(len(z), dtype=np.float64)
    radius = max(float(common['target_radius']), 1e-06)
    for start in range(0, len(z), 256):
        query = z[start:start + 256]
        distances = cdist(query, prototypes, metric='sqeuclidean') / z.shape[1]
        _, compatible = anchor_cost(anchors[start:start + len(query)], common['target_anchors'][accepted])
        distances[~compatible | (distances > radius)] = np.inf
        # Stable ordering makes equal-distance neighbours reproducible.
        selected = np.argsort(distances, axis=1, kind='stable')[:, :neighbours]
        local_distance = np.take_along_axis(distances, selected, axis=1)
        local_reliability = reliability[selected]
        weights = local_reliability / (local_distance + 1e-06)
        weight_sum = weights.sum(axis=1, keepdims=True)
        normalized = np.divide(weights, np.maximum(weight_sum, 1e-12))
        local_delta = np.einsum('nk,nkd->nd', normalized, prototype_delta[selected])
        nearest = local_distance.min(axis=1)
        local_confidence = np.exp(-nearest / radius)
        output[start:start + len(query)] = local_delta * local_confidence[:, None]
        confidence[start:start + len(query)] = local_confidence
    return (output, confidence)


def apply_map(z, anchors, common, choice, eligible=None):
    z = np.asarray(z, dtype=np.float64)
    if choice['method'] == 'identity' or not choice.get('available', True):
        return (z.copy(), np.zeros(len(z), dtype=bool))
    model = choice.get('model') if choice['method'] == 'ot' else None
    supported = support(z, anchors, common, model)
    if eligible is not None:
        supported &= eligible
    if choice['method'] == 'ot':
        delta, confidence = local_ot_delta(z, anchors, common, model)
        supported &= confidence > 0
    else:
        delta = (z - common['target_mean']) @ common['covariance'] + common['source_mean'] - z
    delta *= choice['strength']
    rms = np.sqrt(np.mean(delta ** 2, axis=1))
    delta *= np.minimum(1.0, P['maximum_correction_rms'] / np.maximum(rms, 1e-12))[:, None]
    delta[~supported] = 0
    return (z + delta, supported)


def preservation(original, adapted, truth, rep, seed):
    original_prediction = original @ rep['probe_coef'].T + rep['probe_intercept']
    prediction = adapted @ rep['probe_coef'].T + rep['probe_intercept']
    rows, gates = ([], [])
    for j, name in enumerate(rep['probe_names'].tolist()):
        available = np.isfinite(truth[:, j])
        if not rep['probe_valid'][j] or available.sum() < S['minimum_probe_patients']:
            continue
        before = np.abs(original_prediction[available, j] - truth[available, j])
        after = np.abs(prediction[available, j] - truth[available, j])
        passed = float(after.mean()) <= S['maximum_probe_error_ratio'] * float(before.mean()) + S['probe_error_slack']
        low, high = rep['probe_tails'][j]
        tail = available & ((truth[:, j] <= low) | (truth[:, j] >= high))
        tail_before = tail_after = None
        if tail.sum() >= S['minimum_tail_patients']:
            tail_before = float(np.abs(original_prediction[tail, j] - truth[tail, j]).mean())
            tail_after = float(np.abs(prediction[tail, j] - truth[tail, j]).mean())
            passed &= tail_after <= S['maximum_probe_error_ratio'] * tail_before + S['probe_error_slack']
        sign_agreement = None
        if name.endswith('/change'):
            informative = available & (np.abs(original_prediction[:, j]) > 0.05)
            if informative.sum() >= S['minimum_probe_patients']:
                sign_agreement = float(np.mean(
                    np.sign(prediction[informative, j]) == np.sign(original_prediction[informative, j])))
                passed &= sign_agreement >= S['minimum_slope_sign_agreement']
        rows.append(dict(
            probe=name, patients=int(available.sum()), original_mae=float(before.mean()),
            adapted_mae=float(after.mean()), tail_patients=int(tail.sum()),
            tail_original_mae=tail_before, tail_adapted_mae=tail_after,
            slope_sign_agreement=sign_agreement, passed=bool(passed)))
        gates.append(bool(passed))
    rng = np.random.default_rng(seed)
    left, right = rng.integers(0, len(original), (2, S['pair_samples']))
    before = np.linalg.norm(original[left] - original[right], axis=1)
    after = np.linalg.norm(adapted[left] - adapted[right], axis=1)
    ratio = after[before > 1e-06] / before[before > 1e-06]
    lower, upper = np.quantile(ratio, [0.1, 0.9]) if len(ratio) else (0.0, 0.0)
    geometry = lower >= S['minimum_pair_distance_ratio'] and upper <= S['maximum_pair_distance_ratio']
    return dict(
        passed=bool(len(gates) >= S['minimum_probe_dimensions'] and all(gates) and geometry),
        probes=rows, pair_distance_ratio_p10=float(lower), pair_distance_ratio_p90=float(upper))


def reference_distance(z, anchors, common):
    """Nearest clinically-compatible MIMIC-IV prototype distance."""
    result = np.full(len(z), np.nan, dtype=np.float64)
    for start in range(0, len(z), 256):
        query = z[start:start + 256]
        query_anchors = anchors[start:start + 256]
        _, allowed = anchor_cost(query_anchors, common['reference_anchors'])
        distances = cdist(query, common['reference_z'], metric='sqeuclidean') / z.shape[1]
        distances[~allowed] = np.inf
        nearest = distances.min(axis=1)
        nearest[~np.isfinite(nearest)] = np.nan
        result[start:start + len(query)] = nearest
    return result


def alignment_report(original, adapted, anchors, supported, common):
    """Measure reference proximity on the same supported patients before and after mapping."""
    before = reference_distance(original, anchors, common)
    after = reference_distance(adapted, anchors, common)
    usable = supported & np.isfinite(before) & np.isfinite(after)
    if usable.sum() < S['minimum_validation_patients']:
        return dict(passed=False, patients=int(usable.sum()), original_mean=None, adapted_mean=None, improvement=None)
    original_mean = float(before[usable].mean())
    adapted_mean = float(after[usable].mean())
    improvement = (original_mean - adapted_mean) / original_mean if original_mean > 1e-12 else 0.0
    return dict(passed=bool(improvement > 0.0), patients=int(usable.sum()),
                original_mean=original_mean, adapted_mean=adapted_mean, improvement=float(improvement))


def correction_report(original, adapted, supported):
    delta = adapted - original
    rms = np.sqrt(np.mean(delta ** 2, axis=1))
    values = rms[supported]
    if not len(values):
        return dict(supported_patients=0, mean_rms=0.0, median_rms=0.0, p90_rms=0.0, maximum_rms=0.0)
    return dict(supported_patients=int(len(values)), mean_rms=float(values.mean()),
                median_rms=float(np.median(values)), p90_rms=float(np.quantile(values, 0.9)),
                maximum_rms=float(values.max()))


def assess(rep, rows, common, choice, seed):
    original = rep['z'][rows]
    anchors = rep['anchors'][rows]
    adapted, supported = apply_map(original, anchors, common, choice)
    clinical = preservation(original, adapted, rep['probes'][rows], rep, seed)
    alignment = alignment_report(original, adapted, anchors, supported, common)
    original_view = rep['observed_z'][rows]
    corrected_view = apply_map(original_view, anchors, common, choice)[0]
    stability = {}
    for scenario in ('point', 'block'):
        thinned = rep[scenario + '_z'][rows]
        corrected = apply_map(thinned, rep[scenario + '_anchors'][rows], common, choice)[0]
        before = float(np.mean(np.linalg.norm(original_view - thinned, axis=1)))
        after = float(np.mean(np.linalg.norm(corrected_view - corrected, axis=1)))
        stability[scenario] = dict(original_distance=before, adapted_distance=after,
                                   improvement=(before - after) / before if before > 1e-08 else 0.0)
    minimum_improvement = min((item['improvement'] for item in stability.values()))
    passed = (clinical['passed'] and alignment['passed']
              and float(supported.mean()) >= S['minimum_supported_fraction']
              and minimum_improvement >= S['minimum_stability_improvement']
              and len(rows) >= S['minimum_validation_patients'])
    score = float(alignment['improvement']) if alignment['improvement'] is not None else -1.0
    return dict(passed=bool(passed), supported_fraction=float(supported.mean()), clinical=clinical,
                alignment=alignment, stability=stability, score=score, patients=len(rows),
                correction=correction_report(original, adapted, supported))


def rejection_reasons(assessment):
    """Explain the existing acceptance gates without changing their thresholds."""
    reasons = []
    clinical = assessment['clinical']
    for probe in clinical['probes']:
        if probe['passed']:
            continue
        name = probe['probe']
        for label, before_key, after_key in (
                ('MAE', 'original_mae', 'adapted_mae'),
                ('tail MAE', 'tail_original_mae', 'tail_adapted_mae')):
            before, after = (probe[before_key], probe[after_key])
            if before is not None and after is not None:
                limit = S['maximum_probe_error_ratio'] * before + S['probe_error_slack']
                if after > limit:
                    reasons.append(f'{name} {label} {after:.6g} > {limit:.6g}')
        sign = probe['slope_sign_agreement']
        if sign is not None and sign < S['minimum_slope_sign_agreement']:
            reasons.append(f'{name} slope sign agreement {sign:.4f} below threshold')
    if len(clinical['probes']) < S['minimum_probe_dimensions']:
        reasons.append('insufficient clinical probe dimensions')
    if clinical['pair_distance_ratio_p10'] < S['minimum_pair_distance_ratio']:
        reasons.append('excessive contraction of patient distances')
    if clinical['pair_distance_ratio_p90'] > S['maximum_pair_distance_ratio']:
        reasons.append('excessive expansion of patient distances')
    if not clinical['passed'] and (not reasons):
        reasons.append('clinical preservation failed')
    if not assessment['alignment']['passed']:
        reasons.append('insufficient supported alignment improvement')
    if assessment['supported_fraction'] < S['minimum_supported_fraction']:
        reasons.append('insufficient supported fraction')
    for scenario, result in assessment['stability'].items():
        if result['improvement'] < S['minimum_stability_improvement']:
            reasons.append(f"{scenario} stability improvement {result['improvement']:.2%} "
                           f"< {S['minimum_stability_improvement']:.2%}")
    if assessment['patients'] < S['minimum_validation_patients']:
        reasons.append('insufficient validation patients')
    return reasons


def select_candidate(candidates):
    accepted = [item for item in candidates if item['assessment']['passed']]
    if not accepted:
        return dict(name='identity', method='identity', strength=0.0, available=True)
    return sorted(accepted, key=lambda item: (-item['assessment']['score'], item['strength'], item['name']))[0]


def candidate_summary(choice):
    result = {key: value for key, value in choice.items() if key != 'model'}
    result['rejection_reasons'] = rejection_reasons(choice['assessment'])
    return result


def fit_domain(rep, source_rows, target_rows, validation, domain):
    common = fit_common(rep, source_rows, target_rows, domain)
    candidates = []
    covariance = dict(name='covariance', method='covariance', strength=P['covariance_strength'], available=True)
    covariance['assessment'] = assess(rep, validation, common, covariance, C.CONFIG['seed'] + domain)
    candidates.append(covariance)
    solver_reports = []
    for entropy in P['entropy']:
        model = fit_ot(common, entropy)
        solver_reports.append({key: value for key, value in model.items() if not isinstance(value, np.ndarray)})
        if not model['available']:
            print(f"[03] OT entropy={entropy}: {model['reason']}; "
                  f"final error={model.get('final_error')}; candidate unavailable", flush=True)
            continue
        for strength in strength_grid():
            choice = dict(name=f'ot_e{entropy:g}_s{strength:g}', method='ot',
                          strength=strength, available=True, model=model)
            choice['assessment'] = assess(rep, validation, common, choice, C.CONFIG['seed'] + domain)
            candidates.append(choice)
    selected = select_candidate(candidates)
    ot_candidates = [choice for choice in candidates if choice['method'] == 'ot']
    ot_choice = select_candidate(ot_candidates)
    return dict(common=common, candidates=candidates, selected=selected, ot_comparator=ot_choice,
                covariance=covariance, solver_reports=solver_reports, validation_rows=validation)


def main():
    start = C.begin(3)
    inputs = C.load_npz(C.DATA / 'inputs.npz')
    rep = C.load_npz(C.DATA / 'representations.npz')
    source_rows = np.flatnonzero((inputs['dataset'] == 0) & (inputs['partition'] == 0) & inputs['eligible'])
    result = {name: rep['z'].copy() for name in ('selected', 'ot', 'covariance')}
    result.update({name + '_supported': np.zeros(len(rep['z']), dtype=bool)
                   for name in ('selected', 'ot', 'covariance')})
    models, reports = ({}, {})
    for domain, name in enumerate(C.NAMES[1:], start=1):
        training = np.flatnonzero((inputs['dataset'] == domain) & (inputs['partition'] == 0) & inputs['eligible'])
        validation = np.flatnonzero((inputs['dataset'] == domain) & (inputs['partition'] == 1) & inputs['eligible'])
        C.require(len(training) >= 100 and len(validation) >= S['minimum_validation_patients'],
                  f'{name}: insufficient fitting/selection patients')
        print(f'[03] {name} -> MIMIC-IV: {len(training):,} training patients', flush=True)
        state = fit_domain(rep, source_rows, training, validation, domain)
        models[name] = state
        for candidate in state['candidates']:
            reasons = rejection_reasons(candidate['assessment'])
            status = 'ACCEPTED' if candidate['assessment']['passed'] else 'REJECTED'
            detail = '; '.join(reasons) if reasons else (
                f"alignment improvement={candidate['assessment']['score']:.2%}")
            print(f"[03] {candidate['name']}: {status}; {detail}", flush=True)
        rows = np.flatnonzero(inputs['dataset'] == domain)
        for method, choice in (('selected', state['selected']), ('ot', state['ot_comparator']),
                               ('covariance', state['covariance'])):
            output, supported = apply_map(
                rep['z'][rows], rep['anchors'][rows], state['common'], choice, inputs['eligible'][rows])
            result[method][rows] = output
            result[method + '_supported'][rows] = supported
        reports[name] = dict(
            selected=state['selected']['name'], ot_comparator=state['ot_comparator']['name'],
            candidates=[candidate_summary(item) for item in state['candidates']],
            solvers=state['solver_reports'], source_fit_rows=source_rows.tolist(),
            target_fit_rows=training.tolist(), validation_rows=validation.tolist(),
            supported_patients=int(result['selected_supported'][rows].sum()))
        print(f"[03] {name}: locked {state['selected']['name']}", flush=True)
    joblib.dump(models, C.OUT / 'transport.joblib', compress=3)
    C.save_npz(C.DATA / 'transport.npz', **result)
    C.finish(
        3, start, [C.OUT / 'transport.joblib', C.DATA / 'transport.npz'], domains=reports,
        target_labels_accessed=False, mapping_policy=mapping_policy(),
        selection_rule=C.CONFIG['protocol']['selection']
        + ' Rank validation alignment improvement after clinical preservation, support and stability gates.')
    print('[03] Complete: maps and validation choices locked; '
          'fitting and selection did not access target mortality', flush=True)


if __name__ == '__main__':
    main()
