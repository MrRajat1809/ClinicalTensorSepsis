"""Render the reference atlas, observed clinical networks and transport supplements."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import visualization_common as V


def density_grid(points, bounds, bins, sigma):
    h, xe, ye = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=bounds)
    h = gaussian_filter(h.T.astype(float), sigma)
    if h.sum() > 0:
        h /= h.sum()
    return h, (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2


def contour_levels(density, masses):
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    return sorted(set(float(ordered[min(np.searchsorted(cumulative, m), len(ordered) - 1)]) for m in masses))


def atlas_maps(run):
    table = run.coordinates()
    V.require(table.patient_key.is_unique, 'Duplicate atlas identities')
    eligible = table[table.eligible]
    cols = ['original_x', 'original_y', 'adapted_x', 'adapted_y']
    V.require(np.isfinite(eligible[cols].to_numpy()).all(), 'Nonfinite eligible coordinates')
    V.require(table.loc[~table.eligible, cols].isna().all().all(), 'Ineligible coordinates are populated')
    reference = eligible[eligible.dataset == 'mimiciv']
    V.require(np.array_equal(reference[['original_x', 'original_y']], reference[['adapted_x', 'adapted_y']]),
        'Reference moved')
    unsupported = eligible[(eligible.dataset != 'mimiciv') & ~eligible.transport_supported]
    V.require(np.array_equal(unsupported[['original_x', 'original_y']], unsupported[['adapted_x', 'adapted_y']]),
        'Unsupported target moved')
    all_points = np.concatenate([eligible[['original_x', 'original_y']], eligible[['adapted_x', 'adapted_y']]])
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum((maximum - minimum) * 0.04, 0.1)
    bounds = list(zip(minimum - padding, maximum + padding))
    ref_density, x, y = density_grid(reference[['original_x', 'original_y']].to_numpy(), bounds,
        run.config['density_grid'], run.config['density_sigma_bins'])
    levels = contour_levels(ref_density, run.config['reference_contour_masses'])
    variance = run.json(V.ROOT / 'outputs/atlas/05.json')['explained_variance_ratio']
    targets = run.config['datasets'][1:]
    V.require(len(targets) == 2, 'Figure 5 expects exactly two target datasets')
    grids = {}
    for name in targets:
        patients = eligible[eligible.dataset == name]
        for view in ('original', 'adapted'):
            grids[name, view] = density_grid(patients[[view + '_x', view + '_y']].to_numpy(), bounds,
                run.config['density_grid'], run.config['density_sigma_bins'])[0]
    peak = max(g.max() for g in grids.values())
    figure_width = 8.4
    left, right, bottom, top = 0.85, 0.12, 0.72, 0.38
    column_gap, row_gap = 0.24, 0.25
    plot_width = (figure_width - left - right - column_gap) / 2
    plot_height = plot_width * np.ptp(bounds[1]) / np.ptp(bounds[0])
    figure_height = bottom + top + 2 * plot_height + row_gap
    fig, axes = plt.subplots(2, 2, figsize=(figure_width, figure_height), sharex=True, sharey=True)
    fig.subplots_adjust(left=left / figure_width, right=1 - right / figure_width, top=1 - top / figure_height,
        bottom=bottom / figure_height, wspace=column_gap / plot_width, hspace=row_gap / plot_height)
    fig.supxlabel(f'Reference PC1 ({variance[0]:.1%})', y=0.30 / figure_height, fontsize=9)
    fig.supylabel(f'Reference PC2 ({variance[1]:.1%})', x=0.04 / figure_width, fontsize=9)
    for col, label in enumerate(('Original', 'Adapted')):
        pos = axes[0, col].get_position()
        fig.text(pos.x0 + pos.width / 2, 1 - 0.13 / figure_height, label, ha='center', va='center', fontsize=10,
            fontweight='normal', color=V.INK)
    from matplotlib.colors import LinearSegmentedColormap
    for row, name in enumerate(targets):
        patients = eligible[eligible.dataset == name]
        color = run.config['dataset_colors'][name]
        shade = LinearSegmentedColormap.from_list(name, ['#FFFFFF', V.tint(color, 0.78), color])
        pos = axes[row, 0].get_position()
        fig.text(0.29 / figure_width, pos.y0 + pos.height / 2, run.config['dataset_labels'][name], ha='center',
            va='center', rotation=90, fontsize=9.2, fontweight='normal', color=V.INK)
        if len(patients) > 5000:
            display_index = np.linspace(0, len(patients) - 1, 5000, dtype=int)
            display_patients = patients.iloc[display_index]
        else:
            display_patients = patients
        for col, view in enumerate(('original', 'adapted')):
            ax = axes[row, col]
            panel = chr(65 + row * 2 + col)
            ax.imshow(grids[name, view], origin='lower', extent=[*bounds[0], *bounds[1]], cmap=shade, vmin=0,
                vmax=peak, interpolation='bilinear', rasterized=True, alpha=0.90)
            ax.scatter(display_patients[view + '_x'], display_patients[view + '_y'], color=color, s=2.6, alpha=0.075,
                edgecolors='none', rasterized=True)
            ax.contour(x, y, ref_density, levels=levels, colors='#555555',
                linewidths=[0.75, 0.85, 1.05][: len(levels)], linestyles=[':', '--', '-'][: len(levels)])
            unchanged = patients[~patients.transport_supported]
            ax.scatter(unchanged[view + '_x'], unchanged[view + '_y'], facecolors='none', edgecolors='#444444', s=16,
                linewidths=0.65, rasterized=True, zorder=4)
            ax.set(xlim=bounds[0], ylim=bounds[1], aspect='equal', xlabel='', ylabel='')
            V.clean_axis(ax, grid=None)
            ax.tick_params(axis='x', labelbottom=row == 1, labelsize=7.5)
            ax.tick_params(axis='y', labelleft=col == 0, labelsize=7.5)
            ax.text(0, 1.015, f'{panel}.', transform=ax.transAxes, ha='left', va='bottom', fontsize=9.2,
                fontweight='normal', color=V.INK)
            ax.text(0.025, 0.965, f'n = {len(patients):,}', transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
                fontweight='normal', color=V.INK)
            run.add(5, panel, source='data/processed/atlas/coordinates.parquet', dataset=name,
                metric='coordinate_summary', view=view, n_patients=len(patients),
                n_unsupported=int((~patients.transport_supported).sum()), xmin=bounds[0][0], xmax=bounds[0][1],
                ymin=bounds[1][0], ymax=bounds[1][1], density_peak=peak, median_x=patients[view + '_x'].median(),
                median_y=patients[view + '_y'].median())
    fig.legend([Line2D([], [], color='#555555', ls='-', lw=1.05), Line2D([], [], color='#555555', ls='--', lw=0.85),
            Line2D([], [], color='#555555', ls=':', lw=0.75),
            Line2D([], [], color='#444444', marker='o', markerfacecolor='none', ls='')
        ], ['Reference 50%', 'Reference 80%', 'Reference 95%', 'Unsupported target'], loc='lower center',
        bbox_to_anchor=(0.5, 0.015 / figure_height), ncol=4, frameon=False, fontsize=8, handlelength=2.2,
        columnspacing=1.35)
    run.audit.append(dict(check='shared_reference_projection', patients=len(table), eligible=len(eligible),
            ineligible=int((~table.eligible).sum()), unsupported=len(unsupported), reference_unchanged=True,
            unsupported_unchanged=True, common_axes=True, complete_extent=True,
            density=('within_dataset_probability_per_common_grid_cell'), explained_variance=variance))
    run.save(fig, 5)


def draw_transport_effect(run, ax, rows):
    names = run.config['datasets'][1:]
    for i, name in enumerate(names):
        row = V.one(rows, dataset=name)
        V.interval_point(ax, i, row, run.config['dataset_colors'][name], run.config['dataset_markers'][name])
    ax.axvline(0, color='#777777', lw=0.7, ls='--')
    ax.set_ylim(1.6, -0.6)
    ax.set_yticks(
        range(2), [run.config['dataset_labels'][n] + f' (n={int(V.one(rows, dataset=n).n_patients):,})' for n in names]
    )
    V.clean_axis(ax, categorical='y')
    ax.figure.legend([Line2D([], [], marker='o', ls='', color=V.MUTED, markerfacecolor='white')],
        ['Interval unavailable'], loc='outside lower center')


def reference_distance(run, effects, canvas=None):
    key = 'supp_reference_distance'
    fig, ax = V.panel_canvas(canvas, figsize=(6.5, 3.6))
    rows = V.select(effects, metric='reference_distance_difference', scenario='', stratum='all_eligible')
    draw_transport_effect(run, ax, rows)
    ax.set_xlabel('Squared standardized distance')
    V.title(ax, 'Compatible-reference distance')
    run.add(6, key, rows, 'transport_effects')
    run.save(fig, 6, key)


def point_instability(run, effects, canvas=None):
    key = 'supp_point_instability'
    fig, ax = V.panel_canvas(canvas, figsize=(6.5, 3.6))
    rows = V.select(effects, metric='embedding_instability_difference', scenario='point')
    draw_transport_effect(run, ax, rows)
    ax.set_xlabel('Standardized latent L2')
    V.title(ax, 'Point-removal instability')
    run.add(6, key, rows, 'transport_effects')
    run.save(fig, 6, key)


def block_instability(run, effects, canvas=None):
    key = 'supp_block_instability'
    fig, ax = V.panel_canvas(canvas, figsize=(6.5, 3.6))
    rows = V.select(effects, metric='embedding_instability_difference', scenario='block')
    draw_transport_effect(run, ax, rows)
    ax.set_xlabel('Standardized latent L2')
    V.title(ax, 'Block-removal instability')
    run.add(6, key, rows, 'transport_effects')
    run.save(fig, 6, key)


def clinical_preservation(run, preservation, name, canvas=None):
    key = f'supp_clinical_preservation_{name}'
    fig, ax = V.panel_canvas(canvas, figsize=(8, 7.5))
    probes = [f + '/' + t for f in run.config['clinical_features'] for t in ('early', 'change')]
    rows = V.select(preservation, dataset=name, metric='engineering_constraint_residual')
    for i, probe in enumerate(probes):
        for stratum, marker, offset in (('all_measured', 'o', -0.15), ('measured_tail', '^', 0.15)):
            subset = V.select(rows, feature=probe, stratum=stratum)
            if subset.empty:
                continue
            row = V.one(subset)
            V.interval_point(ax, i, row, run.config['dataset_colors'][name], marker, offset)
    ax.axvline(0, color='#777777', lw=0.8, ls='--')
    ax.set_yticks(range(len(probes)), [V.label(p.split('/')[0]) + ' · ' + p.split('/')[1] for p in probes])
    ax.set_ylim(len(probes) - 0.4, -0.6)
    ax.set_xlabel('Error − allowed error')
    V.clean_axis(ax, categorical='y')
    V.title(ax, 'Probe constraint · ' + run.config['dataset_labels'][name])
    run.add(6, key, rows, 'clinical_preservation')
    fig.legend([Line2D([], [], marker=m, color='#555555', ls='', markerfacecolor=fill)
            for m, fill in [('o', '#555555'), ('^', '#555555'), ('o', 'white')]],
        ['All measured', 'Measured tails', 'Interval unavailable'], loc='outside lower center', ncol=1, fontsize=7,
        labelspacing=0.3)
    run.save(fig, 6, key)


def patient_geometry(run, preservation, canvas=None):
    key = 'supp_patient_geometry'
    fig, ax = V.panel_canvas(canvas, figsize=(8, 3.6))
    names = run.config['datasets'][1:]
    rows = V.select(preservation, metric='pair_distance_ratio_quantile')
    for i, name in enumerate(names):
        values = [V.one(rows, dataset=name, quantile=q).estimate for q in [0.1, 0.5, 0.9]]
        color = run.config['dataset_colors'][name]
        ax.plot([values[0], values[2]], [i] * 2, color=V.tint(color, 0.35), lw=1.5, marker='|', markersize=6,
            markeredgewidth=1)
        ax.plot(values[1], i, run.config['dataset_markers'][name], color=color, markersize=6, markeredgecolor='white',
            markeredgewidth=0.9)
    ax.axvline(1, color='#777777', ls='--', lw=0.8)
    ax.set_ylim(1.6, -0.6)
    ax.set_yticks(range(2), [run.config['dataset_labels'][n] for n in names])
    ax.set_xlabel('Adapted / original distance')
    ax.legend([Line2D([], [], color=V.MUTED, marker='o')], ['Median [10th–90th percentile]'], frameon=False,
        fontsize=7, loc='lower center')
    V.clean_axis(ax, categorical='y')
    V.title(ax, 'Patient geometry')
    run.add(6, key, rows, 'clinical_preservation')
    run.save(fig, 6, key)


def dataset_handles(run):
    return [Line2D([], [], ls='', marker=run.config['dataset_markers'][name], color=run.config['dataset_colors'][name],
            markerfacecolor=run.config['dataset_colors'][name], markeredgecolor='white', markeredgewidth=0.7,
            markersize=6.3, label=run.config['dataset_labels'][name]
        ) for name in run.config['datasets'][1:]]


def ci_handle():
    return Line2D([], [], color=V.MUTED, alpha=0.24, lw=6, solid_capstyle='round', label='95% CI')


def transport_summary(run, effects, canvas=None, panel_letter=None):
    key = 'supp_transport_summary'
    names = run.config['datasets'][1:]
    specifications = [('reference_distance_difference', '', 'Reference distance', 'Δ squared distance'),
        ('embedding_instability_difference', 'point', 'Point removal', 'Δ latent L2'),
        ('embedding_instability_difference', 'block', 'Block removal', 'Δ latent L2')]
    fig = canvas if canvas is not None else plt.figure(figsize=(3.1, 5.3), layout='constrained')
    axes = fig.subplots(3, 1, sharex=True)
    title = 'Adapted − original'
    if panel_letter is not None:
        title = f'{panel_letter}.  {title}'
    fig.suptitle(title, x=0.03, ha='left', fontsize=9, fontweight='normal')
    for ax, (metric, scenario, label, unit) in zip(axes, specifications):
        rows = V.select(effects, metric=metric, scenario=scenario)
        if metric == 'reference_distance_difference':
            rows = V.select(rows, stratum='all_eligible')
        for x, name in enumerate(names):
            row = V.one(rows, dataset=name)
            V.require(np.isfinite(row.estimate), 'Missing transport estimate')
            color = run.config['dataset_colors'][name]
            available = np.isfinite([row.ci_low, row.ci_high]).all()
            ax.bar(x, row.estimate, width=0.48, color=V.tint(color, 0.35) if available else 'white', edgecolor=color,
                linewidth=0.9, zorder=2)
            if available:
                V.require(row.ci_low <= row.ci_high, 'Reversed interval')
                ax.vlines(x, row.ci_low, row.ci_high, color=V.INK, lw=0.9, zorder=3)
                ax.hlines([row.ci_low, row.ci_high], x - 0.08, x + 0.08, color=V.INK, lw=0.9, zorder=3)
            else:
                ax.text(x, 0.04, 'CI unavailable', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                    fontsize=6.5)
        ax.axhline(0, color='#777777', lw=0.7)
        ax.set_xlim(-0.6, 1.6)
        ax.margins(y=0.14)
        ax.set_ylabel(unit, fontsize=7.5, labelpad=4)
        ax.set_title(label, loc='left', fontsize=8, pad=4, fontweight='normal')
        ax.tick_params(axis='both', labelsize=7)
        ax.locator_params(axis='y', nbins=3)
        V.clean_axis(ax, grid='y', categorical='x')
        run.add(6, key, rows, 'transport_effects')
    axes[-1].set_xticks(range(2), [run.config['dataset_labels'][n] for n in names])
    fig.legend([Line2D([], [], color=V.INK, marker='_', lw=0.9)], ['95% CI'], loc='outside lower center', fontsize=7,
        borderaxespad=0.2)
    run.save(fig, 6, key)


def preservation_matrix(run, preservation, canvas=None, panel_letter=None):
    key = 'supp_preservation_matrix'
    probes = [(feature, period) for feature in run.config['clinical_features'] for period in ('early', 'change')]
    names = run.config['datasets'][1:]
    columns = [(name, stratum) for name in names for stratum in ('all_measured', 'measured_tail')]
    ratios = np.full((len(probes), len(columns)), np.nan)
    for i, (feature, period) in enumerate(probes):
        for j, (name, stratum) in enumerate(columns):
            probe = f'{feature}/{period}'
            row = V.one(
                preservation, dataset=name, feature=probe, stratum=stratum, metric='engineering_constraint_residual')
            comparison = V.one(
                preservation, dataset=name, feature=probe, stratum=stratum, metric='probe_mae_difference')
            permitted = float(row.error_ratio_limit) * float(comparison.original_mae) + float(row.absolute_slack)
            V.require(np.isfinite(permitted) and permitted > 0, 'Invalid permitted probe error')
            ratio = float(comparison.adapted_mae) / permitted
            V.require(np.isfinite(ratio) and np.isclose(ratio, 1 + row.estimate / permitted),
                'Probe ratio disagrees with saved constraint')
            ratios[i, j] = ratio
            run.add(6, key, row, 'clinical_preservation', display_error_ratio=ratio, permitted_error=permitted,
                probe_comparison_id=comparison.result_id)
            run.add(6, key, comparison, 'clinical_preservation')
    span = max(0.2, float(np.ceil(np.abs(ratios - 1).max() * 10) / 10))
    fig, ax = V.panel_canvas(canvas, figsize=(4.1, 5.3))
    image = ax.imshow(ratios, cmap=V.cmap(True), vmin=1 - span, vmax=1 + span, aspect='auto', interpolation='nearest',
        rasterized=True)
    for (i, j), value in np.ndenumerate(ratios):
        label = f'{value:.3f}' if 0 < abs(value - 1) < 0.005 else f'{value:.2f}'
        ax.text(j, i, label, ha='center', va='center', fontsize=6.7,
            color='white' if abs(value - 1) > 0.65 * span else V.INK)
    ax.set_yticks(range(len(probes)), [f'{V.label(f)} · {period}' for f, period in probes], fontsize=7)
    ax.set_xticks(range(4), ['Overall', 'Tails'] * 2, fontsize=7)
    V.heatmap_axis(ax)
    ax.axvline(1.5, color='white', lw=2)
    for y in np.arange(1.5, len(probes) - 0.5, 2):
        ax.axhline(y, color='white', lw=0.6)
    for j, name in enumerate(names):
        ax.text(j * 2 + 0.5, 1.012, run.config['dataset_labels'][name], transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8)
    title = 'Probe preservation'
    if panel_letter is not None:
        title = f'{panel_letter}.  {title}'
    ax.set_title(title, loc='left', fontsize=9, pad=22, fontweight='normal')
    bar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025, shrink=0.7, ticks=[1 - span, 1, 1 + span])
    V.colorbar(bar)
    bar.ax.tick_params(labelsize=7)
    bar.set_label('Error / allowed error', fontsize=7.5, labelpad=4)
    run.save(fig, 6, key)


def temporal_agreement(run, preservation, canvas=None, panel_letter=None, show_legend=True):
    key = 'supp_temporal_agreement'
    fig, ax = V.panel_canvas(canvas, figsize=(8.2, 4.8))
    features = run.config['clinical_features']
    names = run.config['datasets'][1:]
    minimum = 1.0
    band_width = 0.13
    for j, name in enumerate(names):
        rows = V.select(preservation, dataset=name, metric=('predicted_change_sign_retention'))
        rows = rows.set_index('feature').reindex([f + '/change' for f in features]).reset_index()
        V.require(np.isfinite(rows.estimate).all(), 'Missing temporal agreement')
        x = np.arange(len(features)) + (j - 0.5) * 0.20
        color = run.config['dataset_colors'][name]
        marker = run.config['dataset_markers'][name]
        for position, row in zip(x, rows.itertuples()):
            available = np.isfinite([row.ci_low, row.ci_high]).all()
            if available:
                V.require(row.ci_low <= row.ci_high, 'Reversed interval')
                minimum = min(minimum, float(row.ci_low))
                ax.fill_between([position - band_width / 2, position + band_width / 2], [row.ci_low, row.ci_low],
                    [row.ci_high, row.ci_high], color=color, alpha=0.20, linewidth=0, zorder=2)
            else:
                minimum = min(minimum, float(row.estimate))
            ax.plot(position, row.estimate, marker=marker, ls='', color=color,
                markerfacecolor=(color if available else 'white'), markeredgecolor=('white' if available else color),
                markeredgewidth=0.8, markersize=6.3, zorder=3)
        run.add(6, key, rows, 'clinical_preservation')
    lower = max(0, np.floor((minimum - 0.025) / 0.05) * 0.05)
    ax.set_xticks(range(len(features)), [V.label(f) for f in features], fontsize=8)
    ax.set(xlim=(-0.55, len(features) - 0.45), ylim=(lower, 1.008), ylabel='Agreement fraction')
    ax.set_yticks(np.arange(lower, 1.001, 0.05))
    V.clean_axis(ax, grid='y', categorical='x', percent=True)
    ax.axhline(1.0, color='#777777', lw=0.8, alpha=0.55, zorder=1)
    title = 'Predicted direction agreement'
    if panel_letter is not None:
        title = f'{panel_letter}.  ' 'Predicted direction agreement'
    ax.set_title(title, loc='left', color=V.INK, fontweight='normal', fontsize=10.5, pad=10)
    if show_legend:
        ax.legend(handles=(dataset_handles(run) + [ci_handle()]), loc='lower left', frameon=False, ncol=3, fontsize=8,
            handletextpad=0.45, columnspacing=1.1)
    run.save(fig, 6, key)


def clinical_relationships(run, name):
    settings = run.config['clinical_network']
    features = [f for group in settings['groups'] for f in group['features']]
    V.require(len(features) == len(set(features)) == 50 and set(features) == set(run.features),
        'Network groups must include all 50 temporal features exactly once')
    minimum, threshold = settings['minimum_pair_patients'], settings['minimum_absolute_rho']
    V.require(minimum >= 3 and 0 <= threshold <= 1, 'Invalid network display settings')
    pairs = V.select(run.plot_table('feature_relationships'), dataset=name)
    nodes = (V.select(run.plot_table('feature_observations'), dataset=name) .set_index('feature') .reindex(features)
        .reset_index())
    V.require(len(pairs) == 1225 and nodes.numerator.notna().all(), 'Incomplete all-feature network')
    edges = [dict(row, displayed=bool(
                np.isfinite(row['estimate']) and row['n_patients'] >= minimum and abs(row['estimate']) >= threshold)
        ) for row in pairs.to_dict('records')]
    run.audit.append(dict(check='clinical_network', dataset=name, features=50, pairs=1225, observed_only=True,
            correlation='pairwise_spearman_average_ties', minimum_pair_patients=minimum,
            display_absolute_rho=threshold, all_features_retained=True, inferential_tests=False))
    return dict(features=features, edges=edges, nodes=nodes, patients=int(nodes.n_patients.iloc[0]),
        source='relationship_plot_data/feature_relationships')


def network_legend(fig, run, ncol=4, assembled=False):
    groups = run.config['clinical_network']['groups']
    handles = [Line2D([], [], marker='o', ls='', color=g['color'], markersize=4, label=g['label']) for g in groups]
    handles += [Line2D([], [], color='#356D89', lw=1.8, alpha=0.65, label='Positive ρ'),
        Line2D([], [], color='#AE5965', lw=1.8, alpha=0.65, label='Negative ρ'), Line2D(
            [], [], marker='o', ls='', color=V.MUTED, markerfacecolor='white', markersize=4, label='No displayed links'
        ), Line2D([], [], marker='s', ls='', color=V.MUTED, markersize=4, label='Derived feature')]
    fig.legend(handles=handles, loc='lower center' if assembled else 'outside lower center', ncol=ncol, fontsize=6.5,
        columnspacing=1, handlelength=1.3, labelspacing=0.25, borderaxespad=0.2)


def clinical_network(run, name, data, canvas=None, show_legend=True, ax=None):
    key = f'clinical_network_{name}'
    if ax is None:
        fig, ax = V.panel_canvas(canvas, figsize=(4.2, 4.5))
    else:
        fig = ax.figure
    settings = run.config['clinical_network']
    features = data['features']
    short = {'hr': 'HR', 'map': 'MAP', 'rr': 'RR', 'spo2': 'SpO₂', 'pao2': 'PaO₂', 'paco2': 'PaCO₂',
        'creatinine': 'Cr', 'bun': 'BUN', 'sodium': 'Na', 'potassium': 'K', 'bicarbonate': 'HCO₃', 'bilirubin': 'Bili',
        'alt': 'ALT', 'albumin': 'Alb', 'platelets': 'Plt', 'wbc': 'WBC', 'hemoglobin': 'Hb', 'lactate': 'Lac',
        'neq': 'NEQ', 'fio2': 'FiO₂', 'pf_ratio': 'P/F', 'vent': 'Vent', 'gcs_eye': 'GCS-E', 'gcs_verbal': 'GCS-V',
        'gcs_motor': 'GCS-M', 'temp_c': 'Temp', 'urine_output': 'Urine', 'chloride': 'Cl', 'anion_gap': 'AG',
        'ph': 'pH', 'glucose': 'Glu', 'calcium_total': 'Ca', 'magnesium': 'Mg', 'phosphate': 'Phos', 'ast': 'AST',
        'alp': 'ALP', 'hematocrit': 'Hct', 'rbc': 'RBC', 'mch': 'MCH', 'mchc': 'MCHC', 'mcv': 'MCV', 'rdw_cv': 'RDW',
        'lymphocytes_pct': 'Lymph', 'monocytes_pct': 'Mono', 'neutrophils_pct': 'Neut', 'basophils_pct': 'Baso',
        'eosinophils_pct': 'Eos', 'pt': 'PT', 'aptt': 'aPTT', 'inr': 'INR'}
    gap = np.deg2rad(3)
    step = (2 * np.pi - len(settings['groups']) * gap) / len(features)
    angle, positions, theta_by_feature, node_colors = np.pi / 2, {}, {}, {}
    for group in settings['groups']:
        first = angle
        for feature in group['features']:
            positions[feature] = 0.88 * np.array([np.cos(angle), np.sin(angle)])
            theta_by_feature[feature] = angle
            node_colors[feature] = group['color']
            angle -= step
        ax.add_patch(Arc((0, 0), 1.9, 1.9, theta1=np.rad2deg(angle + step / 2), theta2=np.rad2deg(first + step / 2),
                edgecolor=group['color'], lw=2, alpha=0.75))
        angle -= gap
    visible = sorted((e for e in data['edges'] if e['displayed']), key=lambda e: abs(e['estimate']))
    for edge in visible:
        start, finish = positions[edge['feature']], positions[edge['other_feature']]
        strength = abs(edge['estimate'])
        curve = Path([start, start * 0.28, finish * 0.28, finish], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        )
        ax.add_patch(PathPatch(curve, facecolor='none', edgecolor='#356D89' if edge['estimate'] >= 0 else '#AE5965',
                lw=0.12 + 1.7 * strength**2, alpha=0.025 + 0.50 * strength**2, zorder=1))
    linked = {feature for edge in visible for feature in (edge['feature'], edge['other_feature'])}
    for j, feature in enumerate(features):
        xy, theta = positions[feature], theta_by_feature[feature]
        node = data['nodes'].iloc[j]
        derived = str(node.derived).lower() == 'true'
        ax.scatter(*xy, s=12, color=node_colors[feature] if feature in linked else 'white',
            marker='s' if derived else 'o', edgecolor=node_colors[feature], lw=0.6, zorder=3)
        degrees = np.rad2deg(theta)
        on_right = np.cos(theta) >= 0
        label_xy = 1.03 * np.array([np.cos(theta), np.sin(theta)])
        ax.text(*label_xy, short.get(feature, V.label(feature)), rotation=degrees if on_right else degrees + 180,
            rotation_mode='anchor', ha='left' if on_right else 'right', va='center', fontsize=6, color=V.INK)
        run.add(6, key, node, 'relationship_plot_data/feature_observations',
            node_label=short.get(feature, V.label(feature)))
    for edge in data['edges']:
        run.add(6, key, edge, data['source'], minimum_pair_patients=settings['minimum_pair_patients'],
            display_absolute_rho=settings['minimum_absolute_rho'])
    ax.set(xlim=(-1.37, 1.37), ylim=(-1.37, 1.37), aspect='equal')
    ax.set_axis_off()
    ax.set_title(run.config['dataset_labels'][name], fontsize=9, pad=3, color=run.config['dataset_colors'][name],
        fontweight='normal')
    if show_legend:
        network_legend(fig, run, ncol=2)
    run.save(fig, 6, key)


def relationship_concordance(run, networks, canvas=None, ax=None):
    key = 'relationship_concordance'
    if ax is None:
        fig, ax = V.panel_canvas(canvas, figsize=(3.6, 3.6))
    else:
        fig = ax.figure
    minimum = run.config['clinical_network']['minimum_pair_patients']
    reference = {(r['feature'], r['other_feature']): r for r in networks['mimiciv']['edges']}
    for name in run.config['datasets'][1:]:
        x, y = [], []
        for row in networks[name]['edges']:
            ref = reference[row['feature'], row['other_feature']]
            if (min(row['n_patients'], ref['n_patients']) >= minimum
                and np.isfinite([row['estimate'], ref['estimate']]).all()):
                x.append(ref['estimate'])
                y.append(row['estimate'])
            for original in (ref, row):
                run.add(6, key, original, networks[name]['source'], comparison_dataset=name)
        ax.scatter(x, y, s=8, alpha=0.25, edgecolors='none', rasterized=True, color=run.config['dataset_colors'][name],
            label=run.config['dataset_labels'][name])
    ax.plot([-1, 1], [-1, 1], color='#777777', lw=0.8, ls='--')
    ax.axhline(0, color='#DDDDDD', lw=0.5)
    ax.axvline(0, color='#DDDDDD', lw=0.5)
    ax.set(xlim=(-1.04, 1.04), ylim=(-1.04, 1.04), aspect='equal', xlabel='Spearman ρ · MIMIC-IV',
        ylabel='Spearman ρ · comparison cohort')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.set_title('Feature-pair concordance', loc='left', fontsize=9, fontweight='normal', pad=6)
    ax.legend(loc='upper left', fontsize=7, markerscale=1.8, handletextpad=0.3)
    run.save(fig, 6, key)


def assembled_relationships(run, networks):
    width, height = 7.0, 7.0
    fig = plt.figure(figsize=(width, height), layout='none')
    boxes = [(0.10, 3.75, 3.00, 3.00), (3.80, 3.75, 3.00, 3.00), (0.10, 0.60, 3.00, 3.00), (4.08, 0.88, 2.72, 2.72)]
    axes = [fig.add_axes([x / width, y / height, w / width, h / height]) for x, y, w, h in boxes]
    network_legend(fig, run, assembled=True)
    renderers = {f'clinical_network_{name}': lambda n=name, c=i: clinical_network(
            run, n, networks[n], ax=axes[c], show_legend=False) for i, name in enumerate(run.config['datasets'])}
    renderers['relationship_concordance'] = lambda: relationship_concordance(run, networks, ax=axes[3])
    run.assemble(fig, 6, renderers)


def assembled_transport_supplement(run, effects, preservation):
    names = run.config['datasets'][1:]
    fig = plt.figure(figsize=(10.8, 15.5), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.04, wspace=0.025, hspace=0.03)
    grid = fig.add_gridspec(4, 6, height_ratios=[2.1, 5.2, 2.9, 5.3])
    renderers = {
        'supp_reference_distance': lambda: reference_distance(run, effects, canvas=fig.add_subfigure(grid[0, :2])),
        'supp_point_instability': lambda: point_instability(run, effects, canvas=fig.add_subfigure(grid[0, 2:4])),
        'supp_block_instability': lambda: block_instability(run, effects, canvas=fig.add_subfigure(grid[0, 4:]))}
    for col, name in enumerate(names):
        renderers[f'supp_clinical_preservation_{name}'] = lambda c=col, n=name: clinical_preservation(
            run, preservation, n, canvas=fig.add_subfigure(grid[1, c * 3 : (c + 1) * 3]))
    renderers['supp_patient_geometry'] = lambda: patient_geometry(
        run, preservation, canvas=fig.add_subfigure(grid[2, :2]))
    renderers['supp_temporal_agreement'] = lambda: temporal_agreement(
        run, preservation, canvas=fig.add_subfigure(grid[2, 2:]))
    renderers['supp_preservation_matrix'] = lambda: preservation_matrix(
        run, preservation, canvas=fig.add_subfigure(grid[3, :3]))
    renderers['supp_transport_summary'] = lambda: transport_summary(run, effects, canvas=fig.add_subfigure(grid[3, 3:])
    )
    run.assemble(fig, 6, renderers, supplementary=True)


def dataset_relationships(run):
    networks = {name: clinical_relationships(run, name) for name in run.config['datasets']}
    run.plan(6, [*[f'clinical_network_{name}' for name in run.config['datasets']], 'relationship_concordance'])
    for name in run.config['datasets']:
        clinical_network(run, name, networks[name])
    relationship_concordance(run, networks)
    assembled_relationships(run, networks)
    effects = run.table('transport_effects')
    preservation = run.table('clinical_preservation')
    run.plan(6, ['supp_reference_distance', 'supp_point_instability', 'supp_block_instability',
            *[f'supp_clinical_preservation_{n}' for n in run.config['datasets'][1:]], 'supp_patient_geometry',
            'supp_temporal_agreement', 'supp_preservation_matrix', 'supp_transport_summary'])
    reference_distance(run, effects)
    point_instability(run, effects)
    block_instability(run, effects)
    for name in run.config['datasets'][1:]:
        clinical_preservation(run, preservation, name)
    patient_geometry(run, preservation)
    temporal_agreement(run, preservation)
    preservation_matrix(run, preservation)
    transport_summary(run, effects)
    assembled_transport_supplement(run, effects, preservation)
    run.audit.append(dict(check='transport_intervals', pointwise=True, preservation_comparisons=64,
            diagnostics='supplementary', main_panels=4, main_question='observed_clinical_relationships',
            geometry_intervals='descriptive_quantiles_not_confidence_intervals'))


def main():
    run = V.Run(4)
    atlas_maps(run)
    dataset_relationships(run)
    run.finish()


if __name__ == '__main__':
    main()
