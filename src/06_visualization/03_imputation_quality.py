"""Render reconstruction comparisons and clinical-unit MAE intervals."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import visualization_common as V

SCENARIOS = {'point': 'Point', 'block6h': '6-h block', 'whole_channel': 'Channel'}


def relative_mae(final, baseline):
    return (final - baseline) / baseline if np.isfinite(baseline) and baseline > 0 else np.nan


def comparison_matrices(run, means, features, columns):
    matrices = {}
    for comparator in run.config['comparators']:
        matrix = np.full((len(features), len(columns)), np.nan)
        for i, feature in enumerate(features):
            for j, (name, scenario) in enumerate(columns):
                final = V.select(means, dataset=name, feature=feature, scenario=scenario, method='final')
                base = V.select(means, dataset=name, feature=feature, scenario=scenario, method=comparator)
                if len(final) == len(base) == 1:
                    V.require(
                        final.iloc[0].n_patients == base.iloc[0].n_patients, 'Different reconstruction denominators')
                    matrix[i, j] = relative_mae(final.iloc[0].estimate, base.iloc[0].estimate)
                    run.add(4, f'relative_mae_{comparator}', source='imputation_comparisons', dataset=name,
                        feature=feature, scenario=scenario, comparator=comparator,
                        metric='relative_patient_mean_mae_change', estimate=matrix[i, j],
                        final_mae=final.iloc[0].estimate, baseline_mae=base.iloc[0].estimate,
                        final_result_id=final.iloc[0].result_id, baseline_result_id=base.iloc[0].result_id,
                        n_patients=final.iloc[0].n_patients, unit='fraction')
        matrices[comparator] = matrix
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
    return matrices, max(0.05, float(np.abs(finite).max()))


def annotation_handles():
    return [Line2D(
            [], [], marker='o', ls='', color='#555555', markerfacecolor='white', markeredgewidth=0.7, markersize=4.1),
        Line2D([], [], marker='+', ls='', color='#555555', markeredgewidth=0.8, markersize=5),
        Line2D([], [], marker='x', ls='', color='#555555', markeredgewidth=0.8, markersize=4.5),
        Line2D([], [], marker=r'$\dagger$', ls='', color='#555555', markersize=5.8)]


def heatmap_panel(run, matrix, limit, features, columns, comparator, primary, flagged, canvas=None, ax=None,
    show_ylabels=True, show_colorbar=True, show_legend=True, panel_letter=None):
    key = f'relative_mae_{comparator}'
    if ax is None:
        fig, ax = V.panel_canvas(canvas, figsize=(8.8, 9.2))
    else:
        fig = ax.figure
    image = ax.imshow(
        matrix, cmap=V.cmap(True), vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest', rasterized=True)
    ax.set_yticks(range(len(features)))
    if show_ylabels:
        ax.set_yticklabels([V.label(f) + (' †' if f in flagged else '') for f in features], fontsize=7)
        ax.tick_params(axis='y', length=0, pad=4)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    ax.set_xticks(range(len(columns)), [SCENARIOS[s] for _, s in columns], fontsize=7.2)
    ax.tick_params(axis='x', length=0, pad=4)
    for center, name in zip((1, 4, 7), run.config['datasets']):
        ax.text(center, 1.008, run.config['dataset_labels'][name], transform=ax.get_xaxis_transform(), ha='center',
            va='bottom', fontsize=8.1, color=run.config['dataset_colors'][name])
    for x in (2.5, 5.5):
        ax.axvline(x, color='white', linewidth=2.4, zorder=3)
    missing_y, missing_x = np.where(~np.isfinite(matrix))
    ax.scatter(missing_x, missing_y, marker='x', color='#666666', s=11, linewidths=0.7, zorder=4)
    intervals = V.select(primary, comparator=comparator, feature=features)
    for i, feature in enumerate(features):
        for j, (name, scenario) in enumerate(columns):
            if not np.isfinite(matrix[i, j]):
                continue
            rows = V.select(intervals, dataset=name, feature=feature, scenario=scenario)
            V.require(len(rows) <= 1, 'Duplicate paired imputation interval')
            row = rows.iloc[0] if len(rows) else None
            if row is None or not np.isfinite([row.ci_low, row.ci_high]).all():
                ax.plot(j, i, '+', color='#555555', markersize=4, markeredgewidth=0.75, zorder=5)
            elif row.ci_low <= 0 <= row.ci_high:
                ax.plot(j, i, 'o', markerfacecolor='white', markeredgecolor='#555555', markersize=3.15,
                    markeredgewidth=0.65, zorder=5)
    V.heatmap_axis(ax)
    title = 'Reconstruction versus ' + comparator.replace('_', ' ')
    if panel_letter:
        title = f'{panel_letter}.  {title}'
    ax.set_title(title, loc='left', color=V.INK, fontweight='normal', fontsize=10.5, pad=18)
    if show_colorbar:
        bar = fig.colorbar(image, ax=ax, shrink=0.74, fraction=0.032, pad=0.018,
            label='Relative MAE change\n' '(final − comparator) / comparator')
        V.colorbar(bar, percent=True)
    if show_legend:
        fig.legend(annotation_handles(),
            ['95% CI includes 0', 'CI unavailable', 'Comparison unavailable', 'Feature review flag'],
            loc='outside lower center', ncol=4, frameon=False, fontsize=8, handletextpad=0.45, columnspacing=1.25)
    run.add(4, key, intervals, 'imputation_comparisons')
    run.save(fig, 4, key)


def clinical_mae_intervals(run, primary, columns, feature, canvas=None):
    key = f'supp_mae_intervals_{feature}'
    fig, ax = V.panel_canvas(canvas, figsize=(6.5, 4.8))
    rows = V.select(primary, feature=feature)
    for j, (name, scenario) in enumerate(columns):
        for comparator, marker, offset in (('median', 'o', -0.14), ('forward_fill', '^', 0.14)):
            subset = V.select(rows, dataset=name, scenario=scenario, comparator=comparator)
            if not subset.empty:
                V.interval_point(ax, j, V.one(subset), run.config['dataset_colors'][name], marker, offset)
    ax.axvline(0, color='#777777', lw=0.7, ls='--')
    ax.set_yticks(range(len(columns)),
        [run.config['dataset_labels'][name] + ' · ' + SCENARIOS[scenario] for name, scenario in columns], fontsize=8)
    ax.set_ylim(len(columns) - 0.4, -0.6)
    ax.set_xlabel(f'MAE difference ({run.units[feature]})')
    V.clean_axis(ax, categorical='y')
    V.title(ax, V.label(feature))
    fig.legend([Line2D([], [], marker='o', ls='', color='#555555'), Line2D([], [], marker='^', ls='', color='#555555'),
            Line2D([], [], marker='o', ls='', color='#555555', markerfacecolor='white')
        ], ['Versus median', 'Versus forward fill', 'Interval unavailable'], loc='outside lower center', ncol=3,
        frameon=False)
    run.add(4, key, rows, 'imputation_comparisons')
    run.save(fig, 4, key)


def assembled_imputation(run, matrices, limit, features, columns, primary, flagged):
    fig = plt.figure(figsize=(15.2, 7.75), facecolor='white')
    left_ax = fig.add_axes([0.080, 0.105, 0.392, 0.790])
    right_ax = fig.add_axes([0.505, 0.105, 0.392, 0.790])
    cax = fig.add_axes([0.920, 0.245, 0.012, 0.515])
    mappable = ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap=V.cmap(True))
    mappable.set_array([])
    bar = fig.colorbar(mappable, cax=cax)
    V.colorbar(bar, percent=True)
    bar.set_label('Relative MAE change\n' '(final − comparator) / comparator', fontsize=8, labelpad=5)
    cax.tick_params(labelsize=7.3)
    fig.legend(annotation_handles(),
        ['95% CI includes 0', 'CI unavailable', 'Comparison unavailable', 'Feature review flag'], loc='lower center',
        bbox_to_anchor=(0.5, 0.018), ncol=4, frameon=False, fontsize=7.7, handletextpad=0.4, columnspacing=1.15,
        borderaxespad=0)
    run.assemble(fig, 4, {'relative_mae_median': lambda: heatmap_panel(run, matrices['median'], limit, features,
                columns, 'median', primary, flagged, ax=left_ax, show_ylabels=True, show_colorbar=False,
                show_legend=False, panel_letter='A'
            ), 'relative_mae_forward_fill': lambda: heatmap_panel(run, matrices['forward_fill'], limit, features,
                columns, 'forward_fill', primary, flagged, ax=right_ax, show_ylabels=False, show_colorbar=False,
                show_legend=False, panel_letter='B')})


def assembled_supplement(run, primary, columns, features):
    ncols = min(3, len(features))
    nrows = (len(features) + ncols - 1) // ncols
    fig = plt.figure(figsize=(6.5 * ncols, 4.8 * nrows), layout='constrained')
    grid = fig.add_gridspec(nrows, ncols)
    renderers = {f'supp_mae_intervals_{feature}': lambda feature=feature, i=i: clinical_mae_intervals(
            run, primary, columns, feature, canvas=fig.add_subfigure(grid[i // ncols, i % ncols]))
        for i, feature in enumerate(features)}
    run.assemble(fig, 4, renderers, supplementary=True)


def main():
    run = V.Run(3)
    table = run.table('imputation_comparisons')
    primary = V.select(table, stratum='all', metric='paired_patient_mae_difference')
    means = V.select(table, stratum='all', metric='patient_mae')
    readiness = V.select(run.table('feature_readiness'), metric='patients_with_any_observation')
    flagged = set(readiness.loc[readiness.review_flags.str.contains('heldout_imputation_regret', na=False), 'feature'])
    features = [feature for feature in run.features if feature in set(primary.feature)]
    forest = run.config['clinical_features'] + [
        feature for feature in features if feature in flagged and feature not in run.config['clinical_features']]
    V.require(flagged.issubset(features), 'Review-flagged feature omitted from main imputation maps')
    columns = [(name, scenario) for name in run.config['datasets'] for scenario in run.config['imputation_scenarios']]
    matrices, limit = comparison_matrices(run, means, features, columns)
    run.plan(4, [f'relative_mae_{comparator}' for comparator in run.config['comparators']])
    for comparator, matrix in matrices.items():
        heatmap_panel(run, matrix, limit, features, columns, comparator, primary, flagged)
    assembled_imputation(run, matrices, limit, features, columns, primary, flagged)
    run.plan(4, [f'supp_mae_intervals_{feature}' for feature in forest])
    for feature in forest:
        clinical_mae_intervals(run, primary, columns, feature)
    assembled_supplement(run, primary, columns, forest)
    run.audit.append(dict(check='imputation_display', features=features, forest_features=forest,
            includes_all_regret_features=flagged.issubset(forest), scenarios=run.config['imputation_scenarios'],
            comparators=run.config['comparators'], scale_limit=limit, pointwise_intervals=True, main_panels=2))
    run.finish()


if __name__ == '__main__':
    main()
