"""Export cohort coverage curves and detailed cohort/site supplements."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import visualization_common as V


def hospital_observation_availability(run, sites, availability, canvas=None):
    key = 'supp_hospital_observation_availability'
    fig, ax = V.panel_canvas(canvas, figsize=(12, 10))
    matrix = availability.pivot(index='stratum', columns='feature', values='estimate').reindex(
        index=sites.stratum, columns=run.features)
    V.require(len(sites) >= 2 and np.isfinite(matrix.to_numpy()).all(), 'Incomplete hospital availability')
    image = ax.imshow(matrix, cmap=V.cmap(), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(run.features)))
    ax.set_xticklabels([V.label(f) for f in run.features], rotation=45, ha='right', rotation_mode='anchor', fontsize=7)
    ax.set_yticks(range(len(sites)),
        [f"{s.replace('hospital=', '')} (n={int(n):,})" for s, n in zip(sites.stratum, sites.group_patients)],
        fontsize=6)
    ax.set_ylabel('Hospital ID')
    V.heatmap_axis(ax)
    V.title(ax, 'eICU · patients with any observation')
    V.colorbar(fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025, label='Patient fraction'), percent=True)
    run.add(7, key, availability, 'coverage_profiles')
    run.save(fig, 7, key)


def hospital_availability_effect(run, tests, canvas=None):
    key = 'supp_hospital_availability_effect'
    fig, ax = V.panel_canvas(canvas, figsize=(5.5, 10))
    rows = V.select(tests, question='hospital_availability').set_index('feature').reindex(run.features).reset_index()
    color = run.config['dataset_colors']['eicu']
    for i, row in rows.iterrows():
        if np.isfinite(row.estimate):
            ax.plot(row.estimate, i, 'o', color=color,
                markerfacecolor=(color if row.test_status == 'tested' else 'white'),
                markeredgecolor=('white' if row.test_status == 'tested' else color), markeredgewidth=0.8, markersize=5,
                zorder=3)
    ax.set_yticks(range(len(run.features)), [V.label(f) for f in run.features], fontsize=7)
    ax.set_ylim(len(run.features) - 0.4, -0.6)
    ax.set(xlim=(-0.02, 1.02), xlabel="Cramér's V")
    V.clean_axis(ax, categorical='y')
    V.title(ax, 'Hospital availability')
    fig.legend([Line2D([], [], marker='o', color=color, markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=0.8, ls='', markersize=4
            )], ['Test unavailable'], loc='outside lower center')
    run.add(7, key, rows, 'site_and_split_tests')
    run.save(fig, 7, key)


def baseline_sofa_sensitivity(run, sensitivity, canvas=None, panel_letter=None):
    key = 'supp_baseline_sofa_sensitivity'
    fig, ax = V.panel_canvas(canvas, figsize=(3.1, 4.6))
    colors = ['#96B8AC', '#DEC3A5', '#D9DDE0']
    strata = ['also_qualifies', 'does_not_qualify', 'not_assessable']
    hatches = ['', '//', '..']
    for i, name in enumerate(run.config['datasets']):
        bottom = 0
        for stratum, color, hatch in zip(strata, colors, hatches):
            row = V.one(sensitivity, dataset=name, stratum=stratum, question='baseline_sofa_sensitivity')
            ax.bar(i, row.estimate, bottom=bottom, color=color, hatch=hatch, edgecolor='white', linewidth=0.7,
                width=0.65, zorder=2)
            if row.estimate > 0.06:
                ax.text(i, bottom + row.estimate / 2, f'{row.estimate:.1%}', ha='center', va='center', fontsize=7)
            bottom += row.estimate
            run.add(7, key, row, 'dataset_readiness')
        V.require(np.isclose(bottom, 1), 'Baseline sensitivity denominator')
    ax.set(xlim=(-0.6, 2.6), ylim=(0, 1), ylabel='Fraction of primary cohort')
    ax.set_xticks(range(3), [run.config['dataset_labels'][n] for n in run.config['datasets']], fontsize=7)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis='y', labelsize=7)
    ax.yaxis.label.set_size(8)
    ax.yaxis.labelpad = 4
    V.clean_axis(ax, grid='y', categorical='x', percent=True)
    title = 'Measured-baseline SOFA'
    if panel_letter is not None:
        title = f'{panel_letter}.  {title}'
    ax.set_title(title, loc='left', fontsize=9, fontweight='normal', pad=8)
    fig.legend([Patch(facecolor=c, hatch=h, edgecolor='white', linewidth=0.7) for c, h in zip(colors, hatches)],
        ['Qualifies', 'Does not qualify', 'Not assessable'], loc='outside lower center', ncol=1, fontsize=7,
        handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    run.save(fig, 7, key)


def draw_selection_difference(run, ax, rows):
    for i, name in enumerate(run.config['datasets'][1:]):
        row = V.one(rows, dataset=name)
        V.interval_point(ax, i, row, run.config['dataset_colors'][name], run.config['dataset_markers'][name])
    ax.axvline(0, color='#777777', ls='--', lw=0.7)
    ax.set_ylim(1.6, -0.6)
    ax.set_yticks(range(2), [run.config['dataset_labels'][n] for n in run.config['datasets'][1:]])
    V.clean_axis(ax, categorical='y')
    ax.figure.legend([Line2D([], [], marker='o', ls='', color=V.MUTED, markerfacecolor='white')],
        ['Interval unavailable'], loc='outside lower center')


def selection_mortality(run, selection, canvas=None):
    key = 'supp_selection_mortality'
    fig, ax = V.panel_canvas(canvas, figsize=(6, 3.6))
    rows = V.select(selection, question='selection_bias', population=('supported_minus_unsupported_eligible'),
        feature='hospital_mortality', metric='risk_difference')
    draw_selection_difference(run, ax, rows)
    ax.set_xlabel('Mortality risk difference')
    V.title(ax, 'Supported minus unsupported · mortality')
    run.add(7, key, rows, 'cohort_selection')
    run.save(fig, 7, key)


def selection_age(run, selection, canvas=None):
    key = 'supp_selection_age'
    fig, ax = V.panel_canvas(canvas, figsize=(6, 3.6))
    rows = V.select(selection, question='selection_bias', population=('supported_minus_unsupported_eligible'),
        feature='age', metric='mean_difference')
    draw_selection_difference(run, ax, rows)
    ax.set_xlabel('Age difference (years)')
    V.title(ax, 'Supported minus unsupported · age')
    run.add(7, key, rows, 'cohort_selection')
    run.save(fig, 7, key)


def partition_balance(run, tests, canvas=None, panel_letter=None):
    key = 'supp_partition_balance'
    fig, ax = V.panel_canvas(canvas, figsize=(8.4, 5.3))
    rows = V.select(tests, question='partition_balance', metric='rank_biserial_difference')
    features = ['age', 'followup_hours', 'observed_fraction', *run.config['clinical_features']]
    columns = [(name, split + '_minus_train') for name in run.config['datasets'] for split in ('validation', 'test')]
    matrix = rows.pivot(index='feature', columns=['dataset', 'population'], values='estimate').reindex(
        index=features, columns=columns)
    V.require(np.isfinite(matrix.to_numpy()).all(), 'Missing partition effect')
    limit = max(0.1, float(np.abs(matrix.to_numpy()).max()))
    image = ax.imshow(
        matrix, cmap=V.cmap(True), vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest', rasterized=True)
    ax.set_yticks(range(len(features)), [V.label(f) for f in features], fontsize=8)
    ax.set_xticks(
        range(len(columns)), [split for _ in run.config['datasets'] for split in ('Validation', 'Test')], fontsize=7.5)
    ax.tick_params(axis='x', pad=5)
    V.heatmap_axis(ax)
    for center, name in zip((0.5, 2.5, 4.5), run.config['datasets']):
        ax.text(center, 1.015, run.config['dataset_labels'][name], transform=(ax.get_xaxis_transform()), ha='center',
            va='bottom', fontsize=8.5, fontweight='normal', color=V.INK)
    for x in (1.5, 3.5):
        ax.axvline(x, color='white', linewidth=2.5)
    title = 'Partition balance'
    if panel_letter is not None:
        title = f'{panel_letter}.  ' + title
    ax.set_title(title, loc='left', color=V.INK, fontweight='normal', fontsize=10.5, pad=27)
    V.colorbar(fig.colorbar(
            image, ax=ax, label=('Rank-biserial difference vs training'), fraction=0.027, shrink=0.88, pad=0.022))
    run.add(7, key, rows, 'site_and_split_tests')
    run.save(fig, 7, key)


def hospital_heterogeneity(run, tests, canvas=None):
    key = 'supp_hospital_heterogeneity'
    fig, ax = V.panel_canvas(canvas, figsize=(7.5, 5.5))
    features = ['age', 'followup_hours', 'observed_fraction', *run.config['clinical_features']]
    rows = (V.select(tests, question='hospital_heterogeneity', metric='between_hospital_eta_squared')
        .set_index('feature') .reindex(features) .reset_index())
    ax.barh(range(len(features)), rows.estimate, color=run.config['dataset_colors']['eicu'], edgecolor='white',
        linewidth=0.8, height=0.7)
    ax.set_yticks(range(len(features)), [V.label(f) for f in features], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Between-hospital variance fraction')
    V.clean_axis(ax, categorical='y', percent=True)
    V.title(ax, 'Hospital variation · eICU')
    run.add(7, key, rows, 'site_and_split_tests')
    run.save(fig, 7, key)


def cohort_profile(run, selection, canvas=None, panel_letter=None):
    key = 'supp_cohort_profile'
    features = ['age', 'hospital_mortality', 'charlson_comorbidity_index', 'followup_hours', 'observed_fraction',
        *run.config['clinical_features']]
    targets = run.config['datasets'][1:]
    rows = V.select(selection, question='source_composition', metric='standardized_mean_difference',
        comparator='mimiciv', feature=features, dataset=targets)
    matrix = (rows.pivot(index='feature', columns='dataset', values='estimate')
        .reindex(index=features, columns=targets) .to_numpy())
    V.require(np.isfinite(matrix).all(), 'Missing shared cohort contrast')
    limit = max(0.5, float(np.ceil(np.abs(matrix).max() * 2) / 2))
    fig, ax = V.panel_canvas(canvas, figsize=(4.1, 4.6))
    image = ax.imshow(
        matrix, cmap=V.cmap(True), vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest', rasterized=True)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:+.2f}', ha='center', va='center', fontsize=7,
            color='white' if abs(value) > 0.65 * limit else V.INK)
    ax.set_yticks(range(len(features)), [V.label(f) for f in features], fontsize=7.5)
    ax.set_xticks(range(len(targets)), [run.config['dataset_labels'][n] for n in targets], fontsize=8)
    V.heatmap_axis(ax)
    for y in (2.5, 4.5):
        ax.axhline(y, color='white', lw=2)
    title = 'Cohort differences'
    if panel_letter is not None:
        title = f'{panel_letter}.  {title}'
    ax.set_title(title, loc='left', fontsize=9, fontweight='normal', pad=8)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, shrink=0.75, ticks=[-limit, 0, limit])
    V.colorbar(bar)
    bar.ax.tick_params(labelsize=7)
    bar.set_label('Standardized mean difference vs MIMIC-IV', fontsize=7.5, labelpad=4)
    run.add(7, key, rows, 'cohort_selection')
    run.save(fig, 7, key)
SUPPORT_METHODS = (('completed', 'Observed + imputed', (0, (4, 2))), ('observed', 'Observed', '-'))


def support_legend(run, fig, assembled=False):
    cohort_handles = [Line2D([], [], color=run.config['dataset_colors'][name],
            marker=run.config['dataset_markers'][name], markersize=3, linewidth=1.4
        ) for name in run.config['datasets']]
    cohort_labels = [run.config['dataset_labels'][name] for name in run.config['datasets']]
    method_handles = [
        Line2D([], [], color=V.INK, linestyle=style, linewidth=1.4) for _, _, style in reversed(SUPPORT_METHODS)]
    method_labels = [label for _, label, _ in reversed(SUPPORT_METHODS)]
    options = dict(loc='lower center', fontsize=7, handlelength=2, columnspacing=1.2, borderaxespad=0, borderpad=0.2)
    if assembled:
        fig.legend(cohort_handles + method_handles, cohort_labels + method_labels, ncol=5, bbox_to_anchor=(0.5, 0.015),
            **options)
    else:
        fig.legend(cohort_handles, cohort_labels, ncol=3, bbox_to_anchor=(0.5, 0.105), **options)
        fig.legend(method_handles, method_labels, ncol=2, bbox_to_anchor=(0.5, 0.015), **options)


def study_support(run, table, minimum_hours, title, canvas=None, show_y=True, ax=None, show_x=True):
    key = f'study_support_{minimum_hours}h'
    if ax is not None:
        fig = ax.figure
    elif canvas is None:
        fig = plt.figure(figsize=(3.6, 2.8), layout='none')
        ax = fig.add_axes([0.60 / 3.6, 0.92 / 2.8, 2.88 / 3.6, 1.58 / 2.8])
        support_legend(run, fig)
    else:
        fig = canvas
        ax = fig.subplots()
    thresholds = np.arange(1, 51)
    rows = V.select(table, minimum_hours=minimum_hours)
    for name in run.config['datasets']:
        cohort = V.select(rows, dataset=name)
        V.require(cohort.n_patients.nunique() == 1 and cohort.n_patients.iloc[0] > 0,
            'Study-support denominators differ or are empty')
        values = {}
        for method, _, _ in SUPPORT_METHODS:
            subset = V.select(cohort, method=method).sort_values('minimum_features')
            V.require(np.array_equal(subset.minimum_features.to_numpy(), thresholds),
                'Study-support curve requires each feature threshold exactly once')
            y = subset.estimate.to_numpy(dtype=float)
            V.require(np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all() and (np.diff(y) <= 1e-12).all(),
                'Invalid study-support curve')
            values[method] = y
            run.add(7, key, subset, 'relationship_plot_data/study_support')
        V.require((values['completed'] >= values['observed'] - 1e-12).all(),
            'Completed coverage falls below observed coverage')
        for method, _, style in SUPPORT_METHODS:
            observed = method == 'observed'
            ax.plot(thresholds, values[method], color=run.config['dataset_colors'][name], linestyle=style,
                linewidth=1.45 if observed else 1.2, marker=run.config['dataset_markers'][name],
                markevery=([0, 9, 19, 29, 39, 49] if observed else [4, 14, 24, 34, 44]), markersize=3,
                markeredgewidth=0.65, markerfacecolor=(run.config['dataset_colors'][name] if observed else 'white'),
                alpha=1 if observed else 0.85, zorder=3 if observed else 2)
    ax.set(xlim=(0.5, 50.5), ylim=(-0.015, 1.025))
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    V.clean_axis(ax, grid='y', percent=True)
    ax.tick_params(labelsize=7, length=2.5, pad=3)
    if show_x:
        ax.set_xlabel('Minimum number of features', fontsize=8, labelpad=5)
    if show_y:
        ax.set_ylabel('Patients meeting requirement', fontsize=8, labelpad=5)
    else:
        ax.tick_params(axis='y', labelleft=False)
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='normal', pad=7)
    run.save(fig, 7, key)


def study_support_1h(run, table, canvas=None, show_y=True, ax=None, show_x=True):
    study_support(
        run, table, minimum_hours=1, title='≥ 1 h / feature', canvas=canvas, show_y=show_y, ax=ax, show_x=show_x)


def study_support_6h(run, table, canvas=None, show_y=True, ax=None, show_x=True):
    study_support(
        run, table, minimum_hours=6, title='≥ 6 h / feature', canvas=canvas, show_y=show_y, ax=ax, show_x=show_x)


def study_support_12h(run, table, canvas=None, show_y=True, ax=None, show_x=True):
    study_support(
        run, table, minimum_hours=12, title='≥ 12 h / feature', canvas=canvas, show_y=show_y, ax=ax, show_x=show_x)


def assembled_study_support(run, table):
    width, height = 7.2, 2.65
    left, right, bottom, top, gap = 0.64, 0.12, 0.70, 0.30, 0.20
    plot_width = (width - left - right - 2 * gap) / 3
    fig = plt.figure(figsize=(width, height), layout='none')
    axes = [fig.add_axes([(left + i * (plot_width + gap)) / width, bottom / height, plot_width / width,
                (height - bottom - top) / height
            ]) for i in range(3)]
    fig.supxlabel('Minimum number of features', x=(left + width - right) / (2 * width), y=0.28 / height, fontsize=8)
    support_legend(run, fig, assembled=True)
    run.assemble(fig, 7, {'study_support_1h': lambda: study_support_1h(run, table, ax=axes[0], show_x=False),
            'study_support_6h': lambda: study_support_6h(run, table, ax=axes[1], show_y=False, show_x=False),
            'study_support_12h': lambda: study_support_12h(run, table, ax=axes[2], show_y=False, show_x=False)})


def assembled_site_supplement(run, sites, availability, tests, selection, sensitivity):
    site_height = max(6.4, 0.10 * len(sites) + 1.0)
    fig = plt.figure(figsize=(12, site_height + 10.0), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.04, wspace=0.025, hspace=0.03)
    grid = fig.add_gridspec(4, 6, height_ratios=[site_height, 2.0, 3.4, 4.6])
    run.assemble(fig, 7, {'supp_hospital_observation_availability': lambda: hospital_observation_availability(
                run, sites, availability, canvas=fig.add_subfigure(grid[0, :4])),
            'supp_hospital_availability_effect': lambda: hospital_availability_effect(
                run, tests, canvas=fig.add_subfigure(grid[0, 4:])),
            'supp_selection_mortality': lambda: selection_mortality(
                run, selection, canvas=fig.add_subfigure(grid[1, :3])),
            'supp_selection_age': lambda: selection_age(run, selection, canvas=fig.add_subfigure(grid[1, 3:])),
            'supp_hospital_heterogeneity': lambda: hospital_heterogeneity(
                run, tests, canvas=fig.add_subfigure(grid[2, :2])),
            'supp_partition_balance': lambda: partition_balance(run, tests, canvas=fig.add_subfigure(grid[2, 2:])),
            'supp_cohort_profile': lambda: cohort_profile(run, selection, canvas=fig.add_subfigure(grid[3, :3])),
            'supp_baseline_sofa_sensitivity': lambda: baseline_sofa_sensitivity(
                run, sensitivity, canvas=fig.add_subfigure(grid[3, 3:]))
        }, supplementary=True)


def main():
    run = V.Run(5)
    tests = run.table('site_and_split_tests')
    selection = run.table('cohort_selection')
    sensitivity = run.table('dataset_readiness')
    coverage = run.table('coverage_profiles')
    support = run.plot_table('study_support')
    run.plan(7, ['study_support_1h', 'study_support_6h', 'study_support_12h'])
    study_support_1h(run, support)
    study_support_6h(run, support)
    study_support_12h(run, support)
    assembled_study_support(run, support)
    profiles = run.table('site_and_split_summaries')
    sites = V.select(profiles, question='hospital_heterogeneity', feature='age').sort_values('stratum')
    availability = V.select(
        coverage, dataset='eicu', question='feature_availability', metric='patients_with_any_observation')
    availability = availability[availability.stratum.isin(sites.stratum)]
    run.plan(7, ['supp_hospital_observation_availability', 'supp_hospital_availability_effect',
            'supp_selection_mortality', 'supp_selection_age', 'supp_hospital_heterogeneity', 'supp_partition_balance',
            'supp_cohort_profile', 'supp_baseline_sofa_sensitivity'])
    hospital_observation_availability(run, sites, availability)
    hospital_availability_effect(run, tests)
    selection_mortality(run, selection)
    selection_age(run, selection)
    hospital_heterogeneity(run, tests)
    partition_balance(run, tests)
    cohort_profile(run, selection)
    baseline_sofa_sensitivity(run, sensitivity)
    assembled_site_supplement(run, sites, availability, tests, selection, sensitivity)
    run.audit.append(dict(check='site_sensitivity', hospitals=len(sites),
            hospital_patients=int(sites.group_patients.sum()), features=50,
            missing_availability_tests='hollow_markers', unadjusted_case_mix=True))
    run.audit.append(dict(check='study_support', main_panels=3, features=50, display='coverage_curves',
            minimum_hours=[1, 6, 12], views=['observed', 'completed'], cohort_denominator='all_clinical_patients',
            selected_feature_set=False, consecutive_or_simultaneous_bins=False, diagnostics='supplementary'))
    run.finish()


if __name__ == '__main__':
    main()
