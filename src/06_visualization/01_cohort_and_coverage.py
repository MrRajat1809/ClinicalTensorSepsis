"""Render patient flow (Fig1) and observation coverage/provenance (Fig3)."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import visualization_common as V


def flow(run):
    table = V.select(run.table('cohort_selection'), question='cohort_flow')
    fig, axes = plt.subplots(1, 3, figsize=(14, 8.5), layout='constrained')
    steps = {'mimiciv': [('after_excluding_elective_admission', 'Base cohort'),
            ('within_icu_presentation_window', 'Culture + IV antimicrobial'),
            ('operational_sepsis_cohort', 'Operational sepsis cohort')
        ], 'mimic3-carevue': [('after_excluding_elective_admission', 'Base cohort'),
            ('suspected_infection', 'Suspected infection'), ('operational_sepsis_cohort', 'Operational sepsis cohort')
        ], 'eicu': [('base_cohort', 'Base cohort'), ('suspected_infection', 'Documented infection + IV proxy'),
            ('operational_sepsis_cohort', 'Operational sepsis cohort')]}

    def box(ax, x, y, text, color, width=0.88, height=0.073):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                boxstyle='round,pad=.008,rounding_size=.012', facecolor=V.tint(color), edgecolor=color, linewidth=0.9))
        ax.text(x, y, text, ha='center', va='center', fontsize=8, linespacing=1.5, color=V.INK)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=0.9, color=V.MUTED))
    for index, (name, ax) in enumerate(zip(run.config['datasets'], axes)):
        color = run.config['dataset_colors'][name]
        V.panel(ax, chr(65 + index), run.config['dataset_labels'][name])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        counts = []
        for i, (step, title) in enumerate(steps[name]):
            row = V.one(table, dataset=name, step=step)
            n = int(row['count'])
            counts.append(n)
            y = 0.93 - i * 0.17
            box(ax, 0.5, y, f'{title}\nn = {n:,}', color)
            run.add(1, name, row, 'cohort_selection')
            if i:
                arrow(ax, 0.5, y + 0.115, 0.5, y + 0.046)
                ax.text(0.53, y + 0.084, f'Excluded: {counts[i-1]-n:,}', fontsize=7, va='center')
        final = counts[-1]
        eligible = V.one(table, dataset=name, step='eligible_representation')
        retained = V.one(table, dataset=name, step='released_observed_and_imputed_patients')
        V.require(int(retained['count']) == final, 'Clinical release patient loss')
        box(ax, 0.5, 0.43, f'Observed + imputed release\nn = {final:,}', color)
        arrow(ax, 0.5, 0.55, 0.5, 0.477)
        en = int(eligible['count'])
        absent = final - en
        box(ax, 0.25, 0.27, f'Atlas representation\nn = {en:,}', color, width=0.43)
        box(ax, 0.75, 0.27, f'No representation\nRetained: {absent:,}', V.MUTED, width=0.43)
        arrow(ax, 0.5, 0.388, 0.25, 0.317)
        arrow(ax, 0.5, 0.388, 0.75, 0.317)
        if name == 'mimiciv':
            box(ax, 0.25, 0.11, f'Reference unchanged\nn = {en:,}', color, width=0.43)
            arrow(ax, 0.25, 0.224, 0.25, 0.157)
        else:
            supported = V.one(table, dataset=name, step='selected_map_supported')
            unchanged = V.one(table, dataset=name, step='eligible_target_unchanged')
            V.require(int(supported['count'] + unchanged['count']) == en, 'Atlas support denominator')
            box(ax, 0.25, 0.11, f'Transported\nn = {int(supported["count"]):,}', color, width=0.43)
            box(ax, 0.75, 0.11, f'Original embedding\nn = {int(unchanged["count"]):,}', V.MUTED, width=0.43)
            arrow(ax, 0.25, 0.224, 0.25, 0.157)
            arrow(ax, 0.25, 0.224, 0.75, 0.157)
            run.add(1, name, supported, 'cohort_selection')
            run.add(1, name, unchanged, 'cohort_selection')
        if name == 'eicu':
            strict = V.one(table, dataset=name, step='strict_culture_sepsis3')
            ax.text(0.96, 0.51, f'Strict-culture subset: {int(strict["count"]):,}', ha='right', fontsize=7)
            arrow(ax, 0.84, 0.552, 0.84, 0.527)
            run.add(1, name, strict, 'cohort_selection')
        run.add(1, name, eligible, 'cohort_selection')
        run.add(1, name, retained, 'cohort_selection')
        run.audit.append(dict(check='patient_flow', dataset=name, retained=final, eligible=en, ineligible=absent))
    run.save(fig, 1)
HATCHES = ['', '', '///', '...', 'xxx']
METHOD_LABELS = ['Observed', 'SAITS', 'Forward fill', 'Median', 'Unfilled']


def provenance_handles(run, methods):
    V.require(len(methods) == 5, 'Expected five provenance methods')
    return [Patch(facecolor=run.config['method_colors'][m], hatch=h, edgecolor='white', linewidth=0.6)
        for m, h in zip(methods, HATCHES)]


def hourly_coverage(run, hourly, name, canvas=None):
    key = f'supp_hourly_coverage_{name}'
    fig, ax = V.panel_canvas(canvas, figsize=(7, 10))
    rows = V.select(hourly, dataset=name)
    values = rows.pivot(index='feature', columns='hour', values='estimate').reindex(
        index=run.features, columns=range(24))
    V.require(values.shape == (50, 24), 'Incomplete hourly coverage')
    image = ax.imshow(values, aspect='auto', cmap=V.cmap(), vmin=0, vmax=1, interpolation='nearest', rasterized=True)
    ax.set_yticks(range(50), [V.label(f) for f in run.features], fontsize=7.2)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel('Hour after onset')
    ax.tick_params(axis='x', labelsize=8)
    V.heatmap_axis(ax)
    V.title(ax, 'Hourly coverage · ' + run.config['dataset_labels'][name])
    bar = fig.colorbar(image, ax=ax, shrink=0.80, fraction=0.032, pad=0.022, label='Observed fraction')
    V.colorbar(bar, percent=True)
    run.add(3, key, rows, 'coverage_profiles')
    run.save(fig, 3, key)


def cell_provenance(run, provenance, hourly, name, canvas=None):
    key = f'supp_cell_provenance_{name}'
    fig, ax = V.panel_canvas(canvas, figsize=(7, 10))
    methods = list(run.config['method_colors'])
    rows = V.select(provenance, dataset=name)
    totals = (
        rows.groupby(['feature', 'method'])['numerator'].sum().unstack().reindex(index=run.features, columns=methods))
    denominator = V.select(hourly, dataset=name).groupby('feature')['n_patients'].sum().reindex(run.features)
    V.require(np.array_equal(totals.sum(axis=1).to_numpy(), denominator.to_numpy()),
        'Provenance counts do not partition exposed cells')
    values = totals.div(denominator, axis=0)
    left = np.zeros(50)
    for method, hatch in zip(methods, HATCHES):
        x = values[method].to_numpy()
        ax.barh(range(50), x, left=left, height=0.72, color=run.config['method_colors'][method], hatch=hatch,
            edgecolor='white', linewidth=0.35)
        left += x
        for feature in run.features:
            run.add(3, key, source='coverage_profiles', dataset=name, feature=feature,
                metric='within_followup_cell_fraction', method=method, estimate=values.loc[feature, method],
                numerator=totals.loc[feature, method], denominator=denominator.loc[feature])
    ax.set_yticks(range(50), [V.label(f) for f in run.features], fontsize=7.2)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set(xlim=(0, 1), xlabel='Fraction of within-follow-up cells')
    ax.invert_yaxis()
    V.clean_axis(ax, categorical='y', percent=True)
    V.title(ax, 'Cell provenance · ' + run.config['dataset_labels'][name])
    fig.legend(provenance_handles(run, methods), METHOD_LABELS, loc='outside lower center', ncol=3, frameon=False,
        fontsize=8, handlelength=1.8, columnspacing=1.2)
    run.save(fig, 3, key)


def measurement_landscape(run, table, canvas=None):
    key = 'measurement_landscape'
    metrics = ['patients_with_any_observation', 'mean_patient_observed_fraction']
    rows = V.select(table, question='feature_availability', stratum='all', metric=metrics)
    columns = [(metric, name) for name in run.config['datasets'] for metric in metrics]
    matrix = rows.pivot(index='feature', columns=['metric', 'dataset'], values='estimate').reindex(
        index=run.features, columns=columns)
    V.require(matrix.shape == (50, 6) and np.isfinite(matrix.to_numpy()).all(), 'Incomplete measurement landscape')
    fig, ax = V.panel_canvas(canvas, figsize=(9.8, 10))
    image = ax.imshow(matrix, cmap=V.cmap(), vmin=0, vmax=1, aspect='auto', interpolation='nearest', rasterized=True)
    ax.set_yticks(range(50), [V.label(f) for f in run.features], fontsize=7.4)
    labels = []
    for name in run.config['datasets']:
        label = run.config['dataset_labels'][name]
        labels.extend([f'{label}\nPatients observed', f'{label}\nHourly coverage'])
    ax.set_xticks(range(6), labels, fontsize=8)
    ax.tick_params(axis='x', length=0, pad=7)
    for tick, (_, name) in zip(ax.get_xticklabels(), columns):
        tick.set_color(run.config['dataset_colors'][name])
        tick.set_linespacing(1.15)
    for x in (1.5, 3.5):
        ax.axvline(x, color='white', linewidth=2.4)
    V.heatmap_axis(ax)
    V.title(ax, 'A. Observation coverage')
    bar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018, aspect=32, label='Observed fraction')
    bar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    V.colorbar(bar, percent=True)
    run.add(3, key, rows, 'coverage_profiles')
    run.save(fig, 3, key)


def cohort_provenance(run, provenance, hourly, canvas=None):
    key = 'cohort_provenance'
    fig, ax = V.panel_canvas(canvas, figsize=(8.2, 3.6))
    methods = list(run.config['method_colors'])
    V.require(len(methods) == 5, 'Expected five provenance methods')
    for i, name in enumerate(run.config['datasets']):
        rows = V.select(provenance, dataset=name)
        totals = (rows.groupby(['feature', 'method'])['numerator'] .sum() .unstack()
            .reindex(index=run.features, columns=methods))
        denominator = V.select(hourly, dataset=name).groupby('feature')['n_patients'].sum().reindex(run.features)
        V.require(np.array_equal(totals.sum(axis=1).to_numpy(), denominator.to_numpy()),
            'Provenance counts do not partition exposed cells')
        total = float(denominator.sum())
        left = 0.0
        for method, hatch in zip(methods, HATCHES):
            numerator = float(totals[method].sum())
            fraction = numerator / total
            ax.barh(i, fraction, left=left, height=0.50, color=run.config['method_colors'][method], hatch=hatch,
                edgecolor='white', linewidth=0.8, zorder=2)
            if fraction >= 0.05:
                ax.text(left + fraction / 2, i, f'{fraction:.0%}', ha='center', va='center', fontsize=8.2,
                    color='white' if method == 'observed' else V.INK, zorder=3)
            run.add(3, key, source='coverage_profiles', dataset=name, feature='all_50_features',
                metric='within_followup_cell_fraction', method=method, estimate=fraction, numerator=numerator,
                denominator=total)
            left += fraction
        V.require(np.isclose(left, 1), 'Cohort provenance does not sum to one')
    ax.set_yticks(range(3), [run.config['dataset_labels'][n] for n in run.config['datasets']], fontsize=9)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set(xlim=(0, 1), xlabel='Fraction of within-follow-up cells')
    ax.invert_yaxis()
    ax.margins(y=0.22)
    V.clean_axis(ax, categorical='y', percent=True)
    V.title(ax, 'B. Cell provenance')
    fig.legend(provenance_handles(run, methods), METHOD_LABELS, loc='outside lower center', ncol=5, frameon=False,
        fontsize=8, handlelength=1.7, columnspacing=1.05)
    run.save(fig, 3, key)


def followup_comparison(run, followup, canvas=None):
    key = 'followup_comparison'
    fig, ax = V.panel_canvas(canvas, figsize=(8.2, 4.8))
    minimum = 1.0
    for name in run.config['datasets']:
        rows = V.select(followup, dataset=name, metric='exposed').sort_values('hour')
        V.require(rows.hour.tolist() == list(range(24)), 'Incomplete follow-up curve')
        color = run.config['dataset_colors'][name]
        minimum = min(minimum, float(rows.ci_low.min()))
        ax.fill_between(rows.hour, rows.ci_low, rows.ci_high, color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(rows.hour, rows.estimate, color=color, linewidth=1.9, marker=run.config['dataset_markers'][name],
            markersize=4, markevery=3, markeredgecolor='white', markeredgewidth=0.55,
            label=run.config['dataset_labels'][name], zorder=3)
        run.add(3, key, rows, 'coverage_profiles')
    lower = max(0, np.floor((minimum - 0.02) / 0.05) * 0.05)
    ax.set(xlim=(0, 23), ylim=(lower, 1.005), xlabel='Hour after onset', ylabel='Patients with follow-up available')
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_yticks(np.arange(lower, 1.001, 0.05))
    V.clean_axis(ax, grid='y', percent=True)
    V.title(ax, 'C. Follow-up availability')
    ax.legend(loc='lower left', frameon=False, fontsize=8, handlelength=2.1, handletextpad=0.55)
    run.save(fig, 3, key)


def assembled_coverage(run, table, provenance, hourly, followup):
    fig = plt.figure(figsize=(15, 8.8), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.14, 0.86], height_ratios=[0.78, 1.22], wspace=0.055, hspace=0.085)
    run.assemble(fig, 3, {
            'measurement_landscape': lambda: measurement_landscape(run, table, canvas=fig.add_subfigure(grid[:, 0])),
            'cohort_provenance': lambda: cohort_provenance(
                run, provenance, hourly, canvas=fig.add_subfigure(grid[0, 1])),
            'followup_comparison': lambda: followup_comparison(run, followup, canvas=fig.add_subfigure(grid[1, 1]))})


def assembled_coverage_supplement(run, hourly, provenance):
    names = run.config['datasets']
    fig = plt.figure(figsize=(7 * len(names), 20), layout='constrained')
    grid = fig.add_gridspec(2, len(names), wspace=0.05, hspace=0.08)
    renderers = {}
    for column, name in enumerate(names):
        renderers[f'supp_hourly_coverage_{name}'] = lambda c=column, n=name: hourly_coverage(
            run, hourly, n, canvas=fig.add_subfigure(grid[0, c]))
        renderers[f'supp_cell_provenance_{name}'] = lambda c=column, n=name: cell_provenance(
            run, provenance, hourly, n, canvas=fig.add_subfigure(grid[1, c]))
    run.assemble(fig, 3, renderers, supplementary=True)


def coverage(run):
    table = run.table('coverage_profiles')
    hourly = V.select(table, question='hourly_coverage')
    provenance = V.select(table, question='cell_provenance')
    followup = V.select(table, question='followup')
    run.plan(3, ['measurement_landscape', 'cohort_provenance', 'followup_comparison'])
    measurement_landscape(run, table)
    cohort_provenance(run, provenance, hourly)
    followup_comparison(run, followup)
    assembled_coverage(run, table, provenance, hourly, followup)
    run.plan(3,
        [f'supp_{kind}_{name}' for name in run.config['datasets'] for kind in ('hourly_coverage', 'cell_provenance')])
    for name in run.config['datasets']:
        hourly_coverage(run, hourly, name)
        cell_provenance(run, provenance, hourly, name)
    assembled_coverage_supplement(run, hourly, provenance)
    run.audit.append(dict(check='coverage_denominators', features=50, hours=24, structural_separate=True,
            provenance_weighting='exposed feature-hour cells', main_panels=3))


def main():
    run = V.Run(1)
    flow(run)
    coverage(run)
    run.finish()


if __name__ == '__main__':
    main()
