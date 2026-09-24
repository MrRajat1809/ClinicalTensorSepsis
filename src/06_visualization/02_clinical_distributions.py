"""Render observed patient-mean distributions without pooling repeated hours."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import visualization_common as V


def patient_means(values):
    count = np.isfinite(values).sum(axis=1)
    return np.divide(np.nansum(values, axis=1, dtype=float), count, out=np.full(len(values), np.nan), where=count > 0)


def panel_card(fig, bounds):
    left, bottom, width, height = bounds
    figure_width, figure_height = fig.get_size_inches()
    inset_left, inset_right = 0.58, 0.12
    inset_bottom, inset_top = 0.52, 0.60
    plot_width = width - inset_left - inset_right
    plot_height = height - inset_bottom - inset_top
    ax = fig.add_axes([(left + inset_left) / figure_width, (bottom + inset_bottom) / figure_height,
            plot_width / figure_width, plot_height / figure_height])
    card = FancyBboxPatch((left, bottom), width, height, transform=fig.dpi_scale_trans, clip_on=False,
        boxstyle='round,pad=0,rounding_size=.04', facecolor='white', edgecolor='#000000', linewidth=0.85, zorder=-20)
    card.set_in_layout(False)
    fig.add_artist(card)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('none')
    ax.yaxis.set_label_coords((0.12 - inset_left) / plot_width, 0.5, transform=ax.transAxes)
    return ax


def panel_title(ax, letter, title):
    ax.text(0, 1.075, f'{letter}. {title}', transform=ax.transAxes, ha='left', va='top', fontsize=10.5, color=V.INK)
    ax.plot([0, 1], [1.015, 1.015], transform=ax.transAxes, color='#D8D8D8', linewidth=0.7, clip_on=False)


def violin_summary(ax, values, color, marker, position):
    if len(np.unique(values)) >= 5:
        parts = ax.violinplot([values], positions=[position], widths=0.76, showextrema=False, points=180)
        for body in parts['bodies']:
            body.set(facecolor=V.tint(color, 0.84), edgecolor=color, linewidth=1.15, alpha=1)
    else:
        unique, counts = np.unique(values, return_counts=True)
        ax.scatter(np.full(len(unique), position), unique, s=12 + 52 * counts / counts.max(),
            color=V.tint(color, 0.70), edgecolor=color, linewidth=0.8, zorder=3)
    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    ax.plot([position] * 2, [q10, q90], color=color, linewidth=1, alpha=0.42, zorder=4)
    ax.plot([position] * 2, [q25, q75], color=color, linewidth=2.8, solid_capstyle='round', zorder=5)
    ax.plot(position, median, marker=marker, linestyle='None', markerfacecolor='white', markeredgecolor=color,
        markeredgewidth=1.25, markersize=5.8, zorder=6)


def main():
    run = V.Run(2)
    features = run.config['clinical_features']
    datasets = run.config['datasets']
    summaries = V.select(run.table('cohort_selection'), question='cohort_description', metric='patient_mean')
    figure_width, figure_height = 23.2, 5.3
    outer_margin, gutter = 0.16, 0.18
    card_width = (figure_width - 2 * outer_margin - (len(features) - 1) * gutter) / len(features)
    card_height = figure_height - 2 * outer_margin
    fig = plt.figure(figsize=(figure_width, figure_height), layout='none')
    fig.set_facecolor('white')
    axes = [panel_card(fig, (outer_margin + i * (card_width + gutter), outer_margin, card_width, card_height))
        for i in range(len(features))]
    for i, (feature, ax) in enumerate(zip(features, axes)):
        panel_title(ax, chr(65 + i), V.label(feature))
    for domain, name in enumerate(datasets):
        folder = V.ROOT / 'data/processed' / name
        with np.load(run.track(folder / 'tensor_support.npz'), allow_pickle=False) as support:
            order = support['features'].tolist()
            V.require(len(np.unique(support['subject_ids'])) == len(support['stay_ids']), 'Repeated patients')
        raw = np.load(run.track(folder / 'tensor_observed.npy'), mmap_mode='r', allow_pickle=False)
        for i, (feature, ax) in enumerate(zip(features, axes)):
            values = patient_means(raw[:, :, order.index(feature)])
            values = values[np.isfinite(values)]
            expected = V.one(summaries, dataset=name, feature=feature)
            V.require(
                len(values) == int(expected.n_patients) and np.isclose(values.mean(), expected.estimate, rtol=1e-6),
                'Patient summary disagrees with frozen analysis')
            logarithmic = feature in run.config['log_features']
            V.require(not logarithmic or (values >= 0).all(), 'Negative value in log1p distribution')
            display = np.log1p(values) if logarithmic else values
            color = run.config['dataset_colors'][name]
            violin_summary(ax, display, color, run.config['dataset_markers'][name], domain)
            ax.text(domain, 0.966, f'n={len(values):,}', transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=6.7, color=color)
            qs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
            for q, value in zip(qs, np.quantile(values, qs)):
                run.add(2, chr(65 + i), source=f'data/processed/{name}/tensor_observed.npy', dataset=name,
                    feature=feature, metric='patient_mean_quantile', quantile=q, estimate=value,
                    n_patients=len(values), unit=run.units[feature],
                    display_transform=('log1p' if logarithmic else 'identity'))
    for feature, ax in zip(features, axes):
        ax.set_xlim(-0.72, 2.72)
        ax.set_xticks(range(3), [run.config['dataset_labels'][n] for n in datasets])
        ax.tick_params(axis='x', labelsize=7.2, length=0, pad=5)
        ax.tick_params(axis='y', labelsize=7.1, length=2, width=0.6, pad=2, color='#999999')
        ylabel = run.units[feature]
        if feature in run.config['log_features']:
            ylabel += ' · log(1+x) spacing'
        ax.set_ylabel(ylabel, fontsize=7.8, labelpad=2)
        ax.grid(axis='y', color='#E7E7E7', linestyle=':', linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        ax.margins(y=0.17)
        upper = ax.get_ylim()[1]
        ax.set_ylim(0, upper)
        if feature in run.config['log_features']:
            ticks = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200])
            ticks = ticks[np.log1p(ticks) <= upper]
            ax.set_yticks(np.log1p(ticks), [str(t) for t in ticks])
    run.audit.append(dict(check='clinical_distributions', features=len(features), independent_patient_means=True,
            input='observed_only', matches_analysis=True, layout='1x8_publication_cards'))
    run.save(fig, 2)
    run.finish()


if __name__ == '__main__':
    main()
