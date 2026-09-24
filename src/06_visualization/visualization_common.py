"""Shared styling, certified inputs and figure provenance."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import PercentFormatter
from matplotlib.text import Text
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ANALYSIS = ROOT / 'outputs/statistical_analysis'
FIGURES = {1: [1, 3], 2: [2], 3: [4], 4: [5, 6], 5: [7]}
SPLIT_FIGURES = {3, 4, 6, 7}
MAIN_PANELS = {3: {'measurement_landscape', 'cohort_provenance', 'followup_comparison'},
    4: {'relative_mae_median', 'relative_mae_forward_fill'}, 6: {'clinical_network_mimiciv',
        'clinical_network_mimic3-carevue', 'clinical_network_eicu', 'relationship_concordance'
    }, 7: {'study_support_1h', 'study_support_6h', 'study_support_12h'}}
INK = '#222222'
MUTED = '#777777'
GRID = '#DDDDDD'
MISSING = '#E0E0E0'
LABELS = {'hr': 'HR', 'map': 'MAP', 'rr': 'RR', 'temp_c': 'Temperature', 'spo2': 'SpO₂', 'fio2': 'FiO₂',
    'pao2': 'PaO₂', 'paco2': 'PaCO₂', 'pf_ratio': 'P/F ratio', 'neq': 'NEQ', 'vent': 'VENT', 'pt': 'PT',
    'aptt': 'aPTT', 'inr': 'INR', 'wbc': 'WBC', 'rbc': 'RBC', 'alt': 'ALT', 'ast': 'AST', 'alp': 'ALP',
    'gcs_eye': 'GCS eye', 'gcs_verbal': 'GCS verbal', 'gcs_motor': 'GCS motor', 'rdw_cv': 'RDW-CV', 'mcv': 'MCV',
    'mch': 'MCH', 'mchc': 'MCHC', 'ph': 'pH', 'charlson_comorbidity_index': 'Charlson index',
    'followup_hours': 'Follow-up (h)', 'observed_fraction': 'Observed fraction',
    'hospital_mortality': 'Hospital mortality'}


def label(name):
    return LABELS.get(name, name.replace('_pct', ' (%)').replace('_', ' ').capitalize())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'size_bytes': Path(path).stat().st_size}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def scalar(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ''
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    return str(value)


def data_digest(rows):
    normalized = [{k: scalar(v) for k, v in row.items() if scalar(v) != ''} for row in rows]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def style(config):
    plt.rcParams.update({'font.family': config['font_family'], 'font.size': config['font_size'],
            'font.weight': 'normal', 'axes.titleweight': 'normal', 'axes.labelweight': 'normal', 'axes.titlesize': 11,
            'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.color': INK, 'axes.labelcolor': INK,
            'axes.edgecolor': '#BBBBBB', 'xtick.color': INK, 'ytick.color': INK, 'axes.labelpad': 8,
            'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
            'axes.grid': False, 'axes.facecolor': 'white', 'grid.color': GRID, 'grid.linestyle': ':',
            'grid.linewidth': 0.7, 'grid.alpha': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out',
            'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
            'lines.linewidth': 1.5, 'lines.solid_capstyle': 'round',
            'axes.prop_cycle': plt.cycler(color=list(config['dataset_colors'].values())), 'legend.frameon': False,
            'legend.fontsize': 8, 'legend.labelspacing': 0.7, 'legend.handletextpad': 0.6, 'legend.columnspacing': 1.8,
            'figure.constrained_layout.w_pad': 0.08, 'figure.constrained_layout.h_pad': 0.09,
            'savefig.dpi': config['dpi'], 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white',
            'savefig.facecolor': 'white', 'axes.formatter.useoffset': False})


def cmap(diverging=False):
    result = (matplotlib.colormaps['RdBu_r'].copy() if diverging else LinearSegmentedColormap.from_list(
            'reference_warm', ['#FDF4D6', '#FCE082', '#F3B032', '#E97233', '#D74241']))
    result.set_bad(MISSING)
    return result


def tint(color, amount=0.85):
    return tuple((1 - amount) * channel + amount for channel in to_rgb(color))


def clean_axis(ax, grid='x', categorical=None, percent=False):
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linestyle=':', linewidth=0.7, alpha=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if categorical:
        ax.spines['left' if categorical == 'y' else 'bottom'].set_visible(False)
        ax.tick_params(axis=categorical, length=0)
    if percent:
        axis = ax.xaxis if grid == 'x' else ax.yaxis
        axis.set_major_formatter(PercentFormatter(1, decimals=0))


def heatmap_axis(ax):
    ax.grid(False, which='both')
    ax.tick_params(axis='both', which='both', length=0, pad=5)
    for spine in ax.spines.values():
        spine.set(visible=True, color='#CCCCCC', linewidth=0.6)


def colorbar(bar, percent=False):
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=3, width=0.6, labelsize=8)
    if percent:
        bar.formatter = PercentFormatter(1, decimals=0)
        bar.update_ticks()
    return bar


def panel(ax, letter, title):
    ax.set_title(f'{letter}.  {title}', loc='left', color=INK, fontweight='normal', pad=12)


def title(ax, text):
    ax.set_title(text, loc='left', color=INK, fontweight='normal', pad=12)


def panel_canvas(canvas=None, *, figsize):
    if canvas is None:
        canvas = plt.figure(figsize=figsize, layout='constrained')
    return canvas, canvas.subplots()


def artifact_stem(number, panel_name=None):
    require(number in {n for numbers in FIGURES.values() for n in numbers}, 'Unknown figure')
    if panel_name is None:
        return f'Fig{number}'
    if number in SPLIT_FIGURES:
        if panel_name == 'supplement':
            return f'SuppFig{number}'
        require(isinstance(panel_name, str) and re.fullmatch(r'[a-z][a-z0-9_-]*', panel_name),
            'Standalone panels need a descriptive filename-safe name')
        if panel_name.startswith('supp_'):
            require(re.fullmatch(r'[a-z][a-z0-9_-]*', panel_name[5:]), 'Empty or invalid supplementary name')
            return f'SuppFig{number}_{panel_name[5:]}'
        return f'Fig{number}_{panel_name}'
    require(panel_name is None, 'This figure remains assembled')
    return f'Fig{number}'


def artifact_path(number, panel_name=None, extension='pdf'):
    filename = f'{artifact_stem(number, panel_name)}.{extension}'
    return filename if panel_name in (None, 'supplement') else f'panels/{filename}'


def validate_artifacts(stage, artifacts, rows):
    require({a['number'] for a in artifacts.values()} == set(FIGURES[stage]), 'Missing figure export plan')
    for stem, item in artifacts.items():
        require(stem == artifact_stem(item['number'], item['panel']), 'Invalid artifact name')
    for number in SPLIT_FIGURES.intersection(FIGURES[stage]):
        assembled = artifacts.get(artifact_stem(number))
        require(assembled is not None and assembled['panel'] is None, f'Fig{number}: missing assembled figure')
        require(
            set(assembled.get('panels', [])) == MAIN_PANELS[number], f'Fig{number}: assembled panel inventory differs')
        planned = {
            a['panel'] for a in artifacts.values() if a['number'] == number and a['panel'] not in (None, 'supplement')}
        require({p for p in planned if not p.startswith('supp_')} == MAIN_PANELS[number],
            f'Fig{number}: compact main panel inventory differs')
        supplement_panels = {p for p in planned if p.startswith('supp_')}
        supplement = artifacts.get(artifact_stem(number, 'supplement'))
        require(supplement_panels and supplement is not None and supplement['panel'] == 'supplement',
            f'Fig{number}: missing supplementary figure')
        require(set(supplement.get('panels', [])) == supplement_panels,
            f'Fig{number}: supplementary assembly differs from panel exports')
        plotted = {r['panel'] for r in rows if r['figure'] == f'Fig{number}'}
        require(planned == plotted, f'Fig{number}: panel data and export plan differ')


def select(frame, **criteria):
    result = frame
    for key, value in criteria.items():
        result = result[result[key].isin(value) if isinstance(value, (tuple, list)) else result[key].eq(value)]
    return result.copy()


def one(frame, **criteria):
    rows = select(frame, **criteria)
    require(len(rows) == 1, f'Expected one row for {criteria}, found {len(rows)}')
    return rows.iloc[0]


def interval_point(ax, y, row, color='#005B96', marker='o', offset=0):
    x = pd.to_numeric(row.get('estimate'), errors='coerce')
    lo = pd.to_numeric(row.get('ci_low'), errors='coerce')
    hi = pd.to_numeric(row.get('ci_high'), errors='coerce')
    if not np.isfinite(x):
        ax.text(0.98, y + offset, 'NA', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7)
        return
    has_interval = np.isfinite(lo) and np.isfinite(hi)
    if has_interval:
        require(lo <= hi, 'Reversed interval')
        ax.plot([lo, hi], [y + offset] * 2, color=tint(color, 0.35), lw=1.3, marker='|', markersize=4,
            markeredgewidth=0.9, zorder=2)
    ax.plot(x, y + offset, marker=marker, linestyle='none', color=color, markersize=5.5,
        markerfacecolor=color if has_interval else 'white', markeredgecolor='white' if has_interval else color,
        markeredgewidth=0.9, zorder=3)


class Run:

    def __init__(self, stage, argv=None):
        parser = argparse.ArgumentParser(description=f'Frozen dataset visualization {stage:02d}')
        parser.add_argument('--config', type=Path, default=HERE / 'visualization_config.json')
        parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/visualization')
        parser.add_argument('--supplementary', action='store_true', help=argparse.SUPPRESS)
        self.args = parser.parse_args(argv)
        self.stage = stage
        self.config = read_json(self.args.config)
        require(self.config['version'] == '1.0.0' and self.config['dpi'] == 300, 'Require v1.0.0 and 300 dpi')
        require(self.config['datasets'] == ['mimiciv', 'mimic3-carevue', 'eicu'], 'Dataset order changed')
        self.out = self.args.outdir.resolve()
        base = (ROOT / 'outputs').resolve()
        require(self.out.is_relative_to(base) and self.out != base, 'Figure output must be inside outputs/')
        require(not any(self.out == base / n or (base / n) in self.out.parents
                for n in [*self.config['datasets'], 'atlas', 'statistical_analysis']), 'Protected processing output')
        self.inputs, self.outputs, self.rows, self.audit = {}, {}, [], []
        self.artifacts = {artifact_stem(n): dict(number=n, panel=None) for n in FIGURES[stage]}
        self.artifacts.update({artifact_stem(n, 'supplement'): dict(number=n, panel='supplement')
                for n in FIGURES[stage] if n in SPLIT_FIGURES})
        self._assembly_panel = None
        self.expected = {}
        manifest = read_json(ANALYSIS / 'analysis_manifest.json')
        qc = read_json(ANALYSIS / 'qc.json')
        require(qc['status'] == 'PASS' and qc['failures'] == 0 and manifest['status'] == 'VERIFIED',
            'Statistical QC must pass first')
        require(record(ANALYSIS / 'analysis_manifest.json') == qc['analysis_manifest'], 'Stale statistical QC')
        require(record(ROOT / 'src/05_statistical_analysis/07_analysis_qc.py') == qc['qc_code'],
            'Changed statistical QC code')
        for number in range(1, 7):
            entry = manifest['stages'][str(number)]
            require(entry['status'] == 'COMPLETE', 'Incomplete analysis stage')
            for section in ('inputs', 'code'):
                self.merge(entry[section])
            self.merge({'outputs/statistical_analysis/' + k: {f: v[f] for f in ('sha256', 'size_bytes')}
                    for k, v in entry['outputs'].items()})
        self.merge({'src/05_statistical_analysis/analysis_config.json': manifest['config_record']})
        self.track(ANALYSIS / 'qc.json')
        self.track(ANALYSIS / 'analysis_manifest.json')
        for n in range(1, 6):
            atlas = self.json(ROOT / f'outputs/atlas/{n:02d}.json')
            self.merge(atlas['outputs'])
        registry = self.json(ROOT / 'src/common/feature_registry.json')['temporal_features']
        self.features = [r['name'] for r in registry]
        self.units = {r['name']: r['canonical_unit'] for r in registry}
        require(len(self.features) == 50, 'Expected 50 clinical features')
        require(self.config['clinical_features'] == manifest['config']['clinical_features'],
            'Clinical selection differs from analysis')
        self.code = {p.relative_to(ROOT).as_posix(): record(p)
            for p in [HERE / 'visualization_common.py', *HERE.glob(f'{stage:02d}_*.py')]}
        self.config_record = record(self.args.config)
        self.out.mkdir(parents=True, exist_ok=True)
        self.receipt = self.out / 'manifest.json'
        self.manifest = read_json(self.receipt) if self.receipt.exists() else {'version': '1.0.0', 'stages': {}}
        if self.manifest.get('config_record') != self.config_record or self.manifest.get('supplementary') is not True:
            self.manifest['stages'] = {}
        self.manifest.update(
            status='INCOMPLETE', config=self.config, config_record=self.config_record, supplementary=True)
        self.manifest['stages'].pop(str(stage), None)
        write_json(self.receipt, self.manifest)
        write_json(self.out / 'qc.json', {'status': 'STALE', 'reason': f'Visualization {stage:02d} started'})
        style(self.config)
        print(f'[VIS {stage:02d}] v1.0.0; frozen inputs; PDF and 300 dpi PNG', flush=True)

    def merge(self, records):
        for key, value in records.items():
            require(key not in self.expected or self.expected[key] == value, f'Conflicting source binding: {key}')
            self.expected[key] = value

    def track(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), 'Input outside project')
        key = path.relative_to(ROOT).as_posix()
        if key not in self.inputs:
            actual = record(path)
            require(key not in self.expected or actual == self.expected[key], f'Changed certified input: {key}')
            self.inputs[key] = actual
        return path

    def json(self, path):
        return read_json(self.track(path))

    def table(self, name):
        frame = pd.read_csv(self.track(ANALYSIS / (name + '.csv')), float_precision='round_trip', keep_default_na=False
        )
        for key in ('estimate', 'ci_low', 'ci_high', 'n_patients', 'numerator', 'hour', 'quantile', 'count',
            'denominator', 'group_patients', 'n_hospitals', 'n_reference', 'n_group', 'n_comparator_group'):
            if key in frame:
                frame[key] = pd.to_numeric(frame[key], errors='coerce')
        return frame

    def coordinates(self):
        return pd.read_parquet(self.track(ROOT / 'data/processed/atlas/coordinates.parquet'))

    def plot_table(self, name):
        folder = ANALYSIS / 'relationship_plot_data'
        if not getattr(self, '_plot_data_bound', False):
            require((folder / 'manifest.json').exists(),
                'Run python src/05_statistical_analysis/08_relationship_plot_data.py first')
            receipt = self.json(folder / 'manifest.json')
            require(receipt['status'] == 'VERIFIED' and receipt['version'] == '1.0.0',
                'Relationship plot data is incomplete; rerun 08_relationship_plot_data.py')
            require(receipt['datasets'] == self.config['datasets'] and receipt['features'] == self.features,
                'Relationship plot-data schema differs')
            for section in ('inputs', 'code'):
                self.merge(receipt[section])
                for key in receipt[section]:
                    self.track(ROOT / key)
            self.merge({(folder / key) .relative_to(ROOT)
                    .as_posix(): {field: value[field] for field in ('sha256', 'size_bytes')}
                    for key, value in receipt['outputs'].items()})
            self._plot_data_bound = True
        require(name in ('feature_relationships', 'feature_observations', 'study_support'),
            'Unknown relationship plot-data table')
        frame = self.table('relationship_plot_data/' + name)
        for column in ('minimum_features', 'minimum_hours', 'cohort_patients', 'joint_observed_fraction'):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors='coerce')
        return frame

    def add(self, figure, panel_name, frame=None, source='', **extra):
        if self._assembly_panel is not None:
            require((figure, panel_name) == self._assembly_panel, 'Unexpected assembled panel data')
            return
        records = (frame.to_dict('records') if isinstance(frame, pd.DataFrame) else [dict(frame)] if frame is not None
            else [{}])
        for row in records:
            self.rows.append({k: scalar(v)
                    for k, v in dict(row, figure=f'Fig{figure}', panel=panel_name, source=source, **extra).items()})

    def plan(self, number, panel_names):
        require(number in FIGURES[self.stage] and number in SPLIT_FIGURES, 'Unexpected split figure')
        for name in panel_names:
            require(name != 'supplement', 'Reserved supplementary assembly name')
            stem = artifact_stem(number, name)
            require(stem not in self.artifacts, f'Duplicate panel: {stem}')
            self.artifacts[stem] = dict(number=number, panel=name)

    def assemble(self, fig, number, renderers, *, supplementary=False):
        require(number in FIGURES[self.stage] and number in SPLIT_FIGURES, 'Unexpected assembled figure')
        require(self._assembly_panel is None, 'Nested assembly is not supported')
        expected = ({a['panel'] for a in self.artifacts.values()
                if a['number'] == number and (a['panel'] or '').startswith('supp_')} if supplementary
            else MAIN_PANELS[number])
        require(expected and set(renderers) == expected, 'Incomplete assembled panel layout')
        for key in renderers:
            require(artifact_path(number, key) in self.outputs,
                f'Export the standalone panel before assembly: {key}')
        try:
            for key, render in renderers.items():
                self._assembly_panel = (number, key)
                render()
        finally:
            self._assembly_panel = None
        assembly_name = 'supplement' if supplementary else None
        self.artifacts[artifact_stem(number, assembly_name)]['panels'] = list(renderers)
        self.save(fig, number, assembly_name)

    def save(self, fig, number, panel_name=None):
        if self._assembly_panel is not None:
            require((number, panel_name) == self._assembly_panel, 'Unexpected panel export during assembly')
            return
        require(number in FIGURES[self.stage], 'Unexpected figure')
        stem = artifact_stem(number, panel_name)
        require(stem in self.artifacts, f'Panel not declared in export plan: {stem}')
        require(artifact_path(number, panel_name) not in self.outputs, f'Duplicate export: {stem}')
        for text in fig.findobj(Text):
            require(text.get_fontweight() in ('normal', 'regular', 400), 'Bold text in figure')
        fig.canvas.draw()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for axis in fig.axes:
            bounds = axis.get_tightbbox(renderer)
            if bounds is not None:
                require(bounds.x0 >= -3 and bounds.y0 >= -3 and bounds.x1 <= fig.bbox.width + 3
                    and bounds.y1 <= fig.bbox.height + 3, f'{stem}: labels extend beyond the canvas')
        fig.set_layout_engine('none')
        width, height = fig.get_size_inches()
        for extension in ('pdf', 'png'):
            filename = artifact_path(number, panel_name, extension)
            path = self.out / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            fig.savefig(temporary, format=extension, dpi=300)
            temporary.replace(path)
            self.outputs[filename] = dict(
                record(path), width_inches=float(width), height_inches=float(height), dpi=300)
        plt.close(fig)
        self.audit.append(dict(check='render_style', figure=f'Fig{number}', panel=panel_name, regular_weight=True,
                no_clipped_axes=True, dpi=300))
        print(f'[VIS {self.stage:02d}] {artifact_path(number, panel_name)} / '
            f'{artifact_path(number, panel_name, "png")}', flush=True)

    def finish(self):
        validate_artifacts(self.stage, self.artifacts, self.rows)
        require(set(self.outputs) == {artifact_path(a['number'], a['panel'], ext)
            for a in self.artifacts.values() for ext in ('pdf', 'png')},
            'Missing figure export')
        for key, expected in {**self.inputs, **self.code}.items():
            require(record(ROOT / key) == expected, f'Input changed during rendering: {key}')
        require(record(self.args.config) == self.config_record, 'Configuration changed during rendering')
        path = self.out / 'figure_data.csv'
        previous = list(csv.DictReader(path.open(encoding='utf-8', newline=''))) if path.exists() else []
        owned = {f'Fig{n}' for n in FIGURES[self.stage]}
        rows = [r for r in previous if r['figure'] not in owned] + self.rows
        fields = ['figure', 'panel', 'source'] + sorted(
            set().union(*(r.keys() for r in rows)) - {'figure', 'panel', 'source'})
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        self.manifest['stages'][str(self.stage)] = dict(status='COMPLETE', inputs=self.inputs, outputs=self.outputs,
            artifacts=self.artifacts, code=self.code, data_sha256=data_digest(self.rows), data_rows=len(self.rows),
            audit=self.audit,
            runtime={n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'matplotlib', 'scipy')})
        write_json(self.receipt, self.manifest)
