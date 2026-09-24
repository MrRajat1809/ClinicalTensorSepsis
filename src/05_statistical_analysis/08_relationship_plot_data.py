"""Export descriptive relationships and study-support surfaces from frozen tensors.

This extension has its own manifest. It does not rewrite the six certified
analyses, their configuration, or their QC receipt. No hypothesis tests are run.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import analysis_common as A


OUT = A.ROOT / 'outputs/statistical_analysis/relationship_plot_data'
TABLE_NAMES = ('feature_relationships', 'feature_observations', 'study_support')
PROTOCOL = dict(version=1, features=50, hours=24, minimum_correlation_patients=3,
                correlation='pairwise_spearman_average_ties',
                summary='mean_of_observed_hourly_values_per_patient',
                support='at_least_k_features_each_available_in_at_least_h_hourly_bins',
                support_denominator='all_clinical_cohort_patients',
                inference='descriptive_only_no_hypothesis_tests')


class PlotDataRun(A.Run):
    """Reuse certified input readers, with an independent output lifecycle."""

    def __init__(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--force', action='store_true', help='Rebuild a current plot-data cache')
        self.args = parser.parse_args()
        self.inputs, self.expected, self.outputs, self.audit = {}, {}, {}, []
        self.config = A.read_json(A.HERE / 'analysis_config.json')
        A.require(self.config['version'] == '1.0.0' and self.config['datasets'] == list(A.SOURCES),
                  'Unexpected analysis version or dataset order')
        self.code = {p.relative_to(A.ROOT).as_posix(): A.record(p)
                     for p in (Path(__file__).resolve(), A.HERE / 'analysis_common.py')}
        self._certificates()
        self.track(A.HERE / 'analysis_config.json')
        base = A.ROOT / 'outputs/statistical_analysis'
        qc, manifest = self.json(base / 'qc.json'), self.json(base / 'analysis_manifest.json')
        A.require(qc['status'] == 'PASS' and manifest['status'] == 'VERIFIED',
                  'Complete the existing statistical analysis and QC first')
        A.require(qc['analysis_manifest'] == A.record(base / 'analysis_manifest.json'),
                  'Stale statistical QC')
        A.require(manifest['config_record'] == A.record(A.HERE / 'analysis_config.json'),
                  'Changed statistical configuration')
        A.require(qc['qc_code'] == A.record(A.HERE / '07_analysis_qc.py'), 'Changed statistical QC code')
        registry = self.json(A.ROOT / 'src/common/feature_registry.json')['temporal_features']
        self.features = [item['name'] for item in registry]
        self.derived = {item['name']: item['derived'] for item in registry}
        A.require(len(set(self.features)) == 50, 'Expected 50 unique temporal features')

    def current(self):
        path = OUT / 'manifest.json'
        if self.args.force or not path.exists():
            return False
        saved = A.read_json(path)
        if (saved.get('status') != 'VERIFIED' or saved.get('code') != self.code
                or saved.get('protocol') != PROTOCOL
                or set(saved.get('outputs', {})) != {n+'.csv' for n in TABLE_NAMES}):
            return False
        for key, record in saved['inputs'].items():
            target = (A.ROOT / key).resolve()
            if not target.is_relative_to(A.ROOT) or not target.is_file() or A.record(target) != record:
                return False
        for key, record in saved['outputs'].items():
            target = OUT / key
            if not target.is_file() or A.record(target) != {k: record[k] for k in ('sha256', 'size_bytes')}:
                return False
        return True

    def write(self, name, rows):
        A.require(name in TABLE_NAMES and rows, 'Unexpected or empty plot-data table')
        fields = A.CORE_FIELDS + sorted(set().union(*(row.keys() for row in rows)) - set(A.CORE_FIELDS))
        destination = OUT / (name + '.csv')
        temporary = destination.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow(A.clean(dict(partition='all', population='clinical_cohort',
                                             interval_method='none', result_id=f'08:{name}:{i}', **row)))
        temporary.replace(destination)
        self.outputs[destination.name] = dict(A.record(destination), rows=len(rows), columns=fields)

    def finish(self):
        A.require(set(self.outputs) == {n+'.csv' for n in TABLE_NAMES}, 'Missing plot-data output')
        for key, record in {**self.inputs, **self.code}.items():
            A.require(A.record(A.ROOT / key) == record, f'Input changed during plot-data export: {key}')
        A.write_json(OUT / 'manifest.json', dict(
            status='VERIFIED', version='1.0.0', protocol=PROTOCOL,
            datasets=self.config['datasets'], features=self.features,
            code=self.code, inputs=self.inputs, outputs=self.outputs, audit=self.audit))


def describe_cohort(run, name):
    folder = A.ROOT / 'data/processed' / name
    support = run.npz(folder / 'tensor_support.npz')
    order, n = support['features'].tolist(), len(support['stay_ids'])
    A.require(n > 0, 'Empty clinical cohort')
    A.require(set(order) == set(run.features) and len(order) == 50, 'Temporal schema differs')
    A.require(len(np.unique(support['subject_ids'])) == n, 'Repeated patient contributions')
    raw = np.load(run.track(folder / 'tensor_observed.npy'), mmap_mode='r', allow_pickle=False)
    filled = np.load(run.track(folder / 'tensor_imputed.npy'), mmap_mode='r', allow_pickle=False)
    A.require(raw.shape == filled.shape == (n, 24, 50), 'Unexpected tensor dimensions')
    means = np.full((n, 50), np.nan)
    observed_hours = np.zeros((n, 50), dtype=np.int16)
    completed_hours = np.zeros_like(observed_hours)
    exposed = support['exposure_seconds'] > 0
    nodes = []
    for j, feature in enumerate(run.features):
        values, completed = raw[:, :, order.index(feature)], filled[:, :, order.index(feature)]
        seen, available = np.isfinite(values), np.isfinite(completed)
        A.require(not np.isinf(values).any() and not np.isinf(completed).any(), 'Infinite tensor value')
        A.require(not seen[~exposed].any() and not available[~exposed].any(), 'Values outside follow-up')
        A.require(np.array_equal(values[seen], completed[seen]), 'Observed values changed by completion')
        observed_hours[:, j], completed_hours[:, j] = seen.sum(axis=1), available.sum(axis=1)
        means[:, j] = np.divide(np.where(seen, values, 0).sum(axis=1), observed_hours[:, j],
                               out=np.full(n, np.nan), where=observed_hours[:, j] > 0)
        count = int((observed_hours[:, j] > 0).sum())
        nodes.append(dict(dataset=name, question='feature_observation', feature=feature,
                          metric='patients_with_observation', unit='fraction', estimate=count/n,
                          numerator=count, n_patients=n, derived=run.derived[feature], status='descriptive'))

    relationships = []
    for i, feature in enumerate(run.features):
        for j in range(i+1, 50):
            complete = np.isfinite(means[:, i]) & np.isfinite(means[:, j])
            pair_n, rho, reason = int(complete.sum()), np.nan, 'insufficient_pairs'
            if pair_n >= PROTOCOL['minimum_correlation_patients']:
                x, y = rankdata(means[complete, i]), rankdata(means[complete, j])
                x, y = x-x.mean(), y-y.mean()
                scale = np.sqrt(np.dot(x, x) * np.dot(y, y))
                if scale > 0:
                    rho, reason = float(np.clip(np.dot(x, y)/scale, -1, 1)), ''
                else:
                    reason = 'constant_feature'
            relationships.append(dict(
                dataset=name, question='clinical_relationships', feature=feature,
                other_feature=run.features[j], metric='observed_patient_mean_spearman', unit='rho',
                estimate=rho, n_patients=pair_n, cohort_patients=n,
                joint_observed_fraction=pair_n/n, not_estimable_reason=reason,
                status='descriptive' if np.isfinite(rho) else 'not_estimable',
                derived_pair=run.derived[feature] or run.derived[run.features[j]]))

    surfaces = []
    grids = {}
    for view, hours in [('observed', observed_hours), ('completed', completed_hours)]:
        grid = np.zeros((24, 50), dtype=np.int64)
        for h in range(1, 25):
            breadth = (hours >= h).sum(axis=1)
            histogram = np.bincount(breadth, minlength=51)
            counts = np.cumsum(histogram[::-1])[::-1][1:]
            grid[h-1] = counts
            for k, count in enumerate(counts, 1):
                surfaces.append(dict(
                    dataset=name, question='study_support', method=view,
                    metric='patients_meeting_breadth_and_depth', unit='fraction',
                    minimum_features=k, minimum_hours=h, numerator=int(count),
                    n_patients=n, estimate=float(count/n), status='descriptive',
                    note='Any k features; feature sets may differ between patients. '
                         'Available hourly bins need not be consecutive or simultaneous across features. '
                         'Completed includes imputed values; partial-exposure bins count when a value is available.'))
        A.require((grid >= 0).all() and (grid <= n).all()
                  and (np.diff(grid, axis=0) <= 0).all() and (np.diff(grid, axis=1) <= 0).all(),
                  'Invalid study-support surface')
        grids[view] = grid
    A.require((grids['completed'] >= grids['observed']).all(), 'Completion lost observed study support')
    A.require(len(relationships) == 1225 and len(nodes) == 50 and len(surfaces) == 2400,
              'Incomplete all-feature plot-data inventory')
    run.audit.append(dict(dataset=name, patients=n, features=50, pairs=1225,
                          unique_patients=True, observed_values_preserved=True,
                          monotone_support_surfaces=True))
    return relationships, nodes, surfaces


def main():
    run = PlotDataRun()
    if run.current():
        print('[PLOT DATA] Current; using verified cached tables', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    A.write_json(OUT / 'manifest.json', dict(status='INCOMPLETE', protocol=PROTOCOL))
    try:
        tables = {name: [] for name in TABLE_NAMES}
        for name in run.config['datasets']:
            print(f'[PLOT DATA] {name}: all 50 temporal features', flush=True)
            for table, rows in zip(TABLE_NAMES, describe_cohort(run, name)):
                tables[table].extend(rows)
        for name, rows in tables.items():
            run.write(name, rows)
        run.finish()
    except Exception as error:
        A.write_json(OUT / 'manifest.json', dict(status='FAIL', error=str(error)))
        raise
    print(f'[PLOT DATA] Complete: {OUT}', flush=True)


if __name__ == '__main__':
    main()
