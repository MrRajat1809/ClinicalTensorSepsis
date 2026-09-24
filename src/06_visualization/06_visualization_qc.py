"""Check rendered artifacts, source bindings and exported numerical summaries."""

import argparse
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image
import visualization_common as V


def equal_value(left, right):
    if left == right:
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-14)
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=V.ROOT / 'outputs/visualization')
    parser.add_argument('--config', type=Path, default=V.HERE / 'visualization_config.json')
    parser.add_argument('--supplementary', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.outdir.resolve()
    V.require(out.is_relative_to((V.ROOT / 'outputs').resolve())
        and out.name not in ('atlas', 'statistical_analysis', 'mimiciv', 'mimic3-carevue', 'eicu'),
        'Invalid visualization output')
    checks = []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        V.require(condition, f'Visualization QC failed: {name}')
    try:
        manifest = V.read_json(out / 'manifest.json')
        config = V.read_json(args.config)
        check('presentation_mode', manifest.get('supplementary') is True)
        check('configuration', manifest['version'] == config['version'] == '1.0.0' and config['dpi'] == 300
            and V.record(args.config) == manifest['config_record'])
        check('complete_stages', set(manifest['stages']) == {'1', '2', '3', '4', '5'})
        qc = V.read_json(V.ANALYSIS / 'qc.json')
        analysis = V.read_json(V.ANALYSIS / 'analysis_manifest.json')
        check('frozen_analysis', qc['status'] == 'PASS' and qc['failures'] == 0
            and V.record(V.ANALYSIS / 'analysis_manifest.json') == qc['analysis_manifest'])
        check('analysis_qc_code', V.record(V.ROOT / 'src/05_statistical_analysis/07_analysis_qc.py') == qc['qc_code'])
        records = {}

        def merge(section):
            for key, value in section.items():
                V.require(key not in records or records[key] == value, f'Conflicting binding: {key}')
                records[key] = value
        for entry in analysis['stages'].values():
            merge(entry['inputs'])
            merge(entry['code'])
            merge({'outputs/statistical_analysis/' + k: {f: v[f] for f in ('sha256', 'size_bytes')}
                    for k, v in entry['outputs'].items()})
        merge({'src/05_statistical_analysis/analysis_config.json': analysis['config_record']})
        with (out / 'figure_data.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        check('figure_inventory', {r['figure'] for r in rows} == {f'Fig{i}' for i in range(1, 8)})
        for stage, numbers in V.FIGURES.items():
            entry = manifest['stages'][str(stage)]
            check(f'stage_{stage}_complete', entry['status'] == 'COMPLETE')
            merge(entry['inputs'])
            merge(entry['code'])
            selected = [r for r in rows if r['figure'] in {f'Fig{n}' for n in numbers}]
            check(f'stage_{stage}_summary_binding',
                len(selected) == entry['data_rows'] and V.data_digest(selected) == entry['data_sha256'])
            V.validate_artifacts(stage, entry['artifacts'], selected)
            check(f'stage_{stage}_panel_inventory', True)
            check(f'stage_{stage}_outputs',
                set(entry['outputs']) == {V.artifact_path(a['number'], a['panel'], ext)
                    for a in entry['artifacts'].values() for ext in ('png', 'pdf')})
            for filename, expected in entry['outputs'].items():
                path = out / filename
                check(filename + '_hash', V.record(path) == {k: expected[k] for k in ('sha256', 'size_bytes')})
                if path.suffix == '.pdf':
                    with path.open('rb') as stream:
                        check(filename + '_format', stream.read(5) == b'%PDF-')
                else:
                    with Image.open(path) as image:
                        check(filename + '_resolution',
                            image.format == 'PNG' and all(abs(d - 300) < 0.1 for d in image.info.get('dpi', [0, 0])))
                        check(filename + '_dimensions', all(abs(actual - inch * 300) <= 1 for actual, inch in zip(
                                    image.size, [expected['width_inches'], expected['height_inches']])))
                        image.thumbnail((500, 500))
                        pixels = np.asarray(image.convert('RGB'))
                        check(filename + '_nonblank', (pixels.min(axis=2) < 235).mean() > 0.005)
        print(f'[VIS QC] Verifying {len(records)} bound files', flush=True)
        check('all_source_and_code_hashes', all(V.record(V.ROOT / p) == expected for p, expected in records.items()))
        tables = {}
        for source in {r['source'] for r in rows if r.get('result_id')}:
            with (V.ANALYSIS / (source + '.csv')).open(encoding='utf-8', newline='') as stream:
                tables[source] = {r['result_id']: r for r in csv.DictReader(stream)}
        for row in rows:
            if row.get('result_id'):
                original = tables[row['source']][row['result_id']]
                for field in original:
                    V.require(equal_value(row.get(field, ''), original[field]),
                        f'Changed analysis value: {row["result_id"]}/{field}')
            if row['metric'] == 'within_followup_cell_fraction':
                V.require(equal_value(row['estimate'], float(row['numerator']) / float(row['denominator'])),
                    'Incorrect cell fraction')
            if row['metric'] == 'relative_patient_mean_mae_change':
                baseline = float(row['baseline_mae'])
                expected = (float(row['final_mae']) - baseline) / baseline if baseline > 0 else None
                V.require(row['estimate'] == '' if expected is None else equal_value(row['estimate'], expected),
                    'Incorrect relative MAE')
            if row.get('display_error_ratio'):
                comparison = tables['clinical_preservation'][row['probe_comparison_id']]
                V.require(all(row[k] == comparison[k] for k in ('dataset', 'feature', 'stratum')),
                    'Mismatched probe comparison')
                permitted = float(row['error_ratio_limit']) * float(comparison['original_mae']) + float(
                    row['absolute_slack'])
                V.require(permitted > 0 and equal_value(row['permitted_error'], permitted), 'Incorrect permitted error'
                )
                V.require(equal_value(row['display_error_ratio'], float(comparison['adapted_mae']) / permitted),
                    'Incorrect normalized preservation display')
        check('plotted_values_and_intervals', True)
        manifest['status'] = 'VERIFIED'
        V.write_json(out / 'manifest.json', manifest)
        report = dict(version='1.0.0', status='PASS', failures=0, checks=checks,
            manifest=V.record(out / 'manifest.json'), figure_data=V.record(out / 'figure_data.csv'),
            qc_code=V.record(Path(__file__)),
            interpretation='Rendering and provenance verified; scientific limitations and unavailable intervals remain explicit.'
        )
        V.write_json(out / 'qc.json', report)
        print(f'[VIS QC PASS] {len(checks)} checks; Fig1–Fig7; PDF + 300 dpi PNG', flush=True)
    except Exception as error:
        V.write_json(out / 'qc.json', dict(version='1.0.0', status='FAIL', checks=checks, failures=1, error=str(error))
        )
        raise


if __name__ == '__main__':
    main()
