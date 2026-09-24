#!/usr/bin/env bash
# Build cohorts, atlas, analyses, figures and release in dependency order.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
OUTDIR=release
FROM=mimiciv:01
DEVICE=""
EXPORT_ONLY=0
NO_EXPORT=0
DRY_RUN=0
CHECK_ONLY=0
VISUALIZATION_OUTDIR=""
CURRENT=preflight
usage() {
  cat <<'HELP'
Usage: bash pipeline/01_build_dataset.sh [options]

Fresh build (Bash/WSL, Python 3.10+ and the full source checkout):
  python -m pip install -r docker/requirements.txt
  bash pipeline/01_build_dataset.sh --check-only
  bash pipeline/01_build_dataset.sh --device cuda
Use --device cpu when CUDA is unavailable. Run inside the installed environment.

Required raw exports, relative to the project root:
  data/raw/mimiciv/3.1/hosp/*.csv.gz
  data/raw/mimiciv/3.1/icu/*.csv.gz
  data/raw/mimic3-carevue/1.4/*.csv.gz  (uppercase names; CareVue-only release)
  data/raw/eicu-crd/2.0/*.csv[.gz]     (one copy per table, either format)
Directories can be mounted or linked at these paths. No precomputed tensors,
checkpoints, reports, figures or release files are needed for a full build.
Keep src/ and its JSON contracts/configuration from the same checkout.

  --check-only              Check environment, raw tables and output paths; write nothing
  --export-only             Package completed, QC-passed datasets; do not retrain
  --no-export               Run processing and QC without creating release archives
  --from DATASET:STAGE       Resume at e.g. eicu:04 or atlas:03 (inclusive)
  --outdir PATH             Release directory; replace managed files (default: release)
  --device DEVICE           Training/QC device, e.g. cuda or cpu
  --dry-run                 Print commands without running or writing anything
  --visualization-outdir PATH  Figure directory inside outputs/ (independent of --outdir)
  -h, --help                Show this help
Order: mimiciv (01-09), mimic3-carevue (01-09), eicu (01-09),
       atlas (01-06), analysis (01-08), visualization (01-06), then release export.
Analysis 08 exports separate descriptive plot data after the existing statistical QC.
Visualization resumes also prepare this cached plot data before figures 6 and 7.
Figures run src/06_visualization through 06_visualization_qc.py.
Figures export all main and supplementary panels and their assembled figures as PDF and PNG.
Standalone panels go in the figure directory's panels/ subdirectory.
--from runs the remaining sequence; preceding stages must already be complete
and match the current code/configuration. After shared visualization changes,
resume at visualization:01 to refresh all figures and their provenance.
PYTHON selects one interpreter path, without additional command-line arguments.

Outputs: data/processed/{mimiciv,mimic3-carevue,eicu,atlas}/; corresponding
outputs/ directories with checkpoints/reports/QC; outputs/statistical_analysis/
including relationship_plot_data/; outputs/visualization/; and release/ with
four ZIPs, data_dictionary.csv, cohort_flow.csv and release_manifest.json.
Execution logs append to outputs/logs/build_dataset.log.

Full builds train models and require substantial time and disk space. Existing
stage outputs are regenerated. Use the same raw versions and configuration for
comparable results; runtime/hardware can change trained values and timestamps.
The Docker requirements are a setup recipe, not an exact environment lock of
the historical run. A fresh rebuild is not expected to be byte-identical.
HELP
}
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }; }
while (($#)); do
  case "$1" in
    --outdir) need_value "$@"; OUTDIR="$2"; shift 2 ;;
    --from) need_value "$@"; FROM="$2"; shift 2 ;;
    --device) need_value "$@"; DEVICE="$2"; shift 2 ;;
    --export-only) EXPORT_ONLY=1; shift ;;
    --no-export) NO_EXPORT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --with-supplements) shift ;; # Compatibility: supplements are always included.
    --visualization-outdir) need_value "$@"; VISUALIZATION_OUTDIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$FROM" =~ ^(mimiciv|mimic3-carevue|eicu):0[1-9]$ || "$FROM" =~ ^(atlas|visualization):0[1-6]$ || "$FROM" =~ ^analysis:0[1-8]$ ]] || {
  echo "Invalid --from: $FROM" >&2; exit 2;
}
if (( EXPORT_ONLY && NO_EXPORT )); then
  echo 'Choose either --export-only or --no-export.' >&2; exit 2
fi
if (( CHECK_ONLY && DRY_RUN )); then
  echo 'Choose either --check-only or --dry-run.' >&2; exit 2
fi
if (( EXPORT_ONLY )) && [[ "$FROM" != mimiciv:01 ]]; then
  echo '--from cannot be used with --export-only.' >&2; exit 2
fi
run() {
  printf '[RUN]'; printf ' %q' "$@"; printf '\n'
  if (( ! DRY_RUN )); then "$@"; fi
}
if (( ! DRY_RUN )); then
  command -v "$PY" >/dev/null 2>&1 || { echo "Python interpreter not found: $PY" >&2; exit 2; }
  "$PY" - "$FROM" "$EXPORT_ONLY" "$NO_EXPORT" "${VISUALIZATION_OUTDIR:-outputs/visualization}" "$OUTDIR" "$DEVICE" <<'PY'
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit('Python 3.10 or newer is required.')
start, export_only, no_export, figures, release, device = sys.argv[1:]
root, errors = Path.cwd().resolve(), []
contracts = ['src/common/' + name + '.json' for name in ('feature_registry', 'tensor_schema', 'measurement_rules')]
contracts += ['src/' + folder + '/' + name + '.json' for folder, name in (
    ('01_mimiciv_datasets', 'dataset_config'), ('02_mimic3_datasets', 'dataset_config'),
    ('03_eicu_datasets', 'dataset_config'), ('04_atlas_datasets', 'atlas_config'),
    ('05_statistical_analysis', 'analysis_config'), ('06_visualization', 'visualization_config'))]
for filename in contracts:
    try:
        json.loads((root / filename).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        errors.append(f'Required configuration {filename}: {error}')
if not (root / 'pipeline/export_release.py').is_file():
    errors.append('Missing pipeline/export_release.py (also required by statistical analysis).')
if errors:
    raise SystemExit('[PREFLIGHT FAILED]\n' + '\n'.join('- ' + error for error in errors))
modules = ('numpy duckdb' if export_only == '1' else
           'numpy duckdb pandas pyarrow scipy matplotlib PIL' if start.startswith('visualization:') else
           'numpy duckdb torch pypots pandas pyarrow sklearn scipy ot joblib matplotlib threadpoolctl PIL').split()
for name in modules:
    try:
        module = importlib.import_module(name)
        print(f'[DEPENDENCY] {name} {getattr(module, "__version__", "available")}')
    except Exception as error:
        errors.append(f'{name}: {error}; install dependencies with this interpreter: -m pip install -r docker/requirements.txt')
if 'pypots' in modules:
    try:
        config = json.loads((root / 'src/01_mimiciv_datasets/dataset_config.json').read_text(encoding='utf-8'))
        expected, actual = config['imputation']['required_pypots_version'], importlib.metadata.version('pypots')
        if expected != actual:
            errors.append(f'PyPOTS must be {expected}; found {actual}')
        importlib.import_module('pypots.imputation').SAITS
    except Exception as error:
        errors.append(f'SAITS dependency: {error}')
if device and 'torch' in modules and 'torch' in sys.modules:
    try:
        sys.modules['torch'].empty(0, device=device)
    except Exception as error:
        errors.append(f'Device {device} is unavailable: {error}; use --device cpu if needed')
if export_only == '0':
    groups = ('mimiciv', 'mimic3-carevue', 'eicu', 'atlas', 'analysis', 'visualization')
    folders = ('01_mimiciv_datasets', '02_mimic3_datasets', '03_eicu_datasets',
               '04_atlas_datasets', '05_statistical_analysis', '06_visualization')
    limits = (9, 9, 9, 6, 8, 6)
    first, number = start.split(':')
    for group, folder, limit in zip(groups, folders, limits):
        if groups.index(group) < groups.index(first):
            continue
        for stage in range(int(number) if group == first else 1, limit + 1):
            if len(list((root / 'src' / folder).glob(f'{stage:02d}_*.py'))) != 1:
                errors.append(f'Expected one stage {stage:02d} script in src/{folder}')
    raw = {
        'mimiciv': ('data/raw/mimiciv/3.1', [
            'hosp/patients.csv.gz', 'hosp/admissions.csv.gz', 'hosp/prescriptions.csv.gz',
            'hosp/microbiologyevents.csv.gz', 'hosp/diagnoses_icd.csv.gz', 'hosp/labevents.csv.gz',
            'icu/icustays.csv.gz', 'icu/chartevents.csv.gz', 'icu/inputevents.csv.gz',
            'icu/outputevents.csv.gz', 'icu/procedureevents.csv.gz']),
        'mimic3-carevue': ('data/raw/mimic3-carevue/1.4', [name + '.csv.gz' for name in (
            'PATIENTS', 'ADMISSIONS', 'ICUSTAYS', 'PRESCRIPTIONS', 'MICROBIOLOGYEVENTS',
            'DIAGNOSES_ICD', 'CHARTEVENTS', 'LABEVENTS', 'INPUTEVENTS_CV', 'OUTPUTEVENTS', 'D_ITEMS', 'D_LABITEMS')]),
        'eicu': (json.loads((root / 'src/03_eicu_datasets/dataset_config.json').read_text(encoding='utf-8'))['eicu']['raw_directory'],
                 'patient apachePredVar medication microLab diagnosis vitalPeriodic vitalAperiodic lab nurseCharting '
                 'respiratoryCharting respiratoryCare intakeOutput infusionDrug')}
    for group in groups[groups.index(first):]:
        if group not in raw:
            continue
        directory, tables = raw[group]
        directory = root / directory
        if group == 'eicu':
            files = [p.name.lower() for p in directory.iterdir() if p.is_file()] if directory.is_dir() else []
            for table in tables.split():
                if sum(f in (table.lower() + '.csv', table.lower() + '.csv.gz') for f in files) != 1:
                    errors.append(f'Expected one {directory / table}.csv[.gz]')
        else:
            errors.extend(f'Missing raw table: {directory / table}' for table in tables
                          if not (directory / table).is_file())
    destination, base = Path(figures).resolve(), root / 'outputs'
    if (destination == base or not destination.is_relative_to(base) or any(destination.is_relative_to(base / name)
            for name in ('mimiciv', 'mimic3-carevue', 'eicu', 'atlas', 'statistical_analysis', 'logs'))):
        errors.append('Figures must be inside outputs/, separate from processing and log directories.')
    elif destination.exists() and not destination.is_dir():
        errors.append(f'Figure destination is not a directory: {destination}')
if no_export == '0':
    destination = Path(release).expanduser().resolve()
    if (root.is_relative_to(destination) or any(destination.is_relative_to(root / name)
            for name in ('src', 'data', 'outputs', 'pipeline', 'docker', '.git'))):
        errors.append('Choose a separate release destination, not a project/processing directory or its parent.')
    elif destination.exists() and not destination.is_dir():
        errors.append(f'Release destination is not a directory: {destination}')
if errors:
    raise SystemExit('[PREFLIGHT FAILED]\n' + '\n'.join('- ' + error for error in errors))
print('[PREFLIGHT PASS] Environment, required raw tables and output paths checked; no stages executed.')
PY
  if (( CHECK_ONLY )); then
    if (( EXPORT_ONLY )); then "$PY" -u pipeline/export_release.py --outdir "$OUTDIR" --check-only; fi
    exit 0
  fi
  mkdir -p outputs/logs
  exec > >(tee -a outputs/logs/build_dataset.log) 2>&1
  printf '\n[BUILD] %s; root=%s; python=%s; from=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$ROOT" "$PY" "$FROM"
  trap 'code=$?; printf "[FAILED] %s (exit %s). Fix the error before resuming; see outputs/logs/build_dataset.log.\n" "$CURRENT" "$code" >&2; exit "$code"' ERR
fi
if (( ! EXPORT_ONLY )); then
  names=(mimiciv mimic3-carevue eicu atlas analysis visualization)
  folders=(01_mimiciv_datasets 02_mimic3_datasets 03_eicu_datasets 04_atlas_datasets 05_statistical_analysis 06_visualization)
  started=0
  plot_data_ready=0
  for i in "${!names[@]}"; do
    limit=9; [[ "${names[$i]}" == atlas || "${names[$i]}" == visualization ]] && limit=6
    [[ "${names[$i]}" == analysis ]] && limit=8
    for ((stage=1; stage<=limit; stage++)); do
      printf -v number '%02d' "$stage"
      [[ "${names[$i]}:$number" == "$FROM" ]] && started=1
      (( started )) || continue
      CURRENT="${names[$i]}:$number"
      scripts=("src/${folders[$i]}/${number}_"*.py)
      [[ ${#scripts[@]} -eq 1 && -f "${scripts[0]}" ]] || {
        echo "Expected one stage $number script in ${folders[$i]}" >&2; exit 2;
      }
      args=()
      if [[ "${names[$i]}" == visualization ]]; then
        if [[ -n "$VISUALIZATION_OUTDIR" ]]; then args+=(--outdir "$VISUALIZATION_OUTDIR"); fi
        if [[ ( "$number" == 04 || "$number" == 05 ) && "$plot_data_ready" == 0 ]]; then
          CURRENT=analysis:08
          run "$PY" -u src/05_statistical_analysis/08_relationship_plot_data.py
          plot_data_ready=1
          CURRENT="${names[$i]}:$number"
        fi
      fi
      if [[ -n "$DEVICE" ]]; then
        if [[ "${names[$i]}" == atlas && ( "$number" == 02 || "$number" == 06 ) ]] ||
           [[ "${names[$i]}" == analysis && "$number" == 03 ]] ||
           [[ "${names[$i]}" != atlas && "${names[$i]}" != analysis && "${names[$i]}" != visualization && ( "$number" == 08 || "$number" == 09 ) ]]; then
          args+=(--device "$DEVICE")
        fi
      fi
      run "$PY" -u "${scripts[0]}" "${args[@]}"
      if [[ "${names[$i]}" == analysis && "$number" == 08 ]]; then plot_data_ready=1; fi
    done
  done
fi
if (( ! NO_EXPORT )); then
  CURRENT=release
  run "$PY" -u pipeline/export_release.py --outdir "$OUTDIR"
fi
if (( DRY_RUN )); then
  echo '[DRY RUN] No scripts executed.'
elif (( NO_EXPORT )); then
  echo '[DONE] Processing and dataset-validation QC complete; no release archives created.'
else
  echo "[DONE] Release: $OUTDIR"
fi
