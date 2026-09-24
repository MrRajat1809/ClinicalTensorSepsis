"""Shared input binding, patient-level estimation and flat analysis exports."""
import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {'mimiciv': '01_mimiciv_datasets', 'mimic3-carevue': '02_mimic3_datasets', 'eicu': '03_eicu_datasets'}
TABLES = {
    1: ('cohort_selection',), 2: ('measurement_quality', 'coverage_profiles'),
    3: ('imputation_comparisons', 'imputation_error_profiles'),
    4: ('transport_effects', 'clinical_preservation'),
    5: ('feature_readiness', 'dataset_readiness'),
    6: ('site_and_split_tests', 'site_and_split_summaries'),
}
CORE_FIELDS = ['result_id', 'question', 'dataset', 'partition', 'population', 'stratum', 'feature',
               'scenario', 'method', 'comparator', 'metric', 'unit', 'estimate', 'ci_low', 'ci_high',
               'n_patients', 'n_observations', 'n_reference', 'n_hospitals', 'bootstrap_valid',
               'bootstrap_requested', 'interval_method', 'status', 'note']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return dict(sha256=h.hexdigest(), size_bytes=Path(path).stat().st_size)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return clean(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rng_for(config, *keys):
    digest = hashlib.sha256(':'.join(map(str, (config['seed'], *keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def finite_mean(values, axis=None):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(np.shape(count), np.nan), where=count > 0)


def interval(estimate, draws, config, n, requested=None, method='patient_percentile_bootstrap'):
    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    requested = len(draws) if requested is None else requested
    enough = np.isfinite(estimate) and n >= config['minimum_patients'] and requested > 0 and len(valid) >= requested * config['minimum_valid_bootstrap_fraction']
    lo, hi = np.quantile(valid, [(1-config['confidence_level'])/2, (1+config['confidence_level'])/2]) if enough else (np.nan, np.nan)
    return dict(estimate=float(estimate), ci_low=lo, ci_high=hi, n_patients=int(n),
                bootstrap_valid=len(valid), bootstrap_requested=requested, interval_method=method,
                status='estimated' if enough else 'insufficient_precision' if np.isfinite(estimate) else 'not_estimable')


def bootstrap_mean(values, config, key, replicates=None):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    b = replicates or config['bootstrap_replicates']
    if len(x) < config['minimum_patients']:
        return interval(finite_mean(x), [], config, len(x), requested=b)
    rng, draws = rng_for(config, key), []
    for start in range(0, b, 32):
        indices = rng.integers(0, len(x), (min(32, b-start), len(x)))
        draws.extend(x[indices].mean(axis=1))
    return interval(x.mean(), draws, config, len(x))


def independent_difference(a, b, config, key, standardized=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    def effect(x, y):
        delta = finite_mean(x) - finite_mean(y)
        if standardized:
            scale = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1))/2) if min(len(x),len(y)) > 1 else np.nan
            return delta / scale if scale > 0 else np.nan
        return delta
    draws, rng = [], rng_for(config, key)
    if min(len(a),len(b)) >= config['minimum_patients']:
        for _ in range(config['bootstrap_replicates']):
            draws.append(effect(a[rng.integers(len(a),size=len(a))], b[rng.integers(len(b),size=len(b))]))
    result = interval(effect(a,b), draws, config, min(len(a),len(b)), requested=config['bootstrap_replicates'])
    result.update(n_patients=len(a), n_reference=len(b))
    return result


def proportion(count, total, config):
    if not total:
        return dict(estimate=np.nan, ci_low=np.nan, ci_high=np.nan, n_patients=0,
                    status='not_estimable', interval_method='wilson')
    z = norm.ppf((1+config['confidence_level'])/2)
    p, d = count/total, 1+z*z/total
    center = (p+z*z/(2*total))/d
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return dict(estimate=p, ci_low=max(0,center-radius), ci_high=min(1,center+radius),
                n_patients=int(total), numerator=int(count), status='estimated', interval_method='wilson')


def risk_difference(a,b,config):
    a,b=np.asarray(a,float),np.asarray(b,float)
    a,b=a[np.isfinite(a)],b[np.isfinite(b)]
    if not len(a) or not len(b):
        return dict(estimate=np.nan,ci_low=np.nan,ci_high=np.nan,n_patients=len(a),n_reference=len(b),
                    status='not_estimable',interval_method='newcombe_wilson_difference')
    left,right=proportion(int(a.sum()),len(a),config),proportion(int(b.sum()),len(b),config)
    delta=left['estimate']-right['estimate']
    lower=delta-np.hypot(left['estimate']-left['ci_low'],right['ci_high']-right['estimate'])
    upper=delta+np.hypot(left['ci_high']-left['estimate'],right['estimate']-right['ci_low'])
    return dict(estimate=delta,ci_low=max(-1,lower),ci_high=min(1,upper),n_patients=len(a),n_reference=len(b),
                status='estimated',interval_method='newcombe_wilson_difference')


def descriptive(value, n=0, **extra):
    return dict(estimate=value, n_patients=n, status='descriptive', interval_method='none', **extra)


def test_fields(name, null, p=np.nan, statistic=np.nan, valid=False):
    return dict(test_name=name, test_null=null, p_value=float(p), test_statistic=float(statistic),
                test_status='tested' if valid else 'not_estimable')


def two_sample_tests(left, right, config, binary=False):
    """Independent patients within a source; U tests distributions, not necessarily medians."""
    x, y = np.asarray(left, float), np.asarray(right, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    enough = min(len(x), len(y)) >= config['minimum_patients']
    if binary:
        result = risk_difference(x, y, config)
        p = stats.fisher_exact([[int(x.sum()), len(x)-int(x.sum())],
                               [int(y.sum()), len(y)-int(y.sum())]])[1] if enough else np.nan
        return [dict(metric='tested_risk_difference', **result,
                     **test_fields('fisher_exact', 'Equal outcome/availability proportions', p, valid=enough))]
    delta = finite_mean(x)-finite_mean(y)
    result = dict(metric='welch_mean_difference', **descriptive(delta, len(x)), n_reference=len(y),
                  **test_fields('welch_t', 'Equal population means'))
    if enough:
        vx, vy = np.var(x, ddof=1)/len(x), np.var(y, ddof=1)/len(y)
        if vx+vy > 0:
            df = (vx+vy)**2/(vx*vx/(len(x)-1)+vy*vy/(len(y)-1))
            t = delta/np.sqrt(vx+vy)
            margin = stats.t.ppf((1+config['confidence_level'])/2, df)*np.sqrt(vx+vy)
            result.update(ci_low=delta-margin, ci_high=delta+margin, status='estimated',
                          interval_method='welch_t_interval', degrees_of_freedom=df,
                          **test_fields('welch_t', 'Equal population means', 2*stats.t.sf(abs(t), df), t, True))
    rank = dict(metric='rank_biserial_difference', **descriptive(np.nan, len(x)), n_reference=len(y),
                **test_fields('mann_whitney_u', 'Identical population distributions'))
    if len(x) and len(y):
        sx, sy = np.sort(x), np.sort(y)
        px = (np.searchsorted(sy, x, 'left')+np.searchsorted(sy, x, 'right'))/(2*len(y))
        py = 1-(np.searchsorted(sx, y, 'left')+np.searchsorted(sx, y, 'right'))/(2*len(x))
        effect = 2*px.mean()-1
        rank['estimate'] = effect
        if enough:
            u, p = stats.mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
            se = 2*np.sqrt(np.var(px, ddof=1)/len(x)+np.var(py, ddof=1)/len(y))
            margin = norm.ppf((1+config['confidence_level'])/2)*se
            rank.update(ci_low=max(-1, effect-margin), ci_high=min(1, effect+margin), status='estimated',
                        interval_method='asymptotic_placement_interval',
                        **test_fields('mann_whitney_u', 'Identical population distributions', p, u, True))
    return [result, rank]


def paired_tests(delta, config):
    """One difference per patient: mean-shift t test and symmetry-free exact sign test."""
    x = np.asarray(delta, float)
    x = x[np.isfinite(x)]
    n = len(x)
    enough = n >= config['minimum_patients']
    mean = dict(metric='paired_t_mean_difference', **descriptive(finite_mean(x), n),
                **test_fields('paired_t', 'Mean paired difference equals zero'))
    if enough and np.var(x, ddof=1) > 0:
        se = np.std(x, ddof=1)/np.sqrt(n)
        t = x.mean()/se
        margin = stats.t.ppf((1+config['confidence_level'])/2, n-1)*se
        mean.update(ci_low=x.mean()-margin, ci_high=x.mean()+margin, status='estimated',
                    interval_method='paired_t_interval', degrees_of_freedom=n-1,
                    **test_fields('paired_t', 'Mean paired difference equals zero', 2*stats.t.sf(abs(t), n-1), t, True))
    positive, negative = int((x>0).sum()), int((x<0).sum())
    nonzero = positive+negative
    sign = dict(metric='positive_difference_fraction_among_nonties', **proportion(positive, nonzero, config),
                paired_patients=n, tied_patients=n-nonzero,
                **test_fields('paired_sign', 'Positive and negative differences are equally likely among nonties',
                              stats.binomtest(positive, nonzero, .5).pvalue if enough and nonzero else np.nan,
                              positive, enough and nonzero>0))
    return [mean, sign]


def adjust_pvalues(rows):
    """Benjamini-Hochberg within explicit, predeclared hypothesis families."""
    families = {}
    for row in rows:
        if row.get('test_name'):
            family = row.get('test_family') or ':'.join(str(row.get(k,'')) for k in
                ('dataset','question','population','scenario','comparator','test_name'))
            row['test_family'] = family
            if np.isfinite(row.get('p_value', np.nan)):
                families.setdefault(family, []).append(row)
    for values in families.values():
        values.sort(key=lambda row: row['p_value'])
        p = np.array([r['p_value'] for r in values])
        adjusted = np.minimum.accumulate((p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
        for row, q in zip(values, adjusted):
            row.update(q_value_bh=float(max(row['p_value'],min(1,q))), family_test_count=len(values), multiplicity='Benjamini-Hochberg')


def output_directory(path):
    path = Path(path)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    require(path.is_relative_to(ROOT/'outputs') and path != ROOT/'outputs', 'Use a dedicated folder under outputs/')
    require(not any(path == ROOT/'outputs'/n or (ROOT/'outputs'/n) in path.parents for n in (*SOURCES,'atlas')), 'Do not overwrite processing outputs')
    return path


class Run:
    def __init__(self, stage, argv=None):
        parser = argparse.ArgumentParser(description=f'Dataset-validation statistical analysis {stage:02d}')
        parser.add_argument('--config', type=Path, default=HERE/'analysis_config.json')
        parser.add_argument('--outdir', type=Path, default=ROOT/'outputs/statistical_analysis')
        if stage == 3:
            parser.add_argument('--device', default=None, help='cpu or cuda; never changes the saved recipe')
        self.args = parser.parse_args(argv)
        self.stage, self.inputs, self.outputs, self.audit = stage, {}, {}, []
        self.config = read_json(self.args.config)
        require(self.config['version']=='1.0.0' and self.config['datasets']==list(SOURCES), 'Analysis version/dataset order mismatch')
        require(self.config['reference']=='mimiciv' and isinstance(self.config['seed'],int) and self.config['seed']>=0, 'Invalid reference/seed')
        for key in ('bootstrap_replicates','minimum_patients'):
            require(isinstance(self.config[key],int) and self.config[key]>0, f'Invalid {key}')
        require(0<self.config['confidence_level']<1 and 0<self.config['minimum_valid_bootstrap_fraction']<=1, 'Invalid interval configuration')
        require(self.config['imputation']['scenarios']==['point','block6h','whole_channel'], 'Use the three frozen holdout scenarios')
        require(self.config['imputation']['comparators']==['median','forward_fill'], 'Use the fixed imputation comparisons')
        for key in ('minimum_hospital_patients',):
            require(isinstance(self.config['readiness'][key], int) and self.config['readiness'][key]>0, f'Invalid {key}')
        require(isinstance(self.config['tests']['minimum_hospital_patients'], int)
                and self.config['tests']['minimum_hospital_patients']>=self.config['minimum_patients'],
                'Hospital test threshold must cover the minimum inferential sample size')
        for key in ('minimum_patient_coverage', 'generated_fraction_warning'):
            require(0<self.config['readiness'][key]<1, f'Invalid {key}')
        self.out = output_directory(self.args.outdir)
        self.out.mkdir(parents=True,exist_ok=True)
        self.manifest_path = self.out/'analysis_manifest.json'
        self.code = {p.relative_to(ROOT).as_posix():record(p) for p in [HERE/'analysis_common.py', *HERE.glob(f'{stage:02d}_*.py')]}
        self.config_record = record(self.args.config)
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {'version':'1.0.0','stages':{}}
        if self.manifest.get('config_record') != self.config_record:
            self.manifest['stages'] = {}
        self.manifest.update(config=self.config,config_record=self.config_record, protocol=self.config['protocol'],status='INCOMPLETE')
        self.manifest['stages'].pop(str(stage),None)
        write_json(self.manifest_path,self.manifest)
        write_json(self.out/'qc.json',dict(status='STALE',reason=f'Analysis {stage:02d} started'))
        self.expected = {}
        self._certificates()
        self.atlas_inputs = self.npz(ROOT/'data/processed/atlas/inputs.npz')
        for key in ('clinical_features',):
            require(len(set(self.config[key]))==len(self.config[key]) and set(self.config[key])<=set(self.atlas_inputs['feature_names']),f'Unknown/duplicated {key}')
        print(f'[ANALYSIS {stage:02d}] v1.0.0; dataset validation; post-hoc estimates',flush=True)

    def _certificates(self):
        def merge(records):
            for key,value in records.items():
                require(key not in self.expected or self.expected[key]==value, f'Conflicting input binding: {key}')
                self.expected[key] = value
        qc = self.json(ROOT/'outputs/atlas/qc.json')
        require(qc['status']=='PASS' and qc['failures']==0, 'Run atlas master QC first')
        merge(qc['code']); merge(qc['dependencies'])
        for number in range(1,6):
            report = self.json(ROOT/f'outputs/atlas/{number:02d}.json')
            require(report['version']=='1.0.0' and report['status']=='COMPLETE','Incomplete atlas inputs')
            for section in ('code','dependencies','outputs','external'):
                merge(report.get(section,{}))
        for name in SOURCES:
            folder = ROOT/'data/processed'/name
            manifest = self.json(folder/'manifest.json')
            qc = self.json(ROOT/'outputs'/name/'qc.json')
            require(qc['status']=='PASS' and qc['failed_required_checks']==0, f'{name}: source QC incomplete')
            require(qc['manifest_sha256']==record(folder/'manifest.json')['sha256'],f'{name}: stale source QC')
            require(manifest['dataset_version']=='1.0.0','Wrong source version')
            merge(manifest['observed']['provenance']); merge(manifest['imputation']['artifacts'])
            merge({(folder/k).relative_to(ROOT).as_posix():v for k,v in manifest['observed']['output_manifest'].items()})
        # Validate code/config certificates now; large data files are verified when read.
        for key in self.expected:
            if key.startswith('src/'):
                self.track(ROOT/key)
        for key,value in self.inputs.items():
            require(key not in self.expected or value==self.expected[key],f'Changed certified report: {key}')

    def track(self,path):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT),f'Input outside project: {path}')
        key = path.relative_to(ROOT).as_posix()
        if key not in self.inputs:
            value = record(path)
            require(key not in self.expected or value==self.expected[key],f'Changed certified input: {key}')
            self.inputs[key] = value
        return path

    def json(self,path):
        return read_json(self.track(path))

    def npz(self,path):
        with np.load(self.track(path),allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def parquet(self,path):
        import duckdb
        with duckdb.connect() as con:
            cur = con.execute('SELECT * FROM read_parquet(?)',[str(self.track(path))])
            names = [c[0] for c in cur.description]
            return [dict(zip(names,row)) for row in cur.fetchall()]

    def query_parquet(self,path,sql):
        import duckdb
        with duckdb.connect() as con:
            cur = con.execute(sql,[str(self.track(path))])
            columns = [item[0] for item in cur.description]
            return [dict(zip(columns,row)) for row in cur.fetchall()]

    def csv(self,path):
        with self.track(path).open(encoding='utf-8',newline='') as f:
            return list(csv.DictReader(f))

    def source(self,name):
        folder = ROOT/'data/processed'/name
        a = self.npz(folder/'tensor_support.npz')
        a.update(self.npz(folder/'imputation_support.npz'))
        a['raw'] = np.load(self.track(folder/'tensor_observed.npy'),mmap_mode='r',allow_pickle=False)
        a['patients'] = self.parquet(folder/'cohort.parquet')
        require(np.array_equal(a['stay_ids'],[r['stay_id'] for r in a['patients']]),f'{name}: cohort row order')
        require(len(np.unique(a['subject_ids']))==len(a['stay_ids']),f'{name}: expected one stay per subject')
        domain = list(SOURCES).index(name)
        rows = np.flatnonzero(self.atlas_inputs['dataset']==domain)
        for key,source in [('stay_id','stay_ids'),('subject_id','subject_ids'),('partition','partition')]:
            require(np.array_equal(self.atlas_inputs[key][rows],a[source]),f'{name}: atlas {key} join')
        require(np.array_equal(self.atlas_inputs['source_row'][rows],np.arange(len(rows))),f'{name}: source row join')
        a['atlas_rows'], a['eligible'] = rows, self.atlas_inputs['eligible'][rows]
        a['followup'] = a['exposure_seconds'].sum(axis=1)/3600
        exposed = (a['exposure_seconds']>0).sum(axis=1)*len(a['features'])
        a['coverage'] = np.divide(np.isfinite(a['raw']).sum(axis=(1,2)),exposed,out=np.full(len(rows),np.nan),where=exposed>0)
        self.audit.append(dict(check='patient_join',dataset=name,patients=len(rows),unique_subjects=len(np.unique(a['subject_ids']))))
        return a

    def write(self,name,rows):
        require(name in TABLES[self.stage],f'Unexpected table: {name}')
        rows = list(rows)
        require(rows, f'Empty analysis table: {name}')
        adjust_pvalues(rows)
        fields = CORE_FIELDS + sorted({k for row in rows for k in row}-set(CORE_FIELDS))
        destination = self.out/(name+'.csv')
        temporary = destination.with_name(destination.name+'.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for i,row in enumerate(rows):
                require('metric' in row and 'status' in row, f'Incomplete result: {name}/{i}')
                row = dict({'partition':'all','population':'clinical_cohort'}, **row)
                writer.writerow(clean(dict(row, result_id=f'{self.stage:02d}:{name}:{i}')))
        temporary.replace(destination)
        self.outputs[destination.name] = dict(**record(destination),rows=len(rows),columns=fields)

    def finish(self):
        require(set(self.outputs)=={n+'.csv' for n in TABLES[self.stage]},'Missing analysis tables')
        for name,value in self.inputs.items():
            require(record(ROOT/name)==value,f'Input changed during analysis: {name}')
        require(record(self.args.config)==self.config_record,'Analysis config changed during execution')
        for name,value in self.code.items():
            require(record(ROOT/name)==value,'Analysis code changed during execution')
        runtime = {'python':sys.version.split()[0]}
        for name in ('numpy','scipy','scikit-learn','duckdb','torch','pypots'):
            try: runtime[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: runtime[name] = None
        self.manifest['stages'][str(self.stage)] = dict(status='COMPLETE',inputs=self.inputs,outputs=self.outputs,
            code=self.code,audit=self.audit,runtime=runtime,completed_at=datetime.now(timezone.utc).isoformat())
        self.manifest['status'] = 'AWAITING_QC'
        self.manifest['export_conventions'] = dict(
            missing='Empty CSV field means unavailable/not applicable, never zero.',
            effects='Comparisons use first named method/group minus comparator unless metric description states otherwise.',
            intervals='Pointwise percentile intervals, Wilson proportions, Newcombe-Wilson independent risk differences, or test-specific analytical intervals.',
            uncertainty='Conditional on fixed preprocessing. P-values accompany effects and intervals; BH q-values apply within named families, with no global guarantee under arbitrary dependence.',
            p_values='Two-sided unless the test null states otherwise. Zero can reflect numerical underflow; it is not a literal zero probability. No automatic acceptance or feature deletion from p-values.',
            counts='n_patients is the number contributing to the estimate; n_group can include patients missing the analyzed variable.',
            unavailable='Sparse groups retain available point estimates with blank intervals and an explicit status.',
            method_recipe='Imputation method final is the saved feature-specific SAITS/baseline recipe, including out-of-bounds fallbacks.')
        write_json(self.manifest_path,self.manifest)
        print(f'[ANALYSIS {self.stage:02d}] Complete; {len(self.outputs)} tables in {self.out}',flush=True)
