"""Train and apply gap-aware SAITS with validation-selected baseline fallbacks."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data/processed/mimiciv"
METRICS_DIR = BASE_DIR / "outputs/mimiciv"
COMMON_DIR = BASE_DIR / "src/common"
CONFIG_PATH = Path(__file__).with_name("dataset_config.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
TENSOR_SCHEMA_PATH = BASE_DIR / "src/common/tensor_schema.json"
TENSOR_SCHEMA = json.loads(TENSOR_SCHEMA_PATH.read_text(encoding="utf-8"))
if TENSOR_SCHEMA["schema_version"] != CONFIG["shared_schema_version"]:
    raise ValueError("Dataset and shared tensor schema versions differ")
if TENSOR_SCHEMA["policy"].keys() & CONFIG["tensor"].keys():
    raise ValueError("Dataset configuration must not override shared tensor settings")
CONFIG["tensor"] = dict(TENSOR_SCHEMA["policy"], **CONFIG["tensor"])
CONFIG["imputation"]["observed_only_features"] = TENSOR_SCHEMA["observed_only_features"]
DATASET_VERSION = CONFIG["dataset_version"]
OBSERVED_ONLY = set(TENSOR_SCHEMA["observed_only_features"])
SCENARIOS = ("point", "block6h", "whole_channel")
METHOD_CODES = {"unfilled": 0, "observed": 1, "saits": 2, "forward_fill": 3, "median": 4}
RULES = CONFIG["imputation"]["final_recipe"]["rules"]


def load_inputs(processed_dir=PROCESSED_DIR, common_dir=COMMON_DIR):
    """Read the observed contract and derive redundant masks in memory."""
    processed_dir, common_dir = Path(processed_dir), Path(common_dir)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    observed = manifest["observed"]
    if manifest.get("shared_schema_version") != TENSOR_SCHEMA["schema_version"]:
        raise ValueError("Observed manifest does not use the current shared tensor schema")
    if manifest.get("dataset_version") != DATASET_VERSION or observed.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported dataset version; run stages 01-07")
    inputs = {"observed_contract": hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()}
    for name, expected in observed["output_manifest"].items():
        path = processed_dir / name
        if path.stat().st_size != expected["size_bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Observed artifact changed: {name}")
        inputs[name] = expected["sha256"]
    for recorded, expected in observed["provenance"].items():
        path = (BASE_DIR / recorded).resolve()
        if not path.is_relative_to(BASE_DIR) or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Processing source/evidence changed: {recorded}")
    with np.load(processed_dir / "tensor_support.npz", allow_pickle=False) as support:
        arrays = {name: support[name] for name in support.files}
    arrays["raw"] = np.load(processed_dir / "tensor_observed.npy", allow_pickle=False)
    arrays["mask"] = np.isnan(arrays["raw"])
    arrays["structural_mask"] = arrays["exposure_seconds"] == 0
    arrays["within_followup_missing_mask"] = arrays["mask"] & ~arrays["structural_mask"][:, :, None]
    arrays["label_mask"] = np.isfinite(arrays["labels"])
    registry = json.loads((common_dir / "feature_registry.json").read_text(encoding="utf-8"))
    bounds = json.loads((common_dir / "measurement_rules.json").read_text(encoding="utf-8"))["bounds"]
    raw, exposure = arrays["raw"], arrays["exposure_seconds"]
    if raw.dtype != np.float64 or raw.shape != tuple(observed["tensor_shape"]) or raw.shape[1:] != (24, 50) or np.isinf(raw).any():
        raise ValueError("Invalid observed tensor shape/dtype/values")
    if arrays["features"].tolist() != [row["name"] for row in registry["temporal_features"]] or arrays["units"].tolist() != [row["canonical_unit"] for row in registry["temporal_features"]]:
        raise ValueError("Feature order or units differ from registry")
    for name in ("stay_ids", "subject_ids", "hadm_ids"):
        ids = arrays[name]
        if ids.shape != (len(raw),) or ids.dtype.kind not in "iu" or np.any(ids <= 0):
            raise ValueError(f"Invalid aligned identifiers: {name}")
        if name != "hadm_ids" and len(np.unique(ids)) != len(raw):
            raise ValueError(f"Duplicate identifiers: {name}")
    if np.any(np.diff(arrays["stay_ids"]) <= 0):
        raise ValueError("Patient order must be ascending stay_id")
    if exposure.shape != raw.shape[:2] or not np.isfinite(exposure).all() or np.any((exposure < 0) | (exposure > 3600)):
        raise ValueError("Invalid follow-up exposure")
    if np.any((exposure[:, :-1] < 3600) & (exposure[:, 1:] > 0)) or np.isfinite(raw[arrays["structural_mask"]]).any():
        raise ValueError("Values or exposure extend outside follow-up")
    if not np.array_equal(arrays["observation_counts"] > 0, np.isfinite(raw)):
        raise ValueError("Observation counts disagree with raw values")
    labels = arrays["labels"]
    if labels.shape != (len(raw),) or np.any(~np.isnan(labels) & ~np.isin(labels, [0, 1])):
        raise ValueError("Invalid mortality labels")
    if observed["policy"] != dict(CONFIG["tensor"], release_version=DATASET_VERSION):
        raise ValueError("Tensor policy changed")
    policy = CONFIG["imputation"]
    if set(policy["observed_only_features"]) != OBSERVED_ONLY:
        raise ValueError("Observed-only channel policy changed")
    inputs["dataset_config.json"] = file_hash(CONFIG_PATH)
    return arrays, policy, bounds, inputs


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_patients(subject_ids, policy):
    size = len(subject_ids)
    train_fraction = policy["split"]["train_fraction"]
    validation_fraction = policy["split"]["validation_fraction"]
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Invalid split fractions")
    ordering = np.asarray(sorted(range(size), key=lambda index: hashlib.sha256(
        f"{policy['seed']}:{int(subject_ids[index])}".encode("ascii")
    ).digest()), dtype=np.int64)
    train_end = int(size * train_fraction)
    validation_end = train_end + int(size * validation_fraction)
    splits = dict(train=ordering[:train_end], validation=ordering[train_end:validation_end], test=ordering[validation_end:])
    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError("Insufficient patients for nonempty train/validation/test partitions")
    return splits


def prepare(arrays, policy, bounds):
    raw, features = arrays["raw"], arrays["features"].tolist()
    splits = split_patients(arrays["subject_ids"], policy)
    model_indices, statistics, unsupported = [], [], []
    for index, feature in enumerate(features):
        if feature in OBSERVED_ONLY:
            continue
        lower, upper = bounds[feature]
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid screening bounds: {feature}")
        observed = raw[:, :, index]
        if np.any(np.isfinite(observed) & ((observed < lower) | (observed > upper))):
            raise ValueError(f"Observed value outside shared measurement bounds: {feature}")
        values = raw[splits["train"], :, index]
        training_patients = int(np.isfinite(values).any(axis=1).sum())
        values = values[np.isfinite(values)]
        if (len(values) < policy["minimum_training_observations"] or
                training_patients < policy.get("minimum_training_patients", 1)):
            unsupported.append(feature)
            continue
        logarithmic = feature in policy["log1p_features"]
        if logarithmic and lower < 0:
            raise ValueError(f"log1p policy requires nonnegative bounds: {feature}")
        transformed = np.log1p(values) if logarithmic else values
        deviation = float(np.std(transformed))
        statistics.append(dict(feature=feature, index=index, count=len(values), mean=float(np.mean(transformed)),
            scale=deviation if deviation > 0 else 1.0, constant=deviation == 0, log1p=logarithmic,
            median=float(np.median(values)), lower=lower, upper=upper))
        model_indices.append(index)
    if not model_indices:
        raise ValueError("No continuous channels have sufficient training observations")
    scaled = np.empty((len(raw), 24, len(model_indices)), dtype=np.float32)
    for column, entry in enumerate(statistics):
        values = raw[:, :, entry["index"]]
        transformed = np.log1p(values) if entry["log1p"] else values
        scaled[:, :, column] = (transformed - entry["mean"]) / entry["scale"]
    if np.isinf(scaled).any():
        raise ValueError("Scaling overflow")
    context = np.isfinite(scaled).any(axis=(1, 2))
    fit_indices = splits["train"][context[splits["train"]]]
    if not len(fit_indices):
        raise ValueError("No training patients have continuous context")
    return dict(splits=splits, fit_indices=fit_indices, scaled=scaled, model_indices=model_indices,
        statistics=statistics, unsupported=unsupported, context=context)


def make_model(feature_count, policy, options, *, model_class=None, training_loss=None, validation_metric=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        version = importlib.metadata.version("pypots")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("PyPOTS is missing; use the pinned Docker environment.") from error
    if version != policy["required_pypots_version"]:
        raise RuntimeError(f"Expected pypots=={policy['required_pypots_version']}, found {version}; use the pinned environment")
    import torch
    from pypots.imputation import SAITS
    from pypots.nn.modules.loss import MAE
    from pypots.imputation.saits.model import logger

    random.seed(policy["seed"])
    np.random.seed(policy["seed"])
    torch.manual_seed(policy["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    custom = {} if training_loss is None else {"training_loss": training_loss}
    model = (model_class or SAITS)(n_steps=24, n_features=feature_count, **policy["model"], **options,
        validation_metric=validation_metric or MAE, num_workers=0, saving_path=None,
        model_saving_strategy=None, **custom)
    model.audit_logger = logger
    runtime = dict(pypots=version, torch=torch.__version__, cuda=torch.version.cuda, device=str(model.device))
    return model, runtime


class TrainingLog(logging.FileHandler):
    def __init__(self, path):
        super().__init__(path, encoding="utf-8")
        self.failures = []

    def emit(self, record):
        message = record.getMessage().lower()
        if record.levelno >= logging.ERROR or "interrupted" in message or "nan loss" in message:
            self.failures.append(record.getMessage())
        super().emit(record)


def fit_model(model, train_set, val_set, log_path):
    logger = getattr(model, "audit_logger", None)
    handler = TrainingLog(log_path)
    if logger is not None:
        logger.addHandler(handler)
    try:
        model.fit(train_set=train_set, val_set=val_set)
        if handler.failures:
            raise RuntimeError("SAITS logged a training failure/interruption: " + "; ".join(handler.failures))
    finally:
        if logger is not None:
            logger.removeHandler(handler)
        handler.close()


def predict(model, values):
    result = model.impute({"X": values})
    result = result["imputation"] if isinstance(result, dict) else result
    result = np.asarray(result, dtype=np.float64)
    if result.shape != values.shape:
        raise ValueError(f"Unexpected SAITS output shape: {result.shape}")
    return result


def inverse_predictions(predictions, statistics):
    physical = np.empty_like(predictions, dtype=np.float64)
    projected = np.zeros(predictions.shape, dtype=bool)
    for column, entry in enumerate(statistics):
        values = predictions[:, :, column]
        with np.errstate(over="raise", invalid="raise"):
            transformed = values * entry["scale"] + entry["mean"]
        lower, upper = entry["lower"], entry["upper"]
        if entry["log1p"]:
            lower, upper = np.log1p(lower), np.log1p(upper)
        projected[:, :, column] = (transformed < lower) | (transformed > upper)
        bounded = np.clip(transformed, lower, upper)
        physical[:, :, column] = np.expm1(bounded) if entry["log1p"] else bounded
        physical[:, :, column] = np.clip(physical[:, :, column], entry["lower"], entry["upper"])
    return physical, projected


def point_mask(observed, rng, fraction):
    mask = np.zeros_like(observed)
    locations = np.flatnonzero(observed)
    if len(locations) >= 2:
        count = min(len(locations) - 1, max(1, int(len(locations) * fraction)))
        mask.reshape(-1)[rng.choice(locations, count, replace=False)] = True
    return mask


def gap_mask(observed, rng, scenario, fraction, channel_weights=None):
    if scenario == "point":
        return point_mask(observed, rng, fraction)
    mask = np.zeros_like(observed)
    total = int(observed.sum())
    if scenario == "block6h":
        starts = [start for start in range(19) if 0 < observed[start:start + 6].sum() < total]
        if starts:
            start = int(rng.choice(starts))
            mask[start:start + 6] = observed[start:start + 6]
    elif scenario == "whole_channel":
        counts = observed.sum(axis=0)
        columns = np.flatnonzero((counts > 0) & (counts < total))
        if len(columns):
            weights = None if channel_weights is None else channel_weights[columns]
            probabilities = None if weights is None else weights / weights.sum()
            column = int(rng.choice(columns, p=probabilities))
            mask[:, column] = observed[:, column]
    else:
        raise ValueError(f"Unknown masking scenario: {scenario}")
    return mask


def evidence_for(prepared, arrays, indices, scenario, seed, fraction, channel_cap):
    """Fixed holdouts; each whole-channel example hides exactly one channel."""
    chosen, masks = [], []
    if scenario == "whole_channel":
        observed = np.isfinite(prepared["scaled"][indices])
        totals = observed.sum(axis=(1, 2))
        counts = observed.sum(axis=1)
        for column in range(observed.shape[2]):
            eligible = np.flatnonzero((counts[:, column] > 0) & (counts[:, column] < totals))
            rng = np.random.default_rng(np.random.SeedSequence([seed, column]))
            locals_ = rng.permutation(eligible)[:channel_cap]
            for local in locals_:
                mask = np.zeros_like(observed[local])
                mask[:, column] = observed[local, :, column]
                chosen.append(int(indices[local]))
                masks.append(mask)
    else:
        for index in indices:
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(arrays["subject_ids"][index])]))
            mask = gap_mask(np.isfinite(prepared["scaled"][index]), rng, scenario, fraction)
            if mask.any():
                chosen.append(int(index))
                masks.append(mask)
    if not chosen:
        raise ValueError(f"No usable {scenario} holdouts; cannot score this fixed release recipe")
    chosen = np.asarray(chosen, dtype=np.int64)
    original = prepared["scaled"][chosen].copy()
    mask = np.asarray(masks, dtype=bool)
    masked = original.copy()
    masked[mask] = np.nan
    if np.any(mask & ~np.isfinite(original)) or not np.isfinite(masked).any(axis=(1, 2)).all():
        raise ValueError("Holdouts must hide only observations and retain patient context")
    return dict(indices=chosen, original=original, masked=masked, mask=mask,
                usable=np.ones(len(chosen), dtype=bool))


def make_gap_model(policy, options, train_values):
    # Artifact and preprocessing checks can run without importing PyTorch.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch.utils.data import Dataset, DataLoader
    from pypots.imputation import SAITS
    from pypots.nn.modules.loss import Criterion

    class FeatureMAE(Criterion):
        def forward(self, logits, targets, masks=None):
            masks = torch.ones_like(targets) if masks is None else masks
            counts = masks.sum(dim=(0, 1))
            active = (counts > 0).to(logits.dtype)
            per_feature = (torch.abs(logits - targets) * masks).sum(dim=(0, 1)) / counts.clamp_min(1)
            return (per_feature * active).sum() / active.sum().clamp_min(1)

    class FeatureHuber(Criterion):
        def forward(self, logits, targets, masks=None):
            masks = torch.ones_like(targets) if masks is None else masks
            counts = masks.sum(dim=(0, 1))
            active = (counts > 0).to(logits.dtype)
            element_loss = torch.nn.functional.huber_loss(logits, targets, reduction="none", delta=1.0)
            per_feature = (element_loss * masks).sum(dim=(0, 1)) / counts.clamp_min(1)
            return (per_feature * active).sum() / active.sum().clamp_min(1)

    coverage = np.isfinite(train_values).any(axis=1).sum(axis=0)
    channel_weights = 1 / np.sqrt(np.maximum(coverage, 1))

    class GapDataset(Dataset):
        def __init__(self, values, fixed_original=None):
            self.values = values
            self.fixed_original = fixed_original
            self.rng = np.random.default_rng(policy["seed"])

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            if self.fixed_original is None:
                original = self.values[index]
                observed = np.isfinite(original)
                scenario = SCENARIOS[int(self.rng.integers(len(SCENARIOS)))]
                hidden = gap_mask(observed, self.rng, scenario, policy["holdout_fraction"], channel_weights)
                if not hidden.any():
                    hidden = point_mask(observed, self.rng, policy["holdout_fraction"])
                available = observed & ~hidden
                values = np.where(available, original, 0).astype(np.float32)
            else:
                original = self.fixed_original[index]
                available = np.isfinite(self.values[index])
                hidden = np.isfinite(original) & ~available
                values = np.nan_to_num(self.values[index], nan=0).astype(np.float32)
            return [torch.tensor(index), torch.from_numpy(values),
                    torch.from_numpy(available.astype(np.float32)),
                    torch.from_numpy(np.nan_to_num(original, nan=0).astype(np.float32)),
                    torch.from_numpy(hidden.astype(np.float32))]

    class GapSAITS(SAITS):
        def fit(self, train_set, val_set=None, file_type="hdf5"):
            if not isinstance(train_set, dict) or val_set is None:
                raise ValueError("Final SAITS requires in-memory training and fixed validation holdouts")
            training = DataLoader(GapDataset(train_set["X"]), batch_size=self.batch_size,
                                  shuffle=True, num_workers=0)
            validation = DataLoader(GapDataset(val_set["X"], val_set["X_ori"]),
                                    batch_size=self.batch_size, shuffle=False, num_workers=0)
            self._train_model(training, validation)
            if self.best_model_dict is None or not np.isfinite(self.best_loss):
                raise ValueError("No finite SAITS checkpoint")
            self.model.load_state_dict(self.best_model_dict)

    return make_model(train_values.shape[2], policy, options, model_class=GapSAITS,
                           training_loss=FeatureHuber, validation_metric=FeatureMAE)


def baseline_values(remaining, entry):
    """Physical-unit baselines based exclusively on remaining observations."""
    previous = np.full(len(remaining), entry["median"], dtype=np.float64)
    seen = np.zeros(len(remaining), dtype=bool)
    values = np.empty(remaining.shape, dtype=np.float64)
    codes = np.empty(remaining.shape, dtype=np.uint8)
    for hour in range(remaining.shape[1]):
        measured = np.isfinite(remaining[:, hour])
        previous[measured] = remaining[measured, hour]
        seen |= measured
        values[:, hour] = previous
        codes[:, hour] = np.where(seen, METHOD_CODES["forward_fill"], METHOD_CODES["median"])
    return values, codes


def score_scenario(model, prepared, arrays, evidence, role, scenario, methods=None):
    """Score the exact final per-cell recipe, including out-of-bound fallbacks."""
    totals = [dict(targets=0, patients=set(), saits=0., median=0., forward_fill=0.,
                   saits_median_fallback=0., saits_forward_fill_fallback=0.,
                   projected=0, final=0., final_squared=0.) for _ in prepared["statistics"]]
    for start in range(0, len(evidence["indices"]), 256):
        sl = slice(start, start + 256)
        indices, mask = evidence["indices"][sl], evidence["mask"][sl]
        prediction = predict(model, evidence["masked"][sl])
        if not np.isfinite(prediction[mask]).all():
            raise ValueError(f"Nonfinite {role}/{scenario} SAITS target predictions")
        physical, projected = inverse_predictions(prediction, prepared["statistics"])
        for column, entry in enumerate(prepared["statistics"]):
            selected = mask[:, :, column]
            if not selected.any():
                continue
            truth = arrays["raw"][indices, :, entry["index"]]
            remaining = truth.copy()
            remaining[selected] = np.nan
            # All hidden cells in this feature are removed, never fed to ffill.
            ffill, _ = baseline_values(remaining, entry)
            errors = dict(saits=np.abs(physical[:, :, column][selected] - truth[selected]),
                          median=np.abs(entry["median"] - truth[selected]),
                          forward_fill=np.abs(ffill[selected] - truth[selected]))
            for baseline_name, baseline in (("median", entry["median"]), ("forward_fill", ffill)):
                candidate = np.where(projected[:, :, column], baseline, physical[:, :, column])
                errors[f"saits_{baseline_name}_fallback"] = np.abs(candidate[selected] - truth[selected])
            row = totals[column]
            row["targets"] += int(selected.sum())
            row["patients"].update(map(int, indices[selected.any(axis=1)]))
            row["projected"] += int(projected[:, :, column][selected].sum())
            for method, error in errors.items():
                row[method] += float(error.sum())
            if methods is not None:
                recipe = methods[entry["feature"]]
                baseline = ffill if recipe["baseline"] == "forward_fill" else np.full_like(ffill, entry["median"])
                if recipe["method"] == "saits":
                    final = np.where(projected[:, :, column], baseline, physical[:, :, column])
                else:
                    final = baseline
                difference = final[selected] - truth[selected]
                row["final"] += float(np.abs(difference).sum())
                row["final_squared"] += float(np.square(difference).sum())
    rows = []
    for entry, total in zip(prepared["statistics"], totals):
        count = total["targets"]
        row = dict(split=role, scenario=scenario, feature=entry["feature"], targets=count,
                   target_patients=len(total["patients"]),
                   projected_fraction=total["projected"] / count if count else None)
        for method in ("saits", "median", "forward_fill", "saits_median_fallback", "saits_forward_fill_fallback"):
            row[f"{method}_mae"] = total[method] / count if count else None
        if methods is not None:
            row["selected_method"] = methods[entry["feature"]]["method"]
            row["final_mae"] = total["final"] / count if count else None
            row["final_rmse"] = np.sqrt(total["final_squared"] / count).item() if count else None
        rows.append(row)
    return rows


def select_methods(rows, statistics):
    methods = {}
    for entry in statistics:
        feature_rows = [row for row in rows if row["feature"] == entry["feature"]]
        adequate = len(feature_rows) == len(SCENARIOS) and all(
            row["targets"] >= RULES["minimum_targets"] and
            row["target_patients"] >= RULES["minimum_target_patients"] for row in feature_rows)
        if not adequate:
            methods[entry["feature"]] = dict(method="median", baseline="median", adequate_validation=False,
                                             reason="Insufficient holdout coverage; explicit training-median fallback")
            continue
        # Normalize each scenario by its median MAE, then give scenarios equal weight.
        normalizers = [max(row["median_mae"], 1e-12) for row in feature_rows]
        scores = {method: float(np.mean([row[f"{method}_mae"] / norm
                    for row, norm in zip(feature_rows, normalizers)]))
                  for method in ("saits", "median", "forward_fill")}
        baseline = min(("median", "forward_fill"), key=lambda name: scores[name])
        candidate_key = f"saits_{baseline}_fallback_mae"
        scores["saits_with_fallback"] = float(np.mean([row[candidate_key] / norm
                                    for row, norm in zip(feature_rows, normalizers)]))
        tolerable = all(row[candidate_key] <= (1 + RULES["maximum_scenario_regret"]) *
                       min(row["median_mae"], row["forward_fill_mae"]) + 1e-12 for row in feature_rows)
        projection_ok = all(row["projected_fraction"] <= RULES["maximum_saits_projection_fraction"]
                            for row in feature_rows)
        improves = scores["saits_with_fallback"] <= (1 - RULES["minimum_mean_improvement"]) * scores[baseline] and scores[baseline] > 0
        accept = tolerable and projection_ok and improves
        methods[entry["feature"]] = dict(method="saits" if accept else baseline, baseline=baseline,
            adequate_validation=True, normalized_mae=scores,
            saits_gates=dict(scenario_regret=tolerable, projection=projection_ok, mean_improvement=improves),
            reason="SAITS passes fixed validation gates" if accept else "Locked baseline wins the fixed SAITS acceptance rule")
    return methods


def reconstruct(model, prepared, arrays, methods):
    output = arrays["raw"].copy()
    codes = np.where(np.isfinite(output), METHOD_CODES["observed"], METHOD_CODES["unfilled"]).astype(np.uint8)
    rejected = np.zeros(output.shape, dtype=bool)
    patients = np.flatnonzero(prepared["context"])
    for start in range(0, len(patients), 256):
        indices = patients[start:start + 256]
        inputs = prepared["scaled"][indices]
        prediction = predict(model, inputs)
        targets = np.isnan(inputs) & ~arrays["structural_mask"][indices, :, None]
        if not np.isfinite(prediction[targets]).all():
            raise ValueError("Nonfinite natural-gap SAITS predictions")
        physical, projected = inverse_predictions(prediction, prepared["statistics"])
        for column, entry in enumerate(prepared["statistics"]):
            feature_index = entry["index"]
            selected = targets[:, :, column]
            values = output[indices, :, feature_index]
            recipe = methods[entry["feature"]]
            fallback, fallback_codes = baseline_values(arrays["raw"][indices, :, feature_index], entry)
            if recipe["baseline"] == "median":
                fallback.fill(entry["median"])
                fallback_codes.fill(METHOD_CODES["median"])
            if recipe["method"] == "saits":
                reject = projected[:, :, column]
                fills = np.where(reject, fallback, physical[:, :, column])
                fill_codes = np.where(reject, fallback_codes, METHOD_CODES["saits"])
                rejected[indices, :, feature_index] = selected & reject
            else:
                fills, fill_codes = fallback, fallback_codes
            values[selected] = fills[selected]
            output[indices, :, feature_index] = values
            cell_codes = codes[indices, :, feature_index]
            cell_codes[selected] = fill_codes[selected]
            codes[indices, :, feature_index] = cell_codes
    return output, codes, rejected


def accept_tensor(arrays, prepared, output, codes, rejected):
    raw = arrays["raw"]
    observed = np.isfinite(raw)
    generated = codes >= METHOD_CODES["saits"]
    eligible = np.zeros(raw.shape, dtype=bool)
    for entry in prepared["statistics"]:
        index = entry["index"]
        eligible[:, :, index] = np.isnan(raw[:, :, index]) & ~arrays["structural_mask"] & prepared["context"][:, None]
        values = output[:, :, index][generated[:, :, index]]
        if not np.isfinite(values).all() or np.any((values < entry["lower"]) | (values > entry["upper"])):
            raise ValueError(f"Invalid final generated values: {entry['feature']}")
    checks = dict(shape_dtype=output.shape == raw.shape and output.dtype == raw.dtype,
        observed_unchanged=np.array_equal(output[observed], raw[observed]),
        no_infinity=not np.isinf(output).any(),
        structural_unfilled=not np.isfinite(output[arrays["structural_mask"]]).any(),
        exact_eligible_fills=np.array_equal(generated, eligible),
        generated_mask_matches_values=np.array_equal(generated, np.isnan(raw) & np.isfinite(output)),
        observed_method_codes=np.array_equal(codes == METHOD_CODES["observed"], observed),
        unfilled_method_codes=np.array_equal(codes == METHOD_CODES["unfilled"], np.isnan(output)),
        valid_method_codes=bool(np.isin(codes, list(METHOD_CODES.values())).all()),
        rejection_provenance=not np.any(rejected & ~np.isin(codes, [3, 4])), generated_in_bounds=True)
    if not all(checks.values()):
        raise ValueError(f"Final tensor acceptance failed: {[name for name, ok in checks.items() if not ok]}")
    return checks


def run_imputation(epochs=None, patience=None, batch_size=None, device=None):
    arrays, policy, bounds, inputs = load_inputs()
    prepared = prepare(arrays, policy, bounds)
    options = {key: value if value is not None else policy["defaults"][key]
               for key, value in (("epochs", epochs), ("patience", patience), ("batch_size", batch_size))}
    if any(not isinstance(value, int) or value <= 0 for value in options.values()):
        raise ValueError("Training options must be positive integers")
    if patience is None:
        options["patience"] = min(options["patience"], options["epochs"])
    if options["patience"] > options["epochs"]:
        raise ValueError("Patience cannot exceed epochs")
    options["device"] = device
    validation = prepared["splits"]["validation"]
    midpoint = len(validation) // 2
    stopping, selection = validation[:midpoint], validation[midpoint:]
    if not len(stopping) or not len(selection):
        raise ValueError("Insufficient patients for disjoint validation subsets")
    print(f"[08] SAITS v{DATASET_VERSION}: {len(arrays['raw']):,} patients, {len(prepared['statistics'])} model channels; "
          f"at most {options['epochs']} epochs, patience {options['patience']}", flush=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROCESSED_DIR / "manifest.json").read_text(encoding="utf-8"))
    script_sha = file_hash(__file__)
    holdouts = {role + "_indices": indices for role, indices in prepared["splits"].items()}
    holdouts.update(early_stopping_indices=stopping, method_selection_indices=selection,
                    model_feature_indices=np.asarray(prepared["model_indices"], dtype=np.int64))
    partition = np.empty(len(arrays["raw"]), dtype=np.uint8)
    for code, indices in enumerate(prepared["splits"].values()):
        partition[indices] = code

    def evidence(indices, scenario, role):
        offset = {"early_stopping": 100, "validation": 200, "test": 300}[role]
        if role == "early_stopping" and scenario != "whole_channel":
            indices = indices[:RULES["stopping_patients_per_scenario"]]
        cap = RULES["stopping_channel_patients_per_feature"] if role == "early_stopping" else RULES["whole_channel_patients_per_feature"]
        result = evidence_for(prepared, arrays, indices, scenario,
            policy["seed"] + offset + SCENARIOS.index(scenario), policy["holdout_fraction"], cap)
        prefix = role + "_" + scenario
        holdouts[prefix + "_indices"] = result["indices"]
        holdouts[prefix + "_mask"] = result["mask"]
        return result

    with tempfile.TemporaryDirectory(prefix=".saits_", dir=METRICS_DIR) as temporary:
        staging = Path(temporary)
        stopping_evidence = [evidence(stopping, scenario, "early_stopping") for scenario in SCENARIOS]
        val_set = {"X": np.concatenate([item["masked"] for item in stopping_evidence]),
                   "X_ori": np.concatenate([item["original"] for item in stopping_evidence])}
        del stopping_evidence
        train_values = prepared["scaled"][prepared["fit_indices"]]
        model, runtime = make_gap_model(policy, options, train_values)
        (METRICS_DIR / "training.log").write_text("", encoding="utf-8")
        fit_model(model, {"X": train_values}, val_set, METRICS_DIR / "training.log")
        del train_values, val_set
        model.save(str(staging / "saits.pypots"))
        if not (staging / "saits.pypots").stat().st_size:
            raise ValueError("Empty checkpoint")
        validation_rows = []
        for scenario in SCENARIOS:
            print(f"[08] Validation: {scenario}", flush=True)
            validation_rows.extend(score_scenario(model, prepared, arrays,
                evidence(selection, scenario, "validation"), "validation", scenario))
        methods = select_methods(validation_rows, prepared["statistics"])
        method_hash = hashlib.sha256(json.dumps(methods, sort_keys=True).encode()).hexdigest()
        print(f"[08] Selected feature methods: {dict(Counter(item['method'] for item in methods.values()))}", flush=True)
        test_rows = []
        for scenario in SCENARIOS:
            print(f"[08] Test: {scenario}", flush=True)
            test_rows.extend(score_scenario(model, prepared, arrays,
                evidence(prepared["splits"]["test"], scenario, "test"), "test", scenario, methods))
        if hashlib.sha256(json.dumps(methods, sort_keys=True).encode()).hexdigest() != method_hash:
            raise ValueError("Method selection changed during test scoring")
        output, codes, rejected = reconstruct(model, prepared, arrays, methods)
        checks = accept_tensor(arrays, prepared, output, codes, rejected)
        np.save(staging / "tensor_imputed.npy", output, allow_pickle=False)
        np.savez_compressed(staging / "imputation_support.npz", method_codes=codes,
                            saits_out_of_bounds_fallback_mask=rejected, partition=partition)
        np.savez_compressed(staging / "holdouts.npz", **holdouts)
        configuration = dict(dataset_version=DATASET_VERSION, policy=policy, model_options=options,
            statistics=prepared["statistics"], input_hashes=inputs,
            model_feature_indices=prepared["model_indices"], model_features=arrays["features"][prepared["model_indices"]].tolist(),
            model_units=arrays["units"][prepared["model_indices"]].tolist(), unsupported_features=prepared["unsupported"],
            method_codes=METHOD_CODES, partition_codes={"train": 0, "validation": 1, "test": 2},
            runtime=runtime, numpy_version=np.__version__, best_epoch=int(model.best_epoch), best_loss=float(model.best_loss))
        test_summary = {}
        for scenario in SCENARIOS:
            rows = [row for row in test_rows if row["scenario"] == scenario and row["targets"]]
            test_summary[scenario] = dict(scored_features=len(rows), targets=sum(row["targets"] for row in rows),
                final_worse_than_median=sum(row["final_mae"] > row["median_mae"] + 1e-12 for row in rows),
                final_worse_than_forward_fill=sum(row["final_mae"] > row["forward_fill_mae"] + 1e-12 for row in rows))
        report = dict(schema_version="1.0.0", dataset_version=DATASET_VERSION, status="COMPLETE",
            configuration=configuration, feature_methods=methods,
            metrics={"validation": validation_rows, "test": test_rows}, acceptance_checks=checks,
            feature_method_counts=dict(Counter(item["method"] for item in methods.values())),
            cell_method_counts={name: int((codes == code).sum()) for name, code in METHOD_CODES.items()},
            saits_out_of_bounds_fallback_cells=int(rejected.sum()), test_summary=test_summary)
        write_json(staging / "saits.json", report)
        current = json.loads((PROCESSED_DIR / "manifest.json").read_text(encoding="utf-8"))
        if current["observed"] != manifest["observed"] or file_hash(__file__) != script_sha:
            raise ValueError("Observed contract or imputation code changed during training")
        for recorded, expected in manifest["observed"]["provenance"].items():
            if file_hash(BASE_DIR / recorded) != expected["sha256"]:
                raise ValueError(f"Source changed during training: {recorded}")
        destinations = {name: PROCESSED_DIR for name in ("tensor_imputed.npy", "imputation_support.npz")}
        destinations.update({name: METRICS_DIR for name in ("saits.pypots", "saits.json", "holdouts.npz")})
        artifacts = {}
        for name, destination in destinations.items():
            path = destination / name
            artifacts[path.relative_to(BASE_DIR).as_posix()] = {
                "sha256": file_hash(staging / name), "size_bytes": (staging / name).stat().st_size}
        manifest["imputation"] = dict(dataset_version=DATASET_VERSION, artifacts=artifacts,
            script_sha256=script_sha, feature_recipe_sha256=method_hash,
            generated_at_utc=datetime.now(timezone.utc).isoformat(), status="COMPLETE")
        write_json(staging / "manifest.json", manifest)
        for name, destination in destinations.items():
            (staging / name).replace(destination / name)
        (staging / "manifest.json").replace(PROCESSED_DIR / "manifest.json")
    print(f"[08] Complete: {output.shape}; cell methods={report['cell_method_counts']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", help="PyTorch device, for example cpu or cuda:0")
    args = parser.parse_args()
    run_imputation(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
