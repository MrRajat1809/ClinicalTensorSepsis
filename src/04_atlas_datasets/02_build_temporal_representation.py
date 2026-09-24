"""Fit a reference-only temporal encoder; retain masks, ordered summaries and clinical probes."""
import argparse
import importlib
import os
import random

import numpy as np
import torch
from torch import nn

C = importlib.import_module("01_prepare_atlas_inputs")
P = C.CONFIG["representation"]


def seed_all():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(C.CONFIG["seed"])
    np.random.seed(C.CONFIG["seed"])
    torch.manual_seed(C.CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.CONFIG["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def robust_stats(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    center = float(np.median(finite))
    scale = float(np.diff(np.quantile(finite, [0.25, 0.75]))[0])
    if scale < 1e-6:
        scale = max(float(np.std(finite)), 1.0)
    return center, scale


def fit_scaling(arrays):
    train = arrays["partition"] == 0
    entries = []
    for j, name in enumerate(arrays["features"].tolist()):
        values = np.asarray(arrays["raw"][train, :, j])
        logarithmic = name in P["log1p_features"]
        if logarithmic:
            C.require(not np.any(values[np.isfinite(values)] < 0), f"Negative log feature: {name}")
            values = np.log1p(values)
        center, scale = robust_stats(values)
        entries.append(dict(name=name, log1p=logarithmic, center=center, scale=scale,
                            observations=int(np.isfinite(values).sum())))
    age = arrays["static"][:, arrays["static_features"].tolist().index("age")]
    center, scale = robust_stats(age[train])
    return dict(features=entries, age=dict(center=center, scale=scale),
                transform="asinh((transformed_value - training_median) / training_IQR); transformed_value is log1p(value) only for declared features; zero IQR uses training SD or 1")


def scale_values(values, scaling):
    output = np.empty(values.shape, dtype=np.float32)
    for j, entry in enumerate(scaling["features"]):
        column = np.asarray(values[:, :, j])
        if entry["log1p"]:
            column = np.log1p(column)
        output[:, :, j] = np.arcsinh((column - entry["center"]) / entry["scale"])
    return output


def masked_mean(values, axis):
    observed = np.isfinite(values)
    result = np.nansum(values, axis=axis) / np.maximum(1, observed.sum(axis=axis))
    return np.where(observed.any(axis=axis), result, np.nan)


def clinical_views(raw, age, names, scaling):
    columns, anchor_names = [age], ["age"]
    for name in P["anchor_features"]:
        j = names.index(name)
        columns.append(masked_mean(raw[:, :6, j], axis=1))
        anchor_names.append(name)
    anchors = np.column_stack(columns).astype(np.float32)
    probes, probe_names = [], []
    for name in P["probe_features"]:
        j = names.index(name)
        early = masked_mean(raw[:, :6, j], axis=1)
        late = masked_mean(raw[:, 18:, j], axis=1)
        probes.extend([early, late - early])
        probe_names.extend([name + "/early", name + "/change"])
    return anchors, np.column_stack(probes).astype(np.float32), anchor_names, probe_names


def prepare_domain(arrays, scaling, indices):
    raw = scale_values(arrays["raw"], scaling)
    imputed = scale_values(arrays["imputed"], scaling)
    observed = np.isfinite(raw[:, :, indices])
    codes = arrays["method_codes"][:, :, indices]
    C.require(np.isin(codes, np.arange(5)).all(), "Unknown imputation method code")
    weights = np.asarray(P["method_weights"], dtype=np.float32)[codes]
    weights *= np.isfinite(imputed[:, :, indices])
    exposure = arrays["exposure_seconds"].astype(np.float32) / 3600
    weights *= (exposure > 0)[:, :, None]
    age = arrays["static"][:, arrays["static_features"].tolist().index("age")]
    age = np.arcsinh((age - scaling["age"]["center"]) / scaling["age"]["scale"])
    anchors, probes, an, pn = clinical_views(raw, age, arrays["features"].tolist(), scaling)
    return dict(values=np.nan_to_num(imputed[:, :, indices], nan=0),
                truth=np.nan_to_num(raw[:, :, indices], nan=0), observed=observed,
                weights=weights, counts=arrays["observation_counts"][:, :, indices].astype(np.float32),
                exposure=exposure, anchors=anchors, probes=probes, anchor_names=an, probe_names=pn,
                raw=raw, age=age, names=arrays["features"].tolist())


def input_tensor(domain, rows, *, observed_only=False, removed=None):
    observed = domain["observed"][rows].copy()
    weights = observed.astype(np.float32) if observed_only else domain["weights"][rows].copy()
    values = domain["truth"][rows] if observed_only else domain["values"][rows]
    counts = domain["counts"][rows].copy()
    if removed is not None:
        observed[removed] = False
        weights[removed] = 0
        counts[removed] = 0
    exposure = domain["exposure"][rows]
    hours = np.broadcast_to(np.arange(24, dtype=np.float32)[None, :, None] / 23, (*exposure.shape, 1))
    x = np.concatenate([values * weights, observed, weights, np.log1p(counts) / np.log(11),
                        exposure[:, :, None], hours], axis=2).astype(np.float32)
    x *= (exposure > 0)[:, :, None]
    return np.ascontiguousarray(x.transpose(0, 2, 1))


class TemporalEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        hidden = P["hidden_channels"]
        self.layers = nn.ModuleList([nn.Conv1d(features * 4 + 2, hidden, 3, padding=1),
                                    nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
                                    nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4)])
        self.decoder = nn.Conv1d(hidden, features, 1)

    def forward(self, values, exposure):
        mask = (exposure > 0).unsqueeze(1)
        h = values
        for layer in self.layers:
            h = torch.nn.functional.gelu(layer(h)) * mask
        pooled = []
        width = 24 // P["pool_blocks"]
        for block in range(P["pool_blocks"]):
            sl = slice(block * width, (block + 1) * width)
            weight = exposure[:, None, sl]
            pooled.append((h[:, :, sl] * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6))
        return torch.cat(pooled, dim=1), h.transpose(1, 2), self.decoder(h).transpose(1, 2)


def reconstruction_loss(prediction, truth, mask):
    errors = torch.nn.functional.smooth_l1_loss(prediction, truth, reduction="none")
    counts = mask.sum(dim=(0, 1))
    present = counts > 0
    return ((errors * mask).sum(dim=(0, 1))[present] / counts[present]).mean()


def forward_batch(model, domain, rows, device, observed_only=False, removed=None):
    x = torch.from_numpy(input_tensor(domain, rows, observed_only=observed_only, removed=removed)).to(device)
    exposure = torch.from_numpy(domain["exposure"][rows]).to(device)
    return model(x, exposure)


def validation_loss(model, domain, rows, device):
    rng = np.random.default_rng(C.CONFIG["seed"] + 11)
    model.eval()
    sums = np.zeros(domain["truth"].shape[2], dtype=np.float64)
    counts = np.zeros_like(sums)
    with torch.no_grad():
        for start in range(0, len(rows), P["batch_size"]):
            batch = rows[start:start + P["batch_size"]]
            mask = domain["observed"][batch] & (rng.random(domain["observed"][batch].shape) < P["mask_fraction"])
            _, _, prediction = forward_batch(model, domain, batch, device, True, mask)
            truth = torch.from_numpy(domain["truth"][batch]).to(device)
            errors = torch.nn.functional.smooth_l1_loss(prediction, truth, reduction="none").cpu().numpy()
            sums += (errors * mask).sum(axis=(0, 1))
            counts += mask.sum(axis=(0, 1))
    C.require(np.any(counts > 0), "No source validation reconstruction targets")
    return float(np.mean(sums[counts > 0] / counts[counts > 0]))


def train_encoder(domain, train, validation, device):
    model = TemporalEncoder(domain["truth"].shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=P["learning_rate"], weight_decay=P["weight_decay"])
    rng = np.random.default_rng(C.CONFIG["seed"])
    history, best, state, stale = [], float("inf"), None, 0
    for epoch in range(1, P["epochs"] + 1):
        model.train()
        order = rng.permutation(train)
        losses = []
        for start in range(0, len(order), P["batch_size"]):
            batch = order[start:start + P["batch_size"]]
            mask = domain["observed"][batch] & (rng.random(domain["observed"][batch].shape) < P["mask_fraction"])
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            _, _, reconstruction = forward_batch(model, domain, batch, device, True, mask)
            truth = torch.from_numpy(domain["truth"][batch]).to(device)
            loss = reconstruction_loss(reconstruction, truth, torch.from_numpy(mask).to(device))
            # No SAITS-derived values enter masked reconstruction: they could reveal hidden targets.
            full, _, _ = forward_batch(model, domain, batch, device)
            with torch.no_grad():
                measured, _, _ = forward_batch(model, domain, batch, device, True)
            loss += P["consistency_weight"] * torch.nn.functional.mse_loss(full, measured)
            C.require(bool(torch.isfinite(loss)), "Nonfinite encoder loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        value = validation_loss(model, domain, validation, device)
        history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)), validation_loss=value))
        if value < best - 1e-6:
            best, stale = value, 0
            state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        print(f"[02] Epoch {epoch:02d}: train={np.mean(losses):.5f}; validation={value:.5f}", flush=True)
        if stale >= P["patience"]:
            break
    C.require(state is not None, "Encoder did not produce a checkpoint")
    model.load_state_dict(state)
    model.eval()
    return model, history, best


def perturbation_mask(domain, scenario, domain_index):
    rng = np.random.default_rng(C.CONFIG["seed"] + 100 + domain_index * 10 + (scenario == "block"))
    observed = domain["observed"]
    if scenario == "point":
        return observed & (rng.random(observed.shape) < P["mask_fraction"])
    mask = np.zeros_like(observed)
    starts = rng.integers(0, 19, len(observed))
    for row, start in enumerate(starts):
        mask[row, start:start + 6] = observed[row, start:start + 6]
    return mask


def encode(model, domain, device, *, observed_only=False, removed=None):
    z, hourly = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(domain["values"]), P["batch_size"]):
            rows = np.arange(start, min(start + P["batch_size"], len(domain["values"])))
            latent, h, _ = forward_batch(model, domain, rows, device, observed_only,
                                          None if removed is None else removed[rows])
            z.append(latent.cpu().numpy())
            hourly.append(h.cpu().numpy())
    return np.concatenate(z), np.concatenate(hourly)


def ordered_baseline(domain):
    result = []
    width = 24 // P["pool_blocks"]
    for start in range(0, 24, width):
        weights = domain["weights"][:, start:start + width] * domain["exposure"][:, start:start + width, None]
        values = domain["values"][:, start:start + width]
        result.extend([(values * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-6),
                       domain["observed"][:, start:start + width].mean(axis=1)])
    return np.concatenate(result, axis=1).astype(np.float32)


def fit_probes(z, truth, train):
    from sklearn.linear_model import Ridge

    coefficients = np.zeros((truth.shape[1], z.shape[1]))
    intercept = np.zeros(truth.shape[1])
    valid = np.zeros(truth.shape[1], dtype=bool)
    tails = np.full((truth.shape[1], 2), np.nan)
    fit_rows = []
    for j in range(truth.shape[1]):
        rows = train[np.isfinite(truth[train, j])]
        fit_rows.append(rows)
        if len(rows) < C.CONFIG["eligibility"]["minimum_training_patients"] or np.std(truth[rows, j]) < 1e-6:
            continue
        model = Ridge(alpha=P["probe_ridge"]).fit(z[rows], truth[rows, j])
        coefficients[j], intercept[j], valid[j] = model.coef_, model.intercept_, True
        tails[j] = np.quantile(truth[rows, j], [0.1, 0.9])
    return dict(coef=coefficients, intercept=intercept, valid=valid, tails=tails, fit_rows=fit_rows)


def standardize(values, train):
    center = values[train].mean(axis=0)
    scale = values[train].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return ((values - center) / scale).astype(np.float32), center, scale


def checkpoint(device):
    saved = torch.load(C.OUT / "encoder.pt", map_location="cpu", weights_only=False)
    model = TemporalEncoder(len(saved["indices"])).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    start = C.begin(2)
    seed_all()
    inputs = C.load_npz(C.DATA / "inputs.npz")
    indices = inputs["common_indices"]
    reference = C.source_arrays(C.CONFIG["reference"])
    scaling = fit_scaling(reference)
    domain = prepare_domain(reference, scaling, indices)
    train = np.flatnonzero((inputs["dataset"] == 0) & (inputs["partition"] == 0) & inputs["eligible"])
    validation = np.flatnonzero((inputs["dataset"] == 0) & (inputs["partition"] == 1) & inputs["eligible"])
    C.require(len(train) >= 100 and len(validation) >= 30, "Insufficient eligible source training/validation patients")
    print(f"[02] Reference encoder: {len(indices)} features, {len(train):,} training patients, device={args.device}", flush=True)
    model, history, best = train_encoder(domain, train, validation, args.device)
    saved = dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, scaling=scaling,
                 indices=indices.tolist(), fit_rows=train.tolist(), validation_rows=validation.tolist(), best_loss=best)
    torch.save(saved, C.OUT / "encoder.pt")
    del domain, reference
    combined = {key: [] for key in ("z", "hourly", "baseline", "anchors", "probes", "observed_z", "point_z", "block_z", "point_anchors", "block_anchors")}
    for domain_index, name in enumerate(C.NAMES):
        domain = prepare_domain(C.source_arrays(name), scaling, indices)
        z, hourly = encode(model, domain, args.device)
        combined["z"].append(z)
        combined["hourly"].append(hourly)
        combined["baseline"].append(ordered_baseline(domain))
        combined["anchors"].append(domain["anchors"])
        combined["probes"].append(domain["probes"])
        combined["observed_z"].append(encode(model, domain, args.device, observed_only=True)[0])
        for scenario in ("point", "block"):
            mask = perturbation_mask(domain, scenario, domain_index)
            combined[scenario + "_z"].append(encode(model, domain, args.device, observed_only=True, removed=mask)[0])
            masked_raw = domain["raw"].copy()
            for column, j in enumerate(indices):
                masked_raw[:, :, j][mask[:, :, column]] = np.nan
            anchors = clinical_views(masked_raw, domain["age"], domain["names"], scaling)[0]
            combined[scenario + "_anchors"].append(anchors)
        anchor_names, probe_names = domain["anchor_names"], domain["probe_names"]
        print(f"[02] Encoded {name}", flush=True)
        del domain
    result = {key: np.concatenate(value) for key, value in combined.items()}
    result["z"], center, scale = standardize(result["z"], train)
    for key in ("observed_z", "point_z", "block_z"):
        result[key] = ((result[key] - center) / scale).astype(np.float32)
    result["baseline"], baseline_center, baseline_scale = standardize(result["baseline"], train)
    probes = fit_probes(result["z"], result["probes"], train)
    result.update(z_center=center, z_scale=scale, baseline_center=baseline_center, baseline_scale=baseline_scale,
                  probe_coef=probes["coef"], probe_intercept=probes["intercept"], probe_valid=probes["valid"],
                  probe_tails=probes["tails"], anchor_names=np.asarray(anchor_names), probe_names=np.asarray(probe_names))
    for key in ("z", "baseline", "hourly", "observed_z", "point_z", "block_z"):
        C.require(np.isfinite(result[key]).all(), f"Nonfinite representation: {key}")
    C.save_npz(C.DATA / "representations.npz", **result)
    C.finish(2, start, [C.OUT / "encoder.pt", C.DATA / "representations.npz"], history=history,
             best_validation_loss=best, fit_rows=train.tolist(), validation_rows=validation.tolist(),
             probe_fit_rows=[rows.tolist() for rows in probes["fit_rows"]], scaling=scaling,
             dimensions=int(result["z"].shape[1]), device=args.device,
             perturbations="Fixed observed-only views; all generated values removed from both views; original and thinned anchors computed separately")
    print(f"[02] Complete: {result['z'].shape}; source-only encoder and scaling saved", flush=True)


if __name__ == "__main__":
    main()
