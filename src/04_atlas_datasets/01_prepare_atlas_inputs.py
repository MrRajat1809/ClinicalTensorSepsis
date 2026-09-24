"""Bind the three completed datasets and establish training-only atlas eligibility."""
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/processed/atlas"
OUT = ROOT / "outputs/atlas"
CONFIG = json.loads((HERE / "atlas_config.json").read_text(encoding="utf-8"))
VERSION = CONFIG["version"]
NAMES = tuple(CONFIG["datasets"])
STAGES = {
    1: "01_prepare_atlas_inputs.py", 2: "02_build_temporal_representation.py",
    3: "03_fit_domain_transport.py", 4: "04_evaluate_domain_adaptation.py",
    5: "05_build_trajectory_atlas.py", 6: "06_atlas_master_qc.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(recorded):
    path = (ROOT / recorded).resolve()
    require(path.is_relative_to(ROOT), f"Path outside project: {recorded}")
    return path


def record(path):
    path = Path(path).resolve()
    return {"sha256": sha(path), "size_bytes": path.stat().st_size}


def bind(paths):
    paths = sorted(set(Path(path).resolve() for path in paths), key=lambda path: path.as_posix())
    return {path.relative_to(ROOT).as_posix(): record(path) for path in paths}


def verify_files(records):
    for name, expected in records.items():
        path = project_path(name)
        require(path.is_file() and record(path) == expected, f"Missing or changed artifact: {name}")


def code_binding(number=6):
    return bind([HERE / name for stage, name in STAGES.items() if stage <= number] + [HERE / "atlas_config.json"])


def runtime():
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "POT"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def receipt_path(number):
    return OUT / f"{number:02d}.json"


def verify_stage(number, external=False):
    result = read_json(receipt_path(number))
    require(result["version"] == VERSION and result["status"] == "COMPLETE", f"Stage {number} incomplete")
    require(result["code"] == code_binding(number), f"Atlas code/config changed; rerun the earliest changed stage and its dependants (stage {number})")
    verify_files(result["outputs"])
    verify_files(result["dependencies"])
    if external:
        verify_files(read_json(receipt_path(1))["external"])
    return result


def begin(number):
    validate_config()
    print(f"[{number:02d}] Verifying inputs and provenance v{VERSION}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if number < 6 and (OUT / "qc.json").exists():
        write_json(OUT / "qc.json", dict(version=VERSION, status="STALE", reason=f"Stage {number:02d} is being rerun; rerun 06 after completion"))
    for previous in range(1, number):
        verify_stage(previous)
    if number > 1:
        verify_files(read_json(receipt_path(1))["external"])
    dependencies = bind([receipt_path(previous) for previous in range(1, number)])
    return {"code": code_binding(number), "dependencies": dependencies}


def validate_config():
    require(VERSION == "1.0.0" and NAMES == ("mimiciv", "mimic3-carevue", "eicu"), "Unsupported atlas version or dataset order")
    r, t, s = CONFIG["representation"], CONFIG["transport"], CONFIG["selection"]
    require(r["pool_blocks"] > 0 and 24 % r["pool_blocks"] == 0, "Temporal blocks must divide 24")
    require(all(isinstance(r[key], int) and r[key] > 0 for key in ("epochs", "patience", "batch_size", "hidden_channels")), "Invalid encoder sizes")
    require(r["patience"] <= r["epochs"] and 0 < r["mask_fraction"] < 1, "Invalid stopping/masking policy")
    require(len(r["method_weights"]) == 5 and r["method_weights"][:2] == [0, 1] and
            all(0 < weight < 1 for weight in r["method_weights"][2:]), "Invalid imputation confidence weights")
    require(all(value > 0 for value in t["entropy"]) and all(0 < value <= 1 for value in t["strengths"]), "Invalid OT candidates")
    require(t["maximum_prototypes"] >= 2 and t["maximum_correction_rms"] > 0 and 0 < t["support_quantile"] < 1, "Invalid OT support policy")
    require(t["mass_regularization"] > 0 and t["ridge"] > 0 and t["solver_tolerance"] > 0, "Invalid OT regularization")
    require(0 < s["minimum_stability_improvement"] < 1 and 0 < s["minimum_supported_fraction"] <= 1, "Invalid validation acceptance thresholds")
    require(s["minimum_pair_distance_ratio"] <= 1 <= s["maximum_pair_distance_ratio"], "Invalid preservation bounds")
    require(2 <= CONFIG["atlas"]["clusters"] <= 10, "Atlas needs 2-10 groups")


def finish(number, start, outputs, **detail):
    require(start["code"] == code_binding(number), "Atlas code changed during execution")
    verify_files(start["dependencies"])
    result = dict(version=VERSION, status="COMPLETE", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=runtime(), outputs=bind(outputs), **start, **detail)
    write_json(receipt_path(number), result)
    return result


def save_npz(path, **arrays):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def source_arrays(name, labels=False):
    folder = ROOT / "data/processed" / name
    with np.load(folder / "tensor_support.npz", allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files if labels or key != "labels"}
    result.update(load_npz(folder / "imputation_support.npz"))
    result["raw"] = np.load(folder / "tensor_observed.npy", mmap_mode="r", allow_pickle=False)
    result["imputed"] = np.load(folder / "tensor_imputed.npy", mmap_mode="r", allow_pickle=False)
    return result


def source_binding(name):
    folder, output = ROOT / "data/processed" / name, ROOT / "outputs" / name
    manifest = read_json(folder / "manifest.json")
    qc = read_json(output / "qc.json")
    require(qc["status"] == "PASS" and qc["failed_required_checks"] == 0, f"{name}: dataset QC has not passed")
    require(qc["manifest_sha256"] == sha(folder / "manifest.json"), f"{name}: QC is stale")
    require(manifest["dataset_version"] == VERSION and qc["dataset_version"] == VERSION, f"{name}: version mismatch")
    require(manifest["imputation"]["status"] == "COMPLETE", f"{name}: imputation incomplete")
    paths = [folder / "manifest.json", output / "qc.json"]
    for filename, expected in manifest["observed"]["output_manifest"].items():
        path = (folder / filename).resolve()
        require(path.is_relative_to(folder.resolve()) and record(path) == expected, f"{name}: changed {filename}")
        paths.append(path)
    for section in (manifest["observed"]["provenance"], manifest["imputation"]["artifacts"]):
        verify_files(section)
        paths.extend(project_path(filename) for filename in section)
    qc_script = list((ROOT / CONFIG["datasets"][name]).glob("09_*_master_qc.py"))
    require(len(qc_script) == 1 and sha(qc_script[0]) == qc["qc_script_sha256"], f"{name}: QC source changed")
    imputation_script = list((ROOT / CONFIG["datasets"][name]).glob("08_*_saits_imputation.py"))
    require(len(imputation_script) == 1 and sha(imputation_script[0]) == manifest["imputation"]["script_sha256"], f"{name}: imputation source changed")
    paths.extend(qc_script)
    paths.extend(imputation_script)
    paths.append(ROOT / CONFIG["datasets"][name] / "dataset_config.json")
    return paths, manifest


def feature_eligibility(arrays):
    policy = CONFIG["eligibility"]
    train = arrays["partition"] == 0
    observed = np.isfinite(arrays["raw"][train])
    patients = observed.any(axis=1).sum(axis=0)
    fractions = patients / max(1, train.sum())
    available = (patients >= policy["minimum_training_patients"]) & (fractions >= policy["minimum_training_prevalence"])
    return available, patients, fractions


def patient_eligibility(raw, common):
    observed = np.isfinite(raw[:, :, common])
    p = CONFIG["eligibility"]
    return ((observed.sum(axis=(1, 2)) >= p["minimum_patient_observed_cells"]) &
            (observed.any(axis=1).sum(axis=1) >= p["minimum_patient_observed_features"]))


def main():
    import pandas as pd

    print(f"[01] Binding atlas inputs v{VERSION}", flush=True)
    start = begin(1)
    registry = read_json(ROOT / "src/common/feature_registry.json")
    schema = read_json(ROOT / "src/common/tensor_schema.json")
    require(registry["schema_version"] == schema["schema_version"] == VERSION, "Shared schema version mismatch")
    require(NAMES[0] == CONFIG["reference"] == "mimiciv", "Reference must be first and remain MIMIC-IV")
    external = [ROOT / "src/common" / name for name in ("feature_registry.json", "tensor_schema.json", "measurement_rules.json")]
    eligibility, summaries, manifests = [], [], {}
    for name in NAMES:
        paths, manifests[name] = source_binding(name)
        external.extend(paths)
        arrays = source_arrays(name)
        raw, part = arrays["raw"], arrays["partition"]
        require(raw.shape[1:] == (24, 50) and raw.shape == arrays["imputed"].shape, f"{name}: tensor shape")
        require(arrays["features"].tolist() == [x["name"] for x in registry["temporal_features"]], f"{name}: feature order")
        require(arrays["units"].tolist() == [x["canonical_unit"] for x in registry["temporal_features"]], f"{name}: units")
        require(part.shape == (len(raw),) and set(part.tolist()) == {0, 1, 2}, f"{name}: partitions")
        require(len(np.unique(arrays["subject_ids"])) == len(raw), f"{name}: repeated patients")
        holdouts = load_npz(ROOT / "outputs" / name / "holdouts.npz")
        for code, role in enumerate(("train", "validation", "test")):
            require(np.array_equal(np.sort(holdouts[role + "_indices"]), np.flatnonzero(part == code)), f"{name}: {role} split changed")
        require(np.array_equal(np.isfinite(raw), arrays["observation_counts"] > 0), f"{name}: observed mask")
        require(np.array_equal(arrays["imputed"][np.isfinite(raw)], raw[np.isfinite(raw)]), f"{name}: observations changed")
        require(not np.isfinite(arrays["imputed"][arrays["exposure_seconds"] == 0]).any(), f"{name}: structural values")
        available, count, fraction = feature_eligibility(arrays)
        eligibility.append(available)
        summaries.append(dict(dataset=name, patients=len(raw), training_observed_patients=count.tolist(),
                              training_observed_fraction=fraction.tolist()))
        del arrays
    common = np.flatnonzero(np.all(eligibility, axis=0))
    require(len(common) >= CONFIG["eligibility"]["minimum_shared_features"], "Too few reliably observed shared features")
    metadata, partitions, valid, domains, rows, stays, subjects = [], [], [], [], [], [], []
    for domain, name in enumerate(NAMES):
        arrays = source_arrays(name)
        cohort = pd.read_parquet(ROOT / "data/processed" / name / "cohort.parquet")
        require(np.array_equal(cohort["stay_id"].to_numpy(), arrays["stay_ids"]), f"{name}: cohort row order")
        require("sepsis_onset_time" in cohort, f"{name}: missing operational onset metadata")
        if name == "eicu":
            require({"strict_culture_sepsis3", "strict_culture_onset_time", "infection_time_is_proxy", "hospital_id"}.issubset(cohort.columns),
                    "eicu: missing required proxy/sensitivity/hospital context")
        accepted = patient_eligibility(arrays["raw"], common)
        columns = [x for x in ("gender", "hospital_id", "source_subject_id", "cohort_definition", "infection_evidence_type",
                   "infection_time_is_proxy", "strict_culture_sepsis3", "strict_culture_onset_time", "strict_onset_matches_primary",
                   "sepsis_onset_time", "observation_end", "death_time_is_proxy", "death_time_basis") if x in cohort]
        table = cohort[columns].copy()
        table.insert(0, "patient_key", [f"{name}:{int(stay)}" for stay in arrays["stay_ids"]])
        table["dataset"], table["stay_id"], table["subject_id"] = name, arrays["stay_ids"], arrays["subject_ids"]
        table["partition"], table["eligible"] = arrays["partition"], accepted
        table["age"] = arrays["static"][:, arrays["static_features"].tolist().index("age")]
        table["followup_hours"] = arrays["exposure_seconds"].sum(axis=1) / 3600
        table["observed_cells"] = np.isfinite(arrays["raw"]).sum(axis=(1, 2))
        metadata.append(table)
        partitions.append(arrays["partition"])
        valid.append(accepted)
        domains.append(np.full(len(accepted), domain, dtype=np.uint8))
        rows.append(np.arange(len(accepted)))
        stays.append(arrays["stay_ids"])
        subjects.append(arrays["subject_ids"])
        summaries[domain]["eligible_patients"] = int(accepted.sum())
        print(f"[01] {name}: {len(accepted):,} retained; {accepted.sum():,} eligible for representation", flush=True)
    table = pd.concat(metadata, ignore_index=True)
    require(table["patient_key"].is_unique, "Duplicate composite patient keys")
    table.to_parquet(DATA / "patients.parquet", index=False)
    save_npz(DATA / "inputs.npz", dataset=np.concatenate(domains), partition=np.concatenate(partitions),
             eligible=np.concatenate(valid), source_row=np.concatenate(rows), stay_id=np.concatenate(stays),
             subject_id=np.concatenate(subjects), common_indices=common, feature_eligibility=np.asarray(eligibility),
             dataset_names=np.asarray(NAMES), feature_names=np.asarray([x["name"] for x in registry["temporal_features"]]))
    finish(1, start, [DATA / "patients.parquet", DATA / "inputs.npz"], external=bind(set(external)),
           datasets=summaries, common_features=[registry["temporal_features"][i]["name"] for i in common],
           protocol=CONFIG["protocol"], source_versions={name: manifests[name]["source_version"] for name in NAMES})
    print(f"[01] Complete: {len(common)} shared encoder channels; all 50 clinical channels retained upstream", flush=True)


if __name__ == "__main__":
    main()
