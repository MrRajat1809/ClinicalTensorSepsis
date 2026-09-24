"""Verify provenance, fitting partitions, checkpoint replay and the locked atlas contract."""
import argparse
import importlib
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch

C = importlib.import_module("01_prepare_atlas_inputs")
R = importlib.import_module("02_build_temporal_representation")
T = importlib.import_module("03_fit_domain_transport")
E = importlib.import_module("04_evaluate_domain_adaptation")
A = importlib.import_module("05_build_trajectory_atlas")


def near(a, b, *, atol=2e-5, rtol=2e-4):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(near(a[key], b[key], atol=atol, rtol=rtol) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(near(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
    if isinstance(a, (np.ndarray, int, float, np.number)) and isinstance(b, (np.ndarray, int, float, np.number)):
        return np.shape(a) == np.shape(b) and bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    return a == b


class QC:
    def __init__(self):
        self.checks = []
        self.section = ""

    def check(self, name, passed, detail=None):
        self.checks.append(dict(section=self.section, name=name, passed=bool(passed), detail=detail))
        if not passed:
            raise ValueError(f"QC failed: {name}")

    def phase(self, name, action):
        self.section = name
        print(f"[ATLAS QC] {name}", flush=True)
        action()


def input_checks(qc, state):
    state["start"] = C.begin(6)
    state["inputs"] = C.load_npz(C.DATA / "inputs.npz")
    state["rep"] = C.load_npz(C.DATA / "representations.npz")
    state["transport"] = C.load_npz(C.DATA / "transport.npz")
    state["maps"] = joblib.load(C.OUT / "transport.joblib")
    state["patients"] = pd.read_parquet(C.DATA / "patients.parquet")
    qc.check("all_stage_artifacts_and_code_bindings", True)
    inputs = state["inputs"]
    qc.check("composite_patient_keys_unique", state["patients"]["patient_key"].is_unique)
    feature_flags = []
    for domain, name in enumerate(C.NAMES):
        C.source_binding(name)
        arrays = C.source_arrays(name)
        rows = np.flatnonzero(inputs["dataset"] == domain)
        qc.check(f"{name}/source_row_order", np.array_equal(inputs["source_row"][rows], np.arange(len(rows))))
        for name_key, source_key in (("stay_id", "stay_ids"), ("subject_id", "subject_ids"), ("partition", "partition")):
            qc.check(f"{name}/{name_key}", np.array_equal(inputs[name_key][rows], arrays[source_key]))
        qc.check(f"{name}/unique_patient_partition", len(np.unique(arrays["subject_ids"])) == len(rows))
        qc.check(f"{name}/eligibility_replay", np.array_equal(inputs["eligible"][rows], C.patient_eligibility(arrays["raw"], inputs["common_indices"])))
        qc.check(f"{name}/structural_values_absent", not np.isfinite(arrays["imputed"][arrays["exposure_seconds"] == 0]).any())
        qc.check(f"{name}/observations_unchanged", np.array_equal(arrays["raw"][np.isfinite(arrays["raw"])], arrays["imputed"][np.isfinite(arrays["raw"])]))
        feature_flags.append(C.feature_eligibility(arrays)[0])
    qc.check("training_only_feature_eligibility", np.array_equal(inputs["feature_eligibility"], feature_flags))
    qc.check("shared_feature_intersection", np.array_equal(inputs["common_indices"], np.flatnonzero(np.all(feature_flags, axis=0))))
    qc.check("metadata_identifiers", np.array_equal(state["patients"]["stay_id"], inputs["stay_id"]))
    qc.check("metadata_partitions", np.array_equal(state["patients"]["partition"], inputs["partition"]))
    qc.check("metadata_eligibility", np.array_equal(state["patients"]["eligible"], inputs["eligible"]))
    for key in ("z", "hourly", "baseline", "observed_z", "point_z", "block_z"):
        qc.check(f"finite_representation/{key}", len(state["rep"][key]) == len(inputs["dataset"]) and np.isfinite(state["rep"][key]).all())


def representation_checks(qc, state, device):
    inputs, rep = state["inputs"], state["rep"]
    R.seed_all()
    model, saved = R.checkpoint(device)
    report = C.read_json(C.receipt_path(2))
    reference = C.source_arrays(C.CONFIG["reference"])
    scaling = R.fit_scaling(reference)
    train = np.flatnonzero((inputs["dataset"] == 0) & (inputs["partition"] == 0) & inputs["eligible"])
    validation = np.flatnonzero((inputs["dataset"] == 0) & (inputs["partition"] == 1) & inputs["eligible"])
    qc.check("reference_only_encoder_fit_rows", np.array_equal(saved["fit_rows"], train) and report["fit_rows"] == train.tolist())
    qc.check("reference_only_stopping_rows", np.array_equal(saved["validation_rows"], validation) and report["validation_rows"] == validation.tolist())
    qc.check("training_only_scaling_replay", near(scaling, saved["scaling"]) and near(scaling, report["scaling"]))
    qc.check("encoder_feature_binding", np.array_equal(saved["indices"], inputs["common_indices"]))
    full_z, full_baseline = [], []
    for domain_index, name in enumerate(C.NAMES):
        print(f"[ATLAS QC] Encoder replay: {name}", flush=True)
        rows = np.flatnonzero(inputs["dataset"] == domain_index)
        domain = R.prepare_domain(C.source_arrays(name), scaling, inputs["common_indices"])
        z, hourly = R.encode(model, domain, device)
        full_z.append(z)
        full_baseline.append(R.ordered_baseline(domain))
        qc.check(f"{name}/checkpoint_reconstruction", near((z - rep["z_center"]) / rep["z_scale"], rep["z"][rows]))
        qc.check(f"{name}/hourly_reconstruction", near(hourly, rep["hourly"][rows]))
        qc.check(f"{name}/structural_encoder_states_zero", np.count_nonzero(hourly[domain["exposure"] == 0]) == 0)
        qc.check(f"{name}/clinical_anchors_replay", near(domain["anchors"], rep["anchors"][rows]))
        qc.check(f"{name}/clinical_probes_replay", near(domain["probes"], rep["probes"][rows]))
        measured = R.encode(model, domain, device, observed_only=True)[0]
        qc.check(f"{name}/observed_view_replay", near((measured - rep["z_center"]) / rep["z_scale"], rep["observed_z"][rows]))
        for scenario in ("point", "block"):
            removed = R.perturbation_mask(domain, scenario, domain_index)
            replay = R.encode(model, domain, device, observed_only=True, removed=removed)[0]
            qc.check(f"{name}/{scenario}_view_replay", near((replay - rep["z_center"]) / rep["z_scale"], rep[scenario + "_z"][rows]))
            masked_raw = domain["raw"].copy()
            for column, index in enumerate(inputs["common_indices"]):
                masked_raw[:, :, index][removed[:, :, column]] = np.nan
            anchors = R.clinical_views(masked_raw, domain["age"], domain["names"], scaling)[0]
            qc.check(f"{name}/{scenario}_anchor_replay", near(anchors, rep[scenario + "_anchors"][rows]))
            # Verify that changed held-out values cannot change masked input channels.
            probe_rows = np.arange(min(8, len(rows)))
            before = R.input_tensor(domain, probe_rows, observed_only=True, removed=removed[probe_rows])
            modified = dict(domain, truth=domain["truth"].copy(), values=domain["values"].copy())
            modified["truth"][removed] += 100
            modified["values"] += 100
            after = R.input_tensor(modified, probe_rows, observed_only=True, removed=removed[probe_rows])
            qc.check(f"{name}/{scenario}_masked_input_no_imputation_or_target_leak", np.array_equal(before, after))
        if domain_index == 0:
            value = R.validation_loss(model, domain, validation, device)
            qc.check("best_checkpoint_validation_loss", near(value, saved["best_loss"]) and near(value, report["best_validation_loss"]))
            qc.check("early_stopping_history", near(min(item["validation_loss"] for item in report["history"]), saved["best_loss"]))
        del domain
    z, center, scale = R.standardize(np.concatenate(full_z), train)
    baseline, bc, bs = R.standardize(np.concatenate(full_baseline), train)
    qc.check("reference_only_latent_standardization", near(center, rep["z_center"]) and near(scale, rep["z_scale"]) and near(z, rep["z"]))
    qc.check("reference_only_ordered_baseline", near(bc, rep["baseline_center"]) and near(bs, rep["baseline_scale"]) and near(baseline, rep["baseline"]))
    fitted = R.fit_probes(rep["z"], rep["probes"], train)
    qc.check("reference_only_probe_fit_rows", [rows.tolist() for rows in fitted["fit_rows"]] == report["probe_fit_rows"])
    for key in ("coef", "intercept", "valid", "tails"):
        qc.check(f"clinical_probe_replay/{key}", near(fitted[key], rep["probe_" + key]))


def transport_checks(qc, state):
    inputs, rep, output = state["inputs"], state["rep"], state["transport"]
    report = C.read_json(C.receipt_path(3))
    source_rows = np.flatnonzero((inputs["dataset"] == 0) & (inputs["partition"] == 0) & inputs["eligible"])
    for domain, name in enumerate(C.NAMES[1:], start=1):
        model = state["maps"][name]
        train = np.flatnonzero((inputs["dataset"] == domain) & (inputs["partition"] == 0) & inputs["eligible"])
        validation = np.flatnonzero((inputs["dataset"] == domain) & (inputs["partition"] == 1) & inputs["eligible"])
        qc.check(f"{name}/transport_source_training_rows", np.array_equal(model["common"]["source_fit_rows"], source_rows))
        qc.check(f"{name}/transport_target_training_rows", np.array_equal(model["common"]["target_fit_rows"], train))
        qc.check(f"{name}/unlabelled_selection_rows", np.array_equal(model["validation_rows"], validation))
        print(f"[ATLAS QC] Transport fitting replay: {name}", flush=True)
        fitted = T.fit_domain(rep, source_rows, train, validation, domain)
        qc.check(f"{name}/reference_and_target_training_support", near(fitted["common"], model["common"]))
        qc.check(f"{name}/solver_and_local_map_replay", near(fitted["candidates"], model["candidates"]))
        qc.check(f"{name}/candidate_reports_and_rejection_reasons",
                 near([T.candidate_summary(item) for item in model["candidates"]], report["domains"][name]["candidates"]))
        qc.check(f"{name}/solver_failure_reporting", near(fitted["solver_reports"], model["solver_reports"]))
        qc.check(f"{name}/validation_only_choice", fitted["selected"]["name"] == model["selected"]["name"] == report["domains"][name]["selected"])
        qc.check(f"{name}/validation_only_ot_comparator", fitted["ot_comparator"]["name"] == model["ot_comparator"]["name"] == report["domains"][name]["ot_comparator"])
        if model["selected"]["method"] != "identity":
            qc.check(f"{name}/selected_preservation_and_stability_gates", model["selected"]["assessment"]["passed"])
        rows = np.flatnonzero(inputs["dataset"] == domain)
        for method, choice in (("selected", model["selected"]), ("ot", model["ot_comparator"]), ("covariance", model["covariance"])):
            replay, supported = T.apply_map(rep["z"][rows], rep["anchors"][rows], model["common"], choice, inputs["eligible"][rows])
            qc.check(f"{name}/{method}_reconstruction", near(replay, output[method][rows]))
            qc.check(f"{name}/{method}_support_reconstruction", np.array_equal(supported, output[method + "_supported"][rows]))
            qc.check(f"{name}/{method}_unsupported_identity", np.array_equal(output[method][rows][~supported], rep["z"][rows][~supported]))
            shift = np.sqrt(np.mean((output[method][rows] - rep["z"][rows]) ** 2, axis=1))
            qc.check(f"{name}/{method}_correction_bound", np.all(shift <= C.CONFIG["transport"]["maximum_correction_rms"] + 1e-5))
    for method in ("selected", "ot", "covariance"):
        qc.check(f"{method}/reference_identity", np.array_equal(output[method][inputs["dataset"] == 0], rep["z"][inputs["dataset"] == 0]))
        qc.check(f"{method}/finite_and_row_aligned", output[method].shape == rep["z"].shape and np.isfinite(output[method]).all())
    qc.check("target_labels_excluded_from_transport_contract", report["target_labels_accessed"] is False)
    qc.check("local_transport_policy_recorded", report["mapping_policy"] == T.mapping_policy())


def evaluation_checks(qc, state):
    report = C.read_json(C.receipt_path(4))
    qc.check("test_recipe_binding", report["outcome_binding"] == E.outcome_binding())
    history = report.get("prior_evaluations", [])
    exposure = report.get("test_exposure")
    expected_exposure = "previously_evaluated_" + history[-1].get("kind", "data_repair") if history else "initial_locked_evaluation"
    qc.check("test_exposure_history_retained", exposure == expected_exposure and all(
        bool(item.get("reason", "").strip()) and all(key in item for key in
        ("generated_at_utc", "outcome_binding", "benchmarks", "selected")) for item in history))
    qc.check("method_revision_provenance_retained", all(
        "code" in item and "preservation" in item for item in history if item.get("kind") == "method_revision"))
    state["test_exposure"] = exposure
    qc.check("dataset_validation_scope", report.get("evaluation_scope") == "dataset_preservation"
             and report.get("mortality_models_fitted") is False)
    qc.check("test_preservation_reporting_replayed", near(E.preservation_reports(state["rep"], state["inputs"], state["maps"]), report["preservation"]))
    qc.check("test_did_not_change_validation_choices", report["selected"] == {name: model["selected"]["name"] for name, model in state["maps"].items()})


def atlas_checks(qc, state):
    report = C.read_json(C.receipt_path(5))
    qc.check("atlas_uses_current_evaluated_maps", report["selected"] == {
        name: model["selected"]["name"] for name, model in state["maps"].items()})
    qc.check("atlas_test_exposure_matches_evaluation", report["test_exposure"] == state["test_exposure"])
    saved = joblib.load(C.OUT / "atlas.joblib")
    fitted = A.fit_atlas(state["rep"], state["inputs"])
    qc.check("atlas_reference_training_only", np.array_equal(saved["fit_rows"], fitted["fit_rows"]))
    qc.check("reference_pca_replay", near(saved["pca"].components_, fitted["pca"].components_) and near(saved["pca"].mean_, fitted["pca"].mean_))
    qc.check("reference_groups_replay", near(saved["groups"].cluster_centers_, fitted["groups"].cluster_centers_))
    coordinates = pd.read_parquet(C.DATA / "coordinates.parquet")
    expected = A.coordinates(saved, state["rep"], state["transport"], state["inputs"])
    pd.testing.assert_frame_equal(expected, coordinates[expected.columns], check_exact=False, atol=2e-5, rtol=2e-4)
    pd.testing.assert_frame_equal(state["patients"], coordinates[state["patients"].columns])
    qc.check("coordinates_assignments_and_original_metadata_replayed", True)
    excluded = ~state["inputs"]["eligible"]
    qc.check("unrepresentable_patients_retained_and_flagged", len(coordinates) == len(excluded) and (coordinates.loc[excluded, "adapted_cluster"] == -1).all() and coordinates.loc[excluded, "adapted_x"].isna().all())
    summaries = A.trajectory_summary(coordinates, state["inputs"])
    pd.testing.assert_frame_equal(summaries, pd.read_parquet(C.DATA / "trajectories.parquet"), check_exact=False, atol=1e-8, rtol=1e-8)
    qc.check("all_original_clinical_trajectory_summaries_replayed", True)
    C.verify_files(C.read_json(C.receipt_path(1))["external"])
    qc.check("frozen_source_data_unchanged_after_replay", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    qc, state = QC(), {}
    failure = None
    try:
        qc.phase("1/5 Provenance, identities, partitions and feature eligibility", lambda: input_checks(qc, state))
        qc.phase("2/5 Reference scaling, checkpoint and masked-view replay", lambda: representation_checks(qc, state, args.device))
        qc.phase("3/5 Training-only transport, support and validation choices", lambda: transport_checks(qc, state))
        qc.phase("4/5 Locked clinical preservation replay", lambda: evaluation_checks(qc, state))
        qc.phase("5/5 Reference atlas and original clinical trajectory replay", lambda: atlas_checks(qc, state))
    except Exception:
        failure = traceback.format_exc()
        if not qc.checks or qc.checks[-1]["passed"]:
            qc.checks.append(dict(section=qc.section, name="phase_completed", passed=False, detail=failure))
    passed = failure is None and bool(qc.checks)
    report = dict(version=C.VERSION, status="PASS" if passed else "FAIL", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        required_checks=len(qc.checks), failures=sum(not row["passed"] for row in qc.checks), checks=qc.checks,
        runtime=C.runtime(), device=args.device, code=C.code_binding(),
        exception=failure,
        test_exposure=state.get("test_exposure"),
        dependencies=C.bind([C.receipt_path(i) for i in range(1, 6) if C.receipt_path(i).exists()]),
        eligibility="ATLAS_IMPLEMENTATION_AND_REPLAY_VERIFIED" if passed else "NOT_ELIGIBLE",
        interpretation=C.CONFIG["protocol"]["acceptance"], selected={name: model["selected"]["name"] for name, model in state.get("maps", {}).items()})
    C.write_json(C.OUT / "qc.json", report)
    print(f"\n[ATLAS QC {'PASS' if passed else 'FAIL'}] {report['eligibility']}", flush=True)
    print(f"    Checks: {report['required_checks']}; failures: {report['failures']}; report: {C.OUT / 'qc.json'}", flush=True)
    print(f"    Selected maps: {report['selected']}; test exposure: {report['test_exposure']}", flush=True)
    if failure:
        print(failure, flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
