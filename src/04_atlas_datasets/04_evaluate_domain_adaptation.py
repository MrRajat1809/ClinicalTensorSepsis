"""Evaluate locked transport preservation without fitting outcome-prediction models."""
import importlib
import joblib
import numpy as np

C = importlib.import_module("01_prepare_atlas_inputs")
T = importlib.import_module("03_fit_domain_transport")


def outcome_binding():
    return dict(configuration=C.sha(C.HERE / "atlas_config.json"),
                inputs=C.read_json(C.receipt_path(1))["outputs"],
                representation=C.read_json(C.receipt_path(2))["outputs"],
                transport=C.read_json(C.receipt_path(3))["outputs"])


def evaluation_history(previous, locked, code):
    """Retain prior test exposure, including historical prediction benchmarks."""
    if previous is None:
        return [], "initial_locked_evaluation"
    history = list(previous.get("prior_evaluations", []))
    if previous["outcome_binding"] == locked and previous["code"] == code:
        return history, previous["test_exposure"]
    kind = "dataset_validation_update"
    history.append(dict(kind=kind, reason="Dataset preparation or validation revised; prior test exposure retained",
        generated_at_utc=previous["generated_at_utc"], outcome_binding=previous["outcome_binding"],
        code=previous["code"], test_exposure=previous.get("test_exposure", "initial_locked_evaluation"),
        benchmarks=previous.get("benchmarks", []), selected=previous["selected"],
        preservation=previous.get("preservation", {})))
    return history, "previously_evaluated_" + kind


def preservation_reports(rep, inputs, models):
    result = {}
    for domain, name in enumerate(C.NAMES[1:], start=1):
        rows = np.flatnonzero((inputs["dataset"] == domain) & (inputs["partition"] == 2) & inputs["eligible"])
        C.require(len(rows) >= 2, f"{name}: insufficient test representations")
        state = models[name]
        result[name] = {method: T.assess(rep, rows, state["common"], choice, C.CONFIG["seed"] + domain)
                       for method, choice in (("selected", state["selected"]), ("ot", state["ot_comparator"]), ("covariance", state["covariance"]))}
    return result



def main():
    start = C.begin(4)
    locked = outcome_binding()
    previous = C.read_json(C.receipt_path(4)) if C.receipt_path(4).exists() else None
    history, exposure = evaluation_history(previous, locked, start["code"])
    inputs = C.load_npz(C.DATA / "inputs.npz")
    rep = C.load_npz(C.DATA / "representations.npz")
    maps = joblib.load(C.OUT / "transport.joblib")
    selected = {name: state["selected"]["name"] for name, state in maps.items()}
    preservation = preservation_reports(rep, inputs, maps)
    C.require(locked == outcome_binding(), "Locked inputs changed during evaluation")
    C.finish(4, start, [], outcome_binding=locked, preservation=preservation,
             prior_evaluations=history, test_exposure=exposure, selected=selected,
             evaluation_scope="dataset_preservation", mortality_models_fitted=False,
             interpretation="Locked clinical preservation and recording stability; no outcome-prediction benchmark.")
    for path in (C.DATA / "predictions.parquet", C.OUT / "benchmark.csv",
                 C.OUT / "calibration.csv", C.OUT / "mortality.joblib"):
        path.unlink(missing_ok=True)
    print(f"[04] Complete: locked preservation evaluated; maps={selected}; {exposure}", flush=True)


if __name__ == "__main__":
    main()
