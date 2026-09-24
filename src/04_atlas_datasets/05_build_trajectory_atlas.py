"""Build reference-fitted coordinates and trajectory groups, with original clinical summaries."""
import importlib

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

C = importlib.import_module("01_prepare_atlas_inputs")


def fit_atlas(rep, inputs):
    rows = np.flatnonzero((inputs["dataset"] == 0) & (inputs["partition"] == 0) & inputs["eligible"])
    values = np.ascontiguousarray(rep["z"][rows], dtype=np.float64)
    with threadpool_limits(limits=1):
        pca = PCA(n_components=2, svd_solver="full").fit(values)
        groups = KMeans(n_clusters=C.CONFIG["atlas"]["clusters"], n_init=10,
                       random_state=C.CONFIG["seed"]).fit(values)
    return dict(pca=pca, groups=groups, fit_rows=rows)


def coordinates(state, rep, transport, inputs):
    result = pd.DataFrame(dict(row_index=np.arange(len(inputs["dataset"]))))
    valid = inputs["eligible"]
    for name, values in (("original", rep["z"]), ("adapted", transport["selected"])):
        values = np.ascontiguousarray(values, dtype=np.float64)
        with threadpool_limits(limits=1):
            xy = state["pca"].transform(values)
        xy[~valid] = np.nan
        clusters = np.full(len(values), -1, dtype=np.int16)
        with threadpool_limits(limits=1):
            clusters[valid] = state["groups"].predict(values[valid])
        result[name + "_x"], result[name + "_y"] = xy[:, 0], xy[:, 1]
        result[name + "_cluster"] = clusters
    result["transport_supported"] = transport["selected_supported"]
    return result


def trajectory_summary(table, inputs):
    rows = []
    for domain, name in enumerate(C.NAMES):
        source = C.source_arrays(name)
        assignment = table.loc[inputs["dataset"] == domain, "adapted_cluster"].to_numpy()
        for cluster in range(C.CONFIG["atlas"]["clusters"]):
            selected = np.flatnonzero(assignment == cluster)
            if not len(selected):
                continue
            for feature, label in enumerate(source["features"].tolist()):
                values = source["raw"][selected, :, feature]
                for hour in range(24):
                    measured = values[:, hour][np.isfinite(values[:, hour])]
                    quantiles = np.quantile(measured, [0.25, 0.5, 0.75]).tolist() if len(measured) else [None] * 3
                    rows.append(dict(dataset=name, cluster=cluster, feature=label, unit=str(source["units"][feature]),
                        hour=hour, patients_in_group=len(selected), exposed_patients=int((source["exposure_seconds"][selected, hour] > 0).sum()),
                        observed_patients=len(measured), q25=quantiles[0], median=quantiles[1], q75=quantiles[2]))
    return pd.DataFrame(rows)


def figures(table, trajectories, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    legend = [Line2D([0], [0], color=plt.get_cmap("tab10")(cluster), label=f"Group {cluster}")
              for cluster in range(C.CONFIG["atlas"]["clusters"])]

    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    for column, name in enumerate(C.NAMES):
        subset = table[(table["dataset"] == name) & table["eligible"]]
        for row, view in enumerate(("original", "adapted")):
            axis = axes[row, column]
            axis.scatter(subset[view + "_x"], subset[view + "_y"], c=subset[view + "_cluster"],
                         cmap="tab10", vmin=0, vmax=9, s=3, alpha=0.3, rasterized=True)
            label = "original" if view == "original" else selected.get(name, "reference unchanged")
            axis.set_title(f"{name}: {label}")
            axis.set_xlabel("Reference PC1")
            axis.set_ylabel("Reference PC2")
    figure.suptitle("Reference trajectory groups; visual overlap is not evidence of clinical equivalence")
    figure.legend(handles=legend, loc="outside lower center", ncol=len(legend))
    figure.savefig(C.OUT / "atlas.png", dpi=160)
    plt.close(figure)
    features = C.CONFIG["atlas"]["plot_features"]
    figure, axes = plt.subplots(len(C.NAMES), len(features), figsize=(18, 9), constrained_layout=True, squeeze=False, sharey="col")
    for row, name in enumerate(C.NAMES):
        for column, feature in enumerate(features):
            axis = axes[row, column]
            subset = trajectories[(trajectories["dataset"] == name) & (trajectories["feature"] == feature)]
            axis.set_xlim(0, 23)
            if subset.empty or not subset["median"].notna().any():
                axis.text(0.5, 0.5, "No accepted observations\nCheck source-unit/coverage audit",
                          ha="center", va="center", transform=axis.transAxes, fontsize=8)
            for cluster, group in subset.groupby("cluster"):
                axis.plot(group["hour"], group["median"], color=plt.get_cmap("tab10")(int(cluster)), linewidth=1,
                          marker=".", markersize=2)
            observed = int(subset["observed_patients"].sum()) if len(subset) else 0
            axis.set_title(f"{name}\n{feature}; {observed:,} observed cells")
            axis.set_xlabel("Hour after source-specific onset")
            if len(subset):
                axis.set_ylabel(str(subset["unit"].iloc[0]))
    figure.suptitle("Observed clinical medians by assigned reference group; patient composition changes with follow-up")
    figure.legend(handles=legend, loc="outside lower center", ncol=len(legend))
    figure.savefig(C.OUT / "trajectories.png", dpi=160)
    plt.close(figure)


def main():
    start = C.begin(5)
    inputs = C.load_npz(C.DATA / "inputs.npz")
    rep = C.load_npz(C.DATA / "representations.npz")
    transport = C.load_npz(C.DATA / "transport.npz")
    patients = pd.read_parquet(C.DATA / "patients.parquet")
    evaluation = C.read_json(C.receipt_path(4))
    selected = evaluation["selected"]
    state = fit_atlas(rep, inputs)
    table = pd.concat([patients, coordinates(state, rep, transport, inputs)], axis=1)
    summaries = trajectory_summary(table, inputs)
    table.to_parquet(C.DATA / "coordinates.parquet", index=False)
    summaries.to_parquet(C.DATA / "trajectories.parquet", index=False)
    joblib.dump(state, C.OUT / "atlas.joblib", compress=3)
    figures(table, summaries, selected)
    C.finish(5, start, [C.DATA / "coordinates.parquet", C.DATA / "trajectories.parquet", C.OUT / "atlas.joblib",
                       C.OUT / "atlas.png", C.OUT / "trajectories.png"], fit_rows=state["fit_rows"].tolist(),
             explained_variance_ratio=state["pca"].explained_variance_ratio_.tolist(),
             selected=selected, test_exposure=evaluation["test_exposure"],
             interpretation="Reference-training PCA and KMeans; held-out coordinates are transformations only. Groups are descriptive, not validated phenotypes. Clinical summaries pool partitions descriptively and use original observations only.")
    print(f"[05] Maps: {selected}", flush=True)
    print(f"[05] Complete: {len(table):,} patient rows retained; coordinates, original trajectories and two figures saved", flush=True)


if __name__ == "__main__":
    main()
