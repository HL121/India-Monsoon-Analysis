import os
from pathlib import Path

_mpl_cache = Path(os.environ.get("TMPDIR", "/private/tmp")) / "matplotlib_codex_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "joint": "EncoderHead+Transformer",
}


def sequence_label(config):
    return f"seq_len_{config['sequence']['seq_len']}"


def chunk_years(years, chunk_size=4):
    years = list(years)
    return [years[i : i + chunk_size] for i in range(0, len(years), chunk_size)]


def _encoder_results_by_model(results):
    return {
        model_name: {
            r["test_year"]: r
            for r in results[f"{model_name}_encoder"]
        }
        for model_name in ["lstm", "transformer", "joint"]
    }


def _common_years(by_model, years_to_plot):
    return sorted(
        set(years_to_plot)
        & set(by_model["lstm"].keys())
        & set(by_model["transformer"].keys())
        & set(by_model["joint"].keys())
    )


def _year_range_label(years):
    return f"{years[0]}_{years[-1]}"


def plot_target_timeseries(results, config, target_idx, years_to_plot=None, out_path=None):
    if years_to_plot is None:
        years_to_plot = config["data"]["years"]
    target_name = config["target"]["plot_names"][target_idx]

    by_model = _encoder_results_by_model(results)
    common_years = _common_years(by_model, years_to_plot)
    if not common_years:
        raise ValueError("No overlapping years found across LSTM, Transformer, and Joint results.")

    obs_parts, year_axis_parts = [], []
    for yr in common_years:
        obs_year = by_model["transformer"][yr]["true"]
        n = obs_year.shape[0]
        obs_parts.append(obs_year)
        year_axis_parts.append(yr + (np.arange(n) / n))

    obs_all = np.concatenate(obs_parts, axis=0)
    x_year_all = np.concatenate(year_axis_parts, axis=0)

    def concat_pred(model_name):
        return np.concatenate(
            [by_model[model_name][yr]["pred"] for yr in common_years], axis=0
        )

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, sharey=True)
    for ax, model_name in zip(axes, ["lstm", "transformer", "joint"]):
        pred_all = concat_pred(model_name)
        ax.plot(
            x_year_all,
            obs_all[:, target_idx],
            color="tab:blue",
            lw=1.2,
            label=f"Observed {target_name}",
        )
        ax.plot(
            x_year_all,
            pred_all[:, target_idx],
            color="tab:red",
            lw=1.2,
            ls="--",
            label=f"Predicted {target_name}",
        )
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_ylabel(target_name)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Year")
    axes[-1].set_xlim(common_years[0], common_years[-1] + 1)
    tick_years = common_years[::2]
    if tick_years[-1] != common_years[-1]:
        tick_years.append(common_years[-1])
    axes[-1].set_xticks(tick_years)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        (
            f"Observed vs Predicted {target_name} "
            f"({sequence_label(config).replace('_', ' ')}) | "
            f"Years: {common_years[0]}-{common_years[-1]}"
        ),
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_trajectory(results, config, years_to_plot=None, out_path=None):
    if years_to_plot is None:
        years_to_plot = config["data"]["years"]

    target_names = config["target"]["plot_names"]
    if len(target_names) < 2:
        raise ValueError("Trajectory plots require at least two target columns.")

    by_model = _encoder_results_by_model(results)
    common_years = _common_years(by_model, years_to_plot)
    if not common_years:
        raise ValueError("No overlapping years found across LSTM, Transformer, and Joint results.")

    rows = [
        ("Observed", None, "k"),
        ("LSTM", by_model["lstm"], "tab:blue"),
        ("Transformer", by_model["transformer"], "crimson"),
        ("EncoderHead+Transformer", by_model["joint"], "tab:green"),
    ]

    nrows = len(rows)
    ncols = len(common_years)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.4 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for col_idx, year in enumerate(common_years):
        obs = by_model["transformer"][year]["true"]

        for row_idx, (row_name, res_by_year, color) in enumerate(rows):
            ax = axes[row_idx, col_idx]
            traj = obs if row_name == "Observed" else res_by_year[year]["pred"]

            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.8)
            ax.scatter(traj[0, 0], traj[0, 1], c=color, s=34, zorder=4)

            if row_idx == 0:
                ax.set_title(str(year))
            if col_idx == 0:
                ax.set_ylabel(f"{target_names[1]} ({row_name})")

            ax.axhline(0, color="0.82", lw=0.8)
            ax.axvline(0, color="0.82", lw=0.8)
            ax.grid(alpha=0.25)
            ax.set_aspect("equal", adjustable="box")

    for col_idx in range(ncols):
        axes[-1, col_idx].set_xlabel(target_names[0])

    fig.suptitle(
        (
            f"{config['experiment_name'].upper()} Trajectories "
            f"({sequence_label(config).replace('_', ' ')}): "
            "Observed and Model Predictions"
        ),
        y=1.01,
    )
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_all_targets(results, config):
    out_dir = Path(config["paths"]["figure_dir"])
    out_paths = []
    seq = sequence_label(config)
    experiment_name = config["experiment_name"]
    all_years = config["data"]["years"]
    year_range = _year_range_label(all_years)

    for i, name in enumerate(config["target"]["plot_names"]):
        out_path = (
            out_dir
            / f"{experiment_name}_{name}_{seq}_timeseries_{year_range}.png"
        )
        fig = plot_target_timeseries(
            results,
            config,
            i,
            years_to_plot=all_years,
            out_path=out_path,
        )
        plt.close(fig)
        out_paths.append(out_path)

    if config["target"]["type"].lower() in {"miso", "mjo"}:
        for years in chunk_years(config["data"]["years"], chunk_size=4):
            out_path = (
                out_dir
                / f"{experiment_name}_{seq}_trajectory_{_year_range_label(years)}.png"
            )
            fig = plot_trajectory(results, config, years_to_plot=years, out_path=out_path)
            plt.close(fig)
            out_paths.append(out_path)

    return out_paths
