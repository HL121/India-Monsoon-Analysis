import os
from pathlib import Path

_mpl_cache = Path(os.environ.get("TMPDIR", "/private/tmp")) / "matplotlib_codex_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

from .config import load_config
from .plotting import plot_all_targets, sequence_label
from .training import run_experiment


def run_config(config_path):
    config = load_config(config_path)
    results, summary = run_experiment(config)

    seq = sequence_label(config)
    summary = summary.copy()
    summary.insert(0, "experiment", config["experiment_name"])
    summary.insert(1, "seq_len", config["sequence"]["seq_len"])

    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / f"{config['experiment_name']}_{seq}_summary.csv"
    summary.to_csv(summary_path, index=False)

    figure_paths = plot_all_targets(results, config)

    print(summary)
    print(f"Saved summary to {summary_path}")
    for path in figure_paths:
        print(f"Saved figure to {path}")

    return results, summary, summary_path, figure_paths
