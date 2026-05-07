from __future__ import annotations

import shutil
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
except Exception:  # pragma: no cover - fallback for broken local scientific envs
    plt = None
    pd = None


ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
OUT = DOCS_ROOT / "source" / "_static" / "images" / "benchmarks"
ASSETS = ROOT / "benchmarking" / "report_assets"
RESULTS = ROOT / "benchmarking" / "results"


def copy_existing_assets() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in ASSETS.glob("*.png"):
        shutil.copy2(path, OUT / path.name)


def load_results() -> pd.DataFrame:
    if pd is None:
        return None
    frames = []
    for path in sorted(RESULTS.glob("*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def generated_summary(df) -> None:
    if df is None or df.empty or plt is None:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (1000, 420), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 980, 400), outline="#24546a", width=3)
        draw.text((55, 80), "Benchmark figures are copied from benchmarking/report_assets.", fill="#20333d")
        draw.text((55, 125), "Regenerate in a full docs environment for CSV-derived plots.", fill="#20333d")
        image.save(OUT / "generated_runtime_summary.png")
        return
    sample_col = _first_existing(list(df.columns), ["n_samples", "samples", "sample_size"])
    method_col = _first_existing(list(df.columns), ["method", "configuration", "name"])
    time_col = _first_existing(list(df.columns), ["total_seconds_mean", "mean_seconds", "total_seconds", "fit_seconds_mean"])
    if sample_col is None or method_col is None or time_col is None:
        return
    grouped = df.groupby([sample_col, method_col], as_index=False)[time_col].mean()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for method, part in grouped.groupby(method_col):
        label = str(method)
        if len(label) > 38:
            label = label[:35] + "..."
        ax.plot(part[sample_col], part[time_col], marker="o", label=label)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Benchmark result summary")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(OUT / "generated_runtime_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    copy_existing_assets()
    generated_summary(load_results())


if __name__ == "__main__":
    main()
