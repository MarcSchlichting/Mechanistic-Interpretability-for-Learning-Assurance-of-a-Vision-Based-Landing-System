"""Aggregate per-seed result JSONs into a single dictionary file the paper reads.

Writes ``figs/aggregated.json``; every hand-written typst figure/table loads
this file via ``json()`` and does its own layout. Keeping layout in typst lets
us iterate on formatting without rerunning Python; this script's only job is
numerical crunching (mean/std across seeds, histogram binning).

    {
      "n_seeds": {"pretrained": <int>, "scratch": <int>},
      "regression": {
        "<variant>": {
          "train_mae":    [mean, std],
          "train_median": [mean, std],
          "test_mae":     [mean, std],
          "test_median":  [mean, std]
        } | null
      },
      "content_style": {
        "<variant>": {
          "content": [mean, std],
          "style":   [mean, std]
        } | null
      },
      "cv_hist": {
        "variant": "pretrained", "seed": 0,
        "bin_centers": [...], "bin_counts": [...],
        "median_cv": <float>, "bin_width": <float>
      } | null,
      "ims_sweep": {
        "seeds": [{"seed": <int>, "nz": [...], "auroc": [...],
                   "content_frac": [...]}, ...],
        "mean":  {"nz": [...], "auroc": [...], "content_frac": [...]}
      } | null,
      "content_style_bogo": {           # optional, from *_atoms_bogo.json
        "<variant>": {"content": [mean, std], "style": [mean, std],
                      "undetermined": [mean, std]} | null
      },
      "bogo_selectivity": {             # optional, from *_bogo_selectivity.json
        "<variant>": {"<label>": [mean, std], ..., "kappa_vs_cv": [mean, std]} | null
      }
    }

Atom ids are not comparable across seeds (each seed has its own dictionary),
so BOGO-selectivity is aggregated as label counts, not per-atom agreement.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


VARIANTS = ("pretrained", "scratch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("figs/aggregated.json"))
    return p.parse_args()


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def _load_train(results_dir: Path, variant: str) -> dict[int, dict]:
    """seed -> {train_mae, train_median, test_mae, test_median}."""
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(str(results_dir / f"lard_{variant}_seed*_train.json"))):
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        seed = int(Path(path).stem.split("_seed")[1].split("_")[0])
        final = data.get("final") or data.get("best") or {}
        tr = final.get("train") or {}
        te = final.get("test")  or {}
        if not tr and not te and final:
            te = final
        entry = {
            "train_mae":    float(tr.get("mae_px",    float("nan"))),
            "train_median": float(tr.get("median_px", float("nan"))),
            "test_mae":     float(te.get("mae_px",    float("nan"))),
            "test_median":  float(te.get("median_px", float("nan"))),
        }
        eval_path = results_dir / f"lard_{variant}_seed{seed}_eval.json"
        if eval_path.exists():
            try:
                edata = json.loads(eval_path.read_text())
                etr = edata.get("train", {}); ete = edata.get("test", {})
                if etr:
                    entry["train_mae"]    = float(etr.get("mae_px",    entry["train_mae"]))
                    entry["train_median"] = float(etr.get("median_px", entry["train_median"]))
                if ete:
                    entry["test_mae"]    = float(ete.get("mae_px",    entry["test_mae"]))
                    entry["test_median"] = float(ete.get("median_px", entry["test_median"]))
            except Exception:
                pass
        out[seed] = entry
    return out


def _load_content_style(results_dir: Path, variant: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(str(results_dir / f"{variant}_seed*_atoms.json"))):
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        seed = int(Path(path).stem.split("_seed")[1].split("_")[0])
        c = data.get("eff_score_on_contentful")
        s = data.get("eff_score_on_stylistic")
        if c is None or s is None:
            continue
        out[seed] = {"content": float(c), "style": float(s)}
    return out


def _load_content_style_bogo(results_dir: Path, variant: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(str(results_dir / f"{variant}_seed*_atoms_bogo.json"))):
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        seed = int(Path(path).stem.split("_seed")[1].split("_")[0])
        out[seed] = {"content":      float(data["eff_score_on_contentful"]),
                     "style":        float(data["eff_score_on_stylistic"]),
                     "undetermined": float(data.get("eff_score_on_undetermined", 0.0))}
    return out


BOGO_LABELS = ("selective", "anti", "agnostic", "undetermined", "inactive")


def _load_bogo_selectivity(results_dir: Path, variant: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(str(results_dir / f"{variant}_seed*_bogo_selectivity.json"))):
        try:
            summary = json.loads(Path(path).read_text())["summary"]
        except Exception:
            continue
        seed = int(Path(path).stem.split("_seed")[1].split("_")[0])
        entry = {k: float(summary["counts"].get(k, 0)) for k in BOGO_LABELS}
        kappa = (summary.get("vs_cv") or {}).get("cohen_kappa_selective_vs_cv_content")
        if kappa is not None:
            entry["kappa_vs_cv"] = float(kappa)
        out[seed] = entry
    return out


def _agg(seeds_dict: dict[int, dict], key: str) -> list[float] | None:
    values = [v[key] for v in seeds_dict.values()
              if key in v and not math.isnan(v[key])]
    if not values:
        return None
    mu, sd = mean_std(values)
    return [mu, sd]


def _cv_hist(results_dir: Path, variant: str = "pretrained", seed: int = 0,
             n_bins: int = 40) -> dict | None:
    """Bin the per-atom cross-subset CV distribution into equal-width bins up
    to the 99th percentile (tails past the cutoff pile into the final bin).
    Pretrained seed 0 is the reference figure; we never show scratch here."""
    path = results_dir / f"{variant}_seed{seed}_atoms.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    cv_values = [float(v) for v in (data.get("cv") or []) if v is not None]
    if not cv_values:
        return None
    median = float(data.get("median_cv", sorted(cv_values)[len(cv_values) // 2]))
    srt = sorted(cv_values)
    lo, hi = 0.0, srt[int(0.99 * len(srt))]
    bin_width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in cv_values:
        k = min(int((v - lo) / bin_width), n_bins - 1) if v >= lo else 0
        counts[k] += 1
    centers = [lo + (i + 0.5) * bin_width for i in range(n_bins)]
    return {
        "variant": variant, "seed": seed,
        "bin_centers": centers, "bin_counts": counts,
        "median_cv": median, "bin_width": bin_width,
    }


def _ims_sweep(results_dir: Path, variant: str = "pretrained") -> dict | None:
    """Collect per-seed L1-sweep curves and column-wise means.

    Assumes a common lambda grid across seeds (all sweeps launched with the
    same λ list). If lengths differ we trim each mean column to whatever seeds
    reached that index.
    """
    files = sorted(glob.glob(str(results_dir / f"{variant}_seed*_ims_sweep.json")))
    if not files:
        return None
    seeds: list[dict] = []
    for fp in files:
        try:
            data = json.loads(Path(fp).read_text())
        except Exception:
            continue
        seeds.append({
            "seed":         int(data.get("seed", -1)),
            "nz":           [float(x) for x in data["nz"]],
            "auroc":        [float(x) for x in data["auroc"]],
            "content_frac": [float(x) * 100.0 for x in data["content_frac"]],
        })
    if not seeds:
        return None
    def col_mean(key: str) -> list[float]:
        rows = [s[key] for s in seeds]
        L = max(len(r) for r in rows)
        out = []
        for j in range(L):
            xs = [r[j] for r in rows if len(r) > j]
            out.append(sum(xs) / len(xs))
        return out
    return {
        "seeds": seeds,
        "mean": {
            "nz":           col_mean("nz"),
            "auroc":        col_mean("auroc"),
            "content_frac": col_mean("content_frac"),
        },
    }


def main() -> None:
    args = parse_args()

    regression: dict = {}
    content_style: dict = {}
    content_style_bogo: dict = {}
    bogo_selectivity: dict = {}
    n_seeds: dict[str, int] = {}

    for variant in VARIANTS:
        train = _load_train(args.results_dir, variant)
        cs    = _load_content_style(args.results_dir, variant)
        n_seeds[variant] = len(train)

        reg = {k: _agg(train, k) for k in
               ("train_mae", "train_median", "test_mae", "test_median")}
        regression[variant] = reg if any(v is not None for v in reg.values()) else None

        cs_agg = {k: _agg(cs, k) for k in ("content", "style")}
        content_style[variant] = cs_agg if any(v is not None for v in cs_agg.values()) else None

        csb = _load_content_style_bogo(args.results_dir, variant)
        csb_agg = {k: _agg(csb, k) for k in ("content", "style", "undetermined")}
        content_style_bogo[variant] = csb_agg if any(v is not None for v in csb_agg.values()) else None

        sel = _load_bogo_selectivity(args.results_dir, variant)
        sel_agg = {k: _agg(sel, k) for k in (*BOGO_LABELS, "kappa_vs_cv")}
        bogo_selectivity[variant] = sel_agg if any(v is not None for v in sel_agg.values()) else None

    cv_hist   = _cv_hist(args.results_dir)
    ims_sweep = _ims_sweep(args.results_dir)

    # Refuse to clobber if nothing has landed.
    if (not any(regression.values()) and not any(content_style.values())
            and cv_hist is None and ims_sweep is None):
        print(f"[AGG] no per-seed JSONs under {args.results_dir}/; refusing to "
              f"overwrite {args.out} with empty content.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_seeds":       n_seeds,
        "regression":    regression,
        "content_style": content_style,
        "cv_hist":       cv_hist,
        "ims_sweep":     ims_sweep,
        "content_style_bogo": content_style_bogo,
        "bogo_selectivity":   bogo_selectivity,
    }, indent=2) + "\n")
    ims_n = len(ims_sweep["seeds"]) if ims_sweep else 0
    print(f"[AGG] wrote {args.out}  n_seeds={n_seeds}  "
          f"cv_hist={'yes' if cv_hist else 'no'}  ims_sweep={ims_n} seeds")


if __name__ == "__main__":
    main()
