"""Content/style split and effective regression score (main.typ @sec:content-style, @sec:head-weight).

Combines what were historically `atom_content_style.py` and
`effective_atom_ranking.py` into a single CLI. For every atom we compute

- activation rate per LARDv2 subset, ``r_{j,s}``;
- cross-subset coefficient of variation ``CV_j = std_s r_{j,s} / mean_s r_{j,s}`` (@eq:cv);
- ``mean|x_j|`` over all test patches (the unconditional expectation that
  appears in the effective-score definition @eq:effective);
- head weight ``||W d_j||``;
- effective score ``score_j = ||W d_j|| * mean|x_j|``.

Thresholding CV at its median across active atoms splits the dictionary
into contentful (below) and stylistic (above) halves, and we report the
share of total effective score on each side.

With ``--selectivity`` the split instead comes from the paired BOGO test of
``scripts/bogo_selectivity.py``: contentful = runway-selective atoms,
stylistic = agnostic + anti-selective atoms. Atoms that test as neither
(undetermined / inactive) are reported as a separate effective-score share,
so the two main shares no longer need to sum to 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer


app = typer.Typer(add_completion=False)


LARD_SUBSETS_DEFAULT = ["xplane", "ges", "arcgis", "bingmaps"]
LARD_TEST_SIZES_DEFAULT = [14020, 12791, 11526, 9135]  # total 47472


def _mutual_info(counts: np.ndarray, subset_totals: np.ndarray) -> np.ndarray:
    m, S = counts.shape
    total = subset_totals.sum()
    p_s = subset_totals / total
    p_a1 = counts.sum(1) / total
    p_a0 = 1.0 - p_a1
    p_a1_s = counts / total
    p_a0_s = (subset_totals[None, :] - counts) / total
    eps = 1e-30
    mi = np.zeros(m, dtype=np.float64)
    for s in range(S):
        mask1 = p_a1_s[:, s] > 0
        mi[mask1] += p_a1_s[mask1, s] * np.log(p_a1_s[mask1, s] / (p_a1[mask1] * p_s[s] + eps) + eps)
        mask0 = p_a0_s[:, s] > 0
        mi[mask0] += p_a0_s[mask0, s] * np.log(p_a0_s[mask0, s] / (p_a0[mask0] * p_s[s] + eps) + eps)
    return np.maximum(mi, 0.0)


@app.command()
def main(
    sparse_codes: Annotated[Path, typer.Option(help="Patch codes .h5 (indices + values)")],
    dictionary: Annotated[Path, typer.Option(help="K-SVD dictionary .npy (D x M)")],
    head: Annotated[Path, typer.Option(help="Head .npz with W, b from extract_patch_embeddings.py")],
    output: Annotated[Path, typer.Option()],
    num_patches_per_image: Annotated[int, typer.Option(help="Patches per image after cue exclusion")] = 255,
    subset_sizes: Annotated[str, typer.Option(help="Comma-separated subset image counts (default: LARDv2 test)")] = "",
    subset_names: Annotated[str, typer.Option()] = "",
    selectivity: Annotated[Path | None, typer.Option(help="bogo_selectivity.json; replaces the CV split")] = None,
):
    with h5py.File(sparse_codes, "r") as f:
        indices = f["indices"][:]                                          # [k, n_cols]
        values  = f["values"][:]
        # subset metadata; sparse_code_patches.jl copies these from the
        # patch embeddings H5 so downstream can recover per-subset boundaries.
        attrs_names = f.attrs["subset_names"] if "subset_names" in f.attrs else None
        attrs_sizes = f.attrs["subset_sizes"] if "subset_sizes" in f.attrs else None
        attrs_ppi   = int(f.attrs["patches_per_image"]) if "patches_per_image" in f.attrs else 0
    _, n_cols = indices.shape

    D = np.load(dictionary).astype(np.float64)
    W = np.load(head)["W"].astype(np.float64)
    # Paper convention: D has shape (d, m) with d the backbone dim and m the
    # dictionary size. Flip if stored the other way around.
    if D.shape[0] != W.shape[1]:
        D = D.T
    assert D.shape[0] == W.shape[1], f"dict shape {D.shape} incompatible with W {W.shape}"
    d, m = D.shape

    if subset_sizes:
        ss = [int(x) for x in subset_sizes.split(",")]
    elif attrs_sizes is not None:
        ss = list(map(int, attrs_sizes))
    else:
        ss = LARD_TEST_SIZES_DEFAULT
    if subset_names:
        sn = subset_names.split(",")
    elif attrs_names is not None:
        sn = [s.decode() if isinstance(s, bytes) else str(s) for s in attrs_names]
    else:
        sn = LARD_SUBSETS_DEFAULT[:len(ss)]
    if attrs_ppi:
        num_patches_per_image = attrs_ppi
    boundaries = np.cumsum([0] + [s * num_patches_per_image for s in ss])
    assert boundaries[-1] <= n_cols, (
        f"subset sizes sum to {boundaries[-1]} patches but codes have {n_cols} columns")

    counts = np.zeros((m, len(ss)), dtype=np.int64)
    abs_sums = np.zeros(m, dtype=np.float64)
    sq_sums  = np.zeros(m, dtype=np.float64)
    tots_per_patch = np.array([s * num_patches_per_image for s in ss], dtype=np.int64)
    for s, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        sub_idx = indices[:, lo:hi].ravel()
        sub_val = np.abs(values[:, lo:hi]).ravel()
        counts[:, s] = np.bincount(sub_idx, minlength=m)[:m]
        abs_sums += np.bincount(sub_idx, weights=sub_val,            minlength=m)[:m]
        sq_sums  += np.bincount(sub_idx, weights=sub_val * sub_val, minlength=m)[:m]

    rates = counts / tots_per_patch[None, :]
    mean_rate = rates.mean(1)
    cv = np.where(mean_rate > 0, rates.std(1) / mean_rate, 0.0)

    mi = _mutual_info(counts, tots_per_patch)
    p_s = tots_per_patch / tots_per_patch.sum()
    h_s = -np.sum(p_s * np.log(p_s + 1e-30))
    nmi = mi / h_s

    # ||W d_j|| and unconditional mean|x_j|.
    head_weight = np.linalg.norm(W @ D, axis=0)                            # [m]
    mean_abs_all = abs_sums / n_cols
    activation_count = counts.sum(1)
    active = mean_rate > 0.001

    mean_abs_cond = np.zeros(m); mean_abs_cond[active] = abs_sums[active] / activation_count[active]
    rms_cond      = np.zeros(m); rms_cond[active]      = np.sqrt(sq_sums[active] / activation_count[active])

    effective_score = head_weight * mean_abs_all
    order = np.argsort(-effective_score).tolist()

    median_cv = float(np.median(cv[active]))
    total_eff = effective_score.sum()
    if selectivity is None:
        split_method = "cv"
        content_mask = cv <= median_cv
        style_mask = ~content_mask
    else:
        split_method = "bogo"
        sel = json.loads(Path(selectivity).read_text())
        content_mask = np.zeros(m, dtype=bool); content_mask[sel["content_atom_ids"]] = True
        style_mask = np.zeros(m, dtype=bool); style_mask[sel["style_atom_ids"]] = True
    eff_content = float(effective_score[content_mask].sum() / total_eff) if total_eff > 0 else 0.0
    eff_style   = float(effective_score[style_mask].sum() / total_eff) if total_eff > 0 else 0.0
    eff_undet   = 1.0 - eff_content - eff_style
    print(f"[ATOMS] n_atoms={m}  split={split_method}  median CV={median_cv:.3f}  "
          f"eff_content={eff_content:.3f}  eff_style={eff_style:.3f}  "
          f"eff_undetermined={eff_undet:.3f}", flush=True)

    out = {
        "n_atoms": int(m),
        "subset_names": sn,
        "subset_sizes": ss,
        "cv": cv.tolist(),
        "mi": mi.tolist(),
        "nmi": nmi.tolist(),
        "per_subset_rates": {n: rates[:, i].tolist() for i, n in enumerate(sn)},
        "mean_activation_rate": mean_rate.tolist(),
        "mean_abs_all": mean_abs_all.tolist(),
        "mean_abs_cond": mean_abs_cond.tolist(),
        "rms_cond": rms_cond.tolist(),
        "head_weight": head_weight.tolist(),
        "effective_score": effective_score.tolist(),
        "effective_order": order,
        "median_cv": median_cv,
        "split_method": split_method,
        "content_atom_ids": np.flatnonzero(content_mask).tolist(),
        "style_atom_ids":   np.flatnonzero(style_mask).tolist(),
        "eff_score_on_contentful": eff_content,
        "eff_score_on_stylistic":  eff_style,
        "eff_score_on_undetermined": eff_undet,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"[ATOMS] wrote {output}")


if __name__ == "__main__":
    app()
