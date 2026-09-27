"""Paired BOGO-selectivity test: which atoms fire more when the runway is in view?

Alternative to the cross-subset CV split of ``scripts/atoms.py``. Every test
image contributes a BOGO crop (runway inside) and a negative crop of the same
image (``scripts/extract_paired_patches.py``), so image-level nuisances
(renderer, LARDv2 source, weather, season) cancel within each pair.

Per crop i and atom j the activation a_ij is the number of patches whose
8-sparse code contains j (``--stat count``) or the summed |coef|
(``--stat abs``). Pairs are pooled into clusters (default: airport) because
frames of one approach are strongly dependent. The same airport appears in
all four LARDv2 sources, so airport clusters also span sources. The effect size is the log
rate ratio

    theta_j = log((sum_i a_ij^BOGO + eps_j) / (sum_i a_ij^neg + eps_j)).

Tests per atom:

- **Primary (D).** Sign-flip permutation test that swaps BOGO/negative labels
  cluster-wise. Exact under within-pair exchangeability (all 2^C flips are
  enumerated when that is cheaper than ``--n-perm``). Studentized max-T
  gives family-wise adjusted p-values in both directions.
- **Margin (E).** Cluster bootstrap of theta_j. Percentile-inversion
  p-values for "theta > log M" (selective), "theta < -log M" (anti) and the
  TOST equivalence claim "|theta| < log M" (agnostic), BH-adjusted.
- **Secondary, iid (A, B).** Exact McNemar test on "fires anywhere in the
  crop" and a Wilcoxon signed-rank test (Pratt zeros) on the paired
  difference. Both ignore clustering; reported for comparison only.

Labels (``--alpha`` = a, ``--margin`` = M):

- ``selective``: max-T p(+) < a, BH p(theta > log M) < a, same sign in at
  least ``--min-consistent`` LARDv2 sources            -> contentful
- ``anti``: the mirror image                           -> stylistic
- ``agnostic``: BH TOST p < a                          -> stylistic
- ``undetermined``: none of the above (not enough evidence either way)
- ``inactive``: mean per-patch firing rate <= ``--min-rate``

With fewer than ``--min-clusters`` clusters the cluster bootstrap is
degenerate (with one cluster every resample equals the data and TOST
"passes" trivially), so all active atoms are left ``undetermined``.

Contentful here means runway-selective. Off-runway scene content (grass,
roads, buildings) is agnostic by construction, and "anti" partly reflects
matching pursuit's fixed k slots being taken by runway atoms in BOGO crops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import scipy.sparse as sp
import typer
from scipy.stats import binom, wilcoxon


app = typer.Typer(add_completion=False)

LABELS = ("selective", "anti", "agnostic", "undetermined", "inactive")


def _num_atoms(f: h5py.File, indices_max: int) -> int:
    if "num_atoms" in f:
        return int(f["num_atoms"][()])
    if "num_atoms" in f.attrs:
        return int(f.attrs["num_atoms"])
    return indices_max + 1


def _per_crop(path: Path, m: int | None = None, chunk_images: int = 2048):
    """Per-crop firing counts and summed |coef|, both [N_images, m].

    Padding entries (value exactly 0, written when a column has < k
    non-zeros) are dropped so they don't inflate atom 0.
    """
    with h5py.File(path, "r") as f:
        ppi = int(f.attrs["patches_per_image"])
        idx_ds, val_ds = f["indices"], f["values"]
        _, n_cols = idx_ds.shape
        if m is None:
            m = _num_atoms(f, int(idx_ds[:, : min(n_cols, 100_000)].max()))
        n_img = n_cols // ppi
        counts = np.zeros((n_img, m), dtype=np.float64)
        abs_sum = np.zeros((n_img, m), dtype=np.float64)
        for i0 in range(0, n_img, chunk_images):
            i1 = min(n_img, i0 + chunk_images)
            idx = idx_ds[:, i0 * ppi: i1 * ppi].astype(np.int64)
            val = val_ds[:, i0 * ppi: i1 * ppi]
            img = (np.arange(idx.shape[1]) // ppi)[None, :]
            keep = val != 0
            flat = (img * m + idx)[keep]
            n = (i1 - i0) * m
            counts[i0:i1] = np.bincount(flat, minlength=n).reshape(-1, m)
            abs_sum[i0:i1] = np.bincount(flat, weights=np.abs(val[keep]).astype(np.float64),
                                         minlength=n).reshape(-1, m)
    return counts, abs_sum, ppi, m


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs pass through)."""
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def _log_ratio(num: np.ndarray, den: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return np.log((num + eps) / (den + eps))


def _flip_null(Bc: np.ndarray, Ic: np.ndarray, eps: np.ndarray, n_perm: int,
               rng: np.random.Generator, chunk: int = 1000):
    """theta under cluster-wise label swaps. Returns (theta_null [P, m], exact)."""
    C = Bc.shape[0]
    diff = Bc - Ic
    sB, sI = Bc.sum(0), Ic.sum(0)
    exact = C < 63 and 2 ** C <= n_perm
    if exact:
        S = ((np.arange(2 ** C)[:, None] >> np.arange(C)[None, :]) & 1).astype(np.float64)
    else:
        S = rng.integers(0, 2, size=(n_perm, C)).astype(np.float64)
    out = np.empty((S.shape[0], Bc.shape[1]))
    for lo in range(0, S.shape[0], chunk):
        sd = S[lo: lo + chunk] @ diff
        out[lo: lo + chunk] = _log_ratio(sB - sd, sI + sd, eps)
    return out, exact


def _bootstrap(Bc: np.ndarray, Ic: np.ndarray, eps: np.ndarray, n_boot: int,
               rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
    C = Bc.shape[0]
    out = np.empty((n_boot, Bc.shape[1]))
    for lo in range(0, n_boot, chunk):
        n = min(chunk, n_boot - lo)
        W = rng.multinomial(C, np.full(C, 1.0 / C), size=n).astype(np.float64)
        out[lo: lo + n] = _log_ratio(W @ Bc, W @ Ic, eps)
    return out


def _perm_pvalues(theta, null, exact, active):
    """Marginal and studentized max-T p-values for both tails."""
    P = null.shape[0]
    add = 0 if exact else 1   # identity flip is already in an exhaustive null
    mu, sd = null.mean(0), null.std(0)
    ok = active & (sd > 0)
    m = len(theta)
    res = {k: np.full(m, np.nan) for k in ("p_plus", "p_minus", "p_maxT_plus", "p_maxT_minus")}
    res["p_plus"][ok] = (add + (null[:, ok] >= theta[ok]).sum(0)) / (P + add)
    res["p_minus"][ok] = (add + (null[:, ok] <= theta[ok]).sum(0)) / (P + add)
    z_null = (null[:, ok] - mu[ok]) / sd[ok]
    z_obs = (theta[ok] - mu[ok]) / sd[ok]
    zmax, zmin = z_null.max(1), z_null.min(1)
    res["p_maxT_plus"][ok] = (add + (zmax[:, None] >= z_obs[None, :]).sum(0)) / (P + add)
    res["p_maxT_minus"][ok] = (add + (zmin[:, None] <= z_obs[None, :]).sum(0)) / (P + add)
    return res


def _mcnemar(pos_fire: np.ndarray, neg_fire: np.ndarray):
    b = (pos_fire & ~neg_fire).sum(0)
    c = (~pos_fire & neg_fire).sum(0)
    n = b + c
    p = np.where(n > 0, np.minimum(1.0, 2.0 * binom.cdf(np.minimum(b, c), np.maximum(n, 1), 0.5)), 1.0)
    return b, c, p


def _wilcoxon(d: np.ndarray, active: np.ndarray) -> np.ndarray:
    p = np.full(d.shape[1], np.nan)
    cols = np.flatnonzero(active & (np.abs(d).sum(0) > 0))
    p[active] = 1.0
    if len(cols):
        p[cols] = wilcoxon(d[:, cols], zero_method="pratt", method="approx", axis=0).pvalue
    return p


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _clusters(meta, how: str) -> tuple[np.ndarray, int]:
    image_id = meta["image_id"].astype(str)
    if how == "image":
        keys = image_id
        n_fallback = 0
    else:
        airport = meta["airport"].astype(str)
        if how == "airport":
            keys = airport
        elif how == "airport_runway":
            keys = np.char.add(np.char.add(airport, "/"), meta["runway"].astype(str))
        elif how == "scenario":
            keys = meta["scenario"].astype(str) if "scenario" in meta else airport
        else:
            raise ValueError(f"unknown cluster key {how!r}")
        missing = keys == ""
        n_fallback = int(missing.sum())
        keys = np.where(missing, image_id, keys)
    _, cluster = np.unique(keys, return_inverse=True)
    return cluster, n_fallback


@app.command()
def main(
    pos_codes: Annotated[Path, typer.Option(help="Patch codes of the BOGO crops (.h5)")],
    neg_codes: Annotated[Path, typer.Option(help="Patch codes of the paired negative crops (.h5)")],
    meta: Annotated[Path, typer.Option(help="{prefix}_meta.npz from extract_paired_patches.py")],
    out: Annotated[Path, typer.Option()],
    atoms: Annotated[Path | None, typer.Option(help="atoms.json (CV split) for agreement + head-weight shares")] = None,
    stat: Annotated[str, typer.Option(help="count | abs")] = "count",
    cluster: Annotated[str, typer.Option(help="airport | airport_runway | scenario (approach sequence) | image")] = "airport",
    alpha: Annotated[float, typer.Option()] = 0.05,
    margin: Annotated[float, typer.Option(help="Rate-ratio equivalence margin M; agnostic means 1/M < ratio < M")] = 1.25,
    min_consistent: Annotated[int, typer.Option(help="LARDv2 sources whose theta must share the sign")] = 3,
    min_rate: Annotated[float, typer.Option(help="Mean per-patch firing rate below which an atom is inactive")] = 0.001,
    min_clusters: Annotated[int, typer.Option(help="Below this many clusters no atom gets an inferential label")] = 10,
    n_perm: Annotated[int, typer.Option()] = 10_000,
    n_boot: Annotated[int, typer.Option()] = 10_000,
    seed: Annotated[int, typer.Option()] = 0,
):
    assert stat in ("count", "abs"), stat
    rng = np.random.default_rng(seed)

    cnt_pos, abs_pos, ppi, m = _per_crop(pos_codes)
    cnt_neg, abs_neg, ppi_n, _ = _per_crop(neg_codes, m)
    assert ppi == ppi_n and cnt_pos.shape == cnt_neg.shape, \
        f"unpaired codes: {cnt_pos.shape} vs {cnt_neg.shape}"
    md = np.load(meta)
    N = cnt_pos.shape[0]
    assert len(md["image_id"]) == N, f"meta has {len(md['image_id'])} pairs, codes {N}"

    a_pos, a_neg = (cnt_pos, cnt_neg) if stat == "count" else (abs_pos, abs_neg)
    rate = (cnt_pos.sum(0) + cnt_neg.sum(0)) / (2 * N * ppi)
    active = rate > min_rate
    if stat == "count":
        eps = np.full(m, 0.5)
    else:  # half a typical firing, in |coef| units
        tot_c = cnt_pos.sum(0) + cnt_neg.sum(0)
        eps = 0.5 * np.where(tot_c > 0, (abs_pos.sum(0) + abs_neg.sum(0)) / np.maximum(tot_c, 1), 1.0)

    cl, n_fallback = _clusters(md, cluster)
    C = int(cl.max()) + 1
    G = sp.csr_matrix((np.ones(N), (cl, np.arange(N))), shape=(C, N))
    Bc, Ic = np.asarray(G @ a_pos), np.asarray(G @ a_neg)
    print(f"[SEL] {N} pairs, {C} clusters by {cluster} ({n_fallback} rows without a cluster key "
          f"-> own cluster), {int(active.sum())}/{m} active atoms, stat={stat}", flush=True)
    enough_clusters = C >= min_clusters
    if not enough_clusters:
        print(f"[SEL] WARNING: only {C} clusters (< {min_clusters}); cluster inference is "
              f"not possible, every active atom stays 'undetermined'.", flush=True)
    elif C < 30:
        print(f"[SEL] WARNING: only {C} clusters; bootstrap CIs and permutation "
              f"resolution will be coarse.", flush=True)

    theta = _log_ratio(a_pos.sum(0), a_neg.sum(0), eps)

    # ── D: cluster sign-flip permutation, max-T ─────────────────────────
    null, exact = _flip_null(Bc, Ic, eps, n_perm, rng)
    perm = _perm_pvalues(theta, null, exact, active)
    del null

    # ── E: cluster bootstrap, margin + TOST ─────────────────────────────
    boot = _bootstrap(Bc, Ic, eps, n_boot, rng)
    lm = np.log(margin)
    ci_lo, ci_hi = np.quantile(boot, alpha, axis=0), np.quantile(boot, 1 - alpha, axis=0)
    p_sel = (1 + (boot <= lm).sum(0)) / (n_boot + 1)
    p_anti = (1 + (boot >= -lm).sum(0)) / (n_boot + 1)
    p_tost = np.maximum((1 + (boot >= lm).sum(0)), (1 + (boot <= -lm).sum(0))) / (n_boot + 1)
    del boot
    for p in (p_sel, p_anti, p_tost):
        p[~active] = np.nan
    q_sel, q_anti, q_tost = _bh(p_sel), _bh(p_anti), _bh(p_tost)

    # ── per-source direction consistency ────────────────────────────────
    subset = md["subset"].astype(str)
    src_names = [str(s) for s in md["subset_names"]]
    theta_src = np.full((len(src_names), m), np.nan)
    for si, s in enumerate(src_names):
        mask = subset == s
        bs, is_ = a_pos[mask].sum(0), a_neg[mask].sum(0)
        theta_src[si] = np.where(bs + is_ > 0, _log_ratio(bs, is_, eps), np.nan)
    present = ~np.isnan(theta_src)
    need = np.minimum(min_consistent, present.sum(0))
    agree_plus = ((theta_src > 0) & present).sum(0)
    agree_minus = ((theta_src < 0) & present).sum(0)

    # ── labels ──────────────────────────────────────────────────────────
    label = np.full(m, "undetermined", dtype=object)
    label[~active] = "inactive"
    sel = active & (perm["p_maxT_plus"] < alpha) & (q_sel < alpha) & (agree_plus >= need)
    anti = active & (perm["p_maxT_minus"] < alpha) & (q_anti < alpha) & (agree_minus >= need)
    agn = active & ~sel & ~anti & (q_tost < alpha)
    if not enough_clusters:
        sel[:] = anti[:] = agn[:] = False
    label[sel], label[anti], label[agn] = "selective", "anti", "agnostic"
    n_by_label = {k: int((label == k).sum()) for k in LABELS}
    print("[SEL] " + "  ".join(f"{k}={v}" for k, v in n_by_label.items()), flush=True)

    # ── A, B: iid secondary tests ───────────────────────────────────────
    mc_b, mc_c, mc_p = _mcnemar(cnt_pos > 0, cnt_neg > 0)
    mc_p = np.where(active, mc_p, np.nan)
    wx_p = _wilcoxon((a_pos - a_neg).astype(np.float32), active)
    mc_q, wx_q = _bh(mc_p), _bh(wx_p)
    iid_sig_plus = active & (wx_q < alpha) & (theta > 0)

    # ── crop-geometry diagnostics (issue: crop location confound) ───────
    pos_xy, neg_xy = md["pos_crop_xy"], md["neg_crop_xy"]
    dy = neg_xy[:, 1] - pos_xy[:, 1]
    geometry = {
        "median_abs_dy_px": float(np.median(np.abs(dy))),
        "frac_neg_above_pos": float((dy < 0).mean()),
        "frac_neg_below_pos": float((dy > 0).mean()),
    }

    summary = {
        "n_pairs": int(N), "n_clusters": C, "cluster_by": cluster,
        "rows_without_cluster_key": n_fallback, "perm_exact": bool(exact),
        "enough_clusters": bool(enough_clusters),
        "n_atoms": int(m), "n_active": int(active.sum()),
        "counts": n_by_label,
        "n_iid_wilcoxon_selective_like": int(iid_sig_plus.sum()),
        "n_selective_also_iid": int((sel & iid_sig_plus).sum()),
        "crop_geometry": geometry,
    }

    # ── agreement with the CV split + head-weight shares ────────────────
    if atoms is not None:
        aj = json.loads(Path(atoms).read_text())
        cv_content = np.zeros(m, dtype=bool)
        cv_content[aj["content_atom_ids"]] = True
        confusion = {k: {"cv_content": int(((label == k) & cv_content).sum()),
                         "cv_style": int(((label == k) & ~cv_content).sum())} for k in LABELS}
        both = active
        eff = np.asarray(aj["effective_score"], dtype=np.float64)
        tot = eff.sum()
        summary["vs_cv"] = {
            "confusion": confusion,
            "cohen_kappa_selective_vs_cv_content": _kappa(sel[both], cv_content[both]),
            "eff_score_share": {k: float(eff[label == k].sum() / tot) if tot > 0 else 0.0
                                for k in LABELS},
        }
        print(f"[SEL] kappa(selective, CV-content) = "
              f"{summary['vs_cv']['cohen_kappa_selective_vs_cv_content']:.3f}", flush=True)

    def _l(x):
        return [None if (isinstance(v, float) and np.isnan(v)) else v for v in np.asarray(x, dtype=float).tolist()]

    result = {
        "config": {
            "pos_codes": str(pos_codes), "neg_codes": str(neg_codes), "meta": str(meta),
            "negative": str(md["negative"]), "exclude": str(md["exclude"]),
            "max_dy": int(md["max_dy"]), "stat": stat, "cluster": cluster,
            "alpha": alpha, "margin": margin, "min_consistent": min_consistent,
            "min_rate": min_rate, "min_clusters": min_clusters, "n_perm": n_perm, "n_boot": n_boot, "seed": seed,
        },
        "summary": summary,
        "label": label.tolist(),
        "content_atom_ids": np.flatnonzero(label == "selective").tolist(),
        "style_atom_ids": np.flatnonzero((label == "agnostic") | (label == "anti")).tolist(),
        "undetermined_atom_ids": np.flatnonzero(label == "undetermined").tolist(),
        "theta": _l(theta), "rate_ratio": _l(np.exp(theta)),
        "ci_lo": _l(ci_lo), "ci_hi": _l(ci_hi), "ci_level": 1 - 2 * alpha,
        "mean_rate": _l(rate),
        **{k: _l(v) for k, v in perm.items()},
        "p_selective_margin": _l(p_sel), "q_selective_margin": _l(q_sel),
        "p_anti_margin": _l(p_anti), "q_anti_margin": _l(q_anti),
        "p_tost": _l(p_tost), "q_tost": _l(q_tost),
        "per_source_theta": {s: _l(theta_src[i]) for i, s in enumerate(src_names)},
        "mcnemar_b": mc_b.tolist(), "mcnemar_c": mc_c.tolist(),
        "mcnemar_p": _l(mc_p), "mcnemar_q": _l(mc_q),
        "wilcoxon_p": _l(wx_p), "wilcoxon_q": _l(wx_q),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[SEL] wrote {out}", flush=True)


if __name__ == "__main__":
    app()
