"""Extract patch tokens for paired BOGO / negative crops of the same test images.

Input to the paired BOGO-selectivity test (``scripts/bogo_selectivity.py``).
For every LARDv2 test row with a valid negative crop we run the model on

- the BOGO crop (positive, runway inside), and
- a negative crop of the same image: inverse-BOGO (``--negative inverse``)
  or a second random BOGO crop (``--negative bogo``, null control).

Writes, with row i of every file referring to the same source image:

- ``{out}_pos_patches.h5`` / ``{out}_neg_patches.h5``: patch tokens in the
  same schema as ``{tag}_all_patches.h5`` (cue patch excluded), so
  ``scripts/julia/sparse_code_patches.jl`` encodes them unchanged.
- ``{out}_meta.npz``: per-pair image_id, subset, airport, runway, scenario, crop
  origins and scale, used for clustering and crop-position diagnostics.

Unlike ``extract_patch_embeddings.py`` the embeddings are streamed to disk
batch by batch, since two full test-set patch arrays do not fit in memory.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from src.constants import IMG_SIZE, NUM_KEYPOINTS
from src.cue import render_runway_cue
from src.data import LARDv2PairedDataset, LARDTrainDataset
from src.model import ViTKeypointRegressor


app = typer.Typer(add_completion=False)

SKIPPED_PATCH_INDICES = (0,)  # top-left patch carries the cue


def _write_patches(model, samples, path: Path, ds: LARDv2PairedDataset,
                   dev, batch_size: int, num_workers: int, label: str) -> None:
    loader = DataLoader(LARDTrainDataset(samples, image_size=IMG_SIZE),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    D = model.backbone.num_features
    keep_mask = np.ones(model.num_patches, dtype=bool)
    for idx in SKIPPED_PATCH_INDICES:
        keep_mask[idx] = False
    keep_mask_t = torch.from_numpy(keep_mask).to(dev)
    n_per_image = int(keep_mask.sum())
    n_total = len(samples) * n_per_image
    n_prefix = getattr(model.backbone, "num_prefix_tokens", 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        emb = f.create_dataset("embeddings", shape=(n_total, D), dtype=np.float32,
                               chunks=(min(n_total, 64 * n_per_image), D))
        f.attrs["skipped_patches"] = np.array(SKIPPED_PATCH_INDICES, dtype=np.int64)
        f.attrs["patches_per_image"] = n_per_image
        f.attrs["num_images"] = len(samples)
        f.attrs["subset_names"] = np.array(ds.subset_names, dtype="S")
        f.attrs["subset_sizes"] = np.array(ds.subset_sizes, dtype=np.int64)

        row = 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                images = batch["image"].to(dev)
                render_runway_cue(images, batch["runway_side"].to(dev), patch_size=model.patch_size)
                tokens = model.backbone(images)
                patches = tokens[:, n_prefix:][:, keep_mask_t, :].cpu().numpy()
                end = row + patches.shape[0] * n_per_image
                emb[row:end] = patches.reshape(-1, D)
                row = end
                if (bi + 1) % 50 == 0 or (bi + 1) == len(loader):
                    print(f"[PAIRED {label}] {min((bi + 1) * batch_size, len(samples))}/{len(samples)}",
                          flush=True)
        assert row == n_total, f"wrote {row} rows, expected {n_total}"
    print(f"[PAIRED {label}] wrote {path} ({n_total} x {D})", flush=True)


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Option()],
    out_prefix: Annotated[Path, typer.Option(help="Writes {out}_pos_patches.h5, {out}_neg_patches.h5, {out}_meta.npz")],
    negative: Annotated[str, typer.Option(help="inverse | bogo (null control)")] = "inverse",
    exclude: Annotated[str, typer.Option(help="Inverse crop must avoid: corners | polygon")] = "polygon",
    max_dy: Annotated[int, typer.Option(help="Max |crop_y - BOGO crop_y| in scaled px for the inverse crop; <0 = unrestricted")] = -1,
    split: Annotated[str, typer.Option()] = "test",
    seed: Annotated[int, typer.Option(help="Seeds the crop sampler")] = 0,
    batch_size: Annotated[int, typer.Option()] = 32,
    num_workers: Annotated[int, typer.Option()] = 4,
    device: Annotated[str, typer.Option()] = "cuda:0",
):
    dry = bool(os.environ.get("DRY"))
    max_samples = 50 if dry else None
    if dry:
        num_workers = 0
    random.seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    model = ViTKeypointRegressor(
        num_keypoints=NUM_KEYPOINTS, backbone=ckpt["config"]["backbone"],
        pretrained=False, image_size=IMG_SIZE,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = LARDv2PairedDataset(subset="all", split=split, max_samples=max_samples,
                             negative=negative, exclude=exclude,
                             max_dy=max_dy if max_dy >= 0 else None)

    pos_path = out_prefix.with_name(out_prefix.name + "_pos_patches.h5")
    neg_path = out_prefix.with_name(out_prefix.name + "_neg_patches.h5")
    _write_patches(model, ds.positive, pos_path, ds, dev, batch_size, num_workers, "pos")
    _write_patches(model, ds.negative, neg_path, ds, dev, batch_size, num_workers, "neg")

    subset_of = np.repeat(np.array(ds.subset_names), ds.subset_sizes)
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.npz")
    np.savez(
        meta_path,
        image_id=np.array([s.image_id for s in ds.positive]),
        subset=subset_of,
        airport=np.array([str(s.metadata.get("airport", "")) for s in ds.positive]),
        runway=np.array([str(s.metadata.get("runway", "")) for s in ds.positive]),
        scenario=np.array([str(s.metadata.get("scenario", "")) for s in ds.positive]),
        pos_crop_xy=np.array([[s.metadata["crop_x"], s.metadata["crop_y"]] for s in ds.positive]),
        neg_crop_xy=np.array([[s.metadata["crop_x"], s.metadata["crop_y"]] for s in ds.negative]),
        scale=np.array([s.metadata["scale"] for s in ds.positive]),
        subset_names=np.array(ds.subset_names),
        subset_sizes=np.array(ds.subset_sizes),
        negative=negative, exclude=exclude, max_dy=max_dy, seed=seed,
    )
    print(f"[PAIRED] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    app()
