# DASC 2026 mechinterp paper — reproducibility pipeline
#
# One target per paper heading. Run `make all` for the end-to-end pipeline
# for a single (VARIANT, SEED) combination. Multi-seed sweeps are driven
# by the bash orchestrator ``scripts/run_sweep.sh`` which invokes make
# with SEED=0..4 on both GPUs.
#
# Key variables (overridable on the command line or via .envrc):
#   VARIANT       pretrained | scratch
#   SEED          integer (0, 1, ...)
#   DEVICE        cuda:0 (default) | cpu
#   DRY           set to 1 for local CPU dry-run (~1 min end-to-end)
#
# Paper target map:
#   sec:reg-setup / sec:reg-results  <-  train  +  eval
#   sec:ksvd-setup / sec:ksvd-results <-  extract_patches  +  ksvd  +  codes
#   sec:content-style / sec:head-weight <-  atoms
#   sec:atom-viz                    <-  atom_viz
#   sec:ims-lr / sec:bogo-res       <-  attn_pooled  +  ims_sweep
#   (variant) BOGO-selectivity split <-  bogo_all = paired_patches + paired_codes
#                                        + bogo_select + atoms_bogo  (+ bogo_null)
#   tab:training, tab:content-style,
#   fig:ims-sweep, fig:cv-hist      <-  aggregate  (emits figs/aggregated.json;
#                                        hand-written typst reads it)

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Configuration ─────────────────────────────────────────────────────
VARIANT      ?= pretrained
SEED         ?= 0
DEVICE       ?= cuda:0
DATA_DIR     ?= data
MODEL_DIR    ?= models
RESULTS_DIR  ?= results
FIG_DIR      ?= figs
DRY          ?=

# DRY mode redirects all artifact output to _dry subtrees so the local
# 1-epoch smoke run never pollutes the aggregated paper tables. Paper
# figs ($(FIG_DIR)/tab_*, fig_*) are left pointing at the real figs/
# dir but the aggregate/paper recipes are skipped under DRY (see below).
ifneq ($(strip $(DRY)),)
  DEVICE      := cpu
  DATA_DIR    := $(DATA_DIR)/_dry
  MODEL_DIR   := $(MODEL_DIR)/_dry
  RESULTS_DIR := $(RESULTS_DIR)/_dry
  FIG_DIR     := $(FIG_DIR)/_dry
  export DRY
endif

PY           := uv run python -u
# Default to the juliaup-installed path because the sweep driver runs
# under a non-login shell that doesn't source .bashrc / .envrc.
JULIA        ?= $(firstword $(wildcard $(HOME)/.juliaup/bin/julia) $(shell which julia 2>/dev/null) julia)
JULIA_FLAGS  := --project=. -t auto

TAG          := $(VARIANT)_seed$(SEED)
RUN_NAME     := lard_$(TAG)

# ── Canonical output paths ────────────────────────────────────────────
CKPT         := $(MODEL_DIR)/$(RUN_NAME)_best.pt
TRAIN_JSON   := $(RESULTS_DIR)/$(RUN_NAME)_train.json
EVAL_JSON    := $(RESULTS_DIR)/$(RUN_NAME)_eval.json
PATCHES_H5   := $(DATA_DIR)/$(TAG)_all_patches.h5
POOL_H5      := $(DATA_DIR)/$(TAG)_ksvd_pool.h5
HEAD_NPZ     := $(DATA_DIR)/$(TAG)_head.npz
DICT_NPY     := $(MODEL_DIR)/$(TAG)_ksvd.npy
CODES_H5     := $(DATA_DIR)/$(TAG)_patch_codes.h5
ATTN_NPZ     := $(DATA_DIR)/$(TAG)_attn_codes.npz
ATOMS_JSON   := $(RESULTS_DIR)/$(TAG)_atoms.json
IMS_JSON     := $(RESULTS_DIR)/$(TAG)_ims_sweep.json
VIZ_STAMP    := $(FIG_DIR)/atoms_$(TAG)/.stamp

# Paired BOGO-selectivity split (alternative to the CV split). PAIRED_*
# pairs BOGO with inverse-BOGO crops; CTRL_* pairs BOGO with a second BOGO
# crop and is the null control (should yield ~no selective atoms).
PAIRED_EXCLUDE ?= polygon
PAIRED_MAX_DY  ?= -1
PAIRED_PREFIX  := $(DATA_DIR)/$(TAG)_paired
CTRL_PREFIX    := $(DATA_DIR)/$(TAG)_ctrl
SEL_JSON       := $(RESULTS_DIR)/$(TAG)_bogo_selectivity.json
CTRL_JSON      := $(RESULTS_DIR)/$(TAG)_bogo_selectivity_ctrl.json
ATOMS_BOGO_JSON := $(RESULTS_DIR)/$(TAG)_atoms_bogo.json

# Paper figure / table artifacts (multi-seed aggregates). One JSON feeds all
# hand-written typst tables and figures; no per-figure emitters anymore.
AGG_JSON     := $(FIG_DIR)/aggregated.json


.PHONY: help all clean deep-clean \
        train eval extract_patches ksvd codes atoms attn_pooled ims_sweep \
        atom_viz aggregate paper \
        paired_patches paired_codes bogo_select atoms_bogo bogo_null bogo_all

help:
	@echo "Usage: make <target> [VARIANT=pretrained|scratch] [SEED=0] [DEVICE=cuda:0] [DRY=1]"
	@echo ""
	@echo "Paper-section targets (single seed):"
	@echo "  train            DINOv2-S + soft-argmax + Huber"
	@echo "  eval             Train and test MAE+median on a finished ckpt"
	@echo "  extract_patches  Patch embeddings + K-SVD importance pool + head"
	@echo "  ksvd             Fit K-SVD dictionary (Julia)"
	@echo "  codes            OMP-encode every test patch (Julia)"
	@echo "  atoms            Content/style split + effective regression score"
	@echo "  attn_pooled      BOGO + inverse-BOGO attention-pooled codes"
	@echo "  ims_sweep        Constrained L1 IMS detector lambda-sweep"
	@echo "  atom_viz         Top-activating-patch grids for each atom"
	@echo ""
	@echo "BOGO-selectivity split (variant of the CV content/style split):"
	@echo "  paired_patches   Patch tokens of paired BOGO / inverse-BOGO crops"
	@echo "  paired_codes     Sparse-code both crop sets"
	@echo "  bogo_select      Cluster permutation + TOST selectivity labels per atom"
	@echo "  atoms_bogo       Effective-score shares under the BOGO split"
	@echo "  bogo_null        Null control: BOGO vs second BOGO crop"
	@echo "  bogo_all         All of the above"
	@echo ""
	@echo "Cross-seed paper artifacts (run after multi-seed sweep):"
	@echo "  aggregate        Emit tab_regression + tab_content_style + fig_ims_sweep"
	@echo "  paper            typst compile main.typ"
	@echo "  all              Full pipeline for (VARIANT, SEED)"
	@echo ""
	@echo "Current (VARIANT, SEED) = ($(VARIANT), $(SEED)),  DEVICE=$(DEVICE),  DRY=$(DRY)"

# atom_viz is intentionally omitted from `all`: the paper figure takes a
# single curated seed's atom grids, so we render them once by hand with
# `make atom_viz VARIANT=pretrained SEED=0` rather than on every seed.
all: train eval extract_patches ksvd codes atoms attn_pooled ims_sweep

# ── 1. Regression training (paper @sec:reg-setup, @sec:reg-results) ───
train: $(CKPT) $(TRAIN_JSON)

$(CKPT) $(TRAIN_JSON) &:
	@mkdir -p $(MODEL_DIR) $(RESULTS_DIR)
	$(PY) scripts/train.py --variant $(VARIANT) --seed $(SEED) \
	    --device $(DEVICE) --run-name $(RUN_NAME) \
	    --output-dir $(MODEL_DIR) --results-dir $(RESULTS_DIR)

eval: $(EVAL_JSON)

$(EVAL_JSON): $(CKPT)
	$(PY) scripts/eval.py --checkpoint $(CKPT) --run-name $(RUN_NAME) \
	    --device $(DEVICE) --results-dir $(RESULTS_DIR)

# ── 2. K-SVD dictionary (paper @sec:ksvd-setup, @sec:ksvd-results) ────
extract_patches: $(PATCHES_H5) $(POOL_H5) $(HEAD_NPZ)

$(PATCHES_H5) $(POOL_H5) $(HEAD_NPZ) &: $(CKPT)
	@mkdir -p $(DATA_DIR)
	$(PY) scripts/extract_patch_embeddings.py --checkpoint $(CKPT) \
	    --out-prefix $(DATA_DIR)/$(TAG) --device $(DEVICE)

ksvd: $(DICT_NPY)

ifneq ($(strip $(DRY)),)
$(DICT_NPY): $(POOL_H5)
	$(PY) scripts/dry_stub_ksvd.py ksvd $(POOL_H5) $(DICT_NPY) --dict-size 64
else
$(DICT_NPY): $(POOL_H5)
	$(JULIA) $(JULIA_FLAGS) scripts/julia/ksvd_lard.jl $(POOL_H5) $(DICT_NPY)
endif

codes: $(CODES_H5)

ifneq ($(strip $(DRY)),)
$(CODES_H5): $(PATCHES_H5) $(DICT_NPY)
	$(PY) scripts/dry_stub_ksvd.py codes $(PATCHES_H5) $(DICT_NPY) $(CODES_H5)
else
$(CODES_H5): $(PATCHES_H5) $(DICT_NPY)
	$(JULIA) $(JULIA_FLAGS) scripts/julia/sparse_code_patches.jl \
	    $(PATCHES_H5) $(DICT_NPY) $(CODES_H5)
endif

# ── 3. Content/style + effective score (@sec:content-style, @sec:head-weight) ─
atoms: $(ATOMS_JSON)

$(ATOMS_JSON): $(CODES_H5) $(DICT_NPY) $(HEAD_NPZ)
	$(PY) scripts/atoms.py --sparse-codes $(CODES_H5) \
	    --dictionary $(DICT_NPY) --head $(HEAD_NPZ) --output $(ATOMS_JSON)

# ── 4. Atom visualization (@sec:atom-viz) ─────────────────────────────
atom_viz: $(VIZ_STAMP)

$(VIZ_STAMP): $(CODES_H5) $(ATOMS_JSON)
	@mkdir -p $(FIG_DIR)/atoms_$(TAG)
	$(PY) scripts/visualize_atoms.py --sparse-codes $(CODES_H5) \
	    --atoms $(ATOMS_JSON) --out-dir $(FIG_DIR)/atoms_$(TAG)
	@touch $@

# ── 5. Attention-pooled codes + IMS L1 sweep (@sec:ims-lr, @sec:bogo-res) ─
attn_pooled: $(ATTN_NPZ)

$(ATTN_NPZ): $(CKPT) $(DICT_NPY)
	$(PY) scripts/extract_attn_pooled.py --checkpoint $(CKPT) \
	    --dictionary $(DICT_NPY) --out $(ATTN_NPZ) --device $(DEVICE)

ims_sweep: $(IMS_JSON)

$(IMS_JSON): $(ATTN_NPZ) $(ATOMS_JSON)
	$(PY) scripts/ims_sweep.py --codes $(ATTN_NPZ) --atoms $(ATOMS_JSON) \
	    --seed $(SEED) --out $(IMS_JSON)

# ── 5b. Paired BOGO-selectivity split (variant of @sec:content-style) ─
paired_patches: $(PAIRED_PREFIX)_pos_patches.h5

$(PAIRED_PREFIX)_pos_patches.h5 $(PAIRED_PREFIX)_neg_patches.h5 $(PAIRED_PREFIX)_meta.npz &: $(CKPT)
	@mkdir -p $(DATA_DIR)
	$(PY) scripts/extract_paired_patches.py --checkpoint $(CKPT) \
	    --out-prefix $(PAIRED_PREFIX) --negative inverse \
	    --exclude $(PAIRED_EXCLUDE) --max-dy $(PAIRED_MAX_DY) \
	    --seed $(SEED) --device $(DEVICE)

$(CTRL_PREFIX)_pos_patches.h5 $(CTRL_PREFIX)_neg_patches.h5 $(CTRL_PREFIX)_meta.npz &: $(CKPT)
	@mkdir -p $(DATA_DIR)
	$(PY) scripts/extract_paired_patches.py --checkpoint $(CKPT) \
	    --out-prefix $(CTRL_PREFIX) --negative bogo \
	    --seed $(SEED) --device $(DEVICE)

paired_codes: $(PAIRED_PREFIX)_pos_codes.h5 $(PAIRED_PREFIX)_neg_codes.h5

# {prefix}_{pos,neg}_patches.h5 -> {prefix}_{pos,neg}_codes.h5
ifneq ($(strip $(DRY)),)
$(DATA_DIR)/%_codes.h5: $(DATA_DIR)/%_patches.h5 $(DICT_NPY)
	$(PY) scripts/dry_stub_ksvd.py codes $< $(DICT_NPY) $@
else
$(DATA_DIR)/%_codes.h5: $(DATA_DIR)/%_patches.h5 $(DICT_NPY)
	$(JULIA) $(JULIA_FLAGS) scripts/julia/sparse_code_patches.jl $< $(DICT_NPY) $@
endif

bogo_select: $(SEL_JSON)

$(SEL_JSON): $(PAIRED_PREFIX)_pos_codes.h5 $(PAIRED_PREFIX)_neg_codes.h5 $(PAIRED_PREFIX)_meta.npz $(ATOMS_JSON)
	$(PY) scripts/bogo_selectivity.py --pos-codes $(PAIRED_PREFIX)_pos_codes.h5 \
	    --neg-codes $(PAIRED_PREFIX)_neg_codes.h5 --meta $(PAIRED_PREFIX)_meta.npz \
	    --atoms $(ATOMS_JSON) --seed $(SEED) --out $@

bogo_null: $(CTRL_JSON)

$(CTRL_JSON): $(CTRL_PREFIX)_pos_codes.h5 $(CTRL_PREFIX)_neg_codes.h5 $(CTRL_PREFIX)_meta.npz
	$(PY) scripts/bogo_selectivity.py --pos-codes $(CTRL_PREFIX)_pos_codes.h5 \
	    --neg-codes $(CTRL_PREFIX)_neg_codes.h5 --meta $(CTRL_PREFIX)_meta.npz \
	    --seed $(SEED) --out $@

atoms_bogo: $(ATOMS_BOGO_JSON)

$(ATOMS_BOGO_JSON): $(CODES_H5) $(DICT_NPY) $(HEAD_NPZ) $(SEL_JSON)
	$(PY) scripts/atoms.py --sparse-codes $(CODES_H5) \
	    --dictionary $(DICT_NPY) --head $(HEAD_NPZ) \
	    --selectivity $(SEL_JSON) --output $@

bogo_all: bogo_select atoms_bogo bogo_null

# ── 6. Cross-seed aggregates (after the sweep) ───────────────────────
ifneq ($(strip $(DRY)),)
# Under DRY=1 we deliberately skip the aggregate step so the 1-epoch
# smoke numbers never overwrite real figures. main.pdf is not touched
# under DRY either (see the `paper` guard below).
aggregate:
	@echo "[DRY] skipping aggregate (keeps paper figs intact)"
else
aggregate: $(AGG_JSON)

$(AGG_JSON):
	$(PY) scripts/aggregate.py --results-dir $(RESULTS_DIR) --out $(AGG_JSON)
endif

# ── 7. Paper build ────────────────────────────────────────────────────
paper: main.pdf

ifneq ($(strip $(DRY)),)
main.pdf:
	@echo "[DRY] skipping paper compile (keeps main.pdf intact)"
else
main.pdf: main.typ aggregate
	typst compile main.typ
endif

# ── Housekeeping ─────────────────────────────────────────────────────
clean:
	rm -f main.pdf

deep-clean: clean
	rm -rf $(DATA_DIR) $(MODEL_DIR) $(RESULTS_DIR) $(FIG_DIR)/atoms_*
