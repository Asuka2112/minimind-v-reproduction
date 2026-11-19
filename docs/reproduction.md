# Reproduction guide

## 1. Repeat the analysis on a fresh clone

Use Python 3.12. No GPU or weights are required for archive verification or score replay.

```sh
python3 scripts/verify_archive.py
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/python scripts/verify_archive.py --replay
```

The second command workflow unpacks byte-checked streams into a temporary directory and recalculates the two controlled experiments with their original scoring logic. `--expand` materializes `.jsonl` streams next to their gzip archives when working directly with the raw answers. The packed and expanded digests are listed in [streams.json](../evidence/streams.json).

## 2. Run the local image question answering app

Install the pinned model dependencies and external base/vision files.

```sh
bash setup_local.sh
.venv/bin/python local_app.py --open
```

To use the reproduced SFT model, download the private release files to a local directory using the GitHub release page or an authenticated GitHub CLI. Then install the exported model and pass the export directory to the app.

```sh
gh release download v1.0.0 --repo Asuka2112/minimind-v-reproduction --dir release-assets
python3 scripts/install_checkpoints.py release-assets
.venv/bin/python local_app.py --open --model-path training/results/exports/sft
```

The installer compares every provided model file to [checkpoints.json](../evidence/checkpoints.json) before copying it to the expected location. The upstream vision encoder is loaded separately. Loopback smoke-test examples and runtime scope are in [validation](../validation/).

## 3. Re-run training or evaluation

- Dense training starts from the pinned MiniMind language weights and SigLIP2, using the [training settings](../training/config.json) and [training runner](../training/README.md). Set the SSH and data paths for your GPU environment. The CUDA/BF16 run used Linux, PyTorch DDP and four RTX 4090s. [Run report](../training/results/REPORT.md)
- Continuation source, original data labels, split manifests and schedules are in [targeted_sft](../targeted_sft/) and [balanced_sft](../balanced_sft/). The recorded GPU-side harness used COCO shards pinned by SHA-256, an index of the original corpora for overlap checks, and JPEG renders matching frozen hashes. Supply those inputs in the directory layout documented by the respective `common.py` files. Set `MINIMIND_V_WORKSPACE` for the workspace root and `MINIMIND_V_PRIOR` for the prior-run directory. Rebuilding with modified code or source bytes calls for a new protocol and receipts.
- The archived numeric results use the previously frozen natural-photo benchmark, held-out constructed challenge, and 192-token primary decode setting. The generated predictions contain the 192- and 384-token prefixes, original/swapped/no-image conditions and model receipts.

[Data and model provenance](provenance.md)
