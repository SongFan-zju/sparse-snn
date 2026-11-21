# Running the preserved code

Run commands from the logical repository root (`.`). This package preserves the executed method code; it does not include datasets, public/source/final weights, author checkouts, or all historical launch configurations. `provenance/code_inventory.json` records each copied file's original relative location and SHA256. Use the retained experiment configuration and evidence for the exact budget, batch size, input size, checkpoint and split of a reported result; parser defaults alone are not a complete reproduction recipe.

## Environment and checks

Use Python 3.10 with PyTorch, torchvision and NumPy; `requirements.txt` lists common image dependencies and `requirements-detection.txt` adds detection packages. Install Pillow for image data, requests and SpikingJelly for DVS preparation, and SciPy/OpenCV for tracking. DiMP additionally needs the pinned PyTracking/LTR environment and its native PreciseRoIPooling extension. Optional Triton 3.1.0 accelerates the exact VJP; the torch fallback remains available. Exact versions of the CPU validation environment are recorded in `provenance/code_inventory.json` (torch 2.5.1.post303, torchvision 0.20.1a0, NumPy 2.2.6); this records the existing environment rather than a portable binary lockfile.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m unittest discover -s tests -p 'test_*.py'
CUDA_VISIBLE_DEVICES='' python scripts/train_dvs.py --help
```

All 15 staged script entrypoints (12 training/data/evaluation scripts are included) were checked with `--help` on CPU. The suite passed 42 tests, with 4 skips for optional external assets/CUDA. Small synthetic R18/R20/R34 and full-width VGG construction checks also passed; these checks do not rerun task accuracy or GPU training.

For supported training entrypoints expose exactly one authorized physical GPU with `CUDA_VISIBLE_DEVICES=0`, `2`, or `3` and use logical `cuda:0` where a `--device` option exists. Several frozen runners enforce these machine-specific device choices. GPU1 is excluded. CPU `--help` and tests require no GPU. No training is started by the commands above.

## Main entrypoints

Every script below supports `python scripts/NAME.py --help`. Paths shown as arguments must point to separately supplied assets. Choose a fresh output directory unless the runner's explicit resume option is used.

| Task / script | Required arguments and additional prerequisites |
| --- | --- |
| DVS: `train_dvs.py` | `--architecture resnet18\|resnet20 --stage source\|snn --manifest FILE --output DIR`; SNN additionally needs `--source FILE --method triangular\|density`. Select `--L 2\|4 --T 4`; source stages use `--relu-source` when reusing the completed ReLU source. |
| Fast-SNN CIFAR100: `train_fast_classification.py` | `--stage source\|snn\|preflight --upstream DIR --data DIR --output DIR`; source adaptation uses `--init` with the public CIFAR10 two-bit checkpoint; SNN uses `--source` and `--method triangular\|density`. Fixed L3/T3; the adapted source is local CIFAR100 training, not an official CIFAR100 checkpoint. |
| VOC: `train_detection.py` | Positional `train\|evaluate`, plus `--upstream DIR --source FILE --data VOCdevkit --output DIR`. Select matched `--L 3 --T 3` or `--L 7 --T 7`; source precision must match. Use `--method hard` for direct conversion, `triangular` or `phase_triangular_density` for adaptation; evaluation of an adapted model additionally uses `--checkpoint`. |
| ViStream: `train_tracking.py` | Positional `train\|eval`, plus `--upstream DIR --source FILE --data DIR --output DIR`; choose `--method hard\|triangular\|density`, and `--checkpoint` for an adapted model. Reuse the retained clip/context configuration and held-out MOT16 sequence assignment; these results are not an official SOT leaderboard. |
| DiMP preparation: `prepare_dimp.py` | `--upstream DIR --checkpoint FILE --pairs FILE --output DIR`; explicitly set L4/T4 and the recorded crop size. Preparation produces calibration only (`source_is_trained=False`), not a trained shared source. |
| DiMP training: `train_dimp.py` | `--method source\|triangular\|density --output DIR`; explicitly supply `--upstream`, `--official-checkpoint`, `--pairs`, `--initialization`, and for SNN arms `--source-checkpoint`. The frozen runner fixes crop288/T4/seed2026100321 and reads L from initialization. Full training needs `--purpose full_tracking`, the recorded epoch budget and all 69401 pairs via `--max-pairs 69401`; defaults describe the small pilot. |
| DiMP evaluation: `evaluate_dimp.py` | `--upstream DIR --checkpoint FILE --tracks FILE --output DIR`; use `--model-kind ann` for original ANN or `trained` for adapted models; `--direct-snn` selects direct conversion from the common QCFS source. Optional `--vendor` supplies external compatibility dependencies. Use the complete retained 229-track manifest for full results. |
| Tiny source: `train_tiny_imagenet.py` | `--manifest FILE --architecture resnet34\|vgg16 --stage relu\|qcfs --output DIR --epochs N --batch-size N --lr FLOAT --memory-cap-gib FLOAT`. ReLU additionally requires `--initializer FILE`; QCFS requires `--source FILE` and the selected `--L`. |
| Tiny SNN: `train_tiny_snn.py` | `--source FILE --manifest FILE --output DIR --method triangular\|density --L 2\|4 --T 2\|4 --batch-size N`; use the final queue's exact cases (L2/T2 and L4/T4), recorded epochs and learning rate. Source checkpoint metadata supplies the architecture. |

Keep seed=2026100321, a shared source, the frozen split, and matched training images/epochs between gradient arms. T=15 experiments are permanently retired. Existing generic kernel support is retained only as original implementation compatibility.

## External code, weights and data

- Fast-SNN: [author repository](https://github.com/zju-bmi-lab/Fast-SNN). For classification, `--upstream` points to `image classification/CIFAR10_resnet18`; for detection it points to `object detection`. Obtain the corresponding public quantized checkpoints from its README. VOC uses the public two-bit source for L3/T3 and three-bit source for L7/T7. The author adapters load upstream Python files dynamically; the checkout is required even though the local static import closure is complete. CIFAR data must be extracted in the format expected by `src/data.py`; VOC requires VOC2007 trainval/test and VOC2012 trainval under `VOCdevkit`.
- ViStream: [author repository at faecff832cea4251b6504a1efe4277dc57828f35](https://github.com/Intelligent-Computing-Research-Group/ViStream/tree/faecff832cea4251b6504a1efe4277dc57828f35). Supply its source weights and MOT16 frames/annotations separately. The MIT notice is retained under `src/third_party/`.
- DiMP18: [PyTracking at 7eb9e74bd3d40e29dbcec444902237da13de247b](https://github.com/visionml/pytracking/tree/7eb9e74bd3d40e29dbcec444902237da13de247b) and [official DiMP18 checkpoint](https://drive.google.com/file/d/1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk/view). Build native PreciseRoIPooling and supply compatible external PyTracking dependencies; these sources/binaries are not bundled. Preserve the fixed training-pair JSONL and complete held-out SOT-track JSON from the retained experiment artifacts, together with the image files they reference. Local GT-initialized SOT evaluation on MOT16 must not be labeled an official SOT benchmark or compared directly with MOT HOTA.
- DVS preparation: `prepare_dvs.py --root DIR --T 4` and `prepare_dvs128.py --root DIR --T 4` use SpikingJelly resource definitions and verified archives. CIFAR10-DVS uses the existing fixed 900/100-per-class manifest; retain it rather than choosing a new split. DVS128 Gesture uses official `trials_to_train/test` and original 11-class interval labels. A prepared manifest requires its adjacent relative `.npz` frame files; a JSON file alone is insufficient. Gesture horizontal flipping remains disabled.
- Tiny-ImageNet: obtain the original [Tiny-ImageNet-200 archive](http://cs231n.stanford.edu/tiny-imagenet-200.zip), then use `prepare_tiny_imagenet.py --archive FILE --root DIR --output DIR`. Supply torchvision-compatible public ImageNet ResNet34 or VGG16-BN state dictionaries as initialization, then train local 200-class sources. Tiny uses 100000 official train images and 10000 labeled validation images; these are not ImageNet-1K results.
- QCFS: `src/third_party/qcfs_layer.py` preserves activation code from [ANN_SNN_QCFS](https://github.com/putshua/ANN_SNN_QCFS); its relevant definitions are loaded by the local author-compatible adapter.

## Path and sidecar requirements

There is no bundled global `configs/` directory. The retained direct task runners generally take explicit arguments. DVS loading is a significant exception: source/reused-ReLU checkpoints require the original `config.json` at `checkpoint.parent.parent/config.json`, with matching architecture/L/T/manifest metadata. Preserve that directory relationship and exact manifest bytes. Tiny manifests store an absolute data `root`; source and SNN checkpoints bind manifest hashes. Relocation must be deliberate and recorded, rather than silently editing hash-bound manifests. Tracking pair/track manifests also contain image paths that must resolve locally. Public weights alone cannot reconstruct these local training/evaluation artifacts.

`src/training_dimp.py` and `train_dimp.py` default to repository-relative historical `artifacts/dimp18_preparation_20261119` and `artifacts/tracking_framework_candidates_20261119` paths. Those assets are not in this package: override all corresponding CLI paths explicitly.

Local queue controllers and the monitoring daemon are omitted from this branch. Use the direct task runners above; source result protocols retain the actual historical allocation and commands.

## Historical CIFAR evidence

The selected R18/R20/R34/VGG model and current neuron implementations are included, but the old generic `scripts/train.py`, its global configs and mixed factory are not. That frozen factory eagerly imports retired R50, low-rank, sparsity and unused surrogate branches. `provenance/code_inventory.json` therefore records a local-path/SHA256 reference to one 2026-11-17 R34 frozen runner closure without copying it. Other R18/R20/VGG results must use their own retained configs/snapshots and checkpoints as provenance; they are not asserted to share the R34 revision. Historical CIFAR accuracy remains retained experiment evidence, not a claim that this minimal package contains every executable historical recipe.
