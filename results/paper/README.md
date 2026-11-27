# Experiment results and reproducibility evidence

`RESULTS_TABLE.csv` and `RESULTS_TABLE.json` separate QCFS source ANN, direct unsigned QCFS, direct signed ECL, triangular fine-tuning and the new gradient. Null means unavailable or not run, never zero. Results use predefined final epochs, retaining negative and tied comparisons. Published-paper references remain separate in source comparison evidence. Standard CIFAR100/ResNet18 appears in the column-sum ablation below. `SCOPE_AND_GAPS.json` documents its historical triangular comparison, and Fast-SNN's author three-stage ResNet18 uses a separate architecture.

All currently included experiment rows have their recorded final endpoints. No pending matched pair is recorded in PENDING.json.

CIFAR100 uses official 10000-image test metrics; CIFAR10-DVS uses fixed local9000/1000 membership; DVS128 Gesture uses official trials_to_train/test. Tiny-ImageNet-200 uses official labeled10000-image validation, not ImageNet-1K test. VOC2007 uses official4952-image test and11pointAP50. Both tracking tasks use local MOT16 held-out protocols: ViStream HOTA and DiMP AO are separate metrics and cannot be compared directly. ViStream's truncated ResNet50 is a preserved historical supplementary negative result. T15 and retired R50 classification/Transformer experiments are excluded from the new main results.

`evidence_packed/` contains one UTF-8 JSONL file per logical experiment group, plus shared configs and independent checks. Each line is an object with `path` (the original logical exported filename), `sha256` (SHA256 of the exact UTF-8 `text` bytes), and `text` (the complete exported file). All 121 epoch histories are preserved as full JSONL text inside their records, including intermediate results. Historical protocol status text can reflect an earlier time; the endpoint tables and final assessments define completion.

To retrieve a file, find its entry in `FILE_MANIFEST.json`, read the JSONL container at `bundle_path`, and select the line whose `path` equals `record_path`. Decode `text` as UTF-8. Its SHA256 must equal `content_sha256`; the container SHA256 must equal `bundle_sha256`. Each manifest entry also preserves the original raw `source_sha256`, source identifier, category and prior text-normalization details. Repeated identical exported files can share one record. Packing changes no numeric evidence or raw source hashes.

`IMPLEMENTATION_VARIANTS.json` records actual closed-formT2, horizon-recurrence, nativeFast-SNN and bounded historical variants, forward definitions and code SHA evidence. Coincident-boundary tie conventions can differ; uniform-phase and M3 variants are not relabeled as the current method. `HASH_EVIDENCE.json` preserves recorded source/checkpoint/split hashes. `splits/` preserves all DVS membership and interval labels plus original manifest hashes; Tiny uses official split metadata and its full-manifest hash. Model/data bytes and per-frame predictions are excluded.

`VALIDATION.json` records the CPU packing checks. `BUNDLE_FILES.json` records hashes of all public files except itself. This handoff contains result evidence and does not require the original workstation directory to read.

## CIFAR100 T=4 table supplement

Each pair uses the same QCFS source, the fixed training split of 42,000 images and 160 fine-tuning epochs. Accuracy is Top-1 (%) on the official 10,000-image test set at epoch 160.

| Backbone | L | Direct ECL | Triangular BPTT | Full J | J − triangular (pp) |
|---|---:|---:|---:|---:|---:|
| ResNet34 | 2 | 77.62 | 79.23 | **80.31** | +1.08 |
| ResNet34 | 4 | 78.03 | 79.05 | **80.50** | +1.45 |
| VGG16 | 2 | 76.24 | 76.33 | **77.29** | +0.96 |
| VGG16 | 4 | 76.56 | 76.63 | **77.47** | +0.84 |

[Updated results table](RESULTS_TABLE.csv) · [Configurations, checkpoint hashes, endpoint tests and epoch histories](evidence_packed/table1_completion_20261126.jsonl)

## ATan comparison

The ATan baseline uses the standard slope $1/(1+\pi^2 z^2)$ for the threshold-normalized margin $z$, with $\alpha=2$. It uses the same signed hard forward, QCFS source, training images, fixed seed and 160 epochs as the corresponding comparison. The CSV records the actual batch and test split. [ATan definition](https://spikingjelly.readthedocs.io/zh-cn/latest/APIs/spikingjelly.activation_based.surrogate.html#spikingjelly.activation_based.surrogate.ATan)

| Dataset | Test images per configuration | Full J wins over ATan | Full J − ATan (pp) |
|---|---:|---:|---:|
| CIFAR100, official test | 10,000 | 12/12 | +0.51 to +10.96 |
| CIFAR10-DVS, fixed local test | 1,000 | 4/4 | +1.00 to +3.90 |
| DVS128 Gesture, official test | 288 | 3/4 | −0.69 to +2.43 |

On DVS128 Gesture / ResNet18 at L=4 and T=4, ATan classified 258/288 samples correctly (89.58%) and full J classified 256/288 correctly (88.89%).

[All 20 ATan results](ablations/atan/results.csv) · [Training and evaluation evidence](evidence_packed/atan_comparison_20261126.jsonl)

## DVS128 Gesture ResNet18 retraining

The follow-up updates student BN statistics on event inputs and, where indicated, uses the common QCFS ANN as a frozen teacher. The runs use T=4, the original QCFS source for each L, 160 epochs over the same 1,176 training samples, and the official 288-sample test split. The original 20 ATan comparisons retain their original recipes and endpoints.

| L | Recipe and endpoint | Full J | Control |
|---|---|---:|---:|
| 2 | Adaptive BN, CE, epoch 160 | 90.63% (261/288) | Triangular 90.63% (261/288) |
| 4 | Adaptive BN, CE, epoch 160 | 90.97% (262/288) | ATan 91.67% (264/288) |
| 4 | Adaptive BN, CE+KD, epoch 160 | 91.67% (264/288) | ATan 92.36% (266/288) |
| 4 | CE+KD rerun, unaveraged epoch 160 | 92.36% (266/288) | ATan 92.36% (266/288) |
| 4 | CE+KD rerun, predeclared epoch 101–160 weight average | 91.67% (264/288) | ATan 92.36% (266/288) |

In the rerun, unaveraged J reached 92.36% at epoch 80 versus 91.67% for ATan at the same epoch. This is an intermediate comparison. At epoch 100 and at the unaveraged epoch 160 endpoint, both scored 92.36%. The predeclared average uses a single model and scored two samples below the ATan endpoint. The saved epoch 80, 100 and 160 checkpoints and the averaged model were independently re-evaluated.

[Endpoint and intermediate results](supplements/dvs128_r18_retraining/results.csv) · [Training histories and checkpoint re-evaluations](evidence_packed/dvs128_r18_retraining_20261127.jsonl)

## ResNet18 CE+KD on CIFAR100

Both stages use full J and standard signed ECL at L=2 and T=2, with a frozen ReLU teacher during training and updated student BN statistics. The original 42k CE recipe reached 77.23%. The first CE+KD recipe also changed the training batch and BN policy. The second stage continued from the first stage's SNN weights on all 50,000 official training images with a fresh optimizer.

| Stage | Training images per epoch | Epochs in stage | Official test Top-1 | Difference from Xia Table 4 (pp) |
|---|---:|---:|---:|---:|
| CE+KD from QCFS | 42,000 | 160 | 77.94% | −0.58 |
| CE+KD continuation | 50,000 | 120 | 78.13% | −0.39 |

The continuation reached 78.56% at epoch 100 and finished at 78.13% at the fixed epoch 120 endpoint. Xia et al. reported 78.52% for ResNet18 at T=2 under their own training and conversion protocol. [Xia et al., Table 4](https://doi.org/10.1016/j.neucom.2026.132604)

[Two fixed endpoints](supplements/r18_ce_kd/results.csv) · [Stage configurations, epoch histories and evaluation evidence](evidence_packed/r18_ce_kd_20261127.jsonl)

## Column-sum ablation

The diagonal control $M=\operatorname{diag}(J^\top\mathbf{1})$ preserves the sensitivity of total output charge to each input time. It places that sensitivity on the corresponding diagonal entry, removing the redistribution across output times. The two vector–Jacobian products agree when the upstream gradient is constant across time. The control constrains column sums and leaves gradient norms free to vary.

| Dataset | Backbone | L | T | QCFS ANN | Direct ECL | Triangular | M | Full J | J − M (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR100 | ResNet18 | 2 | 2 | 77.47 | 75.05 | — | 75.58 | **77.23** | +1.65 |
| CIFAR10-DVS | ResNet20 | 2 | 4 | 72.40 | 65.00 | 72.50 | 62.00 | **76.20** | +14.20 |

Accuracy is Top-1 in percent after 160 fine-tuning epochs. Each pair shares the QCFS checkpoint, the signed hard forward, the training images and seed `2026100321`. CIFAR100 uses the official 10,000-image test set and 6.72 million training image presentations per arm. Its matched comparison covers M and full J. CIFAR10-DVS uses a fixed local 9,000/1,000 train/test split and 1.44 million training image presentations per arm.

CIFAR10-DVS M reached 73.30% at epoch 140 and finished at 62.00% at the fixed epoch 160 endpoint. An independent full test reproduced 62.00%, and checks on sampled currents from 19 activation sites matched the explicit column sums and the implemented backward. The cause of the late decline remains undetermined. The two endpoint gaps measure the effect of replacing J with this control under the recorded recipes.

[Results CSV](ablations/column_sum/results.csv) · [Train and test binding evidence](ablations/column_sum/evidence.json)
