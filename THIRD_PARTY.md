# Upstream implementation

`src/third_party/qcfs_layer.py` is the preserved author activation implementation from [ANN_SNN_QCFS](https://github.com/putshua/ANN_SNN_QCFS), originally `Models/layer.py`.

File SHA256: `64e2ba6566f1c252432092166ad5d2d214f73ebbe7314a5ad2e188d02642030a`.

The file is retained byte-for-byte. `src/conversion/qcfs_ecl_resnet18.py` loads only the needed class definitions with AST; unused upstream imports are not executed and do not add dependencies. QCFS quantization, its straight-through derivative, and channelwise source preparation are established components, not the shared-phase contribution.

The CIFAR-10 ordinary ANN source is [edadaltocg/resnet18_cifar10](https://huggingface.co/edadaltocg/resnet18_cifar10), pinned to revision `ccacbae365101b9bfa45e1f17f4acac30c2601b2`. Its historical provenance is retained in the result and code inventories. The loader changes only `.downsample.` tensor-key segments to `.shortcut.` and loads the result strictly.

`src/detection_fast_snn.py` loads the external [Fast-SNN author repository](https://github.com/zju-bmi-lab/Fast-SNN), pinned for the VOC experiment to commit `e4315cd4e74b4e185ab12bbe2dd74bc3fdccc547`. The checkout and public weights are supplied separately and are not bundled here. The YOLOv2–ResNet34 topology, VOC dataset transformations, label assignment, detection losses, NMS, and VOC07 AP50 evaluator remain author components. Runtime import patches replace the removed TorchVision `model_urls` name and old NumPy aliases/empty-array comparisons. An empty prediction class receives AP=0, correcting the upstream evaluator's AP=-1 sentinel; nonempty-class evaluation is unchanged. The shared-phase gradient and the single final prediction on time-mean features are implemented in this project.

The CIFAR100 ordinary ResNet34 ANN is [edadaltocg/resnet34_cifar100](https://huggingface.co/edadaltocg/resnet34_cifar100), revision `50cd2c2d5506e750dfbad7be7c80070515206137`. It has a CIFAR 3x3 stem, BasicBlock depths [3,4,6,3], and 21,328,292 parameters. The local loader strictly loads its state after the existing `.downsample.` to `.shortcut.` key mapping. This is a ReLU source, not a published QCFS or ECL CIFAR100 R34 result; the subsequent scalar-threshold QAT is local source preparation.

## ViStream / UniTrack tracking

The tracking adapter loads the external [ViStream author repository](https://github.com/Intelligent-Computing-Research-Group/ViStream), pinned to `faecff832cea4251b6504a1efe4277dc57828f35`. Its original UniTrack association and Kalman/IoU recovery remain author components. `src/tracking_association.py` adapts the reconstruction-dot score from `core/association/matching.py` without its final NumPy detach, enabling task-loss gradients. The upstream MIT notice is retained in `src/third_party/vistream_LICENSE.txt`. Author code, data and public weights remain separate downloads. Clip-consistent training, the bounded phase VJP, and runtime compatibility adapters are local additions; reported held-out MOT16 train videos are not private-test leaderboard results.

## DiMP18 / PyTracking preparation

`src/tracking_dimp.py` optionally loads the external [PyTracking author repository](https://github.com/visionml/pytracking/tree/7eb9e74bd3d40e29dbcec444902237da13de247b), pinned to `7eb9e74bd3d40e29dbcec444902237da13de247b` (GPL-3.0). Author source and the public DiMP18 checkpoint are separate local artifacts, not bundled in this module. The original classifier, analytic filter optimizer, IoU head, crop geometry and PreciseRoIPooling remain author components. The local adapter separates repeated activation calls, installs the existing QCFS/shared-phase or triangular neurons, and decodes layer2/layer3 features. Native PrROI compatibility changes only its C++ API/error checks and restores original source symlinks; its CUDA numerical kernel is unchanged. The initialization checkpoint produced by `scripts/prepare_dimp.py` is untrained calibration, not a published QCFS/SNN tracking model. Repository licensing does not establish separate checkpoint or dataset licensing.
