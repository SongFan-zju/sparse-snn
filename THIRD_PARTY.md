# Upstream implementation

`src/third_party/qcfs_layer.py` is the preserved author activation implementation from [ANN_SNN_QCFS](https://github.com/putshua/ANN_SNN_QCFS), originally `Models/layer.py`.

File SHA256: `64e2ba6566f1c252432092166ad5d2d214f73ebbe7314a5ad2e188d02642030a`.

The file is retained byte-for-byte. `src/conversion/qcfs_ecl_resnet18.py` loads only the needed class definitions with AST; unused upstream imports are not executed and do not add dependencies. QCFS quantization, its straight-through derivative, and channelwise source preparation are established components, not the shared-phase contribution.

The CIFAR-10 ordinary ANN source is [edadaltocg/resnet18_cifar10](https://huggingface.co/edadaltocg/resnet18_cifar10), pinned to revision `ccacbae365101b9bfa45e1f17f4acac30c2601b2`. Its download URL and checksum are recorded in `configs/weights.json`. The loader changes only `.downsample.` tensor-key segments to `.shortcut.` and loads the result strictly.
