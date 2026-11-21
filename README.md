# Shared-phase trajectory gradients for ANN-to-SNN adaptation

**English** | [简体中文](README.zh-CN.md)

The `paper` branch is a compact research handoff for writing and reproducing the current work. It contains the method, task runners, mathematical derivations and recorded experiment evidence. It is based on remote `main`; datasets, checkpoints, environments, unrelated research and training logs are supplied separately.

Training keeps the deterministic signed spiking forward and changes its backward pass. A triangular initial-phase distribution is shared across the complete neuron trajectory, producing a conditional-mean charge Jacobian. Deployment uses one deterministic SNN trajectory. L denotes source-ANN quantization levels and T the number of SNN timesteps.

Start with these files:

| Purpose | Entry |
|---|---|
| Final endpoint table and evaluation protocols | [results/paper/README.md](results/paper/README.md), [CSV](results/paper/RESULTS_TABLE.csv), [JSON](results/paper/RESULTS_TABLE.json) |
| Actual gradient variants, forward rules and frozen code hashes | [IMPLEMENTATION_VARIANTS.json](results/paper/IMPLEMENTATION_VARIANTS.json) |
| Mathematical motivation, proofs, assumptions and counterexamples | [Chinese research index](MATH_RESEARCH_INDEX_ZH.md), [references](MATH_REFERENCES_ZH.md) |
| Environment, task commands and external assets | [RUNNING.md](RUNNING.md) |
| Selected source files and provenance | [code inventory](provenance/code_inventory.json), [upstream attribution](THIRD_PARTY.md) |
| Missing or unverified historical evidence | [SCOPE_AND_GAPS.json](results/paper/SCOPE_AND_GAPS.json) |

The main operator is [triangular_density_horizon.py](src/conversion/triangular_density_horizon.py); [triangular_density_triton.py](src/conversion/triangular_density_triton.py) provides optional acceleration. The matched conventional baseline is [ecl_triangular_surrogate.py](src/conversion/ecl_triangular_surrogate.py). Historical two-step closed forms and bounded ViStream neurons are explicitly identified rather than relabeled as the current implementation.

Experiment tables separate source ANN, direct conversion and adapted SNN. They retain fixed final epochs, small differences and negative results. CIFAR100 uses the official test set; Tiny-ImageNet uses its official labeled validation set; CIFAR10-DVS uses the fixed saved local split; Gesture uses official trials. Tracking scores come from local MOT16 tasks, not official tracking benchmarks. Historical standard ResNet18 results lacking verified evidence for the current matched method are not substituted with the different Fast-SNN architecture.

The local mathematical result concerns the conditional-mean neuron Jacobian. It does not establish an exact hard-loss gradient, an unbiased whole-network task gradient or a guarantee of classification accuracy. The report includes its assumptions and counterexamples. Final tracking experiments did not show an additional benefit over triangular gradients.

Install the common dependencies with `pip install -r requirements.txt`. Additional task dependencies and upstream checkouts are listed in RUNNING.md. CPU checks:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python artifacts/tracking_gradient_math_20261119/check_unified_count_boundary_cpu.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python artifacts/tracking_gradient_math_20261119/check_positive_mechanism.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python artifacts/tracking_gradient_math_20261119/check_foundations_cpu.py
```

External-author integrations and CUDA tests require their separate assets. Checkpoints are identified by hashes but are not hosted in this branch. The bundled code is not a full archival replay of every historical training environment; the code inventory and result evidence state the scope for each task. No ICML manuscript is included.
