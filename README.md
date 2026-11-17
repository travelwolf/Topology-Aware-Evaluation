# Topology-Aware Evaluation for Through-Hole Counting

This repository contains the method-specific code and metadata used to construct and audit a topology-aware benchmark for exact through-hole counting in native B-rep CAD models. It supports dataset generation, aligned split construction, topology-based label verification, special-case audits, and simple diagnostic baselines.

## Repository contents

- `holecounting_datasets_build/`: parametric CAD generation, hole sampling, STEP export, and final B-rep channel recount.
- `splits/`: deterministic random-split and leave-one-family-out (LOFO) split construction.
- `audit/`: stratified label auditing, dense rechecks, operation-level validation, and special-case tests.
- `baselines/`: topology-linear, cylindrical-rule, and ring-rule diagnostic baselines.
- `tests/`: unit tests for the generation policy and B-rep recount logic.
- `metadata/`: the 5,000-sample manifest and the train/validation split manifests used in the reported experiments.
- `results/construction_audit/`: released construction-audit summaries and family/count statistics.
- `utils/`: optional STEP-to-point and STEP-preview conversion utilities.

The repository intentionally focuses on the benchmark and evaluation pipeline. The compared neural encoders are identified below with their original publications and upstream implementations.

## Environment

The reference environment uses Python 3.10.13, `pythonocc-core`/OCCT 7.5.1, NumPy 1.26.4, Pillow 12.2.0, trimesh 4.11.4, PyTorch 2.3.0 with CUDA 12.1, DGL 2.5.0 with CUDA 12.1, and scikit-learn 1.7.2.

Create the core geometry environment with:

```bash
conda env create -f environment.yml
conda activate topology-aware-evaluation
```

The PyTorch and DGL CUDA builds depend on the local CUDA installation. Their tested versions are recorded in `requirements.txt` and can be installed from the official PyTorch and DGL package indexes.

## Dataset generation

Generate a dataset and manifest from the repository root:

```bash
python -m holecounting_datasets_build.dataset_builder \
  --output-root data/hole_counting_5000 \
  --num-models 5000 \
  --seed 42 \
  --export-step \
  --no-export-point \
  --no-export-image \
  --no-export-label \
  --no-export-bin
```

The generator assigns labels from the verified final B-rep geometry. Generation metadata and rejection reasons are retained in the manifest.

## Split construction

Create the fixed random split:

```bash
python splits/make_random_split.py \
  --manifest metadata/manifest.csv \
  --output-dir outputs/random_split \
  --seed 42
```

Create the 12-family LOFO splits:

```bash
python splits/make_lofo_splits.py \
  --manifest metadata/manifest.csv \
  --output-dir outputs/lofo
```

## Label and construction audit

Run the stratified audit using the generated dataset and a completed LOFO summary:

```bash
python audit/audit_reviewer_1_2.py \
  --dataset-root data/hole_counting_5000 \
  --lofo-summary outputs/lofo_training_summary.json \
  --source-root . \
  --output-dir outputs/audit \
  --samples-per-cell 5 \
  --dense-samples 2048 \
  --seed 20261117 \
  --replay \
  --accepted-audit
```

Run the final B-rep recount audit and special-case checks:

```bash
python audit/audit_final_brep_recount.py \
  --dataset-root data/hole_counting_5000 \
  --manifest metadata/manifest.csv \
  --output-dir outputs/recount_audit

python audit/run_operation_validation_cases.py --output-dir outputs/operation_cases
python audit/run_recount_special_cases.py --output-dir outputs/recount_cases
python -m unittest discover -s tests -v
```

The released audit summarizes 430 stratified models, 1,785 recorded channels, and 7,548 candidates. With a 2,048-sample dense check and a geometric tolerance of `1e-7`, the audit recorded one false acceptance and one false rejection.

## Diagnostic baselines

Evaluate the released simple baselines with:

```bash
python baselines/evaluate_simple_baselines.py \
  --split-root metadata/splits \
  --data-root data/hole_counting_5000 \
  --output-dir outputs/baselines
```

## Compared neural encoders

The paper evaluates the following established architectures. Their source code remains governed by the respective upstream licenses.

- **UV-Net** — native B-rep face/edge geometry with graph message passing: [paper](https://doi.org/10.1109/CVPR46437.2021.01153), [upstream code](https://github.com/AutodeskAILab/UV-Net).
- **Point Transformer** — point-cloud representation: [paper](https://doi.org/10.1109/ICCV48922.2021.01595).
- **MobileViT** — multi-view image representation: [paper](https://doi.org/10.48550/arXiv.2110.02178), [upstream code](https://github.com/apple-aiml-research/ml-cvnets).
- **VGNet** — voxel/graph representation: [upstream code](https://github.com/xzzz011/VGNet).

## License

The original code and metadata in this repository are released under the MIT License. Third-party implementations and dependencies remain subject to their own licenses.
