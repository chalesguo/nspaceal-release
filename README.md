# N-spaceAL

Anonymous release code for the active-learning experiments on CIFAR-100, SVHN, and Caltech-101.

This package contains the cleaned source needed to reproduce the manuscript settings. Datasets, checkpoints, cached bytecode, local paths, IDE files, and historical result logs are intentionally excluded.

## Layout

```text
.
|-- main.py                    # Main training and acquisition loop
|-- relative_innovation.py     # Nullspace and relative-innovation acquisition
|-- badge.py                   # BADGE baseline
|-- summarize_results.py       # Accuracy table summarizer
|-- summarize_runtime_table.py # Runtime table summarizer
|-- train_utils.py
|-- dataset/
|-- models/
|-- requirements.txt
`-- .gitignore
```

## Environment

Use the PyTorch build that matches your CUDA version. The experiments were configured with PyTorch and torchvision backbones.

```bash
conda create -n nspaceal python=3.10 -y
conda activate nspaceal
pip install -r requirements.txt
```

## PDF-Consistent Defaults

`--backbone auto_pdf` is the default and resolves to the manuscript backbone setting per dataset:

| Dataset | Backbone | Input size | Acquisition budgets |
|---|---|---:|---|
| CIFAR-100 | ResNet-18 | 32 | 1000, 2000, 3000, 4000, 5000 |
| SVHN | Wide ResNet-18, width multiplier 2 | 32 | 500, 1000, 1500, 2000, 2500, 3000 |
| Caltech-101 | ImageNet-pretrained ResNet-50 | 224 | 500, 1000, 1500, 2000, 2500, 3000 |

Caltech-101 should be placed in one of these layouts:

```text
data/caltech101/101_ObjectCategories
data/caltech-101/101_ObjectCategories
data/101_ObjectCategories
```

## Main Experiments

Ours:

```bash
python main.py --dataset cifar100 --datapath data --save results/ours --exp_name ours_cifar100 --trial 3 --budgets base --acq_method ours --use_maximin
python main.py --dataset svhn --datapath data --save results/ours --exp_name ours_svhn --trial 3 --budgets base --acq_method ours --use_maximin
python main.py --dataset caltech101 --datapath data --save results/ours --exp_name ours_caltech101 --trial 3 --budgets base --acq_method ours --use_maximin
```

BADGE baseline under the same dataset/backbone budget convention:

```bash
python main.py --dataset cifar100 --datapath data --save results/badge --exp_name badge_cifar100 --trial 3 --budgets base --acq_method badge
python main.py --dataset svhn --datapath data --save results/badge --exp_name badge_svhn --trial 3 --budgets base --acq_method badge
python main.py --dataset caltech101 --datapath data --save results/badge --exp_name badge_caltech101 --trial 3 --budgets base --acq_method badge
```

For a quick smoke run, use fewer cycles and epochs:

```bash
python main.py --dataset svhn --datapath data --save results/smoke --trial 1 --budgets test --epochs 1 --ft_epochs 1 --batch_size 64
```

## Runtime Reporting

Runtime components are logged to `metrics.jsonl` as feature extraction, SVD, density, greedy selection, and total acquisition time. CUDA synchronization is used around timed regions when a CUDA device is available.

The density term for Ours is the exact full unlabeled-pool Gaussian kernel density computed with block-wise GPU matrix operations. It does not use kNN, subsampling, or another density approximation.

Use the summarizer on a results directory that contains experiment subdirectories:

```bash
python summarize_runtime_table.py --input_dir results/ours_runtime
```

## Notes

- CIFAR and SVHN can be downloaded automatically by torchvision.
- Caltech-101 is loaded from local image folders.
- The release is anonymized and contains no author names, machine-specific paths, data, checkpoints, or result files.
- Add a license file before public reuse if required by your venue or institution.
