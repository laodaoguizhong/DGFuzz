# DGFuzz: Defect-Guided Fuzzing

Official implementation of **DGFuzz: Region-level defect feedback for directed fuzz testing of deep neural networks**.

```bash
pip install -r requirements.txt
```


## Workflow

Run from the project root in this order:

**1. Train models (CIFAR-10, optional)**

```bash
cd models
python train.py --model resnet18 --dataset cifar10
# or batch: .\train_all.ps1
```

Weights are saved as `checkpoints/cifar10_{model}.pth`. For MNIST LeNet models, train separately and name checkpoints `checkpoints/mnist_lenet*.pth`.

**2. Build adversarial library (required before fuzzing)**

```bash
python -m utils.advlib --dataset cifar10 --model resnet18 --device cuda
```

Output: `results/{dataset}/{model}/adversarial_library.npz`. Omit `--dataset` / `--model` to run all models.

**3. Run DGFuzz fuzzing (main experiment)**

```bash
python fuzz.py --dataset cifar10 --model resnet18 --device cuda
```

Output: `results/{dataset}/{model}/dgfuzz/dgfuzz_results.json` and plots.

Common flags: `--max_samples`, `--max_time_hours`, `--mutations_per_seed`.

**4. Ablation (optional)**

```bash
python -m utils.ablate --device cuda
```

## Supported models

- **MNIST**: `lenet1`, `lenet4`, `lenet5`
- **CIFAR-10**: `resnet18/34/20/50`, `densenet121`, `googlenet`, `mobilenet_v2`, `vgg13_bn`, `vgg16_bn`

## Notes

- Fuzzing requires `adversarial_library.npz` from step 2.
- Tune hyperparameters in `config.py` or via CLI (e.g. `--dgfuzz_bandwidth`, `--dgfuzz_tau_r`).
