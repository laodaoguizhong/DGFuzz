# DGFuzz

Code for the paper:

**DGFuzz: Defect-Guided Fuzzing for Adversarial Defect Detection in Deep Neural Networks**


## Install

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the project root.

**1. Build the adversarial library (required before fuzzing)**

```bash
python -m utils.advlib --dataset cifar10 --model resnet18 --device cuda
```

Output: `results/{dataset}/{model}/adversarial_library.npz`

**2. Run DGFuzz**

```bash
python fuzz.py --dataset cifar10 --model resnet18 --device cuda
```

Output: `results/{dataset}/{model}/dgfuzz/dgfuzz_results.json`

Omit `--dataset` and `--model` in either command to run all configured models.

## Models in the Paper

The models used in the paper are provided in the code:

| Dataset   | Models |
|-----------|--------|
| MNIST     | `lenet1`, `lenet4`, `lenet5` |
| CIFAR-10  | `resnet18`, `resnet34`, `resnet20`, `resnet50`, `densenet121`, `googlenet`, `mobilenet_v2`, `vgg13_bn`, `vgg16_bn` |

Place trained weights under `checkpoints/{dataset}_{model}.pth` (train with `models/train.py` if needed).

For other models or datasets, please configure them manually in `models/models.py` and the experiment scripts.

## Notes

- The adversarial library must be generated before fuzzing.
- Method hyperparameters are defined in `utils/runtime.py`, `utils/adv.py`, `utils/mutate.py`, and `utils/cluster.py`. `config.py` only sets paths, device, and VRM sample counts.
