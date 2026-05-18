# CIFAR-10 batch training script (PowerShell)
Write-Host "Starting batch training for CIFAR-10..." -ForegroundColor Green
Write-Host "Only best models will be saved with format: {model_name}_cifar10.pth"
Write-Host ""

# Supported models (vgg11_bn, vgg19_bn, resnet50 excluded)
$models = @(
    "resnet18",
    "resnet34",
    "densenet121",
    "googlenet",
    "mobilenet_v2",
    "vgg13_bn",
    "vgg16_bn"
)

foreach ($model in $models) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Training $model..." -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    python train.py --model $model --dataset cifar10
    Write-Host ""
    Write-Host "$model training completed." -ForegroundColor Green
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All models trained successfully." -ForegroundColor Green
Write-Host "Trained models: $($models -join ', ')"
Write-Host "Check ./checkpoints/ for saved weights"
Write-Host "========================================" -ForegroundColor Cyan
