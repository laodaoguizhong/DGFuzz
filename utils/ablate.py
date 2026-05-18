from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

def run_cmd(cmd: List[str]) -> int:
    print('\n>>>', ' '.join(cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    return proc.returncode

def _read_unique_defects(path: Path) -> int:
    data = json.loads(path.read_text(encoding='utf-8'))
    return int((data.get('summary') or {}).get('unique_defects', 0))

def main() -> None:
    parser = argparse.ArgumentParser(description='Run DGFuzz ablation experiments (Python runner)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=15000)
    parser.add_argument('--max_time_hours', type=float, default=0.25)
    parser.add_argument('--early_stop_rounds', type=int, default=500)
    parser.add_argument('--mutations_per_seed', type=int, default=10)
    parser.add_argument('--force-rerun-existing', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fuzz_script = root / 'fuzz.py'
    targets = [{'dataset': 'mnist', 'model': 'lenet4'}, {'dataset': 'cifar10', 'model': 'vgg13_bn'}, {'dataset': 'cifar10', 'model': 'densenet121'}, {'dataset': 'cifar10', 'model': 'googlenet'}]
    variants = [{'name': 'seed_init', 'si': 0, 'aps': 1}, {'name': 'adaptive', 'si': 1, 'aps': 0}]
    out_dir = root / 'results' / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for t in targets:
        for v in variants:
            key = f"{t['dataset']}/{t['model']}/{v['name']}"
            result_file = out_dir / f"{t['model']}_{v['name']}.json"
            dataset_run_file = out_dir / f"{t['dataset']}_{t['model']}_{v['name']}.json"
            print(f'\n==== {key} ====')
            if result_file.exists() and (not args.force_rerun_existing):
                unique = _read_unique_defects(result_file)
                summary[key] = unique
                print(f'[INFO] Skip existing: {result_file.name} (unique={unique})')
                continue
            cmd = [sys.executable, str(fuzz_script), '--dataset', t['dataset'], '--model', t['model'], '--device', args.device, '--max_samples', str(args.max_samples), '--max_time_hours', str(args.max_time_hours), '--early_stop_rounds', str(args.early_stop_rounds), '--mutations_per_seed', str(args.mutations_per_seed), '--enable_seed_init', str(v['si']), '--enable_adaptive_schedule', str(v['aps']), '--ablation_name', v['name']]
            code = run_cmd(cmd)
            if code != 0:
                print(f"[WARN] Run failed for {t['dataset']}-{t['model']} variant={v['name']} (code={code})")
                continue
            if not result_file.exists():
                print(f'[WARN] Result file not found: {result_file}')
                continue
            if dataset_run_file != result_file:
                shutil.copyfile(result_file, dataset_run_file)
            unique = _read_unique_defects(result_file)
            summary[key] = unique
    summary_path = out_dir / 'ablation_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n===== Ablation Summary (unique defects) =====')
    for k, unique in summary.items():
        print(f'{k}: unique={unique}')
    print(f'\nDGFuzz ablation experiments completed. Summary saved to: {summary_path}')

if __name__ == '__main__':
    main()
