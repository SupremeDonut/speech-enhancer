"""Build manifests, train, and evaluate held-out enhancement (no pretrained model)."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import Mixtures, manifest
from training import StreamingDenoiseNet, enhance, objective, si_sdr


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for mixed, clean in loader:
            mixed, clean = mixed.to(device), clean.to(device)
            enhanced = enhance(model, mixed)
            baseline_score = si_sdr(clean, mixed).item()
            enhanced_score = si_sdr(clean, enhanced).item()
            rows.append({'baseline_si_sdr_db': baseline_score,
                         'enhanced_si_sdr_db': enhanced_score,
                         'delta_si_sdr_db': enhanced_score-baseline_score})
    baseline_mean = float(np.mean([row['baseline_si_sdr_db'] for row in rows]))
    enhanced_mean = float(np.mean([row['enhanced_si_sdr_db'] for row in rows]))
    return {'count': len(rows), 'baseline_mean_db': baseline_mean,
            'enhanced_mean_db': enhanced_mean,
            'delta_mean_db': enhanced_mean-baseline_mean, 'clips': rows}


def training_device(value):
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but is unavailable in this PyTorch environment; '
            'install the CUDA-enabled PyTorch wheel or pass --device cpu explicitly'
        )
    return device


CHECKPOINT_VERSION = 1


def _cpu_state_dict(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _capture_rng_state(device):
    numpy_name, numpy_keys, numpy_position, has_gaussian, cached_gaussian = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {
            'algorithm': numpy_name,
            'keys': torch.from_numpy(numpy_keys.copy()),
            'position': numpy_position,
            'has_gaussian': has_gaussian,
            'cached_gaussian': cached_gaussian,
        },
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
    }


def _restore_rng_state(state, device):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy['algorithm'], numpy['keys'].cpu().numpy().astype(np.uint32, copy=False),
                         numpy['position'], numpy['has_gaussian'], numpy['cached_gaussian']))
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        cuda_states = state['cuda']
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the saved checkpoint')
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _atomic_save(value, path):
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _training_checkpoint(model, optimizer, epoch, best_score, best_epoch,
                         best_model, best_validation, unimproved, args,
                         manifest_sha256, device):
    return {
        'kind': 'training',
        'version': CHECKPOINT_VERSION,
        'model': _cpu_state_dict(model.state_dict()),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_score': best_score,
        'best_epoch': best_epoch,
        'best_model': best_model,
        'best_validation': best_validation,
        'unimproved': unimproved,
        'seed': args.seed,
        'train_pairs': args.train_pairs,
        'validation_pairs': args.validation_pairs,
        'manifest_sha256': manifest_sha256,
        'device_type': device.type,
        'cuda_device_count': torch.cuda.device_count() if device.type == 'cuda' else 0,
        'rng_state': _capture_rng_state(device),
    }


def _best_checkpoint(model_state, epoch, validation, seed, manifest_sha256):
    return {
        'kind': 'best',
        'version': CHECKPOINT_VERSION,
        'model': model_state,
        'epoch': epoch,
        'validation': validation,
        'seed': seed,
        'manifest_sha256': manifest_sha256,
    }


def train(args):
    device = training_device(args.device)
    manifest_path = Path(args.manifest)
    manifest_sha256 = _manifest_sha256(manifest_path)
    args.output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = args.output/'last.pt', args.output/'best.pt'
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    if args.resume is None and (last_path.exists() or best_path.exists()):
        raise FileExistsError(
            f'Checkpoint files already exist in {args.output}; use --resume with last.pt '
            'or choose a new output directory'
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_data = Mixtures(manifest_path, 'train', args.train_pairs, args.seed)
    validation = Mixtures(manifest_path, 'validation', args.validation_pairs, args.seed)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=1, num_workers=0)
    model = StreamingDenoiseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    start_epoch = 0
    best_score, best_epoch, best_model, best_validation = None, None, None, None
    unimproved = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('kind') != 'training' or checkpoint.get('version') != CHECKPOINT_VERSION:
            raise ValueError('Resume requires a compatible last.pt training checkpoint')
        expected = {
            'seed': args.seed,
            'train_pairs': args.train_pairs,
            'validation_pairs': args.validation_pairs,
            'manifest_sha256': manifest_sha256,
            'device_type': device.type,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(f'Resume checkpoint {name} does not match this run')
        if device.type == 'cuda' and checkpoint['cuda_device_count'] != torch.cuda.device_count():
            raise ValueError('Resume checkpoint CUDA device count does not match this run')
        start_epoch = checkpoint['epoch']
        if args.epochs <= start_epoch:
            raise ValueError(
                f'--epochs ({args.epochs}) must exceed the checkpoint epoch ({start_epoch})'
            )
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_score = checkpoint['best_score']
        best_epoch = checkpoint['best_epoch']
        best_model = checkpoint['best_model']
        best_validation = checkpoint['best_validation']
        unimproved = checkpoint['unimproved']
        if best_model is None:
            if best_path.exists():
                raise FileExistsError(
                    f'{best_path} exists but the resume checkpoint has no completed best epoch'
                )
        else:
            _atomic_save(
                _best_checkpoint(best_model, best_epoch, best_validation,
                                 args.seed, manifest_sha256),
                best_path,
            )
        _restore_rng_state(checkpoint['rng_state'], device)
    else:
        _atomic_save(
            _training_checkpoint(model, optimizer, 0, best_score, best_epoch,
                                 best_model, best_validation, unimproved, args,
                                 manifest_sha256, device),
            last_path,
        )

    for epoch in range(start_epoch, args.epochs):
        train_data.epoch = epoch
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch')
        for mixed, clean in progress:
            mixed, clean = mixed.to(device), clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            estimate = enhance(model, mixed)
            loss = objective(clean, estimate)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(loss=f'{loss.item():.4f}')
        report = evaluate(model, validation_loader, device)
        completed_epoch = epoch + 1
        print(json.dumps({'epoch': completed_epoch, 'training_loss': float(np.mean(losses)),
                          'validation_delta_si_sdr_db': report['delta_mean_db'],
                          'validation_mean_si_sdr_db': report['enhanced_mean_db']}), flush=True)
        score = report['enhanced_mean_db']
        improved = best_score is None or score > best_score
        if improved:
            best_score, best_epoch = score, completed_epoch
            best_model, best_validation = _cpu_state_dict(model.state_dict()), report
            unimproved = 0
        else:
            unimproved += 1

        _atomic_save(
            _training_checkpoint(model, optimizer, completed_epoch, best_score,
                                 best_epoch, best_model, best_validation, unimproved,
                                 args, manifest_sha256, device),
            last_path,
        )
        if improved:
            _atomic_save(
                _best_checkpoint(best_model, best_epoch, best_validation,
                                 args.seed, manifest_sha256),
                best_path,
            )
        if unimproved >= 5:
            break
    return best_path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('manifest')
    scan.add_argument('--libri', type=Path, required=True)
    scan.add_argument('--noise', type=Path, required=True)
    scan.add_argument('--output', type=Path, required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--manifest', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--train-pairs', type=int, default=8000)
    fit.add_argument('--validation-pairs', type=int, default=1000)
    fit.add_argument('--seed', type=int, default=1729)
    fit.add_argument('--resume', type=Path,
                     help='resume optimizer and RNG state from a last.pt training checkpoint')
    fit.add_argument('--epochs', type=int, default=50,
                     help='total target epoch count; must exceed the checkpoint epoch when resuming')
    fit.add_argument('--device', default='cuda',
                     help='training device (default: cuda; use cpu explicitly for CPU training)')
    score = sub.add_parser('evaluate')
    score.add_argument('--manifest', type=Path, required=True)
    score.add_argument('--checkpoint', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)
    score.add_argument('--pairs', type=int, default=1000)
    score.add_argument('--seed', type=int, default=1729)
    score.add_argument('--device', default='cuda',
                       help='evaluation device (default: cuda; use cpu explicitly for CPU evaluation)')
    args = parser.parse_args()
    if args.command == 'manifest':
        result = manifest(args.libri, args.noise, args.output)
        print(json.dumps({'manifest': str(args.output), 'speech': {s:len(v) for s,v in result['speech'].items()},
                          'noise': {s:len(v) for s,v in result['noise'].items()}}))
    elif args.command == 'train':
        print(f'Best checkpoint: {train(args)}')
    else:
        device = training_device(args.device)
        model = StreamingDenoiseNet().to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        dataset = Mixtures(args.manifest, 'test', args.pairs, args.seed)
        report = evaluate(model, DataLoader(dataset, batch_size=1), device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k != 'clips'}))


if __name__ == '__main__':
    main()
