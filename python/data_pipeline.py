"""Reproducible corpus inventory and four-second reverberant-speech mixtures."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
LENGTH = 4 * SAMPLE_RATE
SEED = 1729
SPEECH_SPLITS = {'train': 'train-clean-100', 'validation': 'dev-clean', 'test': 'test-clean'}


def inventory(root, extensions):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)


def manifest(libri_root, noise_root, destination):
    libri_root, noise_root = Path(libri_root).resolve(), Path(noise_root).resolve()
    speech = {}
    speakers = set()
    for split, directory in SPEECH_SPLITS.items():
        folder = libri_root / directory
        files = inventory(folder, {'.flac', '.wav'})
        if not files:
            raise ValueError(f'No speech files in {folder}')
        people = set()
        for path in files:
            parts = path.relative_to(folder).parts
            people.add(parts[1] if parts[0] == directory else parts[0])
        if speakers & people:
            raise ValueError('Speaker appears in multiple partitions')
        speakers.update(people)
        speech[split] = [str(p) for p in files]
    files = inventory(noise_root, {'.wav', '.flac'})
    if len(files) < 3:
        raise ValueError('At least three noise recordings are required')
    random.Random(SEED).shuffle(files)
    n_train = int(len(files) * .8)
    n_val = int(len(files) * .1)
    if n_val == 0:
        n_val = 1
        n_train -= 1
    noise = {'train': files[:n_train], 'validation': files[n_train:n_train+n_val],
             'test': files[n_train+n_val:]}
    if any(not group for group in noise.values()):
        raise ValueError('All noise partitions must contain recordings')
    result = {'seed': SEED, 'speech': speech,
              'noise': {key: [str(p) for p in group] for key, group in noise.items()},
              'sources': {'speech': 'https://www.openslr.org/12',
                          'noise_recipe': 'https://github.com/microsoft/MS-SNSD/tree/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742'},
              'files': {}}
    for path in sorted(set(sum((list(v) for v in speech.values()), []) +
                           [str(p) for group in noise.values() for p in group])):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result['files'][path] = digest.hexdigest()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def load_audio(path):
    signal, rate = sf.read(path, dtype='float32', always_2d=True)
    mono = signal.mean(axis=1)
    if rate != SAMPLE_RATE:
        factor = math.gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, rate // factor).astype(np.float32)
    return mono


def segment(signal, rng, repeat=False):
    if not len(signal):
        raise ValueError('Empty recording')
    if repeat and len(signal) < LENGTH:
        signal = np.tile(signal, math.ceil(LENGTH / len(signal)))
    if len(signal) >= LENGTH:
        start = rng.randrange(len(signal) - LENGTH + 1)
        return signal[start:start+LENGTH].copy()
    return np.pad(signal, (0, LENGTH-len(signal))).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def random_rir(rng):
    rt60 = rng.uniform(.1, .5)
    # Stochastic decaying reflections with a unit direct path; bounded FIR.
    duration = min(2.0, rt60 * 1.3)
    length = int(SAMPLE_RATE * duration)
    generator = np.random.default_rng(rng.getrandbits(64))
    t = np.arange(length) / SAMPLE_RATE
    rir = generator.standard_normal(length).astype(np.float32)
    rir *= np.exp(-math.log(1000) * t / rt60).astype(np.float32) * .012
    rir[0] = 1
    return rir


class Mixtures(Dataset):
    def __init__(self, manifest_path, split, count, seed=SEED):
        doc = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        self.speech = doc['speech'][split]
        self.noise = doc['noise'][split]
        if not self.speech or not self.noise or count <= 0:
            raise ValueError('Nonempty speech/noise partitions and positive count required')
        self.count, self.seed = count, seed
        self.split = split
        self.epoch = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        epoch = self.epoch if self.split == 'train' else 0
        rng = random.Random(self.seed + 1000003 * epoch + index * 10007)
        for _ in range(40):
            clean = segment(load_audio(rng.choice(self.speech)), rng)
            clean = fftconvolve(clean, random_rir(rng))[:LENGTH].astype(np.float32)
            noise = segment(load_audio(rng.choice(self.noise)), rng, repeat=True)
            cr, nr = rms(clean), rms(noise)
            if min(cr, nr) < 1e-5:
                continue
            snr = rng.uniform(-5, 15)
            mixed = clean + noise * (cr / (nr * 10 ** (snr / 20)))
            peak = float(np.max(np.abs(mixed)))
            if peak == 0:
                continue
            gain = min(1., .99 / peak)
            return (mixed * gain).astype(np.float32), (clean * gain).astype(np.float32)
        raise ValueError('Could not sample nonsilent speech/noise mixture after 40 attempts')
