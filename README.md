# Real-time Speech Enhancer

A trained neural denoiser in PyTorch, exported to ONNX, then ran from a C++ application. It suppresses background noise in speech in real-time, working on microphone input, Windows playback, and audio files.

## Requirements and setup

Tested only on Windows 11

### Python training

- [uv](https://docs.astral.sh/uv/) is required.
- CUDA training/evaluation is the default. Use `--device cpu` for CPU runs.

### Native C++ runtime

Building the C++ application requires Windows, CMake, and Visual Studio 2022 C++ tools. Install FFmpeg and make `ffmpeg.exe` available on `PATH` to convert unsupported WAVs or other audio formats. Files already accepted as mono 16 kHz PCM16/float32 WAV do not need FFmpeg.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Data aquisition

1. Download LibriSpeech SLR12 `train-clean-100`, `dev-clean`, and `test-clean` from [OpenSLR 12](https://www.openslr.org/12). Save the three `.tar.gz` archives under `data/downloads/`.
2. Extract each archive into `data/`; the expected speech paths are `data/LibriSpeech/train-clean-100/`, `data/LibriSpeech/dev-clean/`, and `data/LibriSpeech/test-clean/`.
3. Obtain the MS-SNSD noise recordings from the [Github](https://github.com/microsoft/MS-SNSD/tree/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742), cloning the repo into `data/MS-SNSD`.
4. Generate the manifest:

```powershell
uv run python python/run_training.py manifest --libri data/LibriSpeech --noise data/MS-SNSD/noise_train --output manifests/corpora.json
```

## Training

```powershell
uv run python python/run_training.py train --manifest manifests/corpora.json --output runs/experiment-01 --train-pairs 8000 --validation-pairs 1000 --epochs 50 --device cuda
```

Training shows a `tqdm` batch bar with live loss. It validates after every epoch, saves `last.pt` at epoch boundaries, and saves `best.pt` when validation SI-SDR improves. It stops at `--epochs` (default 50) or after five consecutive epochs without validation improvement.

## Evaluate the checkpoint

The evaluation command measures SI-SDR on test mixtures and writes per-pair and aggregate metrics:

```powershell
uv run python python/run_training.py evaluate --manifest manifests/corpora.json --checkpoint runs/experiment-01/best.pt --output runs/experiment-01/evaluation.json --pairs 1000 --seed 1729 --device cuda
```

## Export and use the model

Export the PyTorch checkpoint to an ONNX graph:

```powershell
uv run python python/export_onnx.py --checkpoint runs/experiment-01/best.pt --output models/denoiser_causal.onnx
```

The C++ CLI accepts audio in formats FFmpeg can decode. Already-supported nonempty mono 16 kHz WAV files encoded as PCM16 or IEEE float32 are read directly, while other audio inputs are converted through FFmpeg to mono 16 kHz PCM16 WAV before enhancement. Offline mode writes mono 16 kHz float32 WAV with the input sample count preserved:

```powershell
build\Release\speech_isolation.exe --model models/denoiser_causal.onnx --input input.mp3 --output enhanced.wav
```

Default live mode denoises the default microphone and plays to the default output:

```powershell
build\Release\speech_isolation.exe --model models/denoiser_causal.onnx --live
```

To denoise Windows playback, route it through a virtual audio cable (e.g. VB-CABLE). Set Windows playback to the cable's playback endpoint, then enumerate devices:

```powershell
build\Release\speech_isolation.exe --list-devices
build\Release\speech_isolation.exe --model models/denoiser_causal.onnx --system-audio --capture-device 0 --playback-device 1
```

Replace the indexes with the listed cable capture endpoint and real speaker/headphone playback endpoint. Install/configure the virtual cable separately and set Windows playback to its playback endpoint. The app downmixes to mono for the current model, so stereo separation is lost.


## Current model and measured results

- Exported ONNX model: `models/denoiser_causal.onnx`.
- An evaluation of the model reported baseline mean SI-SDR **5.31 dB**, enhanced mean SI-SDR **11.68 dB**, and mean improvement **+6.37 dB**.


## How the model works

Audio is mono float32 at 16 kHz. The DSP makes causal 512-sample STFT frames every 160 samples (10 ms), producing 257 bins. Log-power features enter a network consisting of `Linear(257,128)`, layer normalization, PReLU, a two-layer 128-unit GRU, and a linear output for real and imaginary complex-mask components. The complex mask is applied to each spectrum; inverse STFT and weighted overlap-add reconstruct the enhanced waveform while carrying GRU state between hops.

The learned mixtures use four-second LibriSpeech speech clips convolved with randomized reverberation and mixed with MS-SNSD noise at randomly sampled SNRs from -5 to +15 dB. Speech speakers and noise recordings are partitioned between train, validation, and test.
