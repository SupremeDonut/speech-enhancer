"""Export and numerically validate the fixed-shape streaming float graph."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from training import K, StreamingDenoiseNet, enhance

INPUT_NAMES = ['input_mag', 'h_in']
OUTPUT_NAMES = ['mask_r', 'mask_i', 'h_out']


def export(model, output):
    model.eval().cpu()
    x = torch.zeros(1, 1, K, dtype=torch.float32)
    h = torch.zeros(2, 1, 128, dtype=torch.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (x, h), str(output), input_names=INPUT_NAMES,
                      output_names=OUTPUT_NAMES, dynamic_axes=None,
                      opset_version=17, dynamo=False)
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    expected = {'input_mag': [1,1,K], 'h_in': [2,1,128],
                'mask_r': [1,1,K], 'mask_i': [1,1,K], 'h_out': [2,1,128]}
    values = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*graph.graph.input, *graph.graph.output)}
    if values != expected:
        raise ValueError(f'Unexpected ONNX I/O shape/type metadata: {values}')
    for value in (*graph.graph.input, *graph.graph.output):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f'{value.name} is not float32')
    return graph


def ort_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    actual_in = [(v.name, v.shape, v.type) for v in session.get_inputs()]
    actual_out = [(v.name, v.shape, v.type) for v in session.get_outputs()]
    if [x[0] for x in actual_in] != INPUT_NAMES or [x[0] for x in actual_out] != OUTPUT_NAMES:
        raise ValueError('Unexpected ONNX Runtime graph I/O names')
    if any(x[2] != 'tensor(float)' for x in actual_in + actual_out):
        raise ValueError('ONNX Runtime graph I/O must be float32')
    return session


def validate(model, path, seed=1729, hops=1000):
    model.eval().cpu()
    session = ort_session(path)
    rng = np.random.default_rng(seed)
    pt_state = torch.zeros(2,1,128)
    ort_state = np.zeros((2,1,128), dtype=np.float32)
    max_masks = max_state = 0.
    features = []
    with torch.no_grad():
        for _ in range(hops):
            frame = rng.normal(size=(1,1,K)).astype(np.float32)
            features.append(frame)
            pt_r, pt_i, pt_state = model(torch.from_numpy(frame), pt_state)
            ort_r, ort_i, ort_h = session.run(OUTPUT_NAMES,
                {'input_mag': frame, 'h_in': ort_state})
            max_masks = max(max_masks, float(np.max(np.abs(pt_r.numpy()-ort_r))),
                            float(np.max(np.abs(pt_i.numpy()-ort_i))))
            max_state = max(max_state, float(np.max(np.abs(pt_state.numpy()-ort_h))))
            ort_state = ort_h
    if max(max_masks, max_state) > 1e-4:
        raise AssertionError(f'float parity failed: mask={max_masks}, state={max_state}')

    # Convert deterministic sequence into plausible complex STFT frames and compare
    # aligned waveforms from PyTorch and one-hop-at-a-time ORT inference.
    from training import H, N, reconstruct, windows
    gen = torch.Generator().manual_seed(seed + 1)
    spectrum = torch.complex(torch.randn(1,hops,K,generator=gen),
                             torch.randn(1,hops,K,generator=gen))
    spectrum[...,0] = spectrum[...,0].real
    spectrum[...,-1] = spectrum[...,-1].real
    feature_tensor = torch.from_numpy(np.concatenate(features, axis=1))
    pt_h = torch.zeros(2,1,128)
    pt_rs, pt_is = [], []
    ort_h = np.zeros((2,1,128), np.float32)
    ort_rs, ort_is = [], []
    for frame in feature_tensor.split(1, dim=1):
        with torch.no_grad():
            real, imag, pt_h = model(frame, pt_h)
        pt_rs.append(real); pt_is.append(imag)
        real, imag, ort_h = session.run(OUTPUT_NAMES,
            {'input_mag': frame.numpy(), 'h_in': ort_h})
        ort_rs.append(torch.from_numpy(real)); ort_is.append(torch.from_numpy(imag))
    analysis, synthesis = windows(torch.device('cpu'), torch.float32)
    del analysis
    out_len = (hops-1)*H
    pt_audio = reconstruct(spectrum*torch.complex(torch.cat(pt_rs,1),torch.cat(pt_is,1)), synthesis, out_len)
    ort_audio = reconstruct(spectrum*torch.complex(torch.cat(ort_rs,1),torch.cat(ort_is,1)), synthesis, out_len)
    waveform_error = float(torch.max(torch.abs(pt_audio-ort_audio)))
    if waveform_error > 1e-4:
        raise AssertionError(f'aligned waveform parity failed: {waveform_error}')
    return {'hops': hops, 'seed': seed, 'max_mask_abs_error': max_masks,
            'max_state_abs_error': max_state, 'aligned_waveform_max_abs_error': waveform_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/denoiser_causal.onnx'))
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = StreamingDenoiseNet()
    model.load_state_dict(checkpoint['model'])
    export(model, args.output)
    report = validate(model, args.output, args.seed)
    report['artifact'] = str(args.output)
    args.output.with_suffix('.parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
