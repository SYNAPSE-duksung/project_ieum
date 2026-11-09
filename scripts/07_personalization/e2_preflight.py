"""Multi-batch loss/encoder-gradient diagnostics; never trains or finalizes."""
import argparse
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import tempfile

import soundfile as sf
import torch
from transformers import set_seed

from e2_train import ROOT, load_config, project_path, load_data, general_identity, lambda_directory
from src.personalization.e2_trainer import build_e2_model_and_processor, make_e2_trainer
from src.personalization.ctc_utils import E2Collator, encode_ctc, wav_lengths


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure(model, processor, dataset, indices, device, precision, weights):
    batch = E2Collator(processor)([dataset[i] for i in indices])
    batch = {k: v.to(device) for k, v in batch.items()}
    model.to(device).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    seen = {}
    def inspect_ctc(module, inputs):
        seen['ctc_log_probs_dtype'] = str(inputs[0].dtype)
    handle = model.ctc_loss_fn.register_forward_pre_hook(inspect_ctc)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        output = model(**batch)
    handle.remove()
    params = [p for p in model.model.encoder.parameters() if p.requires_grad]
    def gradient_norm(loss, retain_graph):
        gradients = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
        if not all(bool(torch.isfinite(g).all()) for g in gradients if g is not None):
            raise FloatingPointError('Non-finite diagnostic encoder gradient')
        norm = sum(float(g.detach().float().norm()) ** 2 for g in gradients if g is not None) ** .5
        del gradients
        return norm
    asr = gradient_norm(output.asr_loss, True)
    ctc = gradient_norm(output.ctc_loss, False)
    result = dict(indices=indices, texts=[dataset.rows[i][dataset.text_col] for i in indices],
                  device=device, precision=precision, T=batch['ctc_input_lengths'].tolist(),
                  U=batch['ctc_target_lengths'].tolist(),
                  asr_loss=float(output.asr_loss.detach()), ctc_loss=float(output.ctc_loss.detach()),
                  asr_encoder_l2=asr, ctc_encoder_l2=ctc,
                  weighted_ctc_over_asr={str(w): w * ctc / asr for w in weights},
                  finite=True, **seen)
    if device == 'cuda':
        result.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del output, batch
    model.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/e2.yaml')
    parser.add_argument('--speaker', default='KSM_M_45')
    parser.add_argument('--cuda-check', action='store_true', help='also check BF16 GPU batches; no optimizer')
    parser.add_argument('--report', type=Path, help='new JSON report; existing files are refused')
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(4)
    config = load_config(args.config)
    seed = int(config['training']['seed'])
    weights = config['e2']['lambda_ctc_values']
    identity = general_identity(config['paths']['general_model'])
    initialization = []
    for weight in weights:
        set_seed(seed)
        model, processor = build_e2_model_and_processor(identity['path'], weight)
        initialization.append(dict(lambda_ctc=weight,
                                   backbone_sha256=tensor_digest(model.model.state_dict().items()),
                                   head_sha256=tensor_digest(model.ctc_head.state_dict().items()),
                                   run_directory=str(project_path(config['e2']['output_root']) / args.speaker / lambda_directory(weight))))
        if weight != weights[-1]:
            del model
            gc.collect()
    assert len({r['backbone_sha256'] for r in initialization}) == 1
    assert len({r['head_sha256'] for r in initialization}) == 1
    with contextlib.redirect_stdout(io.StringIO()):
        rows, datasets, preflight = load_data(config, args.speaker, processor, model.model.encoder)
    metadata = []
    for index, row in enumerate(rows['train']):
        info = sf.info(row['_resolved_audio'])
        _, _, t = wav_lengths(info.frames, info.samplerate, processor.feature_extractor, model.model.encoder)
        u = len(encode_ctc(row[config['columns']['text']]))
        metadata.append(dict(index=index, seconds=info.duration, T=t, U=u, ratio=t/u))
    selections = {}
    for name, key, reverse in [('short', 'seconds', False), ('long', 'seconds', True),
                               ('low_T_U', 'ratio', False), ('high_T_U', 'ratio', True)]:
        selections[name] = sorted(metadata, key=lambda r: (r[key], r['index']), reverse=reverse)[:2]
    results = []
    for name, selected in selections.items():
        item = measure(model, processor, datasets['train'], [r['index'] for r in selected], 'cpu', 'fp32', weights)
        item.update(selection=name, metadata=selected)
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    cuda = dict(available=torch.cuda.is_available(), torch_version=torch.__version__, runtime=torch.version.cuda)
    if cuda['available']:
        cuda.update(name=torch.cuda.get_device_name(), bf16_supported=torch.cuda.is_bf16_supported(),
                    total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    if args.cuda_check:
        if not cuda['available'] or not cuda['bf16_supported']:
            raise RuntimeError('CUDA BF16 unavailable')
        model.to('cuda')
        for name, selected in selections.items():
            item = measure(model, processor, datasets['train'], [r['index'] for r in selected], 'cuda', 'bf16', weights)
            item.update(selection=name, metadata=selected)
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        with tempfile.TemporaryDirectory(prefix='ieum_e2_args_') as temporary:
            trainer = make_e2_trainer(model, processor, datasets, temporary, config['training'])
            cuda['trainer'] = dict(device=str(trainer.args.device), bf16=trainer.args.bf16,
                                   fp16=trainer.args.fp16, metric_for_best_model=trainer.args.metric_for_best_model,
                                   load_best_model_at_end=trainer.args.load_best_model_at_end,
                                   eval_strategy=str(trainer.args.eval_strategy), save_strategy=str(trainer.args.save_strategy))
    report = dict(general_checkpoint=identity, independent_initialization=initialization,
                  data_splits=preflight['splits'], cuda=cuda, batches=results,
                  optimizer_steps=0, test_evaluated=False, finalized=False,
                  norm_scope='all trainable Encoder parameters; before clipping/accumulation scaling')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(cuda=cuda, independent_initialization=True, optimizer_steps=0,
                         report=str(args.report)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
