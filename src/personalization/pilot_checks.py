"""Read-only data preflight and isolated real-batch regression probes."""
import gc
import math
from pathlib import Path
import time
import torch
from transformers import set_seed


def check(mode, speakers):
    import pilot_runner as p
    cfg = p.config()
    directory = p.ROOT/'reports/pilot_runner_setup_20261107'
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    model, processor = p.build_model_and_processor(cfg['paths']['general_model'])
    report = dict(mode=mode, speakers=[], cuda=torch.cuda.is_available(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  bf16=torch.cuda.is_bf16_supported(), storage=p.storage(p.ROOT/'outputs/personalization/pilot_e1_e3_v1'))
    fields = ['learning_rate', 'lr_scheduler_type', 'optim', 'adam_beta1', 'adam_beta2', 'adam_epsilon',
              'weight_decay', 'max_grad_norm', 'warmup_steps', 'num_train_epochs',
              'per_device_train_batch_size', 'per_device_eval_batch_size', 'gradient_accumulation_steps',
              'seed', 'data_seed', 'bf16', 'fp16', 'gradient_checkpointing', 'generation_max_length',
              'generation_num_beams', 'predict_with_generate', 'metric_for_best_model', 'greater_is_better']
    for speaker in speakers:
        rows, joined, candidates = p.data(speaker, processor, cfg)
        trainers = []
        for exp in ('E1', 'E3'):
            audit = p.E3Audit(directory/speaker/exp)
            t = p.trainer(exp, model, processor, p.datasets(rows, joined, processor, cfg, exp),
                          directory/speaker/exp/'unused_checkpoints', cfg, audit)
            trainers.append(t)
        left, right = trainers
        differences = {k: [str(getattr(left.args,k)),str(getattr(right.args,k))] for k in fields
                       if getattr(left.args,k) != getattr(right.args,k)}
        if differences:
            raise AssertionError(differences)
        report['speakers'].append(dict(speaker=speaker, splits={k:len(v) for k,v in rows.items()},
                                       candidate_counts=candidates['counts'], argument_differences=differences))
        # Remove the E1 observation hook before another trainer shares this model.
        model._forward_hooks.clear(); model._forward_hooks_with_kwargs.clear()
        del trainers, left, right, t
    report['matched_arguments'] = fields
    if mode == 'smoke':
        del model
        gc.collect(); torch.cuda.empty_cache()
        rows, joined, candidate_report = p.data('YHS_F_26', processor, cfg)
        assert candidate_report['counts'].get('eligible',0) == 0
        gradients = None
        probes = []
        for label, exp in (('E1','E1'), ('E1_repeat','E1'), ('E3','E3')):
            set_seed(42)
            model, processor = p.build_model_and_processor(cfg['paths']['general_model'])
            audit = p.E3Audit(directory/'zero_candidate'/exp)
            ds = p.datasets(rows, joined, processor, cfg, exp)
            t = p.trainer(exp, model, processor, ds, directory/'zero_candidate'/exp/'unused', cfg, audit)
            t.current_gradient_accumulation_steps = 4
            model.gradient_checkpointing_enable()
            model.train()
            batch = t.data_collator([ds['train'][i] for i in range(2)])
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            # Outside train(), Accelerate has not installed its autocast wrapper.
            # Apply the configured CUDA precision explicitly for this probe.
            with torch.autocast('cuda', dtype=torch.bfloat16 if t.args.bf16 else torch.float16,
                                enabled=t.args.bf16 or t.args.fp16):
                loss = t.training_step(model, t._prepare_inputs(batch), num_items_in_batch=None)
            current = {n:v.grad.detach().float().cpu() for n,v in model.named_parameters() if v.grad is not None}
            norm = math.sqrt(sum(float(v.double().square().sum()) for v in current.values()))
            probe = dict(experiment=label, loss=float(loss), gradient_l2=norm,
                         seconds=time.perf_counter()-started,
                         gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         model_accepts_loss_kwargs=t.model_accepts_loss_kwargs,
                         accelerate_accumulation=t.accelerator.gradient_accumulation_steps,
                         explicit_autocast='bf16' if t.args.bf16 else 'fp16' if t.args.fp16 else 'fp32',
                         finite=all(bool(torch.isfinite(v).all()) for v in current.values()))
            if gradients is None:
                gradients = current
            else:
                difference = math.sqrt(sum(float((current[n]-v).double().square().sum()) for n,v in gradients.items()))
                probe.update(correction_loss=t.last_components['correction_loss'] if exp == 'E3' else 0.,
                             gradient_difference_l2=difference, relative_gradient_difference=difference/probes[0]['gradient_l2'])
                assert probe['correction_loss'] == 0
                assert probe['relative_gradient_difference'] < .02
                assert abs(probe['loss']-probes[0]['loss']) < 1e-4
            assert probe['finite']
            probes.append(probe)
            del t, model, current, batch, ds
            gc.collect(); torch.cuda.empty_cache()
        report['zero_candidate_probes'] = probes
        report['scope'] = 'two real Train utterances; backward only; no optimizer step, full training or Test evaluation'
    p.write(directory/f'{mode}.json', report)
    print(str(directory/f'{mode}.json'))
