"""Serializable Whisper + auxiliary CTC head; decoder-only generation."""
from dataclasses import dataclass

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from .ctc_utils import BLANK_ID, VOCAB_SIZE, VOCAB_VERSION, encoder_output_lengths

CTC_KEYS = ('ctc_targets', 'ctc_target_lengths', 'ctc_input_lengths')


@dataclass
class E2Output(Seq2SeqLMOutput):
    asr_loss: torch.Tensor | None = None
    ctc_loss: torch.Tensor | None = None
    weighted_ctc_loss: torch.Tensor | None = None


class WhisperWithAuxiliaryCTC(WhisperForConditionalGeneration):
    # Trainer must apply ordinary microbatch gradient-accumulation scaling.
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        if config.d_model != 768:
            raise ValueError(f'E2 requires Whisper Small hidden size 768, got {config.d_model}')
        if getattr(config, 'ctc_vocab_size', VOCAB_SIZE) != VOCAB_SIZE:
            raise ValueError('Incompatible CTC vocabulary size')
        if getattr(config, 'ctc_vocab_version', VOCAB_VERSION) != VOCAB_VERSION:
            raise ValueError('Incompatible CTC vocabulary mapping')
        config.ctc_vocab_size = VOCAB_SIZE
        config.ctc_vocab_version = VOCAB_VERSION
        config.ctc_blank_id = BLANK_ID
        config.lambda_ctc = float(getattr(config, 'lambda_ctc', 0.1))
        if config.lambda_ctc < 0:
            raise ValueError('lambda_ctc must be nonnegative')
        self.ctc_head = nn.Linear(config.d_model, VOCAB_SIZE)
        self._init_weights(self.ctc_head)
        self.ctc_loss_fn = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    def forward(self, input_features=None, attention_mask=None, labels=None,
                ctc_targets=None, ctc_target_lengths=None, ctc_input_lengths=None,
                **kwargs):
        return_dict = kwargs.pop('return_dict', True)
        kwargs.pop('num_items_in_batch', None)
        present = [x is not None for x in (ctc_targets, ctc_target_lengths, ctc_input_lengths)]
        if any(present) and not all(present):
            raise ValueError('All three CTC tensors are required together')
        if all(present) and labels is None:
            raise ValueError('Auxiliary CTC loss requires decoder labels too')
        if labels is not None:
            if not all(present):
                raise ValueError('E2 supervised forward requires CTC targets and lengths')
            kwargs['use_cache'] = False
        output = super().forward(input_features=input_features, attention_mask=attention_mask,
                                 labels=labels, return_dict=True, **kwargs)
        if not all(present):
            return output if return_dict else output.to_tuple()
        hidden = output.encoder_last_hidden_state
        lengths = ctc_input_lengths.to(dtype=torch.long)
        target_lengths = ctc_target_lengths.to(dtype=torch.long)
        if (lengths.ndim != 1 or target_lengths.shape != lengths.shape
                or lengths.numel() != hidden.shape[0] or ctc_targets.ndim != 1
                or int(target_lengths.sum()) != ctc_targets.numel()):
            raise ValueError('Inconsistent CTC batch shapes')
        if ((lengths <= 0).any() or (lengths > hidden.shape[1]).any()
                or (target_lengths <= 0).any()
                or (ctc_targets <= BLANK_ID).any() or (ctc_targets >= VOCAB_SIZE).any()):
            raise ValueError('Invalid CTC IDs or lengths')
        if attention_mask is not None:
            expected = encoder_output_lengths(attention_mask.sum(-1), self.model.encoder)
            if not torch.equal(lengths.to(expected.device), expected):
                raise ValueError('CTC input lengths disagree with actual feature attention mask')
        offset = 0
        for count, available in zip(target_lengths.tolist(), lengths.tolist()):
            target = ctc_targets[offset:offset + count]
            repeats = int((target[1:] == target[:-1]).sum())
            if count + repeats > available:
                raise ValueError('CTC alignment impossible including adjacent repeats')
            offset += count
        # Crop padded encoder tail before the large 11183-way projection.
        logits = self.ctc_head(hidden[:, :int(lengths.max())])
        # Float32 log-softmax/CTC even under fp16/bf16 autocast.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probs = logits.float().log_softmax(-1).transpose(0, 1).contiguous()
            ctc_loss = self.ctc_loss_fn(log_probs, ctc_targets,
                                        lengths.cpu(), target_lengths.cpu())
        weighted = self.config.lambda_ctc * ctc_loss
        total = output.loss + weighted
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite E2 loss')
        values = dict(output)
        values['loss'] = total
        result = E2Output(**values, asr_loss=output.loss, ctc_loss=ctc_loss,
                          weighted_ctc_loss=weighted)
        return result if return_dict else result.to_tuple()

    def generate(self, *args, **kwargs):
        # Seq2SeqTrainer may forward a complete supervised batch to generate.
        # Never send CTC targets, lengths, or decoder references into generation.
        for key in (*CTC_KEYS, 'labels', 'num_items_in_batch'):
            kwargs.pop(key, None)
        return super().generate(*args, **kwargs)
