"""Whisper Encoder-Decoder loader; no BiGRU, CTC or vocabulary adapter."""
import json
import logging
from contextlib import nullcontext
from threading import RLock
import torch
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast

LOG = logging.getLogger(__name__)
SPEAKER = 'KSM_M_45'

class ModelLoader:
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device('cuda' if settings.device=='auto' and torch.cuda.is_available() else 'cpu' if settings.device=='auto' else settings.device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        precision = settings.precision
        if precision=='auto':
            precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if self.device.type=='cuda' else 'fp32'
        if precision not in ('bf16','fp16','fp32') or (self.device.type!='cuda' and precision!='fp32'):
            raise ValueError('Use auto/fp32 on CPU or auto/bf16/fp16/fp32 on CUDA')
        if precision=='bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 is unsupported on this GPU')
        self.precision=precision
        self.models={}
        self.processors={}
        self.lock=RLock()

    @property
    def is_loaded(self):
        return set(self.models)=={'general','personalized'}

    def load(self):
        if self.is_loaded:
            return
        try:
            for name in ('general','personalized'):
                path=getattr(self.settings,name+'_model')
                processor_path=getattr(self.settings,name+'_processor')
                if not (path/'model.safetensors').is_file():
                    raise FileNotFoundError(f'{name} checkpoint missing: {path}')
                LOG.warning('Loading %s Whisper Small Encoder-Decoder: %s; processor=%s; device=%s; precision=%s',name,path,processor_path,self.device,self.precision)
                cfg=json.loads((processor_path/'processor_config.json').read_text(encoding='utf-8'))
                extractor=WhisperFeatureExtractor(**cfg['feature_extractor'])
                if extractor.sampling_rate!=16000 or extractor.feature_size!=80 or extractor.n_samples!=480000:
                    raise ValueError('Processor is incompatible with the E1 16kHz/80mel/30s setup')
                tokenizer=WhisperTokenizerFast.from_pretrained(processor_path,local_files_only=True)
                model=WhisperForConditionalGeneration.from_pretrained(path,local_files_only=True)
                if model.config.model_type!='whisper' or model.config.d_model!=768:
                    raise ValueError('Expected Whisper Small Encoder-Decoder')
                model.generation_config.language='korean'
                model.generation_config.task='transcribe'
                model.generation_config.forced_decoder_ids=None
                model.generation_config.max_length=160
                model.generation_config.num_beams=1
                model.generation_config.do_sample=False
                model.eval().requires_grad_(False).to(self.device)
                self.models[name]=model
                self.processors[name]=(extractor,tokenizer)
        except Exception:
            self.close()
            raise

    def autocast(self):
        return nullcontext() if self.precision=='fp32' else torch.autocast('cuda',dtype=torch.bfloat16 if self.precision=='bf16' else torch.float16)

    def close(self):
        self.models.clear()
        self.processors.clear()
        if self.device.type=='cuda':
            torch.cuda.empty_cache()
