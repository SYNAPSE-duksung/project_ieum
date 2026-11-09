"""Decode uploaded audio once and generate raw ASR text with fixed E1 settings."""
import io
import librosa
import numpy as np
import soundfile as sf
import torch
from .model_loader import SPEAKER

class ASRInference:
    def __init__(self, model_loader):
        self.loader=model_loader

    def decode_audio(self, audio_bytes):
        if not audio_bytes:
            raise ValueError('Empty audio file.')
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                if f.samplerate<=0 or f.frames==0 or f.frames/f.samplerate>30:
                    raise ValueError('Audio must be non-empty and no longer than 30 seconds.')
                audio=f.read(dtype='float32',always_2d=True).mean(axis=1)
                sr=f.samplerate
        except (RuntimeError, sf.LibsndfileError) as exc:
            raise ValueError('Cannot decode audio. PCM WAV is recommended.') from exc
        if not np.isfinite(audio).all():
            raise ValueError('Audio contains NaN/Inf.')
        if sr!=16000:
            audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
        if len(audio)==0 or len(audio)>480000 or not np.isfinite(audio).all():
            raise ValueError('Invalid resampled audio length or values.')
        return audio

    def _generate(self, audio, name):
        if not self.loader.is_loaded:
            raise RuntimeError('ASR model is not ready.')
        extractor,tokenizer=self.loader.processors[name]
        features=extractor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode(), self.loader.autocast():
            ids=self.loader.models[name].generate(input_features=features.input_features.to(self.loader.device),attention_mask=features.attention_mask.to(self.loader.device))
        return tokenizer.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()

    def transcribe(self,audio_bytes,filename=None):
        audio=self.decode_audio(audio_bytes)
        with self.loader.lock:
            return self._generate(audio,'general')

    def transcribe_personalized(self,audio_bytes,speaker_id,filename=None):
        if speaker_id!=SPEAKER:
            raise ValueError('Unsupported personalized speaker.')
        audio=self.decode_audio(audio_bytes)
        with self.loader.lock:
            return self._generate(audio,'personalized')

    def transcribe_compare(self,audio_bytes,speaker_id,filename=None):
        if speaker_id!=SPEAKER:
            raise ValueError('Unsupported personalized speaker.')
        audio=self.decode_audio(audio_bytes)
        with self.loader.lock:
            general=self._generate(audio,'general')
            personalized=self._generate(audio,'personalized')
        return dict(speaker_id=speaker_id,filename=filename,duration_seconds=round(len(audio)/16000,3),general_text=general,personalized_text=personalized)
