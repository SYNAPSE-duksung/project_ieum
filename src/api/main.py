"""E1 demo server. API contracts selectively reused from integration 07f55d1."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from .config import load_settings
from .model_loader import ModelLoader, SPEAKER
from .whisper_inference import ASRInference
from src.tts.piper_tts import PiperTTS

LOG=logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str=Field(max_length=2000)

def create_app(settings=None, loader_factory=ModelLoader):
    @asynccontextmanager
    async def lifespan(app):
        cfg=settings or load_settings()
        loader=loader_factory(cfg)
        app.state.loader=loader
        try:
            await run_in_threadpool(loader.load)
            app.state.inference=ASRInference(loader)
            app.state.tts=PiperTTS(cfg)
            app.state.settings=cfg
            yield
        finally:
            loader.close()

    app=FastAPI(title='IEUM E1 Whisper Small ASR + TTS API',version='1.0.0',lifespan=lifespan)

    @app.get('/health')
    def health():
        loader=getattr(app.state,'loader',None)
        ready=bool(loader and loader.is_loaded)
        return dict(status='ok' if ready else 'not_ready',general_model_loaded=ready,personalized_speaker_loaded=SPEAKER if ready else None,supported_speakers=[SPEAKER],tts_ready=bool(getattr(app.state,'tts',None) and app.state.tts.ready))

    async def read_audio(audio):
        if not audio.filename or not audio.filename.lower().endswith(('.wav','.mp3','.m4a','.flac')):
            raise HTTPException(400,'Unsupported or missing audio filename.')
        content=await audio.read(app.state.settings.max_upload_bytes+1)
        if not content:
            raise HTTPException(400,'Empty audio file.')
        if len(content)>app.state.settings.max_upload_bytes:
            raise HTTPException(413,'Audio upload is too large.')
        return content

    def speaker_check(speaker):
        if speaker!=SPEAKER:
            raise HTTPException(400,dict(message='Unsupported personalized speaker.',requested_speaker=speaker,supported_speakers=[SPEAKER]))

    async def infer(method,content,filename,speaker=None):
        inference=getattr(app.state,'inference',None)
        if inference is None:
            raise HTTPException(503,'ASR model is not ready.')
        kwargs=dict(audio_bytes=content,filename=filename)
        if speaker is not None:
            kwargs['speaker_id']=speaker
        try:
            return await run_in_threadpool(getattr(inference,method),**kwargs)
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc
        except Exception as exc:
            LOG.exception('ASR inference failed')
            raise HTTPException(500,'ASR inference failed; inspect server logs.') from exc

    @app.post('/transcribe')
    async def transcribe(audio: UploadFile=File(...)):
        content=await read_audio(audio)
        return dict(filename=audio.filename,text=await infer('transcribe',content,audio.filename))

    @app.post('/transcribe/personalized')
    async def personalized(speaker_id: str=Form(...),audio: UploadFile=File(...)):
        speaker_check(speaker_id)
        content=await read_audio(audio)
        return dict(speaker_id=speaker_id,filename=audio.filename,text=await infer('transcribe_personalized',content,audio.filename,speaker_id))

    @app.post('/transcribe/compare')
    async def compare(speaker_id: str=Form(...),audio: UploadFile=File(...)):
        speaker_check(speaker_id)
        content=await read_audio(audio)
        return await infer('transcribe_compare',content,audio.filename,speaker_id)

    @app.post('/tts')
    async def tts(request:TTSRequest):
        if not request.text.strip():
            raise HTTPException(400,'TTS text is empty.')
        if not app.state.tts.ready:
            raise HTTPException(503,'Piper TTS is not configured on this server.')
        try:
            path=await run_in_threadpool(app.state.tts.synthesize,request.text)
        except Exception as exc:
            LOG.exception('Piper synthesis failed')
            raise HTTPException(500,'TTS synthesis failed; inspect server logs.') from exc
        return FileResponse(path,media_type='audio/wav',filename=path.name,background=BackgroundTask(path.unlink,missing_ok=True))

    return app

app=create_app()
