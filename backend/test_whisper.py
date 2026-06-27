from whisper_model import transcribe

audio_path = "../sample_audio/sample.wav"

text = transcribe(audio_path)

print("=" * 50)
print(text)
print("=" * 50)