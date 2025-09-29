from TTS.api import TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file("Hello from Coqui!", file_path="test.wav")