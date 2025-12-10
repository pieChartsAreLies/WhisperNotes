import multiprocessing

def transcribe_audio(audio_path):
    import whisper
    print("[Subprocess] Loading model...")
    model = whisper.load_model("base")
    print("[Subprocess] Model loaded. Transcribing...")
    result = model.transcribe(audio_path, fp16=False)
    print("[Subprocess] Transcription done.")
    return result["text"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python whisper_mp_test.py <audio_file>")
        sys.exit(1)
    audio_path = sys.argv[1]
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        result = pool.apply(transcribe_audio, (audio_path,))
        print("Transcription Result:", result)
