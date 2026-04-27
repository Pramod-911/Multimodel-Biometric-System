try:
    import librosa
    import soundfile
    print("[OK] Libraries imported")
    
    # Test loading the specific sample file
    file = '00:00:00:00-00:00:01:00-00:00:59:00.wav'
    print(f"Testing file: {file}")
    
    try:
        y, sr = librosa.load(file, sr=16000)
        print(f"[OK] Librosa loaded audio. Length: {len(y)}")
    except Exception as e:
        print(f"[FAIL] Librosa error: {e}")
        
    try:
        data, samplerate = soundfile.read(file)
        print(f"[OK] Soundfile loaded audio. Length: {len(data)}")
    except Exception as e:
        print(f"[FAIL] Soundfile error: {e}")

except Exception as e:
    print(f"[CRITICAL] Error: {e}")
