#!/usr/bin/env python3
import sys
import os
import json
import logging
import traceback
import time
import whisper

# Configure logging to stdout for the subprocess
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def validate_audio_file(audio_path):
    """Validate that the audio file exists and is readable."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.access(audio_path, os.R_OK):
        raise PermissionError(f"Cannot read audio file (permission denied): {audio_path}")
    if os.path.getsize(audio_path) == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

def main():
    try:
        start_time = time.time()
        logger.info("Starting transcription worker...")
        
        if len(sys.argv) != 4:
            error_msg = f"Usage: {sys.argv[0]} <model_name> <audio_wav_path> <result_path>"
            logger.error(error_msg)
            print(json.dumps({"error": error_msg}), file=sys.stderr)
            sys.exit(1)
            
        model_name = sys.argv[1]
        audio_path = sys.argv[2]
        result_path = sys.argv[3]
        
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Input audio: {audio_path}")
        logger.info(f"Output result: {result_path}")
        
        # Validate input file
        try:
            validate_audio_file(audio_path)
            logger.info(f"Audio file validation passed: {os.path.getsize(audio_path)} bytes")
        except Exception as e:
            error_msg = f"Audio file validation failed: {str(e)}"
            logger.error(error_msg)
            with open(result_path, 'w') as f:
                json.dump({"error": error_msg}, f)
            sys.exit(2)
        
        # Load audio
        logger.info("Loading audio file...")
        try:
            audio_data = whisper.load_audio(audio_path)
            logger.info(f"Audio loaded, shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        except Exception as e:
            error_msg = f"Failed to load audio: {str(e)}"
            logger.error(error_msg)
            with open(result_path, 'w') as f:
                json.dump({"error": error_msg, "traceback": traceback.format_exc()}, f)
            sys.exit(3)
        
        # Load model
        logger.info(f"Loading Whisper model '{model_name}'...")
        try:
            model = whisper.load_model(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            with open(result_path, 'w') as f:
                json.dump({"error": error_msg, "traceback": traceback.format_exc()}, f)
            sys.exit(4)
        
        # Transcribe
        logger.info("Starting transcription...")
        try:
            result = model.transcribe(audio_data, fp16=False)
            logger.info("Transcription completed successfully")
            
            # Ensure the result contains the expected fields
            if not isinstance(result, dict) or "text" not in result:
                raise ValueError("Unexpected result format from Whisper model")
                
            # Write result
            with open(result_path, 'w') as f:
                output_json = {
                    "status": "success",
                    "text": result.get("text", ""), # Ensure text key exists
                    "data": result # Store the full original whisper result
                }
                json.dump(output_json, f)
            
            elapsed = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            with open(result_path, 'w') as f:
                json.dump({"error": error_msg, "traceback": traceback.format_exc()}, f)
            sys.exit(5)
            
    except Exception as e:
        error_msg = f"Unexpected error in transcription worker: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        try:
            with open(result_path, 'w') as f:
                json.dump({"error": error_msg, "traceback": traceback.format_exc()}, f)
        except:
            # If we can't write to the result file, at least print the error
            print(json.dumps({"error": error_msg, "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(99)
    
    logger.info("Worker process completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main()
