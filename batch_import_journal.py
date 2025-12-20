#!/usr/bin/env python3
"""
Batch import script for processing old audio recordings into journal entries.
Processes audio files through Whisper and creates journal entries with original file timestamps.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import whisper
import numpy as np
import soundfile as sf
import librosa
from journaling import JournalingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.m4a', '.mp3', '.flac', '.ogg', '.aac', '.wma'}

def load_audio_file(file_path: str) -> tuple:
    """
    Load an audio file and return the audio data and sample rate.

    Args:
        file_path: Path to the audio file

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Try using soundfile first
        audio_data, sample_rate = sf.read(file_path)
        return audio_data, sample_rate
    except Exception as e:
        logging.warning(f"soundfile failed for {file_path}, trying librosa: {e}")
        try:
            # Fallback to librosa which handles more formats
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            return None, None

def get_file_timestamp(file_path: str) -> datetime:
    """
    Parse timestamp from filename.
    Supports two formats:
    1. "NN - Audio Journal Recordings - MMM DD, YYYY HHMMSS AM.wav"
       Example: "00 - Audio Journal Recordings - Sep 27, 2018 83059 AM.wav"
    2. "NN - Audio Journal Recordings - MM.DD.YYYY HHMMAM.wav"
       Example: "00 - Audio Journal Recordings - 11.26.2022 0727AM.wav"

    Args:
        file_path: Path to the file

    Returns:
        datetime object representing the file's timestamp
    """
    filename = os.path.basename(file_path)

    try:
        # Remove the file extension
        name_without_ext = os.path.splitext(filename)[0]

        # Split by " - " to get parts
        parts = name_without_ext.split(' - ')

        if len(parts) >= 3:
            # The date/time part is everything after "Audio Journal Recordings - "
            datetime_part = parts[2].strip()

            # Try Format 2 first: "11.26.2022 0727AM" (MM.DD.YYYY HHMMAM with no space)
            # Check if the string ends with AM or PM (no space)
            if datetime_part.endswith('AM') or datetime_part.endswith('PM'):
                # Extract AM/PM
                am_pm = datetime_part[-2:]
                datetime_without_ampm = datetime_part[:-2]

                # Split by space to separate date and time
                parts_split = datetime_without_ampm.split()

                if len(parts_split) == 2:
                    date_part = parts_split[0]
                    time_part = parts_split[1]

                    # Check if date has dots (Format 2)
                    if '.' in date_part:
                        # Format 2: "11.26.2022 0727AM" or "12.24.2023 1732PM"
                        # Time is HHMM format (4 digits)
                        if len(time_part) == 4:
                            hour = int(time_part[0:2])
                            minute = int(time_part[2:4])

                            # Check if it's 24-hour format (hour > 12)
                            if hour > 12:
                                # It's 24-hour format, ignore AM/PM
                                formatted_time = f"{hour:02d}:{minute:02d}:00"
                                full_datetime_str = f"{date_part} {formatted_time}"
                                timestamp = datetime.strptime(full_datetime_str, "%m.%d.%Y %H:%M:%S")
                            else:
                                # It's 12-hour format, use AM/PM
                                formatted_time = f"{hour:02d}:{minute:02d}:00"
                                full_datetime_str = f"{date_part} {formatted_time} {am_pm}"
                                timestamp = datetime.strptime(full_datetime_str, "%m.%d.%Y %I:%M:%S %p")
                            return timestamp
                        elif len(time_part) == 3:
                            formatted_time = f"{time_part[0]}:{time_part[1:3]}:00"
                            full_datetime_str = f"{date_part} {formatted_time} {am_pm}"
                            timestamp = datetime.strptime(full_datetime_str, "%m.%d.%Y %I:%M:%S %p")
                            return timestamp
                        else:
                            raise ValueError(f"Unexpected time format: {time_part}")

            # Try Format 1: "Sep 27, 2018 83059 AM" (with space before AM/PM)
            if ' AM' in datetime_part or ' PM' in datetime_part:
                # Split by space to find AM/PM
                parts_split = datetime_part.split()

                # Find AM or PM
                am_pm = None
                time_part = None
                for i, part in enumerate(parts_split):
                    if part in ['AM', 'PM']:
                        am_pm = part
                        if i > 0:
                            time_part = parts_split[i-1]
                        break

                # The date is everything before the time part
                date_part = ' '.join(parts_split[:len(parts_split)-2])

                if time_part and am_pm and len(time_part) >= 5:
                    # Insert colons into time: "83059" -> "8:30:59"
                    # Format is HHMMSS or HMMSS
                    if len(time_part) == 6:
                        # HHMMSS format
                        formatted_time = f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    else:
                        # HMMSS format (single digit hour)
                        formatted_time = f"{time_part[0]}:{time_part[1:3]}:{time_part[3:5]}"

                    # Combine date and time
                    full_datetime_str = f"{date_part} {formatted_time} {am_pm}"

                    # Parse the datetime
                    timestamp = datetime.strptime(full_datetime_str, "%b %d, %Y %I:%M:%S %p")
                    return timestamp

        logging.warning(f"Could not parse timestamp from filename: {filename}, using file metadata")

    except Exception as e:
        logging.warning(f"Error parsing filename {filename}: {e}, using file metadata")

    # Fallback to file metadata if parsing fails
    stat = os.stat(file_path)
    if hasattr(stat, 'st_birthtime'):
        timestamp = stat.st_birthtime
    else:
        timestamp = stat.st_mtime
    return datetime.fromtimestamp(timestamp)

def find_audio_files(directory: str) -> list:
    """
    Find all audio files in a directory and return them sorted by timestamp (oldest first).

    Args:
        directory: Directory to search for audio files

    Returns:
        List of tuples (file_path, timestamp) sorted by timestamp
    """
    audio_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext in AUDIO_EXTENSIONS:
                try:
                    timestamp = get_file_timestamp(file_path)
                    audio_files.append((file_path, timestamp))
                except Exception as e:
                    logging.warning(f"Could not get timestamp for {file_path}: {e}")

    # Sort by timestamp (oldest first)
    audio_files.sort(key=lambda x: x[1])

    return audio_files

def process_audio_files(directory: str, ollama_model: str = "llama3"):
    """
    Process all audio files in a directory and create journal entries.

    Args:
        directory: Directory containing audio files
        ollama_model: Ollama model to use for summaries
    """
    # Find all audio files
    logging.info(f"Scanning directory: {directory}")
    audio_files = find_audio_files(directory)

    if not audio_files:
        logging.error(f"No audio files found in {directory}")
        return

    logging.info(f"Found {len(audio_files)} audio files to process")

    # Load Whisper model
    logging.info("Loading Whisper model (this may take a moment)...")
    try:
        model = whisper.load_model("base")
        logging.info("Whisper model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        return

    # Create journaling manager
    journal_manager = JournalingManager(ollama_model=ollama_model)

    # Process each file
    successful = 0
    failed = 0

    for idx, (file_path, file_timestamp) in enumerate(audio_files, 1):
        filename = os.path.basename(file_path)
        logging.info(f"\n[{idx}/{len(audio_files)}] Processing: {filename}")
        logging.info(f"  File timestamp: {file_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Load audio
            audio_data, sample_rate = load_audio_file(file_path)
            if audio_data is None:
                failed += 1
                continue

            # Check if audio is long enough (at least 0.1 seconds)
            min_samples = int(sample_rate * 0.1)
            if len(audio_data) < min_samples:
                logging.warning(f"  Audio too short ({len(audio_data)} samples = {len(audio_data)/sample_rate:.3f}s), skipping")
                failed += 1
                continue

            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                logging.info(f"  Resampling from {sample_rate}Hz to 16000Hz...")
                # Use kaiser_fast for faster resampling (good enough for speech)
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000, res_type='kaiser_fast')
                sample_rate = 16000

            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Ensure audio is float32 (Whisper requirement)
            audio_data = audio_data.astype(np.float32)

            # Transcribe with Whisper
            logging.info("  Transcribing...")
            result = model.transcribe(audio_data, fp16=False)
            transcription = result["text"].strip()

            if not transcription:
                logging.warning(f"  No transcription generated for {filename}")
                failed += 1
                continue

            logging.info(f"  Transcription: {transcription[:100]}...")

            # Create journal entry with custom timestamp
            logging.info("  Creating journal entry with Ollama summaries...")
            entry = journal_manager.create_journal_entry(
                transcription=transcription,
                audio_data=audio_data,
                sample_rate=sample_rate,
                custom_timestamp=file_timestamp
            )

            logging.info(f"  ✓ Successfully created entry for {filename}")
            successful += 1

        except Exception as e:
            logging.error(f"  ✗ Error processing {filename}: {e}")
            failed += 1
            continue

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"Batch processing complete!")
    logging.info(f"Successfully processed: {successful}/{len(audio_files)}")
    if failed > 0:
        logging.info(f"Failed: {failed}/{len(audio_files)}")
    logging.info(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Batch import audio files into WhisperNotes journal'
    )
    parser.add_argument(
        'directory',
        help='Directory containing audio files to process'
    )
    parser.add_argument(
        '--model',
        default='llama3',
        help='Ollama model to use for summaries (default: llama3)'
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        logging.error(f"Directory not found: {args.directory}")
        sys.exit(1)

    # Process files
    process_audio_files(args.directory, args.model)

if __name__ == "__main__":
    main()
