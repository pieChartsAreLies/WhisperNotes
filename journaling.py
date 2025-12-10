#!/usr/bin/env python3
"""
Journaling module for WhisperNotes application.
Handles saving transcriptions with timestamps to a markdown file.
"""
import os
import logging
import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import soundfile as sf
import ollama

class JournalingManager:
    """
    Handles journaling functionality including saving transcriptions and audio recordings.
    Integrates with Ollama for text summarization and formatting.
    """
    
    def __init__(self, output_dir: Optional[str] = None, summary_prompt: Optional[str] = None, ollama_model: Optional[str] = None):
        """
        Initialize the JournalingManager.

        Args:
            output_dir: Directory to store journal entries and audio files.
                       If None, uses '~/Documents/Personal/Audio Journal/'.
            summary_prompt: Custom prompt to use for generating summaries
            ollama_model: Ollama model to use for AI processing. If None, uses 'llama3'.
        """
        # Use the specified directory or default to ~/Documents/Personal/Audio Journal/
        if output_dir is None:
            home_dir = os.path.expanduser("~")
            self.output_dir = os.path.join(home_dir, "Documents", "Personal", "Audio Journal")
        else:
            self.output_dir = output_dir
            
        # Set default prompts
        self.default_summary_prompt = "Summarize the following in 1-2 sentences from a first-person perspective, as if you are the person who said it:"
        self.summary_prompt = summary_prompt if summary_prompt else self.default_summary_prompt
        
        # Default formatting prompt
        self.default_format_prompt = "Add paragraph breaks and fix punctuation only. Do NOT change, rephrase, or rewrite any words. Keep the exact original wording:"
        self.format_prompt = None  # Will be loaded from settings if available
            
        # Ensure directories exist
        self.ensure_directory_exists(self.output_dir)
        self.recordings_dir = os.path.join(self.output_dir, "recordings")
        self.ensure_directory_exists(self.recordings_dir)
        self.entries_dir = os.path.join(self.output_dir, "entries")
        self.ensure_directory_exists(self.entries_dir)
        
        # Journal file path
        self.journal_file = os.path.join(self.output_dir, "Journal.md")
        
        # Create journal file if it doesn't exist
        if not os.path.exists(self.journal_file):
            with open(self.journal_file, 'w', encoding='utf-8') as f:
                f.write("# Audio Journal\n\n")
                f.write("*Created on {}*\n\n".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
        
        # Configure Ollama
        self.ollama_model = ollama_model if ollama_model else "llama3"  # Use provided model or default to llama3
        logging.info(f"Using Ollama model: {self.ollama_model}")
        self.ollama_available = self._check_ollama_availability()
        if not self.ollama_available:
            logging.warning("Ollama is not available or no models are installed. Summaries will not be generated.")
            
    def _check_ollama_availability(self) -> bool:
        """
        Check if Ollama is available and has models installed.
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            # Try to list models
            models = ollama.list()
            if not models.get('models'):
                logging.warning("No Ollama models found. Please install models with 'ollama pull llama3'")
                return False
            return True
        except Exception as e:
            logging.error(f"Error checking Ollama availability: {e}")
            return False
    
    def ensure_directory_exists(self, directory: str) -> None:
        """Ensure the specified directory exists, create it if it doesn't."""
        os.makedirs(directory, exist_ok=True)
    
    def save_audio(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Save audio data to a file and return the file path.
        
        Args:
            audio_data: Raw audio data as numpy array
            sample_rate: Audio sample rate
            
        Returns:
            str: Path to the saved audio file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(self.recordings_dir, filename)
        
        try:
            # Convert bytes to numpy array if needed
            import numpy as np
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            # Save audio file
            sf.write(filepath, audio_data, sample_rate)
            logging.info(f"Audio saved to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving audio file: {e}")
            return ""
    
    def set_summary_prompt(self, prompt: str) -> None:
        """
        Set a custom prompt for generating summaries.
        
        Args:
            prompt: The custom prompt to use
        """
        self.summary_prompt = prompt
        logging.info(f"Summary prompt updated: {prompt[:50]}..." if len(prompt) > 50 else f"Summary prompt updated: {prompt}")
        
    def set_format_prompt(self, prompt: str) -> None:
        """
        Set a custom prompt for formatting transcriptions.
        
        Args:
            prompt: The custom prompt to use
        """
        self.format_prompt = prompt
        logging.info(f"Format prompt updated: {prompt[:50]}..." if len(prompt) > 50 else f"Format prompt updated: {prompt}")
    
    def _clean_ai_response(self, response: str) -> str:
        """
        Clean up common AI response preambles and formatting.

        Args:
            response: The raw AI response

        Returns:
            Cleaned response text
        """
        # Common preambles to remove
        preambles = [
            "Here is a 1-2 sentence summary of the text:",
            "Here is a summary of the text:",
            "Here is the summary:",
            "Here's a 1-2 sentence summary:",
            "Here's the summary:",
            "Summary:",
            "Here is a formatted version:",
            "Here is the formatted text:",
            "Here's the formatted version:",
        ]

        cleaned = response.strip()

        # Remove preambles (case-insensitive)
        for preamble in preambles:
            if cleaned.lower().startswith(preamble.lower()):
                cleaned = cleaned[len(preamble):].strip()
                break

        return cleaned

    def process_with_ollama(self, text: str) -> Tuple[str, str]:
        """
        Process text with Ollama to get a summary and formatted version.

        Args:
            text: The raw transcription text

        Returns:
            Tuple containing (summary, formatted_text)
        """
        # If Ollama is not available, return the original text
        if not hasattr(self, 'ollama_available') or not self.ollama_available:
            logging.warning("Ollama not available. Using original text without processing.")
            return f"Transcription: {text[:50]}..." if len(text) > 50 else f"Transcription: {text}", text

        try:
            # Get summary from Ollama using system message for better control
            summary_response = ollama.chat(model=self.ollama_model, messages=[
                {
                    "role": "system",
                    "content": "You are a concise summarizer. Provide only the summary without any preamble, introduction, or meta-commentary."
                },
                {
                    "role": "user",
                    "content": f"{self.summary_prompt}\n\n{text}"
                }
            ])
            summary = self._clean_ai_response(summary_response['message']['content'])

            # Get formatted text from Ollama using system message
            if self.format_prompt:
                format_prompt_text = self.format_prompt
            else:
                format_prompt_text = self.default_format_prompt

            format_response = ollama.chat(model=self.ollama_model, messages=[
                {
                    "role": "system",
                    "content": "You are a minimal text formatter. ONLY add paragraph breaks and fix punctuation. Do NOT change, rephrase, rewrite, or paraphrase any words. Preserve the exact original wording."
                },
                {
                    "role": "user",
                    "content": f"{format_prompt_text}\n\n{text}"
                }
            ])
            formatted_text = self._clean_ai_response(format_response['message']['content'])

            logging.info("Successfully processed text with Ollama")
            return summary, formatted_text
        except Exception as e:
            logging.error(f"Error processing text with Ollama: {e}")
            # Create a simple summary as fallback
            simple_summary = f"Transcription: {text[:50]}..." if len(text) > 50 else f"Transcription: {text}"
            return simple_summary, text  # Return original text if processing fails
    
    def create_journal_entry(self, transcription: str, audio_data=None, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Create a new journal entry with optional audio.
        
        Args:
            transcription: The transcribed text
            audio_data: Optional raw audio data
            sample_rate: Audio sample rate
            
        Returns:
            Dict containing entry metadata
        """
        timestamp = datetime.datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        entry_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Process text with Ollama
        summary, formatted_text = self.process_with_ollama(transcription)
        
        # Save audio if provided
        audio_path = None
        relative_audio_path = None
        if audio_data is not None:
            audio_path = self.save_audio(audio_data, sample_rate)
            if audio_path:
                relative_audio_path = os.path.join("recordings", os.path.basename(audio_path))
        
        # Create entry data
        entry = {
            "id": entry_id,
            "date": date_str,
            "timestamp": timestamp_str,
            "transcription": transcription,
            "summary": summary,
            "formatted_text": formatted_text,
            "audio_file": audio_path,
            "relative_audio_path": relative_audio_path
        }
        
        # Save detailed entry markdown file
        entry_file_path = self._save_entry_file(entry)
        entry["entry_file"] = entry_file_path
        
        # The entry_link will be set in _save_entry_file method
        
        # Save to main journal file
        self._save_markdown_entry(entry)
        
        return entry
    
    def _save_entry_file(self, entry: Dict[str, Any]) -> str:
        """
        Save a detailed journal entry to its own markdown file.
        
        Args:
            entry: The journal entry data
            
        Returns:
            str: Path to the saved entry file
        """
        try:
            # Parse the timestamp to create a formatted time string
            timestamp_obj = datetime.datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S")
            time_str = timestamp_obj.strftime("%I:%M:%S %p")
            
            # Create filename based on date and time
            entry_filename = f"{entry['date']} - {time_str} - Audio Journal Entry.md"
            entry_path = os.path.join(self.entries_dir, entry_filename)
            
            with open(entry_path, 'w', encoding='utf-8') as f:
                f.write(f"# Audio Journal Entry - {entry['date']} {time_str}\n\n")
                
                f.write("### Summary\n")
                f.write(f"{entry['summary']}\n\n")
                
                f.write("### Transcript\n")
                f.write(f"{entry['formatted_text']}\n\n")
                
                # Add link to audio recording if available
                if entry.get('relative_audio_path'):
                    f.write(f"🔊 [Listen to recording]({entry['relative_audio_path']})\n\n")
            
            logging.info(f"Detailed journal entry saved to {entry_path}")
            return entry_path
        except Exception as e:
            logging.error(f"Error saving detailed journal entry: {e}")
            return ""
    
    def _save_markdown_entry(self, entry: Dict[str, Any]) -> None:
        """
        Save a journal entry to the main markdown file (newest entries at top).

        Args:
            entry: The journal entry data
        """
        try:
            # Parse the timestamp to create a formatted time string
            timestamp_obj = datetime.datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S")
            time_str = timestamp_obj.strftime("%I:%M:%S %p")

            # Read existing content
            existing_content = ""
            header_lines = []
            if os.path.exists(self.journal_file):
                with open(self.journal_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Separate header from entries
                # Header is the first few lines until we hit the first entry (starts with ###)
                in_header = True
                for line in lines:
                    if in_header:
                        if line.strip().startswith('###'):
                            # Found first entry, switch to content mode
                            in_header = False
                            existing_content += line
                        else:
                            header_lines.append(line)
                    else:
                        existing_content += line

            # Build the new entry content
            new_entry = f"\n### {entry['timestamp']}\n\n"

            # Add link to audio recording if available
            if entry.get('relative_audio_path'):
                new_entry += f"🔊 [Listen to recording]({entry['relative_audio_path']})  (Right-click and select 'Open Link' to play)\n\n"

            # Add summary
            new_entry += f"{entry['summary']}\n\n"

            # Add link to detailed entry
            entry_link = f"[[{entry['date']} - {time_str} - Audio Journal Entry]]"
            new_entry += f"{entry_link}\n\n"

            new_entry += "---\n\n"

            # Write back: header + new entry + old entries
            with open(self.journal_file, 'w', encoding='utf-8') as f:
                # Write header
                f.writelines(header_lines)
                # Write new entry at top
                f.write(new_entry)
                # Write existing entries below
                f.write(existing_content)

            logging.info(f"Journal entry saved to {self.journal_file} (inserted at top)")
        except Exception as e:
            logging.error(f"Error saving journal entry: {e}")
            raise
