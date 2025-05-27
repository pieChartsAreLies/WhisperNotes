# Voice Typer

A lightweight voice-to-text application that runs locally on your Mac. Simply press `Cmd+Shift+R` to start recording your voice, and the transcribed text will be typed at your cursor position. You can also use `Cmd+Shift+J` to create journal entries with your voice.

## Features

- 🎙️ Record audio with a global hotkey (`Cmd+Shift+R`)
- ✍️ Transcribe speech to text using OpenAI's Whisper (runs locally)
- 📝 Automatically type transcribed text at cursor position
- 📊 Keep a log of all transcriptions with timestamps
- 📓 Journal entries with `Cmd+Shift+J` for easy voice journaling
- 🤖 AI-powered summaries and formatting for journal entries
- 🎯 Visual recording indicator
- 🖥️ Runs in the system tray (menu bar)
- 🔒 All processing happens locally - no data leaves your computer

## Installation

1. **Install Python 3.8 or higher** if you haven't already:
   ```bash
   # Using Homebrew (recommended)
   brew install python
   ```

2. **Clone this repository** or download the source code

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional audio dependencies** (required for PyAudio):
   ```bash
   brew install portaudio
   ```

## Usage

1. **Run the application**:
   ```bash
   python voice_typer.py
   ```

2. **Grant Accessibility Permissions**:
   - When prompted, go to System Preferences > Security & Privacy > Privacy > Accessibility
   - Click the lock icon and enter your password
   - Click the + button and add your Terminal or iTerm app
   - If you're using VS Code's terminal, you'll need to add VS Code to the list

3. **Start using Voice Typer**:
   - The app will run in the menu bar (look for a red dot icon)
   - Press `Cmd+Shift+R` to start recording for regular transcription
   - Release the keys to stop recording and transcribe
   - The transcribed text will be typed at your cursor position
   - A visual indicator will show while recording

4. **Using Journal Mode**:
   - Press `Cmd+Shift+J` to start recording for a journal entry
   - Release the keys to stop recording
   - The transcription will be saved to `~/Documents/Personal/Audio Journal/Journal.md`
   - Each entry includes a timestamp, summary, and link to the full transcription
   - Audio recordings are saved in the `recordings` subfolder

## Logging

- Regular transcriptions are automatically saved to the configured output file (default: `~/Documents/VoiceTyperTranscriptions.md`) with timestamps.
- Journal entries are saved to `~/Documents/Personal/Audio Journal/Journal.md` with timestamps, summaries, and links to detailed entries.

## Customization

You can customize the following in the `CONFIG` dictionary in `voice_typer.py`:

- Hotkey combination
- Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
- Audio recording settings
- Log file location

## Troubleshooting

- **App doesn't type text**: Make sure you've granted Accessibility permissions to your terminal app.
- **No audio input detected**: Check your microphone settings in System Preferences > Sound > Input.
- **Installation issues**: Try creating a virtual environment first:
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

## Requirements

- macOS 10.15 or later
- Python 3.8+
- Microphone access
- Internet connection (only for first-time model download)
- Ollama running locally (for journal entry summarization)

## License

MIT
