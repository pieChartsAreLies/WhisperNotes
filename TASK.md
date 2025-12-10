# WhisperNotes Tasks

## Feature Enhancements (2025-05-26)

-   [x] **Add 'Quit' option to tray menu:**
    -   Implement a "Quit" action in the system tray icon's context menu that cleanly exits the application.
-   [x] **Configurable Markdown Output File:**
    -   Add a menu option ("Set Output File...") to allow the user to select a Markdown file for saving transcriptions.
    -   Store the selected file path persistently (e.g., using `QSettings`).
    -   If no file is set, or on first run, prompt the user or use a default.
-   [x] **Persistent Timestamped Transcriptions:**
    -   When a transcription is successful, append it to the configured Markdown file.
    -   Each entry should include a timestamp (e.g., "YYYY-MM-DD HH:MM:SS - Transcription text").
    -   Ensure file operations are robust (e.g., handle cases where the file might be temporarily unavailable).

## Discovered During Work
*(No items yet)*
