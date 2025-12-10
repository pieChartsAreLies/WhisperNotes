import subprocess
import logging

def check_accessibility_permissions() -> bool:
    """Check if the app has accessibility permissions (macOS)."""
    check_cmd = 'tell application "System Events" to return name of processes'
    result = subprocess.run(['osascript', '-e', check_cmd], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    logging.error(f"Accessibility permission error: {result.stderr}")
    return False

def type_text_applescript(text: str) -> bool:
    """Type text using AppleScript keystroke (macOS)."""
    try:
        escaped = text.replace('"', '\\"').replace('\\', '\\\\')
        applescript_cmd = f'tell application "System Events" to keystroke "{escaped}"'
        result = subprocess.run(['osascript', '-e', applescript_cmd], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        logging.error(f"AppleScript error: {result.stderr}")
        return False
    except Exception as e:
        logging.error(f"AppleScript typing error: {e}", exc_info=True)
        return False

def type_text_clipboard(text: str) -> bool:
    """Paste text using clipboard and Cmd+V (macOS)."""
    try:
        escaped = text.replace('"', '\\"').replace('\\', '\\\\')
        copy_cmd = f'set the clipboard to "{escaped}"'
        subprocess.run(['osascript', '-e', copy_cmd], check=True)
        paste_cmd = 'tell application "System Events" to keystroke "v" using command down'
        result = subprocess.run(['osascript', '-e', paste_cmd], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        logging.error(f"Clipboard paste error: {result.stderr}")
        return False
    except Exception as e:
        logging.error(f"Clipboard typing error: {e}", exc_info=True)
        return False
