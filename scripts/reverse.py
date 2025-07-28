import subprocess
import time

def get_clipboard():
    """Get clipboard content using PowerShell."""
    result = subprocess.run(
        ["powershell.exe", "-Command", "Get-Clipboard"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def set_clipboard(content):
    """Set clipboard content using PowerShell."""
    subprocess.run(
        ["powershell.exe", "-Command", f"Set-Clipboard -Value \"{content}\""],
        text=True
    )

def reverse_clipboard_content():
    """Reverse a comma-separated list from the clipboard."""
    clipboard_content = get_clipboard()
    if not clipboard_content:
        return None

    # Split by commas, reverse the list, and join back
    reversed_content = ",".join(reversed(clipboard_content.split(",")))
    return reversed_content

def monitor_clipboard():
    """Continuously monitor the clipboard and reverse its content."""
    last_content = None
    print("Monitoring clipboard... Press Ctrl+C to exit.")
    try:
        while True:
            current_content = get_clipboard()
            if current_content != last_content:  # Check if clipboard content has changed
                reversed_content = reverse_clipboard_content()
                if reversed_content is not None:
                    set_clipboard(reversed_content)
                    print(f"Reversed content: {reversed_content}")
                last_content = current_content
            time.sleep(0.5)  # Check clipboard every 500ms
    except KeyboardInterrupt:
        print("\nExiting clipboard monitor.")

if __name__ == "__main__":
    monitor_clipboard()