#!/usr/bin/env python3
"""Fix soundsig compatibility with scipy >= 1.2.0

Run this ONCE after setting up your environment:
    python fix_soundsig.py
"""
import soundsig
from pathlib import Path

soundsig_path = Path(soundsig.__path__[0])
signal_file = soundsig_path / "signal.py"

print(f"Patching: {signal_file}")

# Read the file
content = signal_file.read_text()

# Backup
backup_file = signal_file.with_suffix(".py.bak")
backup_file.write_text(content)
print(f"Backup saved to: {backup_file}")

# Fix the import
old_import = "from scipy.signal import filter_design, resample, filtfilt, hann"
new_import = """from scipy.signal import resample, filtfilt
from scipy.signal.windows import hann
try:
    from scipy.signal import filter_design
except ImportError:
    from scipy.signal import iirdesign as filter_design"""

if old_import in content:
    content = content.replace(old_import, new_import)
    signal_file.write_text(content)
    print("Patch applied successfully!")
else:
    print("Import line not found or already patched.")

# Test the fix
try:
    # Force reimport
    import importlib
    importlib.reload(soundsig.signal)
    from soundsig.signal import lowpass_filter
    print("Test passed: soundsig imports correctly!")
except Exception as e:
    print(f"Test failed: {e}")
    print("Restoring backup...")
    signal_file.write_text(backup_file.read_text())
