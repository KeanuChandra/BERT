import scipy.signal
import scipy.signal.windows

# Monkey-patch hann into the signal namespace
scipy.signal.hann = scipy.signal.windows.hann

