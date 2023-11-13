import ujson as json  # or 'import ujson as json' if you prefer to use ujson
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Load the JSON file
with open('Project5.json', 'r') as f:
    data = json.load(f)

# Extract cricket sound signal and sampling frequency
Fs = data['Fs']  # Assuming Fs is a single value
crickets = np.array(data['crickets'])  # Convert to NumPy array if not already

# Perform FFT on the cricket sound signal
freq_spectrum = fft(crickets)

# Apply high frequency emphasizing ramp filter
freqs = np.fft.fftfreq(len(crickets), 1 / Fs)
ramp_filter = np.abs(freqs)  # The ramp filter is the absolute value of frequencies
filtered_spectrum = freq_spectrum * ramp_filter

# Plot the magnitude spectrum of the result
plt.figure(figsize=(10, 6))
plt.plot(freqs, np.abs(filtered_spectrum), '*')
plt.title("Magnitude Spectrum of Filtered Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, Fs / 2)  # Display only positive frequencies up to the Nyquist frequency
plt.show()
