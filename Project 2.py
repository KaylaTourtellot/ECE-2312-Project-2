import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
from numpy import cos, sin, pi, absolute, arange, zeros
from scipy.signal import butter,filtfilt
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
#1
sr = 44100  # Hertz
DURATION = 5  # Seconds

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

# Generate a 5000 hertz sine wave that lasts for 5 seconds
x, y = generate_sine_wave(5000, sr, DURATION)
write("q1sinetone.wav",sr, y)
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=5000)
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("5kHz SineWave")
#plt.show()

#2
def generate_chirp_sine_wave(freq, sample_rate, duration):
    x1 = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = np.linspace(0, freq*.5, sample_rate * duration, endpoint=False)
    frequencies = x1 * frequencies
    # 2pi because np.sin takes radians
    y1 = np.sin((2 * np.pi) * frequencies)
    return x1, y1
# Generate a 5000 hertz sine wave that lasts for 5 seconds
x1, y1 = generate_chirp_sine_wave(8000, sr, DURATION)
write("q2chirp.wav",sr, y1)
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=8000)
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("8kHz Chirp SineWave")
#plt.show()

#3
duration = 1
n1, ny1 = generate_sine_wave(3951, sr, duration)
n2, ny2 = generate_sine_wave(4186, sr, duration)
n3, ny3= generate_sine_wave(3322, sr, duration)
n4, ny4 = generate_sine_wave(1661, sr, duration)
n5, ny5 = generate_sine_wave(2489, sr, duration)
cetky= np.concatenate((ny1, ny2, ny3, ny4, ny5))
write("q3cetk.wav", sr, cetky)
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(cetky)), ref=8000)
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Cetk Spectrogram")
#plt.show()

#4
x_lazy, sr = librosa.load("sound1.wav", sr = sr)
Addition = np.add(x_lazy, y)
write("q4speechchirp.wav",sr, Addition)
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(Addition)), ref=8000)
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Speech & Sine Wave Spectrogram")
plt.show()

#5
y4, sr = librosa.load("q4speechchirp.wav")
fc = 4000     
FN = 0.5 * sr       
nfc= fc/FN 
w, z = butter(5, nfc, btype='low')
filtering = filtfilt(w, z, y4)
write("q5filteredspeechsine.wav", sr, y4)
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(filtering)), ref=8000)
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Filtered Speech Specrogram")
plt.show()

#6
x_lazy, sr = librosa.load("sound1.wav")
y3, sr = librosa.load("q5filteredspeechsine.wav")
stereoaudiosignal = np.hstack((x_lazy.reshape(-1, 1), y3.reshape(-1, 1)))
write("q6stereospeechsine.wav", sr, stereoaudiosignal)
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(y3)), ref=8000)
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Stereo Spectrogram y3")
plt.show()
freq_in_db = librosa.amplitude_to_db(np.abs(librosa.stft(x_lazy)), ref=8000)
librosa.display.specshow(freq_in_db, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Stereo Audio Signal Spectrogram x_lazy")
plt.show()

















