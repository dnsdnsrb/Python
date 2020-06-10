import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
# y, sr = librosa.load(librosa.util.example_audio_file())
# D = np.abs(librosa.stft(y))
# print(np.shape(y))
# print(np.shape(D))
# print(D[0, 2])

# test = np.random.normal(size=(400, 10))
# print(test[0])
# print(max(test))
# print(np.shape(test))
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='s', y_axis='log')
print(test.dtype)
librosa.display.specshow(test, x_axis='s', y_axis='linear', sr=800, hop_length=800)
plt.title('Power spectrogram')
plt.colorbar(format = '%2.0f dB')
plt.tight_layout()

plt.savefig('librosa2.png')
