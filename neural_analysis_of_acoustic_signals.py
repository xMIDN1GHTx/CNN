import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, spectrogram

# Параметры
FS = 16000  # Частота дискретизации (16 кГц)
DURATION = 1  # Длина анализа (1 секунда)
LOWCUT = 1500  # Нижняя граница
HIGHCUT = 4500  # Верхняя граница


# Фильтр Баттерворта
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output='sos')
    return sosfilt(sos, data)


# CNN для анализа спектра
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Модель
model = AudioCNN()


# Функция захвата звука
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer = indata[:, 0]
    filtered_audio = bandpass_filter(audio_buffer, LOWCUT, HIGHCUT, FS)

    # FFT для АЧХ
    freqs, times, Sxx = spectrogram(filtered_audio, FS)
    plt.pcolormesh(times, freqs, 10 * np.log10(Sxx))
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [сек]')
    plt.title('АЧХ сигнала')
    plt.colorbar(label='Мощность [дБ]')
    plt.show()

    # Преобразование в мел-спектрограмму
    waveform = torch.tensor(filtered_audio).unsqueeze(0)
    mel_spec = T.MelSpectrogram(sample_rate=FS, n_fft=2048, hop_length=512, n_mels=32)(waveform)
    mel_spec = mel_spec.unsqueeze(0)  # Размерность канала

    # Классификация сигнала
    with torch.no_grad():
        prediction = model(mel_spec.unsqueeze(0))
        if prediction.item() > 0.6:
            print("🔔 Обнаружен сигнал в диапазоне 1500–4500 Гц!")
        else:
            print("🚫 Это шум!")


# Запуск
stream = sd.InputStream(callback=callback, samplerate=FS, channels=1, blocksize=int(FS * DURATION))
stream.start()
input("Нажмите Enter для остановки...")
stream.stop()
