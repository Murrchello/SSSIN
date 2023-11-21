import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import read, write


Fs, data = read('./audio/SDRSharp_20231014_114318Z_145475000Hz_AF.wav')
data = data[:, 0] # Берем левый канал
N = len(data) # Определяем длину массива исходных аудиоданных
time = round(N / Fs, 2) # Вычисляем время длительности аудиофайла в секундах
min_limit_x = 0 # Наименьшее значение оси x
max_limit_x = time # Набольшее значение оси x
t = np.linspace(min_limit_x, max_limit_x, N) # Cоздание временной оси
ampl =  round(np.amax(np.abs(data)), 2) # Определяем амплитуду сигнала

print('Частота дискретизации Fs =', Fs, 'Гц')
print('Длина массива исходных аудиоданных N =', N)
print('Длительность исходного аудиофайла t =', time, 'Сек')
print('Амплитуда исходного аудиофайла A =', ampl, 'Вольт')

# Устанавливаем значения для делений оси X (время) с ценой деления в 1 секунду
x_ticks = np.arange(0, max_limit_x + 1, 1)
# Установите значения для делений оси Y (амплитуда) с ценой деления в 0.1 Вольт
y_ticks = np.arange(-ampl, ampl + 0.1, 0.1)

# Построение графика исходного аудиофайла
plt.figure(1)
plt.title('Исходный аудиофайл')
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, В')
plt.plot(t, data) 
plt.grid("on")
plt.show()


#--- Вычисляем спектр исходного аудиофайла ---#
# Создаем частотную ось
M = N*4
f_range = np.zeros(M) # Частотная ось

for smp_ref in range(-int(M/2), int(M/2)):
    f_range[smp_ref] = (smp_ref / M) * Fs

# Преобразование фурье для исходного аудиофайла
fft_data = np.abs(np.fft.fft(data, M)/N)

# Построение графика спектра исходного аудиофайла
plt.figure(2)
plt.title('Спектр исходного аудиофайла')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.plot(f_range, fft_data)
plt.grid("on")
plt.show()


#--- Генерируем шум в определенном диапазоне частот ---#
mean = 0 # Мат.ожидание
sigma = 0.5 # СКО 
freq_range_noise = (150, 2500)  # Диапазон частот для генерации шума

# Генерация белого шума
noise = np.random.normal(mean, sigma, N)

# Фильтрация шума для ограничения частотного диапазона
# На выходе получаем шум в ограниченном диапазоне частот (freq_range_noise)
b, a = butter(4, Wn=[freq / (Fs / 2) for freq in freq_range_noise], btype='band')
filtered_noise = filtfilt(b, a, noise)

# Строим график сгенерированного шума
plt.figure(3)
plt.title('Сгенерированный шум')
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, В')
plt.plot(t, filtered_noise) 
plt.grid("on")
plt.show()


#--- Построение спектра сгенерированного шума ---#
# Преобразование фурье для сгенерированного шума
fft_filtered_noise = np.abs(np.fft.fft(filtered_noise, M) / N)

# Построение графика спектра сгенерированного шума
plt.figure(4)
plt.title('Спектр сгенерированного шума')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.plot(f_range, fft_filtered_noise)
plt.grid("on")
plt.show()


#--- Наложение белого шума на исходный сигнал ---#
sum_data_filtered_noise = data + filtered_noise

# Построение графика с исходным сигналом и наложенным белым шумом
plt.figure(5)
plt.title('Исходный сигнал с наложенным белым шумом')
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, В')
plt.plot(t, sum_data_filtered_noise)
plt.grid("on")
plt.show()

# Сохраняем аудиофайл с шумом
write('./audio/noise_audio.wav', Fs, sum_data_filtered_noise.astype(np.float32))


#--- Вычисляем спектр суммы аудиофайла и шума ---#
# Преобразование фурье для суммы аудиофайла и шума
fft_sum_data_filtered_noise = np.abs(np.fft.fft(sum_data_filtered_noise, M) / N)

# Построение графика спектра с наложенным белым шумом
plt.figure(6)
plt.title('Спектр аудиофайла с наложенным шумом')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.plot(f_range, fft_sum_data_filtered_noise)
plt.grid("on")
plt.show()

