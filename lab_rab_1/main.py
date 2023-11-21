import numpy as np
import matplotlib.pyplot as plt
plt.close("all") # закрывать окна при последующем новом запуске

# формируем и рисуем массив с синусоидальным сигналом:
ampl = 5 # амплитуда сигнала
f0 = 3 # частота сигнала
min_limit_x = 0 # наименьшее значение оси x
max_limit_x = 10 # набольшее значение оси x
N = 500 # длина массива
# частота дискретизации, отсчётов в секунду
fd = N/max_limit_x
x = np.linspace(min_limit_x, max_limit_x, N) # создаём ось
mean = 10 # мат.ожидание
sigma = 10 # СКО
massive_normal1 = np.random.normal(mean,sigma,N)

massive_sin1 = (ampl+4)*np.sin(2*np.pi*(f0*5)*x)
massive_sin2 = (ampl-5)*np.sin(2*np.pi*(f0*3)*x)
massive_sin3 = (ampl-6)*np.sin(2*np.pi*(f0*2)*x)
massive_sin4 = ampl*np.sin(2*np.pi*(f0*4)*x)
massive_sin5 = (ampl+4)*np.sin(2*np.pi*(f0*13)*x)
massive_sin6 = (ampl-5)*np.sin(2*np.pi*(f0*14)*x)
massive_sin7 = (ampl-6)*np.sin(2*np.pi*(f0*15)*x)
massive_sin8 = ampl*np.sin(2*np.pi*(f0*16)*x)
add_sig1 = massive_sin1 + massive_sin2 + massive_sin3 + massive_sin4 + massive_sin5 + massive_sin6 + massive_sin7 + massive_sin8
add_sig_noise1 = add_sig1 + massive_normal1

plt.figure(1)
plt.title('Синусоида 1')
plt.plot(x, massive_sin1)
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')
plt.grid("on")

plt.figure(2)
plt.plot(x,massive_sin2);plt.title('Синусоида 2')
plt.grid("on")
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

plt.figure(3)
plt.plot(x,massive_sin3);plt.title('Синусоида 3')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

plt.figure(4)
plt.plot(x,massive_sin4);plt.title('Синусоида 4')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

plt.figure(5)
plt.plot(x,massive_sin5);plt.title('Синусоида 5')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

plt.figure(6)
plt.plot(x,massive_sin6);plt.title('Синусоида 6')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

plt.figure(7)
plt.plot(x,massive_sin7);plt.title('Синусоида 7')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

plt.figure(8)
plt.plot(x,massive_sin8);plt.title('Синусоида 8')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

# рисуем полученный массив
plt.figure(9)
plt.plot(x, add_sig1);plt.title('Сумма синусоид')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

plt.figure(10)
plt.plot(x, add_sig_noise1);plt.title('Аддитивная смесь сигналов с шумом')
plt.grid("on") #сетка на графике
plt.xlabel('Время, мс')
plt.ylabel('Амплитуда, В')

# Приведение отчётов к частотам суммы синусоид
M = 4*N
f_range = np.zeros(M)
for smp_ref in range(-int(M/2), int(M/2)):
 f_range[smp_ref] = (smp_ref / M)*fd
# выполняем прямое преобразование Фурье
# первый аргумент функции fft - массив с исходным сигналом
# второй аргумент функции fft - длина выходного массива спектра
fft_result1 = np.fft.fft(add_sig1, M)/N
# полученный спектр - комплексный, график должен содержать только его модуль
# т.е. амплитудный спектр
mod_fftshift_result1 = abs(fft_result1)

# рисуем полученный массив
plt.figure(11)
plt.plot(f_range, mod_fftshift_result1);plt.title('Спектр аддитивной смеси сигнала без шума')
plt.grid("on") #сетка на графике
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, В')
# выполняем прямое преобразование Фурье
# первый аргумент функции fft - массив с исходным сигналом
# второй аргумент функции fft - длина выходного массива спектра
fft_result2 = np.fft.fft(add_sig_noise1, M)/N
# полученный спектр - комплексный, график должен содержать только его модуль
# т.е. амплитудный спектр
mod_fftshift_result2 = abs(fft_result2)

# рисуем полученный массив
plt.figure(12)
plt.plot(f_range, mod_fftshift_result2);plt.title('Спектр аддитивной смеси сигнала с шумом')
plt.grid("on") #сетка на графике
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, В')
# Фильтр
a4h = np.zeros(M)
# резонансная частота фильтра в Гц
freq_rez = 9
# полоса пропускания фильтра в Гц
polosa = 16
# нижняя граница полосы пропускания фильтра в Гц
min_a4h_freq = freq_rez - (polosa/2)
# верхняя граница полосы пропускания фильтра в Гц
max_a4h_freq = freq_rez + (polosa/2)
# в пределах полосы пропускания АЧХ фильтра пропускает сигнал без усиления
# за пределами полосы не пропускает вообще
for smp_ref in range(-int(M/2), int(M/2)):
    #Фильтр в отрицательных частотах
 if (f_range[smp_ref] > min_a4h_freq):
  if (f_range[smp_ref] < max_a4h_freq):
     a4h[smp_ref] = 1
     #Фильтр в положительных частотах
 if (f_range[smp_ref] < -min_a4h_freq):
  if (f_range[smp_ref] > -max_a4h_freq):
      a4h[smp_ref] = 1
      
freq_rez1 = 45
# полоса пропускания фильтра в Гц
polosa1 = 12
# нижняя граница полосы пропускания фильтра в Гц
min_a4h_freq1 = freq_rez1 - (polosa1/2)
# верхняя граница полосы пропускания фильтра в Гц
max_a4h_freq1 = freq_rez1 + (polosa1/2)
# в пределах полосы пропускания АЧХ фильтра пропускает сигнал без усиления
# за пределами полосы не пропускает вообще
for smp_ref1 in range(-int(M/2), int(M/2)):
    #Фильтр в отрицательных частотах
 if (f_range[smp_ref1] > min_a4h_freq1):
  if (f_range[smp_ref1] < max_a4h_freq1):
     a4h[smp_ref1] = 1
     #Фильтр в положительных частотах
 if (f_range[smp_ref1] < -min_a4h_freq1):
  if (f_range[smp_ref1] > -max_a4h_freq1):
      a4h[smp_ref1] = 1
# рисуем полученную АЧХ фильтра
plt.figure(13)
plt.plot(f_range, a4h);plt.title('АЧХ фильтра')
plt.grid("on") #сетка на графике
plt.xlabel('Частота, Гц')
plt.ylabel('Коэффициент передачи, К')
#Спектр отфильтрованной аддитивной смеси
filtered_add_smes_spektr = a4h*fft_result2
#ОБПФ от отфильтрованного спектра
filtered_add_smes = N*np.fft.ifft(filtered_add_smes_spektr)

# рисуем полученный массив
plt.figure(14)
plt.plot(f_range, abs(filtered_add_smes_spektr));plt.title('Отфильтрованный спектр аддитивной смеси сигнала')
plt.grid("on") #сетка на графике
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, В')

# требуется взять количество отсчётов, равное количеству в исходном массиве:
invert_fft_mas_len = int(len(filtered_add_smes)*N/M)

# рисуем полученный массив
plt.figure(15)
plt.plot(x, filtered_add_smes[0:invert_fft_mas_len]);plt.title('Отфильтрованная аддитивная смесь сигналов')
plt.grid("on") #сетка на графике
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, В')



