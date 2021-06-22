"""
Реализация модуля по обработке аудиозаписей.
"""

import os
import pathlib
import random
import operator
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# ==================================================================================================


def zipwith(a, b, f):
    """
    Слияние двух списков с использованием произвольной функции.

    :param a: Первый список.
    :param b: Второй список.
    :param f: Функция объединения двух элементов в одну сущьность.

    :return: Список, полученный в результате соедиения.
    """

    return [f(ai, bi) for (ai, bi) in zip(a, b)]


# --------------------------------------------------------------------------------------------------


def indices_slice_array(ar_len, start, part_len, step):
    """
    Получение списка кортежей, каждый из которых представляет собой индексы подсписков
    при разделении исходного списка на части.

    :param ar_len:   Исходная длина списка, для которого нужно выполнить разделение.
    :param start:    Позиция, с которой следует начать разделение (все элементы до этой позиции
                     игнорирутся).
    :param part_len: Длина каждого кусочка, на которые делится исходный список.
    :param step:     Шаг между точками начала двух соседних кусков.

    :return: Список кортежей с координатами частей списка.
    """

    idx = [(i, i + part_len)
           for i in range(start, ar_len - part_len + 1)
           if (i - start) % step == 0]

    return idx


# --------------------------------------------------------------------------------------------------


def show_graph(data, figsize=(20, 8),
               style='r', linewidth=2.0,
               title='title', xlabel='', ylabel='',
               show_grid='true'):
    """
    Демонстрация графика или набора графиков.

    :param data:      Массив данных для отображения (это могут быть данные одного графика
                      или массив данных для нескольких графиков).
    :param figsize:   Размер картинки.
    :param style:     Стиль графика (или списов стилей, если графиков несколько).
    :param linewidth: Ширина линии (или список ширин линий).
    :param title:     Заголовок графика.
    :param xlabel:    Надпись на оси OX.
    :param ylabel:    Надпись на оси OY.
    :param show_grid: Флаг отображения сетки.
    """

    # Пример кода:
    # https://pythonru.com/biblioteki/pyplot-uroki

    # Примеры стилей графиков:
    #   цвета      : 'b', 'g', 'r', 'y'
    #   маркеры    : '*', '^', 's'
    #   типы линий : '--', '-.'

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(show_grid)

    # Отображение.
    # По типу параметра style определяем, подан ли на отображение один график или несколько.
    if type(style) is str:

        # График один.
        plt.plot(data, style, linewidth=linewidth)

    elif type(style) is list:

        # Несколько графиков.
        for i in range(len(style)):
            plt.plot(data[i], style[i], linewidth=linewidth[i])

    plt.show()


# --------------------------------------------------------------------------------------------------


def min_without_some(ar, part):
    """
    Получение минимального значение с игнорированием части значений.

    :param ar:   Список.
    :param part: Часть значений, которые нужно проигнорировать (0.0 <= part < 1.0).

    :return: Минимальное значение списка с учетом проигнорированной части элементов.
    """

    i = int(len(ar) * part)
    cp = ar.copy()
    cp.sort()

    return cp[i]


# --------------------------------------------------------------------------------------------------


def apply_array_lo_bound(a, lo_bound):
    """
    Применение фильтра нижней границы к списку.

    :param a:        Список.
    :param lo_bound: Нижняя граница значений.

    :return: Новый список после применения фильтра.
    """

    return [max(ai, lo_bound) for ai in a]

# --------------------------------------------------------------------------------------------------


def shift_array_to_min(a):
    """
    Сдвиг списка по нижней границе.

    :param a: Список.

    :return: Список, сдвинутый по нижней границе.
    """

    m = min(a)

    return [ai - m for ai in a]


# ==================================================================================================


class Defect:
    """
    Дефект.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, record_name, channel, defect_name, defect_coords):
        """
        Конструктор дефекта.

        :param record_name:   Имя записи.
        :param channel:       Номер канала.
        :param defect_name:   Имя дефекта.
        :param defect_coords: Координаты дефекта.
        """

        self.RecordName = record_name
        self.Channel = channel
        self.DefectName = defect_name
        self.DefectCoords = defect_coords

    # ----------------------------------------------------------------------------------------------

    def __repr__(self):
        """
        Получение строкового представления дефекта.

        :return: Строка.
        """

        if type(self.DefectCoords) is tuple:
            defect_coords_str = '{0:.3f} s - {1:.3f} s'.format(self.DefectCoords[0],
                                                               self.DefectCoords[1])
        else:
            defect_coords_str = '{0:.3f} s'.format(self.DefectCoords)

        return 'Defect: {0} (ch {1}) : {2} ({3})'.format(self.RecordName, self.Channel,
                                                         self.DefectName, defect_coords_str)

# ==================================================================================================


class Channel:
    """
    Канал.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, y, sample_rate, duration):
        """
        Конструктор канала.

        :param y:           Массив амплитуд.
        :param sample_rate: Частота дисткретизации.
        :param duration:    Продолжительность.
        """

        self.Y = y
        self.Spectre = None
        self.TSpectre = None
        self.SampleRate = sample_rate
        self.Duration = duration

    # ----------------------------------------------------------------------------------------------

    def generate_spectre(self):
        """
        Генерация спектра.
        """

        self.Spectre = librosa.amplitude_to_db(abs(librosa.stft(self.Y, n_fft=2048)))
        self.TSpectre = self.Spectre.transpose()

    # ----------------------------------------------------------------------------------------------

    def show_wave(self, figsize=(20, 8)):
        """
        Демонстрация звуковой волны.

        :param figsize: Размер картинки.
        """

        # Создание картинки и отображение волны на ней.
        plt.figure(figsize=figsize)
        librosa.display.waveplot(self.Y, sr=self.SampleRate)

    # ----------------------------------------------------------------------------------------------

    def show_spectre(self, figsize=(20, 8)):
        """
        Демонстрация спектра.

        :param figsize: Размер картинки.
        """

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        librosa.display.specshow(self.Spectre, sr=self.SampleRate,
                                 x_axis='time', y_axis='hz', cmap='turbo')
        plt.colorbar(format='%+02.0f dB')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_centroid(self, figsize=(20, 8)):
        """
        Демонстрация графика спектрального центроида.

        :param figsize: Размер картинки.
        """

        # Пример кода:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Получение данных для отображения центроида.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y, sr=self.SampleRate)[0]

        # Нормализация данных.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y, sr=self.SampleRate, alpha=0.4)
        plt.plot(t, normalize(spectral_centroids), color='b')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_rolloff(self, figsize=(20, 8)):
        """
        Демонстрация графика спектрального спада.

        :param figsize: Размер картинки.
        """

        # Пример кода:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Получение данных массива спектрального центроида.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y, sr=self.SampleRate)[0]

        # Получение данных массива спектрального спада.
        spectral_rolloff = librosa.feature.spectral_rolloff(self.Y + 0.01, sr=self.SampleRate)[0]

        # Нормализация данных.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y, sr=self.SampleRate, alpha=0.4)
        plt.plot(t, normalize(spectral_rolloff), color='r')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_bandwidth(self, figsize=(20, 8)):
        """
        Демострация спектральной ширины.

        :param figsize: Размер картинки.
        """

        # Пример кода:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Вычисление данных спектрального центроида.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y, sr=self.SampleRate)[0]

        # Вычисление данных спектральной ширины.
        sb_2 = librosa.feature.spectral_bandwidth(self.Y + 0.01, sr=self.SampleRate)[0]
        sb_3 = librosa.feature.spectral_bandwidth(self.Y + 0.01, sr=self.SampleRate, p=3)[0]
        sb_4 = librosa.feature.spectral_bandwidth(self.Y + 0.01, sr=self.SampleRate, p=4)[0]

        # Нормализациия данных.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y, sr=self.SampleRate, alpha=0.4)
        plt.plot(t, normalize(sb_2), color='r')
        plt.plot(t, normalize(sb_3), color='g')
        plt.plot(t, normalize(sb_4), color='y')
        plt.legend(('p = 2', 'p = 3', 'p = 4'))

# ==================================================================================================


class WAV:
    """
    Аудиоззапись.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, filename=None):
        """
        Конструктор аудиозаписи.

        :param filename: Имя файла.
        """

        # Имя файла.
        # (в данном месте инициализировать нельзя, так как задание имени записи не гарантирует
        # ее успешную загрузку, имя файла записывается в момент загрузки).
        self.FileName = None

        # Каналы.
        self.Channels = None

        # Частота дискретизации.
        self.SampleRate = None

        # Продолжительность записи (с).
        self.Duration = None

        # Если подано имя файла, то пытаемся загрузить его.
        if filename is not None:
            self.load(filename)

    # ----------------------------------------------------------------------------------------------

    def is_ok(self):
        """
        Проверка, является ли данная запись нормальной, то есть она загрузилась.

        :return: True  - если запись успешно загружена,
                 False - в противном случае.
        """

        return self.Channels is not None

    # ----------------------------------------------------------------------------------------------

    def load(self, filename):
        """
        Загрузка файла аудиозаписи.

        :param filename: Имя файла.

        :return: True  - если загрузка завершена успешно,
                 False - если во время загрузки произвошел сбой.
        """

        # Проверка расширения файла.
        if pathlib.Path(filename).suffix != '.wav':
            return False

        # Проверка существования файла.
        if not os.path.isfile(filename):
            # print('No such file ({0}).'.format(filename))
            return False

        # Загрузка файла.
        self.FileName = filename
        try:

            # Чтение амплитуд и частоты дискретизации и вычисление продолжительности.
            ys, self.SampleRate = librosa.load(filename, sr=None, mono=False)
            self.Duration = librosa.get_duration(y=ys, sr=self.SampleRate)

            # Создание каналов.
            # Частота дискретизации и продолжительность отправляются в каждый канал.
            self.Channels = [Channel(y, self.SampleRate, self.Duration) for y in ys]

        except BaseException:
            # Если что-то пошло не так, то не разбираемся с этим, а просто игнорим ошибку.
            return False

        # Загрузка прошла успешно.
        return True

    # ----------------------------------------------------------------------------------------------

    def ch0(self):
        """
        Получение канала 0.

        :return: Канал 0.
        """

        return self.Channels[0]

    # ----------------------------------------------------------------------------------------------

    def ch1(self):
        """
        Получение канала 1.

        :return: Канал 1.
        """

        return self.Channels[1]

    # ----------------------------------------------------------------------------------------------

    def summary(self):
        """
        Печать общей информации об аудиозаписи.
        """

        if not self.is_ok():
            print('Bad WAV audio record!')
            return

        n = len(self.Channels)

        print('WAV audio record: FileName       = {0}'.format(self.FileName))
        a = [ch.Y.shape for ch in self.Channels]
        print('                  Channels       = {0} : {1}'.format(n, a))
        print('                  SampleRate     = {0}'.format(self.SampleRate))
        print('                  Duration       = {0:.3f} s'.format(self.Duration))
        a = [ch.Spectre.shape for ch in self.Channels]
        print('                  Spectres       = {0} : {1}'.format(n, a))

    # ----------------------------------------------------------------------------------------------

    def generate_spectres(self):
        """
        Генерация спектров.
        """

        for ch in self.Channels:
            ch.generate_spectre()

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Перевод точки времени в точку в матрице спектра.

        :param tx: Точка сремени.

        :return: Точка в спектре.
        """

        return int(tx * (self.Channels[0].Spectre.shape[-1] / self.Duration))

    # ----------------------------------------------------------------------------------------------

    def specpos_to_time(self, specpos):
        """
        Перевод точки спектра в точку времени.

        :param specpos: Точка в спектре.

        :return: Точка времени.
        """

        return specpos * (self.Duration / self.Channels[0].Spectre.shape[-1])

    # ----------------------------------------------------------------------------------------------

    def normalize_spectre_value(self, idx):
        """
        Значение для нормализации спектра.

        :param idx: Индекс спектра.

        :return: Значение для нормализации.
        """

        return -self.Channels[idx].Spectre.min()

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_total_power(self, idx, figsize=(20, 8)):
        """
        Демонстрация графика суммарной энергии спектра.

        :param idx:     Номер канала.
        :param figsize: Размер картинки.
        """

        m = self.Channels[idx].Spectre.transpose()
        d = [sum(mi) for mi in m]
        show_graph(d, figsize=figsize, title='Spectre Total Power')

    # ----------------------------------------------------------------------------------------------

    def get_min_power_data(self, idx, ignore_min_powers_part):
        """
        Получение данных минимальной энергии.

        :param idx:                    Номер канала.
        :param ignore_min_powers_part: Часть минимальных значений, которые нужно проигнорировать.

        :return: Данные о минимальной энергии.
        """

        m = self.Channels[idx].Spectre.transpose()

        return [min_without_some(mi, part=ignore_min_powers_part) for mi in m]

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_min_max_power(self, idx, ignore_min_powers_part, figsize=(20, 8)):
        """
        Демонстрация графиков минимальной и максимальной силы звука.

        :param idx:                    Номер канала.
        :param ignore_min_powers_part: Часть минимальных значений, которые нужно проигнорировать.
        :param figsize:                Размер картинки.
        """

        m = self.Channels[idx].Spectre.transpose()
        d_min = [min_without_some(mi, part=ignore_min_powers_part) for mi in m]
        d_max = [max(mi) for mi in m]
        show_graph([d_min, d_max], figsize=figsize,
                   title='Spectre Min/Max Power',
                   style=['b', 'r'], linewidth=[2.0, 2.0])

    # ----------------------------------------------------------------------------------------------

    def get_min_power_leap_markers(self, idx,
                                   ignore_min_powers_part, power_lo_bound, leap_threshold):
        """
        Генерация маркеров скачков минимальной силы.

        :param idx:                    Номер канала.
        :param ignore_min_powers_part: Часть минимальных значений, которые нужно проигнорировать.
        :param power_lo_bound:         Нижняя граница силы (DB).
        :param leap_threshold:         Порог определения скачка (DB).

        :return: Массив маркеров скачков.
        """

        d1 = self.get_min_power_data(idx, ignore_min_powers_part)
        d2 = apply_array_lo_bound(d1, power_lo_bound)
        d3 = shift_array_to_min(d2)

        return [int(d3i > leap_threshold) for d3i in d3]

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_min_power_leap_markers(self, idx,
                                                  ignore_min_powers_part, power_lo_bound,
                                                  leap_threshold, figsize=(20, 8)):
        """
        Демонстрация графика скачков минимальной силы.

        :param idx:                    Номер канала.
        :param ignore_min_powers_part: Часть минимальных значений, которые нужно проигнорировать.
        :param power_lo_bound:         Нижняя граница силы (DB).
        :param leap_threshold:         Порог определения скачка (DB).
        :param figsize:                Размер картинки.
        """

        m = self.Channels[idx].Spectre.transpose()
        d = self.get_min_power_leap_markers(idx, ignore_min_powers_part,
                                            power_lo_bound, leap_threshold)
        show_graph(d, figsize=figsize,
                   title='Spectre Min Power Leap Markers')

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_total_power_with_high_accent(self, idx, figsize=(20, 8)):
        """
        Показ графика суммарной силы с акцентом на высокие частоты.

        :param idx:     Номер канала.
        :param figsize: Размер картинки.
        """

        n = self.normalize_spectre_value(idx)
        m = self.Channels[idx].Spectre.transpose()

        # Веса по частотам.
        w = [i * i for i in range(len(m[0]))]

        # Формируем данные для графика.
        d = [sum(zipwith(c, w, lambda ci, wi: (ci + n) * wi))
             for c in m]

        show_graph(d, figsize=figsize, title='Spectre Total Power With High Accent')

    # ----------------------------------------------------------------------------------------------

    def detect_defect_min_power_short_leap(self,
                                           ignore_min_powers_part=0.01,
                                           power_lo_bound=-50.0,
                                           leap_threshold=5.0,
                                           leap_half_width=2):
        """
        Детектирование дефекта скачка минимальной силы.

        :param ignore_min_powers_part: Часть минимальных значений, которые нужно проигнорировать.
        :param power_lo_bound:         Нижняя граница силы (DB).
        :param leap_threshold:         Порог определения скачка (DB).
        :param leap_half_width:        Ограничение на полудлину прыжка.

        :return: Список дефектов.
        """

        dfs = []

        for idx in [0, 1]:

            # Process data.
            d1 = self.get_min_power_data(idx, ignore_min_powers_part)
            d2 = apply_array_lo_bound(d1, power_lo_bound)
            d3 = shift_array_to_min(d2)

            # Leap markers.
            leap_markers = self.get_min_power_leap_markers(idx, ignore_min_powers_part,
                                                           power_lo_bound, leap_threshold)

            n = len(leap_markers)

            # We detect defect only for narrow leap.
            # For defect we have to have leap in i-th position, and
            # no leaps in (i - leap_half_width)-th and (i + leap_half_width)-th.
            for (i, lmi) in enumerate(leap_markers):
                if (i >= leap_half_width) and (i < n - leap_half_width):
                    m_0 = lmi
                    m_left = leap_markers[i - leap_half_width]
                    m_right = leap_markers[i + leap_half_width]
                    if m_0 and (not m_left) and (not m_right):
                        df = Defect(self.FileName, idx,
                                    'min_power_short_leap', self.specpos_to_time(i))
                        dfs.append(df)

        return dfs

    # ----------------------------------------------------------------------------------------------

    def detect_defects(self):
        """
        Определение дефектов.

        :return: Список дефектов.
        """

        return self.detect_defect_min_power_short_leap()

# ==================================================================================================


if __name__ == '__main__':

    # Тесты.

    # indices_slice_array
    assert indices_slice_array(3, 0, 2, 1) == [(0, 2), (1, 3)]
    assert indices_slice_array(10, 3, 3, 2) == [(3, 6), (5, 8), (7, 10)]

    # min_without_part
    assert min_without_some([2, 1, 3, 5, 2], 0.0) == 1

    # Тело основного теста.

    directory = 'wavs/origin'
    # tests = os.listdir(directory)
    tests = ['0015.wav']
    print(tests)
    defects = []

    for test in tests:
        print('... process {0}'.format(test))
        wav = WAV('{0}/{1}'.format(directory, test))
        if wav.is_ok():
            wav.generate_spectres()
            defects = defects + wav.detect_defects()

    for defect in defects:
        print('  {0}'.format(defect))

# ==================================================================================================
