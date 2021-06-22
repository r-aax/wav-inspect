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


class WAV:

    # ----------------------------------------------------------------------------------------------

    def __init__(self, filename=None):
        """
        Init WAV.
        :param filename: name of file
        """

        # Name of file.
        self.FileName = None

        # Array of amplitudes.
        self.Y = None

        # Sample rate.
        self.SampleRate = None

        # Duration.
        self.Duration = None

        # Spectres of two channels.
        # 3-dimensional array:
        #   (channels count) * (Y lines) * (X lines)
        self.Spectres = None

        # If filename if given - load it.
        if filename is not None:
            self.load(filename)

    # ----------------------------------------------------------------------------------------------

    def is_ok(self):
        """
        :return: True - if it is a correct record, and it is loaded, False - otherwise
        """

        return self.Y is not None

    # ----------------------------------------------------------------------------------------------

    def load(self, filename):
        """
        Load WAV file.
        :param filename: name of file
        :return: True - if loading is completed, False - if loading faults.
        """

        # Check for filename.
        if not os.path.isfile(filename):
            # print('No such file ({0}).'.format(filename))
            return False

        # First of all, check file extension.
        if pathlib.Path(filename).suffix != '.wav':
            return False

        # Load file.
        self.FileName = filename
        try:
            self.Y, self.SampleRate = librosa.load(filename, sr=None, mono=False)
        except BaseException:
            # If there is some problem with file, just ignore it.
            return False

        # Calculate duration.
        self.Duration = librosa.get_duration(y=self.Y, sr=self.SampleRate)

        return True

    # ----------------------------------------------------------------------------------------------

    def summary(self):
        """
        Print summary.
        """

        print('WAV audio record: FileName       = {0}'.format(self.FileName))
        print('                  Y.shape        = {0}'.format(self.Y.shape))
        print('                  SampleRate     = {0}'.format(self.SampleRate))
        print('                  Duration       = {0:.3f} s'.format(self.Duration))

        if self.Spectres is not None:
            print('                  Spectres.shape = {0}'.format(self.Spectres.shape))

    # ----------------------------------------------------------------------------------------------

    def generate_spectres(self):
        """
        Generate spectres.
        """

        generate_spectre = lambda d: librosa.amplitude_to_db(abs(librosa.stft(d, n_fft=2048)))

        self.Spectres = np.array([generate_spectre(d) for d in self.Y])

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Translate time position to specpos.
        :param tx: time position
        :return: specpos
        """

        return int(tx * (self.Spectres.shape[-1] / self.Duration))

    # ----------------------------------------------------------------------------------------------

    def specpos_to_time(self, specpos):
        """
        Translate position in spectre to time.
        :param specpos: position in spectre
        :return: time point
        """

        return specpos * (self.Duration / self.Spectres.shape[-1])

    # ----------------------------------------------------------------------------------------------

    def show_wave(self, idx, figsize=(20, 8)):
        """
        Show wave of the sound.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Create figure and plot graph on it.
        plt.figure(figsize=figsize)
        librosa.display.waveplot(self.Y[idx], sr=self.SampleRate)

    # ----------------------------------------------------------------------------------------------

    def show_spectre(self, idx, figsize=(20, 8)):
        """
        Show spectre.
        :param idx: spectre index
        :param figsize: figure size
        """

        # Create figure and plot graphs onto it.
        plt.figure(figsize=figsize)
        librosa.display.specshow(self.Spectres[idx],
                                 sr=self.SampleRate,
                                 x_axis='time', y_axis='hz', cmap='turbo')
        plt.colorbar(format='%+02.0f dB')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_centroid(self, idx, figsize=(20, 8)):
        """
        Show spectral centroid.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Code source:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Calculate centroid.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y[idx],
                                                               sr=self.SampleRate)[0]

        # Normalize data.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Create figure and plot on it.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y[idx], sr=self.SampleRate, alpha=0.4)
        plt.plot(t, normalize(spectral_centroids), color='b')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_rolloff(self, idx, figsize=(20, 8)):
        """
        Show spectral rolloff.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Code source:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Calculate centroid.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y[idx],
                                                               sr=self.SampleRate)[0]

        # Calculate rolloff.
        spectral_rolloff = librosa.feature.spectral_rolloff(self.Y[idx] + 0.01,
                                                            sr=self.SampleRate)[0]

        # Normalize data.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Create figure and plot.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y[idx], sr=self.SampleRate, alpha=0.4)
        plt.plot(t, normalize(spectral_rolloff), color='r')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_bandwidth(self, idx, figsize=(20, 8)):
        """
        Show spectral bandwidth.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Code source:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Calculate centroid.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y[idx],
                                                               sr=self.SampleRate)[0]

        # Construct bandwidth.
        x = self.Y[idx]
        sr = self.SampleRate
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=4)[0]

        # Normalize data.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Create figure and plot on it.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(x, sr=sr, alpha=0.4)
        plt.plot(t, normalize(spectral_bandwidth_2), color='r')
        plt.plot(t, normalize(spectral_bandwidth_3), color='g')
        plt.plot(t, normalize(spectral_bandwidth_4), color='y')
        plt.legend(('p = 2', 'p = 3', 'p = 4'))

    # ----------------------------------------------------------------------------------------------

    def normalize_spectre_value(self, idx):
        """
        Value for normalize spectre to shift minimum power to zero db.
        :param idx: indxes of amplitudes array
        :return: normalize spectre value
        """

        return -self.Spectres[idx].min()

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_total_power(self, idx, figsize=(20, 8)):
        """
        Show graph spectre total power.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        m = self.Spectres[idx].transpose()
        d = [sum(mi) for mi in m]
        show_graph(d, figsize=figsize, title='Spectre Total Power')

    # ----------------------------------------------------------------------------------------------

    def get_min_power_data(self, idx, ignore_min_powers_part):
        """
        Get min power data.
        :param idx: index of amplitudes array
        :param ignore_min_powers_part: part of min values in column those are ignored
                                       when we detect minimum value in column
        :return: min power data
        """

        m = self.Spectres[idx].transpose()

        return [min_without_some(mi, part=ignore_min_powers_part) for mi in m]

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_min_max_power(self, idx, ignore_min_powers_part, figsize=(20, 8)):
        """
        Show graph spectre minimum and maximum power.
        :param idx: index of amplitudes array
        :param ignore_min_powers_part: part of min values in column those are ignored
                                       when we detect minimum value in column
        :param figsize: figure size
        """

        m = self.Spectres[idx].transpose()
        d_min = [min_without_some(mi, part=ignore_min_powers_part) for mi in m]
        d_max = [max(mi) for mi in m]
        show_graph([d_min, d_max], figsize=figsize,
                   title='Spectre Min/Max Power',
                   style=['b', 'r'], linewidth=[2.0, 2.0])

    # ----------------------------------------------------------------------------------------------

    def get_min_power_leap_markers(self, idx,
                                   ignore_min_powers_part, power_lo_bound, leap_threshold):
        """
        Get min power leap markers.
        :param idx: index of amplitudes array
        :param ignore_min_powers_part: part of min values in column those are ignored
                                       when we detect minimum value in column
        :param power_lo_bound: low bound of power for analysis (in DB)
        :param leap_threshold: threshold for leap detection (in DB)
        :return: array of leap markers
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
        Show graph with min power leap markers.
        :param idx: index of amplitudes array
        :param ignore_min_powers_part: part of min values in column those are ignored
                                       when we detect minimum value in column
        :param power_lo_bound: low bound of power for analysis (in DB)
        :param leap_threshold: threshold for leap detection (in DB)
        :param figsize: figure size
        """

        m = self.Spectres[idx].transpose()
        d = self.get_min_power_leap_markers(idx, ignore_min_powers_part,
                                            power_lo_bound, leap_threshold)
        show_graph(d, figsize=figsize,
                   title='Spectre Min Power Leap Markers')

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_total_power_with_high_accent(self, idx, figsize=(20, 8)):
        """
        Show graph spectre total power when high frequences are taken with big weights.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        n = self.normalize_spectre_value(idx)
        m = self.Spectres[idx].transpose()

        # Weights.
        w = [i * i for i in range(len(m[0]))]

        # Form data for plot.
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
        Detect defect due to min power.
        :param ignore_min_powers_part: part of min values in column those are ignored
                                       when we detect minimum value in column
        :param power_lo_bound: low bound of power for analysis (in DB)
        :param leap_threshold: threshold for leap detection (in DB)
        :param leap_half_width: half width for leap (in frames)
        :return: defects list
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
        Detect defects.
        :return: defects list
        """

        return self.detect_defect_min_power_short_leap()

# ==================================================================================================


if __name__ == '__main__':

    # Unit tests.

    # indices_slice_array
    assert indices_slice_array(3, 0, 2, 1) == [(0, 2), (1, 3)]
    assert indices_slice_array(10, 3, 3, 2) == [(3, 6), (5, 8), (7, 10)]

    # min_without_part
    assert min_without_some([2, 1, 3, 5, 2], 0.0) == 1

    # Main test.

    directory = 'wavs/origin'
    # tests = os.listdir('wavs/origin')
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
