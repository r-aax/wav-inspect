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
import keras
import keras.utils
import keras.utils.np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


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


def unzip(a):
    """
    Разделение списка двухэлементных кортежей на два списка.

    :param a: Список двухэлементных кортежей.

    :return: Кортеж из двух списков.
    """

    return tuple([list(ai) for ai in zip(*a)])


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
                      или массив/кортеж данных для нескольких графиков).
    :param figsize:   Размер картинки.
    :param style:     Стиль графика (или списов/кортеж стилей, если графиков несколько).
    :param linewidth: Ширина линии (или список/кортеж ширин линий).
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


def min_max_extended(a, limits_before_sort, limits_after_sort):
    """
    Получение минимального и максимального значения с применением ограничений.
    Сначала массив обрезается по границам limits_before_sort.
    После этого он сортируется.
    После сортировки массив обрезается по границам limits_after_sort.
    После этого возвращается первый и последний элемент массива.

    :param a:                  Массив.
    :param limits_before_sort: Границы, по которым обрубатся массив до сортировки.
    :param limits_after_sort:  Границы, по которым обрубается массив после сортировки.

    :return: Минимальное и максимальнео значения с учетом органичителей.
    """

    n = len(a)
    x = a[int(n * limits_before_sort[0]): int(n * limits_before_sort[1])]
    x.sort()
    n = len(x)
    x = x[int(n * limits_after_sort[0]): int(n * limits_after_sort[1])]

    return x[0], x[-1]


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


class DefectSnapSettings:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 limits_before_sort,
                 limits_after_sort,
                 min_power_lo_threshold,
                 half_snap_len,
                 diff_min_max_powers_hi_threshold):
        """
        Конструктор настроек для дефекта snap.

        :param limits_before_sort:               Границы, по которым обрубаются массивы
                                                 силы звука до сортировки.
        :param limits_after_sort:                Границы, по которым обрубаются массивы
                                                 силы звука после сортировки.
        :param min_power_lo_threshold:           Минимальное отслеживаемое значение скачка
                                                 минимальной силы звука.
        :param half_snap_len:                    Половинная длина склейки
                                                 (чем меньше она, тем более резкую скейку ловим).
        :param diff_min_max_powers_hi_threshold: Максимально допустимая разница в значениях
                                                 максимума и минимума силы звука (определяет
                                                 степень постоянства силы в массиве).
        """

        self.LimitsBeforeSort = limits_before_sort
        self.LimitsAfterSort = limits_after_sort
        self.MinPowerLoThreshold = min_power_lo_threshold
        self.HalfSnapLen = half_snap_len
        self.DiffMinMaxPowersHiThreshold = diff_min_max_powers_hi_threshold

# ==================================================================================================


class Channel:
    """
    Канал.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, filename, channel, y, sample_rate, duration):
        """
        Конструктор канала.

        :param filename:    Имя записи.
        :param channel:     Канал.
        :param y:           Массив амплитуд.
        :param sample_rate: Частота дисткретизации.
        :param duration:    Продолжительность.
        """

        self.FileName = filename
        self.Channel = channel
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

        # Транспонируем матрицу звука, чтобы первым измерением была отметка времени.
        # При этом удобнее работать с матрицей, если в нижних частях массива лежат низкие частоты.
        self.TSpectre = self.Spectre.transpose()

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Перевод точки времени в точку в матрице спектра.

        :param tx: Точка сремени.

        :return: Точка в спектре.
        """

        return int(tx * (self.Spectre.shape[-1] / self.Duration))

    # ----------------------------------------------------------------------------------------------

    def specpos_to_time(self, specpos):
        """
        Перевод точки спектра в точку времени.

        :param specpos: Точка в спектре.

        :return: Точка времени.
        """

        return specpos * (self.Duration / self.Spectre.shape[-1])

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

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_total_power(self, figsize=(20, 8)):
        """
        Демонстрация графика суммарной энергии спектра.

        :param figsize: Размер картинки.
        """

        show_graph([sum(tsi) for tsi in self.TSpectre],
                   figsize=figsize, title='Spectre Total Power')

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_min_max_power(self,
                                         limits_before_sort, limits_after_sort,
                                         show_min=True, show_max=True,
                                         figsize=(20, 8)):
        """
        Демонстрация графиков минимальной и максимальной силы звука.

        :param limits_before_sort: Границы, по которым обрубатся массив до сортировки.
        :param limits_after_sort:  Границы, по которым обрубается массив после сортировки.
        :param show_min:           Флаг демонстрации минимальных значений.
        :param show_max:           Флаг демонстрации максимальных значений.
        :param figsize:            Размер картинки.
        """

        d = [min_max_extended(tsi,
                              limits_before_sort=limits_before_sort,
                              limits_after_sort=limits_after_sort)
             for tsi in self.TSpectre]
        ud = unzip(d)

        if show_min:
            if show_max:
                show_graph(ud, figsize=figsize,
                           title='Spectre Min/Max Power',
                           style=['b', 'r'], linewidth=[2.0, 2.0])
            else:
                show_graph(ud[0], figsize=figsize, title='Spectre Min Power', style='b')
        else:
            if show_max:
                show_graph(ud[1], figsize=figsize, title='Spectre Max Power', style='r')
            else:
                pass

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_min_max_diff_power(self,
                                              limits_before_sort, limits_after_sort,
                                              figsize=(20, 8)):
        """
        Демонстрация графика разницы между максимальной и минимальной силой звука.

        :param limits_before_sort: Границы, по которым обрубается массив до сортировки.
        :param limits_after_sort:  Границы, по которым обрубается массив после сортировки.
        :param figsize:            Размер картинки.
        """

        d = [min_max_extended(tsi,
                              limits_before_sort=limits_before_sort,
                              limits_after_sort=limits_after_sort)
             for tsi in self.TSpectre]
        diffs = [d_max - d_min for (d_min, d_max) in d]

        show_graph(diffs, figsize=figsize, title='Spectre Min/Max Diff Power')

    # ----------------------------------------------------------------------------------------------

    def get_defect_snap_markers(self, s: DefectSnapSettings):
        """
        Получение маркеров дефекта snap.

        :param s: Настройки.

        :return: Список маркеров snap.
        """

        d = [min_max_extended(tsi,
                              limits_before_sort=s.LimitsBeforeSort,
                              limits_after_sort=s.LimitsAfterSort)
             for tsi in self.TSpectre]

        # Создаем массив для разметки дефектов.
        n = len(d)
        markers = [0] * n

        # Производим разметку.
        for i in range(s.HalfSnapLen, n):
            is_snap = (d[i][0] - d[i - s.HalfSnapLen][0] > s.MinPowerLoThreshold)
            is_cnst = (d[i][1] - d[i][0] < s.DiffMinMaxPowersHiThreshold)
            if is_snap and is_cnst:
                markers[i] = 1

        return markers

    # ----------------------------------------------------------------------------------------------

    def show_defect_snap_markers(self, s: DefectSnapSettings, figsize=(20, 8)):
        """
        Демонстрация маркеров дефекта snap.

        :param s:       Настройки.
        :param figsize: Размер картинки.
        """

        markers = self.get_defect_snap_markers(s)
        show_graph(markers, figsize=figsize, title='Defect Snap Markers')

    # ----------------------------------------------------------------------------------------------

    def get_defects_snap(self, s: DefectSnapSettings):
        """
        Получение дефектов snap.

        :param s: Настройки.

        :return: Список дефектов snap.
        """

        markers = self.get_defect_snap_markers(s)

        # Формируем список дефектов.
        objs = [Defect(self.FileName, self.Channel, 'snap', self.specpos_to_time(i))
                for (i, marker) in enumerate(markers)
                if (marker == 1)]

        return objs

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
            self.Channels = [Channel(self.FileName, i, y, self.SampleRate, self.Duration)
                             for (i, y) in enumerate(ys)]

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

    def get_defects_snap(self, s: DefectSnapSettings):
        """
        Получение маркеров дефекта snap.

        :param s: Настройки.

        :return: Список дефектов snap.
        """

        return self.ch0().get_defects_snap(s) + self.ch1().get_defects_snap(s)

    # ----------------------------------------------------------------------------------------------

    def get_defects(self):
        """
        Определение дефектов.

        :return: Список дефектов.
        """

        defect_snap_settings = DefectSnapSettings(limits_before_sort=(0.7, 0.95),
                                                  limits_after_sort=(0.25, 0.75),
                                                  min_power_lo_threshold=5.0,
                                                  half_snap_len=2,
                                                  diff_min_max_powers_hi_threshold=5.0)

        return self.get_defects_snap(defect_snap_settings)

# ==================================================================================================


class NNetTrainer:

    # ----------------------------------------------------------------------------------------------

    def __init__(self, name):
        """
        Конструктор нейронной сети.
        """

        # Имя сети.
        self.Name = name

        # Обучающие и валидационные данные.
        self.XTrain = None
        self.YTrain = None
        self.XTest = None
        self.YTest = None

        # Модель.
        self.Model = None

    # ----------------------------------------------------------------------------------------------

    def init_data(self):
        """
        Инициализация данных для обучения.
        """

        if self.Name == 'mnist':
            self.init_data_mnist()
        else:
            raise Exception('unknown nnet name {0}'.format(self.Name))

    # ----------------------------------------------------------------------------------------------

    def init_data_mnist(self):
        """
        Инициализация данных mnist для обучения.
        """

        # Загрузка данных.
        (self.XTrain, self.YTrain), (self.XTest, self.YTest) = mnist.load_data()

        # Обработка данных X.
        self.XTrain = self.XTrain.reshape(60000, 784)
        self.XTest = self.XTest.reshape(10000, 784)
        self.XTrain = self.XTrain.astype('float32')
        self.XTest = self.XTest.astype('float32')
        self.XTrain /= 255
        self.XTest /= 255

        # Обработка данных Y.
        self.YTrain = keras.utils.np_utils.to_categorical(self.YTrain, 10)
        self.YTest = keras.utils.np_utils.to_categorical(self.YTest, 10)

    # ----------------------------------------------------------------------------------------------

    def init_model(self):
        """
        Иницализация модели.
        """

        if self.Name == 'mnist':
            self.init_model_mnist()
        else:
            raise Exception('unknown nnet {0}'.format(self.Name))

    # ----------------------------------------------------------------------------------------------

    def init_model_mnist(self):
        """
        Инициализация модели mnist.
        """

        # Сборка модели.
        self.Model = Sequential()
        self.Model.add(Dense(512, activation='relu', input_shape=(784,)))
        self.Model.add(Dropout(0.2))
        self.Model.add(Dense(512, activation='relu'))
        self.Model.add(Dropout(0.2))
        self.Model.add(Dense(10, activation='softmax'))

        # Компиляция модели.
        self.Model.compile(loss='categorical_crossentropy',
                           optimizer=RMSprop(),
                           metrics=['accuracy'])

    # ----------------------------------------------------------------------------------------------

    def fit(self):
        """
        Обучение модели.
        """

        if self.Name == 'mnist':
            self.fit_mnist()
        else:
            raise Exception('unknown nnet {0}'.format(self.Name))

    # ----------------------------------------------------------------------------------------------

    def fit_mnist(self):
        """
        Обучение модели mnist.
        """

        self.Model.fit(self.XTrain, self.YTrain,
                       batch_size=128,
                       epochs=20,
                       verbose=1,
                       validation_data=(self.XTest, self.YTest))

    # ----------------------------------------------------------------------------------------------

    def save(self):
        """
        Сохранение модели.
        """

        self.Model.save('nnets/{0}.h5'.format(self.Name))

# ==================================================================================================


def analyze_directory(directory, filter_fun=lambda _x: True,
                      verbose=False):
    """
    Анализ директории с файлами на наличие дефектов.

    :param directory:  Имя директории.
    :param filter_fun: Дополнительная функция для отфильтровывания файлов, которые необходимо
                       анализировать.
    :param verbose:    Признак печати процесса анализа.

    :return: Список дефектов.
    """

    fs = os.listdir(directory)
    ds = []

    for f in fs:
        if filter_fun(f):

            if verbose:
                print('.... process {0}'.format(f))

            wav = WAV('{0}/{1}'.format(directory, f))

            if wav.is_ok():
                wav.generate_spectres()
                ds = ds + wav.get_defects()

    return ds

# ==================================================================================================


def unit_tests():
    """
    Короткие тесты.
    """

    # indices_slice_array
    assert indices_slice_array(3, 0, 2, 1) == [(0, 2), (1, 3)]
    assert indices_slice_array(10, 3, 3, 2) == [(3, 6), (5, 8), (7, 10)]

# ==================================================================================================


def nnet_test():
    """
    Тест нейронки.
    """

    nn = NNetTrainer('mnist')
    nn.init_data()
    nn.init_model()
    nn.fit()
    nn.save()

# ==================================================================================================


def main():
    """
    Головная функция.
    """

    defects = analyze_directory('wavs/origin',
                                filter_fun=lambda f: True,
                                verbose=True)

    for d in defects:
        print(d)


# ==================================================================================================


if __name__ == '__main__':
    nnet_test()


# ==================================================================================================
