"""
Реализация модуля по обработке аудиозаписей.
"""

import os
import time
import itertools
import pathlib
import random
import operator
import sklearn
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import librosa
import librosa.display
# import pocketsphinx
import keras
import keras.utils
import keras.utils.np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten
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


# ==================================================================================================


def unzip(a):
    """
    Разделение списка двухэлементных кортежей на два списка.

    :param a: Список двухэлементных кортежей.

    :return: Кортеж из двух списков.
    """

    return tuple([list(ai) for ai in zip(*a)])


# ==================================================================================================


def split(ar, pos):
    """
    Разделение списка по позиции.

    :param ar:  Список.
    :param pos: Позиция.

    :return: Разделенный список.
    """

    return ar[:pos], ar[pos:]


# ==================================================================================================


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


# ==================================================================================================


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


def predicated_count(a, p):
    """
    Получение количества элементов, удовлетворяющих предикату.

    :param a: Список.
    :param p: Предикат.

    :return: Количество элементов, удовлетворяющих предикату.
    """

    return len([1 for e in a if p(e)])

# ==================================================================================================


def predicated_part(a, p):
    """
    Получение доли элементов, удовлетворяющих предикату.

    :param a: Список.
    :param p: Предикат.

    :return: Доля элементов списка, удовлетворяющих предикату.
    """

    return predicated_count(a, p) / len(a)


# ==================================================================================================


def recognize_speech_text_pocketsphinx(file, s=1):
    """
    Распознать текст в аудиофайле.

    :param file: Имя файла.
    :param s:    Вариант настроек.
    """

    # Если текст распознается с помощью speech_recognition, то
    # файл записи нужно перекодировать, как показано по ссылке:
    # https://github.com/Uberi/speech_recognition/issues/325

    if s == 1:

        # Настройки распознавания.
        # https://habr.com/en/post/351376/
        config = {
            'verbose': True,
            'audio_file': file,
            'buffer_size': 2048,
            'no_search': False,
            'full_utt': False,
            'hmm': '../pocketsphinx/download/zero_ru.cd_cont_4000',
            'lm': '../pocketsphinx/download/ru.lm',
            'dict': '../pocketsphinx/download/ru.dic',
        }

    elif s == 2:

        # Другой вариант настроек распознавания.
        # https://github.com/lavrenkov-sketch/speech-rechnition/blob/master/spech_to_text.py
        config = {
            'verbose': True,
            'audio_file': file,
            'buffer_size': 2048,
            'no_search': False,
            'full_utt': False,
            'hmm': '../pocketsphinx/lavrenkov/zero_ru.cd_cont_4000',
            'lm': False,
            'jsgf': '../pocketsphinx/lavrenkov/calc2.jsgf',
            'dict': '../pocketsphinx/lavrenkov/vocabular.dict',
        }

    else:
        raise Exception('unknown settings №')

    audio = pocketsphinx.AudioFile(**config)

    for phrase in audio:
        print(phrase)

    print('recognize_speech_test finished')


# ==================================================================================================


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
    """
    Настройки дефекта snap.
    """

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


class DefectMutedSettings:
    """
    Настройки дефекта muted.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 case_width,
                 case_learn_step,
                 train_cases_part,
                 case_pred_step,
                 category_detect_limits,
                 part_for_decision):
        """
        Конструктор дефекта глухой записи.

        :param case_width:             Ширина кадра спектра для обучения нейронки.
        :param case_learn_step:        Длина шага между соседними кейсами для обучения нейронки.
        :param train_cases_part:       Доля обучающей выборки.
        :param case_pred_step:         Длина шага между соседними кейсами для предсказания.
        :param category_detect_limits: Пределы на определение категории
                                       (если сигнал выше верхнего порога, то категория детектировна,
                                       если сигнал ниже нижнего порога, то категория не
                                       детектирована, в других случаях решение не принято).
        :param part_for_decision:      Доля детектированных кейсов для определения глухой записи.
        """

        self.CaseWidth = case_width
        self.CaseLearnStep = case_learn_step
        self.TrainCasesPart = train_cases_part
        self.CasePredStep = case_pred_step
        self.CategoryDetectLimits = category_detect_limits
        self.PartForDecision = part_for_decision

        # Грузим нейронку, если она есть.
        if os.path.isfile('nnets/muted.h5'):
            self.NNet = keras.models.load_model('nnets/muted.h5')

# ==================================================================================================


class DefectsSettings:
    """
    Настройки дефектов.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 limits_db,
                 snap,
                 muted):
        """
        Конструктор настроек для всех дефектов.

        :param limits_db: Лимиты по силе (за пределами лимитов вообще
                          не учитываем сигнал).
        :param snap:      Настройки дефекта snap.
        :param muted:     Настройки дефекта muted.
        """

        self.LimitsDb = limits_db
        self.Snap = snap
        self.Muted = muted

# ==================================================================================================


class Channel:
    """
    Канал.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, parent, channel, y):
        """
        Конструктор канала.

        :param parent:  Родительская запись.
        :param channel: Канал.
        :param y:       Массив амплитуд.
        """

        self.Parent = parent
        self.Channel = channel
        self.Y = y
        self.Spectre = None
        self.TSpectre = None
        self.NNetData = None

        # Безусловно генерируем спектры.
        self.generate_spectre()

    # ----------------------------------------------------------------------------------------------

    def generate_spectre(self):
        """
        Генерация спектра.
        """

        # Генерируем спектр.
        self.Spectre = librosa.amplitude_to_db(abs(librosa.stft(self.Y, n_fft=2048)))

        # Транспонируем матрицу звука, чтобы первым измерением была отметка времени.
        # При этом удобнее работать с матрицей, если в нижних частях массива лежат низкие частоты.
        self.TSpectre = self.Spectre.transpose()

        # Генерация данных для нейронки.
        (min_v, max_v) = self.Parent.Settings.LimitsDb
        self.NNetData = self.TSpectre + 0.0
        np.clip(self.NNetData, min_v, max_v, out=self.NNetData)
        self.NNetData = (self.NNetData - min_v) / (max_v - min_v)

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Перевод точки времени в точку в матрице спектра.

        :param tx: Точка сремени.

        :return: Точка в спектре.
        """

        return int(tx * (self.Spectre.shape[-1] / self.Parent.Duration))

    # ----------------------------------------------------------------------------------------------

    def specpos_to_time(self, specpos):
        """
        Перевод точки спектра в точку времени.

        :param specpos: Точка в спектре.

        :return: Точка времени.
        """

        return specpos * (self.Parent.Duration / self.Spectre.shape[-1])

    # ----------------------------------------------------------------------------------------------

    def get_nnet_data_cases(self, width, step):
        """
        Получение данные для нейросетей, порезанных на части.

        :param width: Ширина кейса.
        :param step:  Шаг между кейсами.

        :return: Список кейсов для нейросети.
        """

        idxs = indices_slice_array(self.NNetData.shape[0], 0, width, step)

        return [self.NNetData[fr:to] for (fr, to) in idxs]

    # ----------------------------------------------------------------------------------------------

    def show_wave(self, figsize=(20, 8)):
        """
        Демонстрация звуковой волны.

        :param figsize: Размер картинки.
        """

        # Создание картинки и отображение волны на ней.
        plt.figure(figsize=figsize)
        librosa.display.waveplot(self.Y, sr=self.Parent.SampleRate)

    # ----------------------------------------------------------------------------------------------

    def show_spectre(self, figsize=(20, 8)):
        """
        Демонстрация спектра.

        :param figsize: Размер картинки.
        """

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        librosa.display.specshow(self.Spectre, sr=self.Parent.SampleRate,
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
        spectral_centroids = librosa.feature.spectral_centroid(self.Y,
                                                               sr=self.Parent.SampleRate)[0]

        # Нормализация данных.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y, sr=self.Parent.SampleRate, alpha=0.4)
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
        spectral_centroids = librosa.feature.spectral_centroid(self.Y,
                                                               sr=self.Parent.SampleRate)[0]

        # Получение данных массива спектрального спада.
        spectral_rolloff = librosa.feature.spectral_rolloff(self.Y + 0.01,
                                                            sr=self.Parent.SampleRate)[0]

        # Нормализация данных.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y, sr=self.Parent.SampleRate, alpha=0.4)
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
        spectral_centroids = librosa.feature.spectral_centroid(self.Y, sr=self.Parent.SampleRate)[0]

        # Вычисление данных спектральной ширины.
        sb_2 = librosa.feature.spectral_bandwidth(self.Y + 0.01, sr=self.Parent.SampleRate)[0]
        sb_3 = librosa.feature.spectral_bandwidth(self.Y + 0.01, sr=self.Parent.SampleRate, p=3)[0]
        sb_4 = librosa.feature.spectral_bandwidth(self.Y + 0.01, sr=self.Parent.SampleRate, p=4)[0]

        # Нормализациия данных.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Создание картинки и отображение на ней.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y, sr=self.Parent.SampleRate, alpha=0.4)
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

    def get_defect_snap_markers(self):
        """
        Получение маркеров дефекта snap.

        :return: Список маркеров snap.
        """

        s = self.Parent.Settings.Snap

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

    def show_defect_snap_markers(self, figsize=(20, 8)):
        """
        Демонстрация маркеров дефекта snap.

        :param figsize: Размер картинки.
        """

        markers = self.get_defect_snap_markers()
        show_graph(markers, figsize=figsize, title='Defect Snap Markers')

    # ----------------------------------------------------------------------------------------------

    def get_defects_snap(self):
        """
        Получение дефектов snap.

        :return: Список дефектов snap.
        """

        markers = self.get_defect_snap_markers()

        # Формируем список дефектов.
        objs = [Defect(self.Parent.FileName,
                       self.Channel,
                       'snap',
                       self.specpos_to_time(i))
                for (i, marker) in enumerate(markers)
                if (marker == 1)]

        return objs

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self):
        """
        Получение дефектов muted.

        :return: Список дефектов muted.
        """

        s = self.Parent.Settings.Muted

        if s.NNet is None:
            return []

        xs = self.get_nnet_data_cases(s.CaseWidth, s.CasePredStep)
        xs = np.array(xs)
        shp = xs.shape
        xs = xs.reshape((shp[0], shp[1] * shp[2]))
        xs = xs.astype('float32')

        # Анализ каждого кейса.
        answers = s.NNet.predict(xs)

        # Предикат определения глухого кейса.
        def is_ans_muted(ans):
            lim = s.CategoryDetectLimits
            return (ans[0] < lim[0]) and (ans[1] > lim[1])

        # Часть глухих кейсов.
        muted_part = predicated_part(answers, is_ans_muted)

        # Принимаем решение о глухой записи, если часть глухих кейсов высока.
        if muted_part > s.PartForDecision:
            return [Defect(self.Parent.FileName,
                           self.Channel,
                           'muted',
                           (0.0, self.Parent.Duration))]
        else:
            return []

    # ----------------------------------------------------------------------------------------------

    def get_defects_by_name(self, defect_name):
        """
        Получение дефектов заданного типа.

        :param defect_name: Имя дефекта.

        :return: Список дефектов заданного типа.
        """

        if defect_name == 'snap':
            return self.get_defects_snap()
        elif defect_name == 'muted':
            return self.get_defects_muted()
        else:
            raise Exception('unknown defect name ({0})'.format(defect_name))

# ==================================================================================================


class WAV:
    """
    Аудиоззапись.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, filename, settings=None):
        """
        Конструктор аудиозаписи.

        :param filename: Имя файла.
        :param settings: Настройки.
        """

        # Имя файла.
        # (в данном месте инициализировать нельзя, так как задание имени записи не гарантирует
        # ее успешную загрузку, имя файла записывается в момент загрузки).
        self.FileName = filename

        # Каналы.
        self.Channels = None

        # Частота дискретизации.
        self.SampleRate = None

        # Продолжительность записи (с).
        self.Duration = None

        # Настройки.
        self.Settings = settings
        if self.Settings is None:
            self.Settings = get_settings()

        # Пытаемся загрузить файл.
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
            self.Channels = [Channel(self, i, y) for (i, y) in enumerate(ys)]

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

    def get_defects_from_both_channels(self, defect_name):
        """
        Получение списка дефектов для которого выполняется анализ обоих каналов.

        :param defect_name: Имя дефекта.

        :return: Список дефектов.
        """

        ch0dfs = self.ch0().get_defects_by_name(defect_name)
        ch1dfs = self.ch1().get_defects_by_name(defect_name)

        return ch0dfs + ch1dfs

    # ----------------------------------------------------------------------------------------------

    def get_defects_snap(self):
        """
        Получение маркеров дефекта snap.

        :return: Список дефектов snap.
        """

        return self.get_defects_from_both_channels('snap')

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self):
        """
        Получение маркеров дефекта muted.

        :return: Список дефектов muted.
        """

        return self.get_defects_from_both_channels('muted')

    # ----------------------------------------------------------------------------------------------

    def get_defects_by_name(self, defect_name):
        """
        Получение списка дефектов по имени.

        :param defect_name: Имя дефекта.

        :return: Список дефектов.
        """

        if defect_name == 'snap':
            return self.get_defects_snap()
        elif defect_name == 'muted':
            return self.get_defects_muted()
        else:
            raise Exception('unknown defect name ({0})'.format(defect_name))

    # ----------------------------------------------------------------------------------------------

    def get_defects(self, defects_names):
        """
        Получение списка дефектов по списку имен дефектов.

        :param defects_names: Список имен дефектов.

        :return: Список дефектов.
        """

        m = [self.get_defects_by_name(name) for name in defects_names]

        return list(itertools.chain(*m))
    
    
    # ----------------------------------------------------------------------------------------------
    
    def get_defects_echo(self, filename=None, cor_p=0.9, len_echo=2, times_echo=2, hop_length=512):
        
        """
        Функция обнаружения эхо эффекта в аудиосигнале.

        Аргументы функции:
        root_path - полный путь к файлу + полное название файла с иследуемой аудиозаписью (тип - str)
        cor_p - порог обнаружения эхо: корреляция выше этого значения считается первым признаком наличия эхо (тип - float, диапазон значений [0, 1])
        len_echo - минимальная длительность события эхо для детектирования явления (тип - int, измеряется в фреймах)
        times_echo - минимальное количество повторений эхо для детектирования явления (тип - int, измеряется в фреймах)
        hop_length - размер окна для кратковременного преобразования Фурье (тип - int, кратное степени 2)

        Результатом работы функции является:
        Волновой график исследуемого аудиосигнала, где зеленым цветом закрашена область возможного нахождения эффекта эхо
        или
        Сообщение: 'В данном аудиосигнале эхо не найдено!'

        """
        self.FileName = filename
        
        # чтение файла
        x, sr = librosa.load(filename, mono=True, sr = None)

        # кратковременное преобразование Фурье
        X = librosa.stft(x, hop_length=hop_length)
        Xmag = abs(X)

        # получение корреляционной матрицы
        cor = np.zeros((Xmag.shape[1] - 1, Xmag.shape[1] - 1)) # матрица для записи в нее результатов корреляции

        for i in range(Xmag.shape[1] - 1): # анализ каждого столбца на корреляю с каждым последующим

            seq = Xmag.T[i]
            for j in range(0, Xmag.shape[1] - 1 - i):
                seq2 = Xmag.T[i+j+1]
                result = sp.stats.pearsonr(seq, seq2)
                cor[i][j] = round(result[0], 3) # анализ корреляции двух матриц и запись результата проверки в нулевую матрицу

        # формирование матрици корреляций с булевыми значениями
        cor_true = cor >= cor_p # порог обнаружения эхо

        point_echo = [] # список для записи отсечек эхо
        # проверяем каждую строку матрицы корреляции с булевыми значениями
        for i_num, i_val in enumerate(cor_true[ : -1 * times_echo]): # нужно проверить все строки кроме последних

            skan_core = [] # задаем ядро счетчика эха в строке, сюда записываем номера столбов эхо, далее идет условие его формирования в конкретной строке
            for bool_num, bool_ansver in enumerate(i_val): # скан значений строки - проходимся вдоль строки, ищем начало столба эхо

                if bool_ansver == True: # нашли начало столба эхо

                    tru = 1 # счетчик последовательности совпадений вертикальных Тру, начало отсчета с 1
                    for i_vert_skan in range(1, cor_true.shape[0]-i_num): # от найденной точки идем вниз по матрице корреляций с булевыми значениями

                        if i_num + i_vert_skan >= cor_true.shape[0]: # прекратить цикл, если индекс вышел за пределы массива
                            break
                        elif cor_true[i_num + i_vert_skan][bool_num] == False: # нету продолжения, заканчиваем цикл, продолжаем искать следующее значение в строке
                            break
                        else: # нашли продолжение! переставляем счетчик последовательности - глубины столба эхо
                            tru += 1

                    if tru >= len_echo: # счетчи глубины столба эхо больше порога, предполагаем что это признак эхо!

                        skan_core.append(bool_num) # запоминаем индекс столбца матрицы корелляции, где были замечены признаки эхо

            # действия внутри строки закончились, подытоживаем результат работы со строкой
            if len(skan_core) > times_echo: # проверка на наличие признаков эхов в строке

                steps_echo = [] # записываем сюда расстояние между каждым столбом эхо
                for num_sc, val_cs in enumerate(skan_core[:-1], 1): # цикл записи расстояний между столбами эхо

                    steps_echo.append(skan_core[num_sc] - val_cs) # запоминаем расстояние между каждым столбом

                # проверить длину шагов - равны ли они с учетом области разброса
                echo_puls = 0 # счетчик - сколько раз расстояния между эхо оказались равны с учетом разброса в два фрейма
                for step in range(len(steps_echo) - 1):
                    len_step = round(steps_echo[step+1] - steps_echo[step], 0)
                    if len_step in range(-2, 3, 1): # совпадают ли длины шагов эхо с учетом разброса от -2 до 2 (если шаги равны, то ответ 0)
                        echo_puls += 1

                # зафиксировать результат
                if echo_puls >= times_echo: # В строке есть эхо! нужно запомнить эти отсечки, для последующей визуализации
                    point_echo.append([i_num, skan_core]) # запомним индекс строки матрици корреляций и номера столбцов с эхо для дальнейшей интерпртации

            # !конец цикла прохода одной строки, начинаем все сначала для следующей строки!

        # обработка результатов - преобразование во фреймы, после во время

        if len(point_echo) >= times_echo: # проверка на наличие эхо в записи
            # данные во фреймы
            frames = []
            frames.append(point_echo[0][0]) # начальная точка - номер первой строки, где было замечано эхо
            frames.append(point_echo[-1][0] + point_echo[-1][-1][-1]) # конечная точка - номер последней строкии с эхо + смещение, где последний раз было замечено эхо

            # фреймы во время
            t = librosa.frames_to_time(frames, sr=sr)

            # визуализировать итоговый результат
            # диапозон возможного наличия эхо
            t1 = float(t[0])
            t2 = float(t[-1])

            #  График сигнала
            fig, ax = plt.subplots(figsize = (15, 5))
            librosa.display.waveplot(x, sr = sr) # визуализация исходного сигнала

            ax.axvspan(xmin = t1, xmax = t2, color = 'green', alpha = 0.2) # закрашивание указанной области

            plt.show()

        else:

            print('В данном аудиосигнале эхо не найдено!')
 

    # ----------------------------------------------------------------------------------------------
    
    def get_defects_deaf_audio(self, filename=None, lim = 20, hop_length=512):
        
        """
        Функция обнаружения эффекта глухой записи в аудиосигнале.

        Аргументы функции:
        filename - полный путь к файлу + полное название файла с иследуемой аудиозаписью (тип - str)
        lim - уровень пустоты - определяет сколько верхних пустых слоев подряд должно быть, 
            чтобы сказать, что аудиосигнал имеет участки глухой записи (тип - int, измеряется в фреймах)
        hop_length - размер окна для кратковременного преобразования Фурье (тип - int, кратное степени 2)

        Результатом работы функции является:
        Сообщение: 'В данной аудиозаписи обнаружен эффект глухой записи!'
        или
        Сообщение: 'В данной аудиозаписи эффект глухой записи не обнаружен!'
        """

        # чтение файла
        x, sr = librosa.load(filename, mono=True, sr = None)

        # кратковременное преобразование Фурье
        X = librosa.stft(x, hop_length=hop_length)
        Xmag = abs(X)
        Xdb = librosa.amplitude_to_db(Xmag) 

        void = [] # пустой массив для записи в него факта наличия пстот (1) или их отсутствие (0)
        for n, v in enumerate(Xdb[::-1]): # сичтываем все фреймы записи сверху вних
            tr = v < -37 # фильтр для обнаружения пустых частот, которые имеют значение -37.623936
            if tr.sum() == Xdb.shape[1]: 
                void.append(1)
            else:
                void.append(0)

        nh = 0 # счетчик посторений пустых частот
        for nt, t in enumerate(void): # проверяем сколько раз подряд встречаются пустые частоты

            if t == 1:
                nh += 1
            else:
                nh = 0

            if nh > lim: # если повторений больше, чем установленный лимит, то эффект глухой записи найден!
                print('В данной аудиозаписи обнаружен эффект глухой записи!')
                break

            if nt == int(len(void) - 1):
                print('В данной аудиозаписи эффект глухой записи не обнаружен!')
    

# ==================================================================================================


class NNetTrainer:

    # ----------------------------------------------------------------------------------------------

    def __init__(self, name, settings):
        """
        Конструктор нейронной сети.

        :param name:     Имя дефекта (и соответствующей нейронки).
        :param settings: Настройки.
        """

        # Имя сети.
        self.Name = name

        # Настройки.
        self.Settings = settings

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

        if self.Name == 'muted':
            self.init_data_muted()
        elif self.Name == 'mnist':
            self.init_data_mnist()
        else:
            raise Exception('unknown nnet name {0}'.format(self.Name))

    # ----------------------------------------------------------------------------------------------

    def init_data_muted(self):
        """
        Инициализация данных muted для обучения.
        """

        t0 = time.time()
        print('init_data_muted : start : {0}'.format(time.time() - t0))

        # Директория и набор файлов для позитивных и негативных тестов.
        directory = 'wavs/origin'
        pos_files = ['0003.wav']
        neg_files = ['0004.wav']
        files = pos_files + neg_files

        all_xs = []
        all_ys = []

        # Обработка всех тестов.
        for file in files:

            # Получаем флаг позитивного кейса.
            if file in pos_files:
                is_pos = 1
            else:
                is_pos = 0

            wav = WAV('{0}/{1}'.format(directory, file), self.Settings)

            if wav.is_ok():
                for ch in wav.Channels:
                    loc_xs = ch.get_nnet_data_cases(self.Settings.Muted.CaseWidth,
                                                    self.Settings.Muted.CaseLearnStep)
                    loc_ys = [is_pos] * len(loc_xs)
                    all_xs = all_xs + loc_xs
                    all_ys = all_ys + loc_ys

        print('init_data_muted : collect : {0}'.format(time.time() - t0))

        # Перемешаваем данные.
        all_data = list(zip(all_xs, all_ys))
        random.shuffle(all_data)
        all_xs, all_ys = unzip(all_data)
        all_xs = np.array(all_xs)
        shp = all_xs.shape
        all_xs = all_xs.reshape((shp[0], shp[1] * shp[2]))
        all_xs = all_xs.astype('float32')
        all_ys = keras.utils.np_utils.to_categorical(all_ys, 2)

        print('init_data_muted : shuffle : {0}'.format(time.time() - t0))

        # Позиция для разделения данных на обучающую и тестовую выборки.
        p = int(len(all_xs) * self.Settings.Muted.TrainCasesPart)
        self.XTrain, self.XTest = split(all_xs, p)
        self.YTrain, self.YTest = split(all_ys, p)

        print('init_data_muted : '
              '{0} train and {1} test cases are constructed : '
              '{2}'.format(len(self.XTrain), len(self.XTest), time.time() - t0))

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

    def is_data_inited(self):
        """
        Проверка того, что данные инициализированы.

        :return: True  - если данные инициализированы,
                 False - в противном случае.
        """

        is_x_inited = (self.XTrain is not None) and (self.XTest is not None)
        is_y_inited = (self.YTrain is not None) and (self.YTest is not None)

        return is_x_inited and is_y_inited

    # ----------------------------------------------------------------------------------------------

    def init_model(self):
        """
        Иницализация модели.
        """

        # Не собираем модель, если данные не готовы.
        if not self.is_data_inited():
            return

        if self.Name == 'muted':
            self.init_model_muted()
        elif self.Name == 'mnist':
            self.init_model_mnist()
        else:
            raise Exception('unknown nnet {0}'.format(self.Name))

    # ----------------------------------------------------------------------------------------------

    def init_model_muted(self):
        """
        Инициализация модели muted.
        """

        # Сборка модели.
        self.Model = Sequential()
        self.Model.add(Dense(16, activation='relu', input_shape=(self.XTrain.shape[1],)))
        self.Model.add(Dropout(0.2))
        self.Model.add(Dense(16, activation='relu'))
        self.Model.add(Dropout(0.2))
        self.Model.add(Dense(2, activation='softmax'))

        # Компиляция модели.
        self.Model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.Model.summary()

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

    def is_model_inited(self):
        """
        Проверка того, что модель инициализирована.

        :return: True  - если модель инициализирована,
                 False - в противном случае.
        """

        return self.Model is not None

    # ----------------------------------------------------------------------------------------------

    def fit(self):
        """
        Обучение модели.
        """

        # Не учимся, если данные или модель не готовы.
        if not self.is_data_inited():
            return
        if not self.is_model_inited():
            return

        if self.Name == 'muted':
            self.fit_muted()
        elif self.Name == 'mnist':
            self.fit_mnist()
        else:
            raise Exception('unknown nnet {0}'.format(self.Name))

    # ----------------------------------------------------------------------------------------------

    def fit_muted(self):
        """
        Обучение модели muted.
        """

        self.Model.fit(self.XTrain, self.YTrain,
                       batch_size=128,
                       epochs=20,
                       verbose=1,
                       validation_data=(self.XTest, self.YTest))

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

        if self.Model is not None:
            self.Model.save('nnets/{0}.h5'.format(self.Name))

# ==================================================================================================


def get_settings():
    """
    Получение настроек.

    :return: Настройки.
    """

    defect_snap_settings = DefectSnapSettings(limits_before_sort=(0.7, 0.95),
                                              limits_after_sort=(0.25, 0.75),
                                              min_power_lo_threshold=5.0,
                                              half_snap_len=2,
                                              diff_min_max_powers_hi_threshold=5.0)
    defect_muted_settings = DefectMutedSettings(case_width=16,
                                                case_learn_step=10,
                                                train_cases_part=0.8,
                                                case_pred_step=16,
                                                category_detect_limits=(0.45, 0.55),
                                                part_for_decision=0.9)

    return DefectsSettings(limits_db=(-50.0, 50.0),
                           snap=defect_snap_settings,
                           muted=defect_muted_settings)


# ==================================================================================================


def analyze_directory(directory,
                      filter_fun,
                      defects_names,
                      verbose=False):
    """
    Анализ директории с файлами на наличие дефектов.

    :param directory:     Имя директории.
    :param filter_fun:    Дополнительная функция для отфильтровывания файлов, которые необходимо
                          анализировать.
    :param defects_names: Список имен дефектов.
    :param verbose:       Признак печати процесса анализа.

    :return: Список дефектов.
    """

    s = get_settings()

    fs = os.listdir(directory)
    ds = []

    for f in fs:
        if filter_fun(f):

            if verbose:
                print('.... process {0}'.format(f))

            wav = WAV('{0}/{1}'.format(directory, f), s)

            if wav.is_ok():
                ds = ds + wav.get_defects(defects_names)

    return ds


# ==================================================================================================


def unit_tests():
    """
    Короткие тесты.
    """

    # zipwith
    assert zipwith([1, 2, 3], [2, 3, 4], operator.add) == [3, 5, 7]
    assert zipwith(['a', 'b'], ['1', '2'], lambda x, y: (x, y)) == list(zip(['a', 'b'], ['1', '2']))

    # unzip
    assert unzip([('a', 1), ('b', 2)]) == (['a', 'b'], [1, 2])

    # min_max_extended
    assert min_max_extended([6, 3, 8, 2, 6, 3, 9, 2, 9, 1],
                            (0.15, 0.85), (0.25, 0.75)) == (2, 6)

    # indices_slice_array
    assert indices_slice_array(3, 0, 2, 1) == [(0, 2), (1, 3)]
    assert indices_slice_array(10, 3, 3, 2) == [(3, 6), (5, 8), (7, 10)]

    # predicated_count
    assert predicated_count([0, 0, 0, 1, 1, 1], lambda e: e > 0.5) == 3
    assert predicated_count([1, 'a', 2, 'b', 3], lambda e: type(e) is str) == 2


# ==================================================================================================


def nnet_test():
    """
    Тест нейронки.
    """

    nn = NNetTrainer('muted', get_settings())
    nn.init_data()
    nn.init_model()
    nn.fit()
    nn.save()


# ==================================================================================================


def main(filter_fun, defects_names):
    """
    Головная функция.

    :param filter_fun:    Функция отбора файлов для детектирования дефектов.
    :param defects_names: Список имен дефектов.
    """

    defects = analyze_directory('wavs/origin',
                                filter_fun=filter_fun,
                                defects_names=defects_names,
                                verbose=True)

    for d in defects:
        print(d)


# ==================================================================================================


if __name__ == '__main__':
    # unit_tests()
    # nnet_test()
    main(filter_fun=lambda f: True, defects_names=['snap', 'muted'])


# ==================================================================================================
