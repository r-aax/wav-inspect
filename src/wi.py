"""
Реализация модуля по обработке аудиозаписей.
"""

import os
import time
import itertools
import pathlib
import sklearn
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wi_utils
import wi_settings


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

        # Исходный спектр, который строит librosa.
        self.Spectre = None

        # Транспонированный спектр от librosa.
        self.TSpectre = None

        # Транспонированный спектр после нормализации.
        self.NSpectre = None

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

        # Генерация нормализованного спектра.
        (min_v, max_v) = self.Parent.Settings.LimitsDb
        min_v = max(min_v, self.TSpectre.min())
        max_v = min(max_v, self.TSpectre.max())
        self.NSpectre = self.TSpectre + 0.0
        np.clip(self.NSpectre, min_v, max_v, out=self.NSpectre)
        self.NSpectre = (self.NSpectre - min_v) / (max_v - min_v)

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Перевод точки времени в точку в матрице спектра.

        :param tx: Точка времени.

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

        idxs = wi_utils.indices_slice_array(self.NSpectre.shape[0], 0, width, step)

        return [self.NSpectre[fr:to] for (fr, to) in idxs]

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

        wi_utils.show_graph([sum(tsi) for tsi in self.TSpectre],
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

        d = [wi_utils.min_max_extended(tsi,
                                       limits_before_sort=limits_before_sort,
                                       limits_after_sort=limits_after_sort)
             for tsi in self.TSpectre]
        ud = wi_utils.unzip(d)

        if show_min:
            if show_max:
                wi_utils.show_graph(ud, figsize=figsize,
                                    title='Spectre Min/Max Power',
                                    style=['b', 'r'], linewidth=[2.0, 2.0])
            else:
                wi_utils.show_graph(ud[0], figsize=figsize,
                                    title='Spectre Min Power',
                                    style='b')
        else:
            if show_max:
                wi_utils.show_graph(ud[1], figsize=figsize,
                                    title='Spectre Max Power',
                                    style='r')
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

        d = [wi_utils.min_max_extended(tsi,
                                       limits_before_sort=limits_before_sort,
                                       limits_after_sort=limits_after_sort)
             for tsi in self.TSpectre]
        diffs = [d_max - d_min for (d_min, d_max) in d]

        wi_utils.show_graph(diffs, figsize=figsize, title='Spectre Min/Max Diff Power')

    # ----------------------------------------------------------------------------------------------

    def get_defect_snap_markers(self):
        """
        Получение маркеров дефекта snap.

        :return: Список маркеров snap.
        """

        s = self.Parent.Settings.Snap

        d = [wi_utils.min_max_extended(tsi,
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

    def get_defect_snap2_markers(self):
        """
        Получение маркеров дефекта snap2.

        :return: Список маркеров snap2.
        """

        s = self.Parent.Settings.Snap2

        # Применяем фильтр Собеля для выявления границ.
        ns2 = wi_utils.apply_filter_2d(self.NSpectre,
                                       wi_utils.operator_sobel_gy())

        # Получаем маркеры.
        w = s.FreqBlockWidth
        v = [(max(c[-w:]), max(c[-2 * w:-w]), max(c[-3 * w:-2 * w]), max(c[-4 * w:-3 * w]))
             for c in ns2]
        y = [min(vi) for vi in v]
        hi, lo = s.HiThreshold, s.LoThreshold
        markers = [(i > s.HalfSnapLen) and (i < len(v) - s.HalfSnapLen)
                   and (y[i] > hi) and (y[i - s.HalfSnapLen] < lo) and (y[i + s.HalfSnapLen] < lo)
                   for i in range(len(v))]

        return markers

    # ----------------------------------------------------------------------------------------------

    def get_defect_comet_markers(self):
        """
        Получение маркеров дефекта comet.

        :return: Список маркеров comet.
        """

        s = self.Parent.Settings.Comet

        # Получаем маркеры.
        orth = [wi_utils.array_orthocenter(c) for c in self.NSpectre]
        lev = [max(c) for c in self.NSpectre]
        qu = [wi_utils.array_weight_quartile(c) for c in self.NSpectre]

        return [orth[i] * (lev[i] > s.SignalThreshold) * qu[i] > s.OrthQuartileThreshold
                for i in range(len(orth))]

    # ----------------------------------------------------------------------------------------------

    def show_defect_snap_markers(self, figsize=(20, 8)):
        """
        Демонстрация маркеров дефекта snap.

        :param figsize: Размер картинки.
        """

        markers = self.get_defect_snap_markers()
        wi_utils.show_graph(markers, figsize=figsize, title='Defect Snap Markers')

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

    def get_defects_snap2(self):
        """
        Получение дефектов snap2.

        :return: Список дефектов snap2.
        """

        markers = self.get_defect_snap2_markers()

        # Формируем список дефектов.
        objs = [Defect(self.Parent.FileName,
                       self.Channel,
                       'snap2',
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

        ns = self.NSpectre
        h = ns.shape[1]
        weights = np.array([range(h)] * ns.shape[0])
        ns2 = ns * weights
        y = [sum(ns2[i]) / (sum(ns[i]) + 1e-10) for i in range(ns.shape[0])]
        ind = sum(y) / len(y)

        # Принимаем решение о глухой записи, по порогу среднего значения ортоцентра.
        if sum(y) / len(y) < s.OrthocenterThreshold:
            return [Defect(self.Parent.FileName,
                           self.Channel,
                           'muted',
                           (0.0, self.Parent.Duration))]
        else:
            return []

    # ----------------------------------------------------------------------------------------------

    def get_defects_comet(self):
        """
        Получение дефектов comet.

        :return: Список дефектов comet.
        """

        markers = self.get_defect_comet_markers()
        ivs = wi_utils.markers_true_intervals(markers)

        objs = [Defect(self.Parent.FileName,
                       self.Channel,
                       'comet',
                       (self.specpos_to_time(iv[0]), self.specpos_to_time(iv[1])))
                for iv in ivs]

        return objs

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted2(self, lim_down=6, lim_db=-37, hop_length=512):
        """
        Функция обнаружения эффекта глухой записи в аудиосигнале.

        :param lim: уровень пустоты - определяет сколько верхних пустых слоев подряд должно быть,
            чтобы сказать, что аудиосигнал имеет участки глухой записи (тип - int, измеряется в фреймах)
        :param hop_length: размер окна кратковременного преобразования Фурье (тип - int, предпочтительно значениее равное степени 2)

        :return: список пар фреймов (начало и конец) события
        """

        # чтение файла
        # x, sr = librosa.load(filename, mono=True, sr=None)

        # кратковременное преобразование Фурье
        X = librosa.stft(self.Y, hop_length=hop_length) # вместо х - self.Y
        Xmag = abs(X)
        Xdb = librosa.amplitude_to_db(Xmag)

        # отбросить элементы, содержащие тишину
        # поиск тишины
        silence = self.get_silence(self.Y)
        frame_silence = librosa.samples_to_frames(silence, hop_length=hop_length)

        # обнулить фреймы с тишиной
        for fs in frame_silence:

            # приравниваем диапозон фреймов к нулю
            Xdb[-1][fs[0]: fs[-1]] = 0

        # пустой массив для записи в него факта наличия пустот (1) или их отсутствие (0)
        void = []

        # контейнер для записи фреймов конца и начала события
        frame = []
        # индикатор наличия эффекта
        deaf = 0

        # считываем элементы верхней строки
        for inum, ival in enumerate(Xdb[-1]):

            # фильтр для обнаружения пустых частот, которые имеют значение -37.623936
            if ival < lim_db:

                # условие выполнено - на верхних частототах обнаружен ноль
                # начинаем анализировать глубину отсутствующих частот
                # инициализируем (обновляем) счетчики
                # счетчик глубины глухой записи (на первой строке исходного массива уже есть первый признак)
                zero = 1

                # начинаем идти вниз по исходной матрице, анализируя глубину обнаруженного эффекта
                for num, val in enumerate(Xdb[::-1], 1):

                    # если счетчик вышел за диапозон значений массива, то завершить цикл
                    if num == len(Xdb):
                        break

                    # если следующее значение нулевое, то обновить счетчик глубины события
                    if Xdb[::-1][num][inum] < lim_db:
                        zero += 1

                    # если следующее значение не нулевое, то завершаем цикл, ибо это уже не подпадает под критерии эффекта
                    else:
                        break

                    # известно, что анализ происходит внутри найденного эффекта (deaf = 1)
                    # если глубина соответсвует фильру обнаружения эффекта, то завершаем цикл для экономии ресурсов
                    if zero >= lim_down and deaf == 1:
                        break

                    # если счетчик глубины больше или равен заданному порогу детектирования эффекта
                    # и до этого не было зарегистрировано наличия эффекта (deaf = 0)
                    # то начало эффекта глухой записи найдено!
                    if zero >= lim_down and deaf == 0:
                        # записывает номер фрейма, где встретили эффект
                        frame.append([inum])

                        # далее все действия идут с информацией о том, что анализируется продолжительность найденного эффекта
                        deaf = 1

                        # завершаем цикл спуска, ибо начало эффекта уже зарегистрированно и нет смысла трать ресурсы
                        break

                # цикл анализа глубины события окончен
                # проверяем результат анализа глубины события

                # известно, что спуск происходил в найденном эффекте
                # но исследуемый столбец не привысил порог фильтра обнаружения эффекта
                # значит это конец эффекта
                if zero < lim_down and deaf == 1:

                    # обновляем счетчик - выходим из эффекта
                    deaf = 0

                    # записываем анализируемый фрейм как окончание эффекта
                    frame[-1].append(inum)

                # известно, что спуск происходил в найденном эффекте
                # исследуемый элемен является последним
                # тогда просто закрываем запись данных
                elif deaf == 1 and inum == len(Xdb[-1]) - 1:

                    # обновляем счетчик - выходим из эффекта
                    deaf = 0

                    # записываем анализируемый фрейм как окончание эффекта
                    frame[-1].append(inum)

            # известно, что спуск происходил в найденном эффекте
            # исследуемый элемент больше порога
            # это означает конец эффекта
            elif ival > lim_db and deaf == 1:

                # обновляем счетчик - выходим из эффекта
                deaf = 0

                # записываем анализируемый фрейм как окончание эффекта
                frame[-1].append(inum)

        # вывод резудьтата
        if len(frame) > 0:
            t2 = librosa.frames_to_time(frame, sr=self.Parent.SampleRate, hop_length=512)
            for td in t2:

                return [Defect(self.Parent.FileName,
                           self.Channel,
                           'muted2',
                           (td[0], td[-1]))]
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
        elif defect_name == 'snap2':
            return self.get_defects_snap2()
        elif defect_name == 'muted':
            return self.get_defects_muted()
        elif defect_name == 'comet':
            return self.get_defects_comet()
        elif defect_name == 'muted2':
            return self.get_defects_muted2()
        else:
            raise Exception('unknown defect name ({0})'.format(defect_name))

    # ----------------------------------------------------------------------------------------------

    def get_silence(self, x, limx = 0.01, hop_length=512):
        """
        Функция обнаружения тишины

        :param x: исследуемая запись
        :param limx: порог обнаружения тишины (тип - float)
        :param hop_length: размер окна преобразования Фурье (тип - int, предпочтительно значениее равное степени 2)

        :return: список пар начала и конца тишины
        """

        # объявление внутренних переменных
        # индекс явления (0 - звук, 1 - тишина)
        into = 0

        #  список для записи начала и конца явления
        t = []

        # анализируем исходный файл записи
        for inum, iv in enumerate(x):

            # если сейчас есть звук
            # и найдено значение ниже фильтра
            # то это тишина
            if into == 0 and iv < limx:

                # тишина встретилась впервые
                # сколько раз она повторилась
                cont = 1

                # теперь исследуется тишина
                into = 1

                # запись начала явления
                t.append([inum])

            # если сейчас тишина
            # и найдено значение ниже фильтра
            # то это продолжение тишины
            elif into == 1 and iv < limx:

                # считаем сколько раз это повторится
                cont += 1

            # если сейчас тишина
            # и значение последнее
            # то фиксируем данные
            if into == 1 and inum == len(x) - 1:

                # обнуляем счетчики
                cont = 0
                into = 0

                # записываем конец тишины
                t[-1].append(inum)

            # если сейчас тишина
            # и значение выше фильтра
            # и кол-во повторений тишины больше или равно размеру фрейма
            # то тишина закончилась
            elif into == 1 and iv >= limx and cont >= hop_length:

                # обнуляем счетчик
                cont = 0

                # теперь исследуем звук
                into = 0

                # стираем запись начала тишины
                t[-1].append(inum)

            # если сейчас тишина
            # и значение выше фильтра
            # и кол-во повторений тишины меньше размера фрейма
            # то пренебрегаем этой тишиной
            elif into == 1 and iv >= limx and cont < hop_length:

                # обнуляем счетчик
                cont = 0

                # теперь исследуем звук
                into = 0

                # стираем запись начала тишины
                t.pop(-1)

        return t

# ==================================================================================================


class WAV:
    """
    Аудиоззапись.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, filename, settings=wi_settings.defects_settings):
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

    def get_defects_snap2(self):
        """
        Получение маркеров дефекта snap2.

        :return: Список дефектов snap2.
        """

        return self.get_defects_from_both_channels('snap2')

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self):
        """
        Получение маркеров дефекта muted.

        :return: Список дефектов muted.
        """

        return self.get_defects_from_both_channels('muted')

    # ----------------------------------------------------------------------------------------------

    def get_defects_comet(self):
        """
        Получение маркеров дефекта comet.

        :return: Список дефектов comet.
        """

        return self.get_defects_from_both_channels('comet')

    # ----------------------------------------------------------------------------------------------

    def get_defects_by_name(self, defect_name):
        """
        Получение списка дефектов по имени.

        :param defect_name: Имя дефекта.

        :return: Список дефектов.
        """

        if defect_name == 'snap':
            return self.get_defects_snap()
        elif defect_name == 'snap2':
            return self.get_defects_snap2()
        elif defect_name == 'muted':
            return self.get_defects_muted()
        elif defect_name == 'muted2':
            return self.get_defects_muted2()
        elif defect_name == 'comet':
            return self.get_defects_comet()
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
        t - время начала и конца эффекта эхо (тип - list, len(t) => 2, t[0] - начало эха (float), t[1] - конец эхо (float))
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
            return t
            
        else:

            print('В данном аудиосигнале эхо не найдено!')
 
    # ----------------------------------------------------------------------------------------------
    
    def get_defects_muted2(self):

        """
        Получение маркеров дефекта muted2.

        :return: Список дефектов muted2.
        """

        return self.get_defects_from_both_channels('muted2')

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

    fs = os.listdir(directory)
    ds = []
    records_count = 0
    records_time = 0.0
    ts = time.time()

    for f in fs:
        if filter_fun(f):

            if verbose:
                print('.... process {0}'.format(f))

            wav = WAV('{0}/{1}'.format(directory, f))

            if wav.is_ok():
                records_count = records_count + 1
                records_time = records_time + wav.Duration
                ds = ds + wav.get_defects(defects_names)

    print('Process finished:')
    print('    {0} records processed'.format(records_count))
    print('    {0} s of audio records processed'.format(records_time))
    print('    {0} defects found'.format(len(ds)))
    print('    {0} s time estimated'.format(time.time() - ts))

    return ds


# ==================================================================================================


def run(directory, filter_fun, defects_names):
    """
    Головная функция.

    :param directory:     Имя директории.
    :param filter_fun:    Функция отбора файлов для детектирования дефектов.
    :param defects_names: Список имен дефектов.
    """

    defects = analyze_directory(directory,
                                filter_fun=filter_fun,
                                defects_names=defects_names,
                                verbose=True)

    for d in defects:
        print(d)


# ==================================================================================================


if __name__ == '__main__':

    run(directory='wavs/origin',
        filter_fun=lambda f: True,
        defects_names=['snap', 'snap2', 'muted'])


# ==================================================================================================
