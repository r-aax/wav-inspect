"""
Реализация модуля по обработке аудиозаписей.
"""

import math
import os
import time
import itertools
import pathlib
import scipy.ndimage
import sklearn
import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wi_utils
import wi_settings


# ==================================================================================================


class Separator:
    """
    Класс для разделения массива амплитуд.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, whole_size, sep_sizes):
        """
        Конструктор.

        :param whole_size: Полный размер (целое).
        :param sep_sizes:  Кортеж из настроек разделения
                           (размер отрезка - целое, минимальный размер хвоста - целое).
        """

        self.WholeSize = whole_size
        self.ChunkSize = sep_sizes[0]
        self.MinSize = sep_sizes[1]
        self.CurStart = 0

    # ----------------------------------------------------------------------------------------------

    def reset(self):
        """
        Сброс разделителя.
        """

        self.CurStart = 0

    # ----------------------------------------------------------------------------------------------

    def get_next(self):
        """
        Получение индексов следующего фрагмента.

        :return: Кортеж из индексов следующего фрагмента (либо None).
        """

        # Случай, когда фрагмент полностью укладывается.
        if self.CurStart + self.ChunkSize <= self.WholeSize:
            res = (self.CurStart, self.CurStart + self.ChunkSize)
            self.CurStart = self.CurStart + self.ChunkSize
            return res

        # Фрагмент полностью не укладываается, но хвост можно вернуть.
        if self.WholeSize - self.CurStart > self.MinSize:
            res = (self.CurStart, self.WholeSize)
            self.CurStart = self.WholeSize
            return res

        # Фрагмент взять не получится.
        return None

# ==================================================================================================


class DefectsList:
    """
    Список дефектов.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Конструктор.
        """

        # Описание дефектов храним обычным списком.
        self.L = []

    # ----------------------------------------------------------------------------------------------

    def get(self):
        """
        Получение списка описаний дефектов.

        :return: Список описаний дефектов.
        """

        return self.L

    # ----------------------------------------------------------------------------------------------

    def add(self, rec, ch, name, beg, end):
        """
        Добавление дефекта к списку описаний.

        :param rec:  Имя записи.
        :param ch:   Номер канала.
        :param name: Имя дефекта.
        :param beg:  Начало дефекта.
        :param end:  Конец дефекта.
        """

        # Попытка соединить дефект с предыдущим, если совпадает запись, канал и имя.
        if len(self.L) > 0:
            d = self.L[-1]
            if (d['rec'] == rec) and (d['ch'] == ch) and (d['name'] == name):
                if abs(d['end'] - beg) < 1.0e-3:
                    # Текущий дефект можно прицепить к предыдущему.
                    d['end'] = end
                    return

        # Требуется создать новый дефект.
        self.L.append({'rec': rec, 'ch': ch, 'name': name, 'beg': beg, 'end': end})

# ==================================================================================================


class Chunk:
    """
    Часть массива амплитуд.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, parent, channel, offset, duration, y):
        """
        Конструктор канала.

        :param parent:   Родительская запись.
        :param channel:  Номер канала.
        :param offset:   Смещение от начала канала (секунды).
        :param duration: Длительность фрагмента (секунды).
        :param y:        Массив амплитуд.
        """

        self.Parent = parent
        self.Channel = channel
        self.Offset = offset
        self.Duration = duration
        self.Y = y

        # Исходный и транспонированный спектр, который строит librosa.
        self.ASpectre = None
        self.Spectre = None
        self.TSpectre = None

        # Матрица нормализованного спектр, которая является массивом горизонтальных линий.
        self.H = None

        # Матрица нормализованного спектра, которая является массивом вертикальных линий.
        self.V = None

    # ----------------------------------------------------------------------------------------------

    def generate_spectres(self):
        """
        Генерация спектра.
        """

        # Генерируем спектр.
        self.ASpectre = abs(librosa.stft(self.Y, n_fft=2048))
        self.Spectre = librosa.amplitude_to_db(self.ASpectre)
        self.TSpectre = self.Spectre.transpose()

        # Нормализуем матрицу спектра.
        self.H = self.Spectre + 0.0
        (min_val, max_val) = self.Parent.Settings.LimitsDb
        min_val = max(min_val, self.H.min())
        max_val = min(max_val, self.H.max())
        np.clip(self.H, min_val, max_val, out=self.H)
        self.H = (self.H - min_val) / (max_val - min_val)

        # Транспонируем матрицу, чтобы получился массив вертикальных линий,
        # при этом в каждой линии в нижних индексах лежат низкие частоты и наоборот.
        self.V = self.H.transpose()

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Перевод точки времени в точку в матрице спектра.

        :param tx: Точка времени.

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

    def music_to_voice(self,
                       margin_v=11.75,
                       power=3,
                       lim_percent=24,
                       top_db=32):
        """
        разделяет аудиозапись на музыку и голос.

        :param margin_v: Множетель фильтра шумов.
        :param power:  Число мощности для выделения маски (>0, целое).
        :param lim_percent: лимит голоса в записи в процентах (от 0 до 100)
        :param top_db: предел громкости для отсеивания тишины
        """

        # убираем тишину и формируем семпл без тишины
        semp = librosa.effects.split(y=self.Y, top_db=top_db)

        index_semp = []
        for seq in semp:
            index_semp += [i for i in range(seq[0], seq[-1])]

        # новый семпл без тишины
        y = self.Y[index_semp]

        S_full, phase = librosa.magphase(librosa.stft(y))

        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine'
                                               )

        S_filter = np.minimum(S_full, S_filter)

        margin_v = margin_v
        power = power

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)

        S_foreground = mask_v * S_full

        stft = S_foreground * phase

        # в записи оставили только голос
        y_foreground = librosa.istft(stft)

        lim_percent = lim_percent

        lim_percent = len(self.Y)*lim_percent/100

        # из голоса вырезаем тишину
        semp = librosa.effects.split(y=y_foreground, top_db=top_db)

        coin = 0

        for c in semp:
            for num in range(c[0], c[1]):
                coin += 1

        # если голоса больше порога, то
        if coin >= lim_percent:
            return True

        # если голоса осталось мало
        else:
            return False

    # ----------------------------------------------------------------------------------------------

    def get_silence2(self, x, limx = 0.02, hop_length=512, Xdb=None):

        '''
        :param x: аудиозапись
        :param limx: предел амплитуды для однаружения тишины
        :param hop_length: ширина одно фрейма
        :return: матрица частот с обнуленными столбцами в местах нахождения тишины
        '''

        tms = []
        for wf in range(0, len(x) + 1, hop_length):

            bm = x[wf:wf + hop_length] <= limx

            # если это тишина, то
            if all(bm):
                tms.append(0)

            # если это звук, то
            else:
                tms.append(1)

        # обнуляем все тихие места
        res = np.multiply(Xdb, tms)

        # print(res)
        return res

    # ----------------------------------------------------------------------------------------------

    def get_volume_level(self,
                         Tg = 0.4, # 400 мс
                         T_itr = 10, # количество строблений
                         ):

        '''

        :param Tg: Размер стробления (сек).
        :param T_itr: Количество строблений на участке Т.

        :return: Список громкости звука на участках длиной Т.
        '''

        T = round(T_itr*0.1+Tg-0.1, 1)

        x, sr = librosa.load(self.FileName, mono=False, sr=48000)

        # К-фильтр частот
        # коэф-ты при sr = 48000:
        a1 = 1.69065929318241 * (-1)
        a2 = 0.73248077421585
        b0 = 1.53512485958697
        b1 = 2.69169618940638 * (-1)
        b2 = 1.19839281085285

        # цифровой биквадратный блок
        # через каскадное включение
        # рекурсивного и нерекурсивного фильтров
        y = []
        for xi in x[:5]:

            # промежуточный сигнал
            wi = []

            # итоговый сигнал
            yi = []

            # включение рекурсивного фильтра
            for num, val in enumerate(xi):
                if num <= 1:
                    wi.append(val)  # val или 0
                else:
                    wi.append(xi[num] - a1 * wi[num - 1] - a2 * wi[num - 2])

            # включение нерекурсивного фильтра
            for numw, valw in enumerate(wi):
                if numw <= 1:
                    yi.append(valw)  # val или 0
                else:
                    yi.append(b0 * wi[numw] + b1 * wi[numw - 1] + b2 * wi[numw - 2])

            y.append(yi)

        y = np.array(y)

        # К-фильтр частот
        # коэф-ты при sr = 48000:
        a1 = 1.99004745483398 * (-1)
        a2 = 0.99007225036621
        b0 = 1.0
        b1 = 2.0 * (-1)
        b2 = 1.0

        x = y

        # цифровой биквадратный блок
        # через каскадное включение
        # рекурсивного и нерекурсивного фильтров
        y = []
        for xi in x[:5]:

            # промежуточный сигнал
            wi = []

            # итоговый сигнал
            yi = []

            # включение рекурсивного фильтра
            for num, val in enumerate(xi):
                if num <= 1:
                    wi.append(val)  # val или 0
                else:
                    wi.append(xi[num] - a1 * wi[num - 1] - a2 * wi[num - 2])

            # включение нерекурсивного фильтра
            for numw, valw in enumerate(wi):
                if numw <= 1:
                    yi.append(valw)  # val или 0
                else:
                    yi.append(b0 * wi[numw] + b1 * wi[numw - 1] + b2 * wi[numw - 2])

            y.append(yi)

        y = np.array(y)

        # среднеквадратичное значение

        # Энергия j-го стробирующего блока по каналам
        zij = [[], []]

        # дважды отфильтрованный сигнал разбиваем на два канала
        for zni, zi in enumerate(y):

            # канал разбиваем на Т-интерваллы (без хвоста)

            step_seg = int(sr * T)
            segments = np.array([zi[i:i + step_seg] for i in range(0, len(zi), int(step_seg))])
            if len(segments[-1]) < step_seg:
                segments = segments[:-1]

            # Т-интерваллы (каждый) разбиваем на Тg-интерваллы с перекрытием в 75%
            for T_int in segments:

                step_seg = int(sr * Tg)
                segments_Tg = [T_int[i:i+step_seg] for i in range(0, T_itr*int(step_seg/4), int(step_seg/4))]

                # вычисляем энергию каждого Tg-интервала
                for Tg_int in segments_Tg:
                    zj = (1 / len(Tg_int)) * sum(Tg_int * Tg_int)

                    zij[zni].append(zj)

        G = [1, 1, 1, 1.41, 1.41]  # весовые коэф-ты каналов

        # Стробированная громкость в интервале измерения T
        lkg = []

        for tj in range(len(segments)):

            sumij = []

            for ti in range(len(y)):
                sumij.append(G[ti] * (sum(zij[ti][tj * T_itr:tj * T_itr + T_itr]) / T_itr))

            lkg.append(-0.691 + 10 * math.log10(sum(sumij)))

        return lkg

    # ----------------------------------------------------------------------------------------------

    def get_defects_click(self, dlist):
        """
        Получение дефектов click.

        :param dlist: Список дефектов.
        """

        s = self.Parent.Settings.Click
        qs, win_h, win_w = s.Quartiles, s.WinH, s.WinW

        # Квартиль списка.
        def q(a, ind, width):
            return a[ind * width: (ind + 1) * width]

        # Отрезаем верхнюю часть частот и прогоняем через фильтр Собеля для выявления границ.
        v = self.V[:, -qs * win_h:]
        v = cv2.filter2D(v, -1, np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]))

        # Проходим по всем окнам и ищем в них щелчки.
        # Каждое окно нужно нормализовть отдельно, чтобы щелчки разной интенсивности
        # не экранировали друг друга.
        for i in range(len(v) // win_w):
            vi = q(v, i, win_w)
            np.clip(vi, 0.0, 1.0, out=vi)
            mm = vi.max()
            if mm > 0.0:
                vi = vi / mm
            y = np.array([min([max(q(c, qi, win_h)) for qi in range(qs)]) for c in vi])
            if (y.max() - y.mean() > s.Thr) and (y.mean() < s.MeanThr):
                t = self.specpos_to_time(win_w * i + np.argmax(y))
                dlist.add(self.Parent.FileName, self.Channel, 'click',
                          self.Offset + t, self.Offset + t)

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self, dlist):
        """
        Получение дефектов muted.

        :param dlist: Список дефектов.
        """

        s = self.Parent.Settings.Muted

        ns = self.V
        w, h = ns.shape[0], ns.shape[1]
        weights = np.array([range(h)] * w)
        nsw = ns * weights
        y = np.array([sum(nsw[i]) / (sum(ns[i]) + 1e-10) for i in range(w)])
        p = 100.0 * y.mean() / h

        # Принимаем решение о глухой записи, по порогу среднего значения ортоцентра.
        if p < s.Thr:
            dlist.add(self.Parent.FileName, self.Channel, 'muted',
                      self.Offset, self.Offset + self.Duration)

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted2(self, dlist):
        """
        Получение дефектов muted2.

        :param dlist: Список дефектов.
        """

        s = self.Parent.Settings.Muted2

        # кратковременное преобразование Фурье
        Xdb = self.Spectre

        # порог db, ниже которых детектируем пустоту
        lim_db = int((int(Xdb.max()) - int(Xdb.min())) / 100 * s.PercentageOfLimDb) + int(Xdb.min())

        # обнулить фреймы с тишиной
        Xdb = self.get_silence2(x=self.Y, limx=s.Muted2Silence, Xdb=Xdb)

        # нашли пустоты во всей матрице - все что ниже пороговой амплитуды
        Xdb = Xdb <= lim_db

        # отсекаем нижню часть спектрограммы
        val_void_in_song = int(Xdb.shape[0] / 100 * s.PercentNotVoid)
        Xdb = Xdb[val_void_in_song:]

        # процент погрешности
        val_of_error = int(Xdb.shape[0] / 100 * s.PercentageOfError)

        # отсортировать и срезать погрешность
        Xdb.sort(axis=0)  # сортировка по столбцам
        Xdb = Xdb[val_of_error:]

        # проверяем наличие пустот в верхней части с учетом ввычета погрешности
        void_frame = []
        for nf in range(Xdb.shape[1]):
            if all(Xdb.T[nf]):
                void_frame.append(nf)

        # высчитываем процент глухих фреймов
        lim_frame = int(s.LimPercentFrame * Xdb.T.shape[0] / 100)

        # вывод резудьтата
        # если глухих фреймов больше лимита, то запись глухая
        if len(void_frame) >= lim_frame:
            dlist.add(self.Parent.FileName, self.Channel, 'muted2',
                      self.Offset, self.Offset + self.Duration)

    # ----------------------------------------------------------------------------------------------

    def get_defects_hum(self, dlist):
        """
        Получение дефектов hum.

        :param dlist: Список дефектов.
        """

        s = self.Parent.Settings.Hum

        # Отрезаем совсем нижние частоты.
        h = self.H[int((s.LoIgnore / 100.0) * len(self.H)):880]
        r = [0] * len(h)

        # Номера квантилей не передаются через настройки, это magic numbers,
        # как и параметры сглаживания.

        for i in range(len(r)):
            hi = h[i]
            hi.sort()
            q55 = hi[55 * len(hi) // 100]
            if q55 > 0.0:
                r[i] = hi[len(hi) // 10] / q55

        rsm = sp.signal.savgol_filter(r, 15, 3)
        d = r - rsm

        if d.max() > s.Thr:
            dlist.add(self.Parent.FileName, self.Channel, 'hum',
                      self.Offset, self.Offset + self.Duration)

    # ----------------------------------------------------------------------------------------------

    def get_defects_dense(self, dlist):
        """
        Получение дефектов dense.

        :param defects: Список дефектов.
        """

        # нормализованная спектрограмма
        xh = self.H

        # задаем процент погрешности
        perset = 3

        # выбираем рабочий диапозон частот
        xh = xh[500:900]

        # смещаем частоты вправо сортировкой
        xh.sort(axis=1)

        # условие нахождения звукового события
        # сейчас: если в данном окне частот что-то есть, то True
        # надо: детектор горизонтальных линий
        if xh.T[0:int(xh.shape[1] / 100 * perset)].max() > 0:

            # условие выполненно, записать фреймы начала и конца
            dlist.add(self.Parent.FileName, self.Channel, 'dense',
                      self.Offset, self.Offset + self.Duration)

    # ----------------------------------------------------------------------------------------------

    def get_defects_satur(self, dlist):
        """
        Получение дефектов satur.

        :param dlist: Список дефектов.
        """

        r = librosa.feature.rms(S=self.ASpectre)[0]
        rs = scipy.ndimage.uniform_filter1d(r, size=self.Parent.Settings.Satur.FilterWidth)
        m = [(rsi > self.Parent.Settings.Satur.PowerThr) for rsi in rs]
        intervals = wi_utils.markers_true_intervals(m)

        for interval in intervals:
            dlist.add(self.Parent.FileName, self.Channel, 'satur',
                      self.Offset + self.specpos_to_time(interval[0]),
                      self.Offset + self.specpos_to_time(interval[1]))

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

        # Считанный массив амплитуд.
        # Если запись моно, то массив одномерный.
        # Если запись стерео, то массив двумерный.
        self.Ys = None

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

        return self.Ys is not None

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
            self.Ys, self.SampleRate = librosa.load(filename, sr=None, mono=False)
            self.Duration = librosa.get_duration(y=self.Ys, sr=self.SampleRate)

        except BaseException:
            # Если что-то пошло не так, то не разбираемся с этим, а просто игнорим ошибку.
            return False

        # Загрузка прошла успешно.
        return True

    # ----------------------------------------------------------------------------------------------

    def channels_count(self):
        """
        Получение количества каналов.

        :return: Количество каналов.
        """

        return self.Ys.shape[0]

    # ----------------------------------------------------------------------------------------------

    def get_chunk(self, channel_num, chunk_coords):
        """
        Получение фрагмента.

        :param channel_num:  Номер канала.
        :param chunk_coords: Координаты фрагмента.

        :return: Фрагмент.
        """

        offset = chunk_coords[0]
        duration = chunk_coords[1] - offset
        y_beg = int(chunk_coords[0] * self.SampleRate)
        y_end = int(chunk_coords[1] * self.SampleRate)

        return Chunk(self, channel_num, offset, duration, self.Ys[channel_num][y_beg:y_end])

    # ----------------------------------------------------------------------------------------------

    def get_chunks_pair(self, coords):
        """
        Получение пары фрагментов из соседних каналов.

        :param coords: Координаты фрагментов.

        :return: Пара фрагментов.
        """

        return (self.get_chunk(0, coords), self.get_chunk(1, coords))

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

    def get_defects_click(self, defects):
        """
        Получение дефектов click.

        :param defects: Список дефектов.
        """

        # Обнаружение дефекта кратковременного щелчка.
        # Щелчок ищется в диапазоне верхних частот, который разделяется на равные по ширине полосы
        # (они называются квартили, так как по умолчанию их 4, однако количество и ширина
        # квартилей передается в качестве опций).
        # Статистика по минимальному скачку в каждом из квартилей ищется на нормализованной
        # спектрограмме после применения свертки с помощью оператора Собеля (для выделения
        # вертикальных границ.
        # При этом сканирование выполняется в ограниченном временном окне, с нормализацией
        # данного окна (это делается для избежания экранирования слабых щелчков более сильными).
        # Критерием щелчка является сильный кратковременный скачок (превышение локального
        # предела по интенсивности и низкое значение средней интенсивности по всему окну).

        for channel_num in range(self.channels_count()):
            s = Separator(self.Duration, self.Settings.Click.Sep)

            chunk_coords = s.get_next()
            while chunk_coords:
                ch = self.get_chunk(channel_num, chunk_coords)
                ch.generate_spectres()
                ch.get_defects_click(defects)
                chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self, defects):
        """
        Получение маркеров дефекта muted.

        :param defects: Список дефектов.
        """

        # Обнаружение дефекта глухой записи.
        # Решение принимается на основе среднего значения ортоцентра нормализованной спектрограммы.
        # При этом ортоцентр спектрограммы дает адекватное решение для учатков тишины
        # (ортоцентр на абсолютной тишине находится на уровне 50%, что позволяет не трактовать
        # тишину как глухую запись).
        # При этом ортоцентр вычисляется из нормализованной спектрограммы прямым счетом
        # (получение центроида средствами librosa дает принципиально другой результат,
        # поэтому данная функция не используется).
        # Причина глухоты записи не определяется.

        for channel_num in range(self.channels_count()):
            s = Separator(self.Duration, self.Settings.Muted.Sep)

            chunk_coords = s.get_next()
            while chunk_coords:
                ch = self.get_chunk(channel_num, chunk_coords)
                ch.generate_spectres()
                ch.get_defects_muted(defects)
                chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted2(self, defects):

        """
        Получение маркеров дефекта muted2.

        :param defects: Список дефектов.
        """

        for channel_num in range(self.channels_count()):
            s = Separator(self.Duration, self.Settings.Muted2.Sep)

            chunk_coords = s.get_next()
            while chunk_coords:
                ch = self.get_chunk(channel_num, chunk_coords)
                ch.generate_spectres()
                ch.get_defects_muted2(defects)
                chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_echo(self, dlist):

        """
        Получение маркеров дефекта echo.

        :param dlist: Список дефектов.
        """

        # Сильное эхо детектируется по значению автокорреляционной фунции по короткому окну.
        # Если использовать только данный признак, то возникает слишком много ложных срабатываний
        # на музыке, для которой характерно повторение звуков (темп, барабаны и т.д.).
        # Для отсечения музыки вычисляется глобальное значение автокорреляционной функции
        # для темпограммы - сохранение данной величины на высоком уровне свидетельствует
        # о сохранении темпа в записи.
        # Для скорости отсечение по глобальной автокорреляции темпограммы делаем по одному
        # каналу.

        s = Separator(self.Duration, self.Settings.Echo.Sep)
        chunk_coords = s.get_next()

        while chunk_coords:
            chunks = self.get_chunks_pair(chunk_coords)

            # Ориентируемся по каналу 0 для определения подходящей темпограммы.
            # Для этого не требуется построение спектрограмм.
            oenv = librosa.onset.onset_strength(y=chunks[0].Y, sr=self.SampleRate)
            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.SampleRate)
            acg = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
            acg = librosa.util.normalize(acg)
            if (acg[:len(acg) // 2].mean()) > self.Settings.Echo.GlobNormCorrThr:
                return

            for ch in chunks:
                # Для анализа эхо не строим никакие спектрограммы.
                ln = int(self.SampleRate * self.Settings.Echo.LocCorrWin)
                parts = len(ch.Y) // ln
                for i in range(parts):
                    yp = ch.Y[i * ln: (i + 1) * ln]
                    ac = librosa.autocorrelate(yp)
                    ac = ac[ln // 5:]
                    if max(ac) > self.Settings.Echo.LocCorrThr:
                        dlist.add(self.FileName, ch.Channel, 'echo',
                                  ch.Offset + i * (ln / self.SampleRate),
                                  ch.Offset + (i + 1) * (ln / self.SampleRate))

            chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_asnc(self, dlist):
        """
        Получение дефектов asnc.

        :param dlist: Список дефектов.
        """

        # Дефект рассинхронизации каналов выполняется практически бесплатно на
        # основе анализа коэффициента корреляции Пирсона.
        # На основании данного коэффициента могут быть приняты решения:
        #   - если коэффициент близок к 1.0, то имеет место дефект ложного стерео
        #     (мы это не фиксируем, так как таковы практически все аудиозаписи);
        #   - если коэфициент слишком низкий и положительный, то имеет место
        #     эффект сильного расхождения, плывущий звук, ощутимые выпадения звука,
        #     запись на слух очевидно дефектная;
        #   - отрицательный коэффициент означает сдвиг по фазе между каналами
        #     (среди тестовых записей данного дефекта нет).
        #
        # Источник:
        # Pablo Alonzo-Jimenez, Luis Joglar-Ongay, Xavier Serra, Dmitry Bogdanov.
        # Automatic detection of audio problems for quality control in
        # digital music distribution.
        # Audio Engineering Society, Convention paper 10205.
        # 146-th Convention, 2019 March 20-23, Dublin, Ireland.

        s = Separator(self.Duration, self.Settings.Asnc.Sep)
        chunk_coords = s.get_next()

        while chunk_coords:
            ch0, ch1 = self.get_chunks_pair(chunk_coords)

            # Рассинхрон не требует анализа спектров - ничего не строим.

            c = sp.stats.pearsonr(ch0.Y, ch1.Y)

            if c[0] < self.Settings.Asnc.Thr:
                # Вместо номера канала ставим (-1),
                # так как дефект не относится к какому-то одному каналу.
                dlist.add(self.FileName, -1, 'asnc', ch0.Offset, ch0.Offset + ch0.Duration)

            chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_diff(self, dlist):
        """
        Получение дефектов diff.

        :param dlist: Список дефектов.
        """

        # Определение дефекта слишком значимого расхождения каналов
        # по среднему значению силы сигнала на спектрограмме.
        # Помогает отследить, например, резкое выпадение звука на одном из каналов,
        # рассинхронизацию и другие значительные отклонения в звучании каналов.
        # Работает с нормализованной спектрограммой.

        s = Separator(self.Duration, self.Settings.Diff.Sep)
        chunk_coords = s.get_next()

        while chunk_coords:
            ch0, ch1 = self.get_chunks_pair(chunk_coords)

            # Требуется пересчитать нормализованную спектрограмму V.
            ch0.generate_spectres()
            ch1.generate_spectres()

            m0 = np.array([vi.mean() for vi in ch0.V])
            m1 = np.array([vi.mean() for vi in ch1.V])
            d = np.abs(m0 - m1)
            r = sp.ndimage.minimum_filter1d(d, self.Settings.Diff.WidthMin)

            if r.max() > self.Settings.Diff.Thr:
                # Вместо номера канала ставим (-1),
                # так как дефект не относится к какому-то одному каналу.
                dlist.add(self.FileName, -1, 'diff', ch0.Offset, ch0.Offset + ch0.Duration)

            chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_hum(self, defects):
        """
        Получение дефектов hum.

        :param defects: Список дефектов.
        """

        # Определение дефекта гула на некоторой частоте должно определятся
        # на достаточно продолжительном участке записи (не менее 15 секунд).
        # Характеризуется отношением 10-го и 55-го квантилей сигнала на спектрограмме.
        # Скачок разности оригинального и сглаженного отношения сигнализирует о гуле.
        # Сглаживание выполняется с помощью кубического фильтра Савицкого-Голея
        # (можно попробовать простую усредняющую свертку, хотя бы из соображений скорости).
        #
        # Источник:
        # Matthias Brandt, Joerg Bitzer.
        # Automatic detection of hum in audio signals.
        # J. Audio Eng. Socc., Vol. 62, No. 9, 2014, September.

        for channel_num in range(self.channels_count()):
            s = Separator(self.Duration, self.Settings.Hum.Sep)

            chunk_coords = s.get_next()
            while chunk_coords:
                ch = self.get_chunk(channel_num, chunk_coords)
                ch.generate_spectres()
                ch.get_defects_hum(defects)
                chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_dense(self, defects):
        """
        Получение дефектов dense.

        :param defects: Список дефектов.
        """

        for channel_num in range(self.channels_count()):
            s = Separator(self.Duration, self.Settings.Dense.Sep)

            chunk_coords = s.get_next()
            while chunk_coords:
                ch = self.get_chunk(channel_num, chunk_coords)
                ch.generate_spectres()
                ch.get_defects_dense(defects)
                chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects_satur(self, defects):
        """
        Получение дефектов satur.

        :param defects: Список дефектов.
        """

        # Определение дефекта saturation (перегрузка).
        # Выполняется поиск участков, в которых превышен порог по энергии звука,
        # полученной по спектрограмме амплидуд.

        for channel_num in range(self.channels_count()):
            s = Separator(self.Duration, self.Settings.Satur.Sep)

            chunk_coords = s.get_next()
            while chunk_coords:
                ch = self.get_chunk(channel_num, chunk_coords)
                ch.generate_spectres()
                ch.get_defects_satur(defects)
                chunk_coords = s.get_next()

    # ----------------------------------------------------------------------------------------------

    def get_defects(self, defects_names, defects):
        """
        Получение списка дефектов по списку имен дефектов.

        :param defects_names: Список имен дефектов.
        :param defects:       Список дефектов.
        """

        m = {'click'  : self.get_defects_click,
             'muted'  : self.get_defects_muted,
             'muted2' : self.get_defects_muted2,
             'echo'   : self.get_defects_echo,
             'asnc'   : self.get_defects_asnc,
             'diff'   : self.get_defects_diff,
             'hum'    : self.get_defects_hum,
             'dense'  : self.get_defects_dense,
             'satur'  : self.get_defects_satur}

        for dn in defects_names:
            fun = m.get(dn)
            if fun:
                fun(defects)

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

    :return: Список описаний дефектов.
    """

    dlist = DefectsList()
    fs = os.listdir(directory)
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
                wav.get_defects(defects_names, dlist)

    dd = dlist.get()

    print('Process finished:')
    print('    {0} records processed'.format(records_count))
    print('    {0} s of audio records processed'.format(records_time))
    print('    {0} defects found'.format(len(dd)))
    print('    {0} s time estimated'.format(time.time() - ts))

    return dd

# ==================================================================================================


def run(directory, filter_fun, defects_names):
    """
    Головная функция.

    :param directory:     Имя директории.
    :param filter_fun:    Функция отбора файлов для детектирования дефектов.
    :param defects_names: Список имен дефектов.
    """

    # Получаем описания дефектов.
    dd = analyze_directory(directory,
                           filter_fun=filter_fun,
                           defects_names=defects_names,
                           verbose=True)

    m = {}
    fs = os.listdir(directory)
    for f in fs:
        if filter_fun(f):
            m['{0}/{1}'.format(directory, f)] = {'click': False, 'muted': False, 'muted2': False,
                                                 'echo': False, 'asnc': False, 'diff': False,
                                                 'hum': False, 'dense': False, 'satur': False}
    for d in dd:
        m[d['rec']][d['name']] = True
        print(d)

    def col(b):
        if b:
            return ' bgcolor="darkgreen"'
        else:
            return ''

    # Print file.
    with open('report.html', 'w') as f:
        f.write('<html>')
        f.write('<head>')
        f.write('</head>')
        f.write('<body>')
        f.write('<table border="1">')
        f.write('<tr><th>record</th>'
                '<th>click</th><th>muted</th><th>muted2</th>'
                '<th>echo</th><th>asnc</th><th>diff</th>'
                '<th>hum</th><th>dense</th><th>satur</th>'
                '</tr>')
        for mi in m:
            f.write('<tr>')
            f.write('<td>{0}</td>'
                    '<td{1}>&nbsp;</td><td{2}>&nbsp;</td><td{3}>&nbsp;</td>'
                    '<td{4}>&nbsp;</td><td{5}>&nbsp;</td><td{6}>&nbsp;</td>'
                    '<td{7}>&nbsp;</td>'
                    '<td{8}>&nbsp;</td>'
                    '<td{9}>&nbsp;</td>'.format(mi,
                                                col(m[mi]['click']),
                                                col(m[mi]['muted']),
                                                col(m[mi]['muted2']),
                                                col(m[mi]['echo']),
                                                col(m[mi]['asnc']),
                                                col(m[mi]['diff']),
                                                col(m[mi]['hum']),
                                                col(m[mi]['dense']),
                                                col(m[mi]['satur'])))
            f.write('</tr>')
        f.write('</table>')
        f.write('</body>')
        f.write('</html>')
        f.close()

# ==================================================================================================


if __name__ == '__main__':

    # При запуске по умолчанию данного скрипта
    # рабочая директория должна быть установлена в корень проекта,
    # набор тестовых аудиозаписей не включен в репозиторий, его нужно поместить
    # в корень проекта в папку wavs/origin.
    # Сгенерированные оригинальные и нормализованные спектрограммы находятся
    # в директории docs.

    run(directory='wavs/origin',
        filter_fun=lambda f: True,
        defects_names=['click',
                       'muted',
                       'muted2',
                       'echo',
                       'asnc',
                       'diff',
                       'hum',
                       'dense',
                       'satur'])

# ==================================================================================================
