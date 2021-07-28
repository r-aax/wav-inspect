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

        return 'Defect: {0} (ch {1}) : {2} ({3:.3f} s - {4:.3f} s)'.format(self.RecordName,
                                                                           self.Channel,
                                                                           self.DefectName,
                                                                           self.DefectCoords[0],
                                                                           self.DefectCoords[1])

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

        # Исходный и транспонированный спектр, который строит librosa.
        self.Spectre = None
        self.TSpectre = None

        # Матрица нормализованного спектр, которая является массивом горизонтальных линий.
        self.H = None

        # Матрица нормализованного спектра, которая является массивом вертикальных линий.
        self.V = None

        # Безусловно генерируем спектры.
        self.generate_spectre()

    # ----------------------------------------------------------------------------------------------

    def generate_spectre(self):
        """
        Генерация спектра.
        """

        # Генерируем спектр.
        self.Spectre = librosa.amplitude_to_db(abs(librosa.stft(self.Y, n_fft=2048)))
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

        idxs = wi_utils.indices_slice_array(self.V.shape[0], 0, width, step)

        return [self.V[fr:to] for (fr, to) in idxs]

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
        ns2 = wi_utils.apply_filter_2d(self.V,
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
        orth = [wi_utils.array_orthocenter(c) for c in self.V]
        lev = [max(c) for c in self.V]
        qu = [wi_utils.array_weight_quartile(c) for c in self.V]

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

    def get_defects_snap(self, defects):
        """
        Получение дефектов snap.

        :param defects: Список дефектов.
        """

        markers = self.get_defect_snap_markers()

        # Формируем список дефектов.
        for i, marker in enumerate(markers):
            if marker:
                defects.append(Defect(self.Parent.FileName,
                                      self.Channel,
                                      'snap',
                                      (self.specpos_to_time(i), self.specpos_to_time(i))))

    # ----------------------------------------------------------------------------------------------

    def get_defects_snap2(self, defects):
        """
        Получение дефектов snap2.

        :param defects: Список дефектов.
        """

        markers = self.get_defect_snap2_markers()

        # Формируем список дефектов.
        for i, marker in enumerate(markers):
            if marker:
                defects.append(Defect(self.Parent.FileName,
                                      self.Channel,
                                      'snap2',
                                      (self.specpos_to_time(i), self.specpos_to_time(i))))

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self, defects):
        """
        Получение дефектов muted.

        :param defects: Список дефектов.
        """

        s = self.Parent.Settings.Muted

        ns = self.V
        h = ns.shape[1]
        weights = np.array([range(h)] * ns.shape[0])
        ns2 = ns * weights
        y = [sum(ns2[i]) / (sum(ns[i]) + 1e-10) for i in range(ns.shape[0])]
        ind = sum(y) / len(y)

        # Принимаем решение о глухой записи, по порогу среднего значения ортоцентра.
        if sum(y) / len(y) < s.OrthocenterThreshold:
            defects.append(Defect(self.Parent.FileName,
                                  self.Channel,
                                  'muted',
                                  (0.0, self.Parent.Duration)))

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted2(self, defects):

        '''
        Получение дефектов muted2.

        :param defects: Список дефектов.
        '''

        s = self.Parent.Settings.Muted2

        # кратковременное преобразование Фурье
        Xdb = self.Spectre

        # порог db, ниже которых детектируем пустоту
        lim_db = int((int(Xdb.max()) - int(Xdb.min())) / 100 * s.PercentageOfLimDb) + int(Xdb.min())

        # обнулить фреймы с тишиной
        Xdb = self.get_silence2(x = self.Y, limx = s.Muted2Silence, Xdb = Xdb)

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
            defects.append(Defect(self.Parent.FileName,
                                  self.Channel,
                                  'muted2',
                                  (0.0, self.Parent.Duration)))

    # ----------------------------------------------------------------------------------------------

    def get_defects_comet(self, defects):
        """
        Получение дефектов comet.

        :param defects: Список дефектов.
        """

        markers = self.get_defect_comet_markers()
        ivs = wi_utils.markers_true_intervals(markers)

        for iv in ivs:
            defects.append(Defect(self.Parent.FileName,
                                  self.Channel,
                                  'comet',
                                  (self.specpos_to_time(iv[0]), self.specpos_to_time(iv[1]))))

    # ----------------------------------------------------------------------------------------------

    def get_defects_by_name(self, defect_name, defects):
        """
        Получение дефектов заданного типа.

        :param defect_name: Имя дефекта.
        :param defects: Список дефектов.
        """

        if defect_name == 'snap':
            self.get_defects_snap(defects)
        elif defect_name == 'snap2':
            self.get_defects_snap2(defects)
        elif defect_name == 'muted':
            self.get_defects_muted(defects)
        elif defect_name == 'muted2':
            self.get_defects_muted2(defects)
        elif defect_name == 'comet':
            self.get_defects_comet(defects)
        else:
            raise Exception('unknown defect name ({0})'.format(defect_name))

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

    def get_defects_from_both_channels(self, defect_name, defects):
        """
        Получение списка дефектов для которого выполняется анализ обоих каналов.

        :param defect_name: Имя дефекта.
        :param defects: Список дефектов.
        """

        self.ch0().get_defects_by_name(defect_name, defects)
        self.ch1().get_defects_by_name(defect_name, defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_snap(self, defects):
        """
        Получение маркеров дефекта snap.

        :param defects: Список дефектов.
        """

        self.get_defects_from_both_channels('snap', defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_snap2(self, defects):
        """
        Получение маркеров дефекта snap2.

        :param defects: Список дефектов.
        """

        self.get_defects_from_both_channels('snap2', defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self, defects):
        """
        Получение маркеров дефекта muted.

        :param defects: Список дефектов.
        """

        self.get_defects_from_both_channels('muted', defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted2(self, defects):

        """
        Получение маркеров дефекта muted2.

        :param defects: Список дефектов.
        """

        self.get_defects_from_both_channels('muted2', defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_comet(self, defects):
        """
        Получение маркеров дефекта comet.

        :param defects: Список дефектов.
        """

        self.get_defects_from_both_channels('comet', defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_by_name(self, defect_name, defects):
        """
        Получение списка дефектов по имени.

        :param defect_name: Имя дефекта.
        :param defects:     Список дефектов.
        """

        if defect_name == 'snap':
            self.get_defects_snap(defects)
        elif defect_name == 'snap2':
            self.get_defects_snap2(defects)
        elif defect_name == 'muted':
            self.get_defects_muted(defects)
        elif defect_name == 'muted2':
            self.get_defects_muted2(defects)
        elif defect_name == 'comet':
            self.get_defects_comet(defects)
        else:
            raise Exception('unknown defect name ({0})'.format(defect_name))

    # ----------------------------------------------------------------------------------------------

    def get_defects(self, defects_names, defects):
        """
        Получение списка дефектов по списку имен дефектов.

        :param defects_names: Список имен дефектов.
        :param defects:       Список дефектов.
        """

        for name in defects_names:
            self.get_defects_by_name(name, defects)

# ==================================================================================================


def analyze_directory(directory,
                      filter_fun,
                      defects_names,
                      defects,
                      verbose=False):
    """
    Анализ директории с файлами на наличие дефектов.

    :param directory:     Имя директории.
    :param filter_fun:    Дополнительная функция для отфильтровывания файлов, которые необходимо
                          анализировать.
    :param defects_names: Список имен дефектов.
    :param defects:       Список, в который записывать дефекты.
    :param verbose:       Признак печати процесса анализа.
    """

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
                wav.get_defects(defects_names, defects)

    print('Process finished:')
    print('    {0} records processed'.format(records_count))
    print('    {0} s of audio records processed'.format(records_time))
    print('    {0} defects found'.format(len(defects)))
    print('    {0} s time estimated'.format(time.time() - ts))


# ==================================================================================================


def run(directory, filter_fun, defects_names):
    """
    Головная функция.

    :param directory:     Имя директории.
    :param filter_fun:    Функция отбора файлов для детектирования дефектов.
    :param defects_names: Список имен дефектов.
    """

    defects = []
    analyze_directory(directory,
                      filter_fun=filter_fun,
                      defects_names=defects_names,
                      verbose=True,
                      defects=defects)

    for d in defects:
        print(d)


# ==================================================================================================


if __name__ == '__main__':

    run(directory='wavs/origin',
        filter_fun=lambda f: f in ['0001.wav', '0003.wav'],
        defects_names=['snap', 'snap2', 'muted', 'muted2'])


# ==================================================================================================
