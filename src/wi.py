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
import cv2
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wi_utils
import wi_settings


# ==================================================================================================


def defect_descr(rec, ch, name, beg, end):
    """
    Конструирование описания дефекта.

    :param rec:  Имя записи.
    :param ch:   Номер канала.
    :param name: Имя дефекта.
    :param beg:  Начало дефекта.
    :param end:  Конец дефекта.

    :return: Описание дефекта.
    """

    return {'rec': rec, 'ch': ch, 'name': name, 'beg': beg, 'end': end}


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
        self.generate_spectres()

    # ----------------------------------------------------------------------------------------------

    def generate_spectres(self):
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

    def get_defects_click(self, defects):
        """
        Получение дефектов click.

        :param defects: Список дефектов.
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
                defects.append({'rec': self.Parent.FileName, 'ch': self.Channel,
                                'name': 'click', 'beg': t, 'end': t})

    # ----------------------------------------------------------------------------------------------

    def get_defects_deaf(self, defects):
        """
        Получение дефектов deaf.

        :param defects: Список дефектов.
        """

        s = self.Parent.Settings.Deaf

        ns = self.V
        w, h = ns.shape[0], ns.shape[1]
        weights = np.array([range(h)] * w)
        nsw = ns * weights
        y = np.array([sum(nsw[i]) / (sum(ns[i]) + 1e-10) for i in range(w)])
        p = 100.0 * y.mean() / h

        # Принимаем решение о глухой записи, по порогу среднего значения ортоцентра.
        if p < s.Thr:
            defects.append({'rec': self.Parent.FileName, 'ch': self.Channel,
                            'name': 'deaf', 'beg': 0.0, 'end': self.Parent.Duration})

    # ----------------------------------------------------------------------------------------------

    def get_defects_deaf2(self, defects):

        '''
        Получение дефектов deaf2.

        :param defects: Список дефектов.
        '''

        s = self.Parent.Settings.Deaf2

        # кратковременное преобразование Фурье
        Xdb = self.Spectre

        # порог db, ниже которых детектируем пустоту
        lim_db = int((int(Xdb.max()) - int(Xdb.min())) / 100 * s.PercentageOfLimDb) + int(Xdb.min())

        # обнулить фреймы с тишиной
        Xdb = self.get_silence2(x = self.Y, limx = s.Deaf2Silence, Xdb = Xdb)

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
            defects.append(defect_descr(self.Parent.FileName,
                                        self.Channel,
                                        'deaf2',
                                        0.0,
                                        self.Parent.Duration))

    # ----------------------------------------------------------------------------------------------

    def get_defects_comet(self, defects):
        """
        Получение дефектов comet.

        :param defects: Список дефектов.
        """

        markers = self.get_defect_comet_markers()
        ivs = wi_utils.markers_true_intervals(markers)

        for iv in ivs:
            defects.append(defect_descr(self.Parent.FileName,
                                        self.Channel,
                                        'comet',
                                        self.specpos_to_time(iv[0]),
                                        self.specpos_to_time(iv[1])))

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

    def get_defects_click(self, defects):
        """
        Получение дефектов click.

        :param defects: Список дефектов.
        """

        for ch in self.Channels:
            ch.get_defects_click(defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_deaf(self, defects):
        """
        Получение маркеров дефекта deaf.

        :param defects: Список дефектов.
        """

        for ch in self.Channels:
            ch.get_defects_deaf(defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_deaf2(self, defects):

        """
        Получение маркеров дефекта deaf2.

        :param defects: Список дефектов.
        """

        for ch in self.Channels:
            ch.get_defects_deaf2(defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_comet(self, defects):
        """
        Получение маркеров дефекта comet.

        :param defects: Список дефектов.
        """

        for ch in self.Channels:
            ch.get_defects_comet(defects)

    # ----------------------------------------------------------------------------------------------

    def get_defects_asnc(self, defects):
        """
        Получение маркеров дефекта asnc.

        :param defects: Список дефектов.
        """

        # Источник:
        # Pablo Alonzo-Jimenez, Luis Joglar-Ongay, Xavier Serra, Dmitry Bogdanov.
        # Automatic detection of audio problems for quality control in
        # digital music distribution.
        # Audio Engineering Society, Convention paper 10205.
        # 146-th Convention, 2019 March 20-23, Dublin, Ireland.

        c = np.corrcoef(self.ch0().Y, self.ch1().Y)

        if c[0][1] < self.Settings.Asnc.Thr:
            # Вместо номера канала ставим 2,
            # так как дефект не относится к какому-то одному каналу.
            defects.append({'rec': self.FileName, 'ch': 2,
                            'name': 'async', 'beg': 0.0, 'end': self.Duration})

    # ----------------------------------------------------------------------------------------------

    def get_defects(self, defects_names, defects):
        """
        Получение списка дефектов по списку имен дефектов.

        :param defects_names: Список имен дефектов.
        :param defects:       Список дефектов.
        """

        if 'click' in defects_names:
            self.get_defects_click(defects)
        if 'deaf' in defects_names:
            self.get_defects_deaf(defects)
        if 'deaf2' in defects_names:
            self.get_defects_deaf2(defects)
        if 'comet' in defects_names:
            self.get_defects_comet(defects)
        if 'asnc' in defects_names:
            self.get_defects_asnc(defects)

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
        filter_fun=lambda f: True,
        # filter_fun=lambda f: f in ['0001.wav', '0002.wav', '0003.wav', '0004.wav', '0005.wav'],
        defects_names=['click', 'deaf', 'asnc'])


# ==================================================================================================
