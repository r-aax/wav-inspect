"""
Реализация модуля по обработке аудиозаписей.
"""

import os
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

        idxs = wi_utils.indices_slice_array(self.NNetData.shape[0], 0, width, step)

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
        muted_part = wi_utils.predicated_part(answers, is_ans_muted)

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

    for f in fs:
        if filter_fun(f):

            if verbose:
                print('.... process {0}'.format(f))

            wav = WAV('{0}/{1}'.format(directory, f))

            if wav.is_ok():
                ds = ds + wav.get_defects(defects_names)

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
        defects_names=['snap', 'muted'])


# ==================================================================================================
