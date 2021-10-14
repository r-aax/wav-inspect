import os
import time
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


class WAV:
    """
    Аудиоззапись.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self, seg, settings=wi_settings.defects_settings):
        """
        Конструктор аудиозаписи.

        :param settings: Настройки.
        """

        # Имя файла.
        self.FileName = seg[2]

        # Считанный массив амплитуд.
        self.Y = seg[0]

        # Количество каналов в сегменте
        self.FlagChannel = None
        self.flag_for_channels()

        # Частота дискретизации.
        self.SampleRate = seg[1]

        # Время начала и конца сегмента (с).
        self.StartTime = seg[3]
        self.EndTime = seg[4]

        # Длительность сегмента (с).
        self.Duration = self.EndTime - self.StartTime

        # Настройки.
        self.Settings = settings

    # ----------------------------------------------------------------------------------------------

    def is_ok(self):
        """
        Проверка, является ли данная запись нормальной, то есть она загрузилась.

        :return: True  - если запись успешно загружена,
                 False - в противном случае.
        """

        return self.Y is not None

    # ----------------------------------------------------------------------------------------------

    def flag_for_channels(self):
        """
        Определяет количество каналов в сегменте.

        :return:
        """

        # проверка на моно
        if len(self.Y.shape) == 1:  # моно
            self.FlagChannel = 1
        # проверка на многоканальность
        elif len(self.Y.shape) == 2:
            self.FlagChannel = 2

    # ----------------------------------------------------------------------------------------------

    def separator(self, sep_sizes):

        res = []  # контайнер для результата сегментации

        step_seg = int(self.SampleRate * sep_sizes[0])  # разбиваем на сегменты по заданным настройкам.

        # проверка на моно
        if self.FlagChannel == 1:  # моно

            for n, i in enumerate(range(0, len(self.Y), int(step_seg))):
                seg = self.Y[i:i + step_seg]
                # проверка размера сегмента
                # если он меньше 1 секунды - не учитывать
                if len(seg) < self.SampleRate * 1:
                    break
                elif len(seg) <= self.SampleRate * sep_sizes[1]:
                    break
                t0 = self.StartTime + n * sep_sizes[0]
                t1 = self.StartTime + n * sep_sizes[0] + sep_sizes[0]
                if t1 > self.EndTime:
                    t1 = self.EndTime
                res += [[seg, t0, t1]]

            return res

        # многоканальная запись
        else:

            for n, i in enumerate(range(0, len(self.Y[0]), int(step_seg))):
                seg = [[] for _ in range(len(self.Y))]
                for ch in range(len(self.Y)):
                    seg[ch] = self.Y[ch][i:i + step_seg]
                seg = np.array(seg)
                # проверка размера сегмента
                # если он меньше 1 секунды - не учитывать

                if len(seg[0]) < self.SampleRate * 1:
                    break
                elif len(seg[0]) <= self.SampleRate * sep_sizes[1]:
                    break
                t0 = self.StartTime + n * sep_sizes[0]
                t1 = self.StartTime + n * sep_sizes[0] + sep_sizes[0]
                if t1 > self.EndTime:
                    t1 = self.EndTime
                res += [[seg, t0, t1]]

            return res

    # ----------------------------------------------------------------------------------------------

    def generate_a_spectre(self, x):

        # Генерируем амплитудный спектр.
        return abs(librosa.stft(x, n_fft=2048))

    # ----------------------------------------------------------------------------------------------

    def generate_spectre(self, x):

        # Генерируем частотный спектр.
        return librosa.amplitude_to_db(self.generate_a_spectre(x))

    # ----------------------------------------------------------------------------------------------

    def generate_h_spectre(self, x):

        # Нормализованная матрица спектра.
        h = self.generate_spectre(x) + 0.0
        (min_val, max_val) = self.Settings.LimitsDb
        min_val = max(min_val, h.min())
        max_val = min(max_val, h.max())
        np.clip(h, min_val, max_val, out=h)
        h = (h - min_val) / (max_val - min_val)

        return h

    # ----------------------------------------------------------------------------------------------

    def get_silence2(self, x, limx, xdb, hop_length=512):
        """
        :param x: аудиозапись
        :param limx: предел амплитуды для однаружения тишины
        :param hop_length: ширина одно фрейма

        :return: матрица частот с обнуленными столбцами в местах нахождения тишины
        """

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
        res = np.multiply(xdb, tms)

        return res

    # ----------------------------------------------------------------------------------------------

    def get_defects_click(self):

        """
        Получение дефектов click.
        """

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Click.Sep)

        # загрузка настроек
        setting = self.Settings.Click

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    qs, win_h, win_w = setting.Quartiles, setting.WinH, setting.WinW

                    # Квартиль списка.
                    def q(a, ind, width):
                        return a[ind * width: (ind + 1) * width]

                    # нормализованная спектрограмма
                    xh = self.generate_h_spectre(x)

                    ns = xh.T

                    # Отрезаем верхнюю часть частот и прогоняем через фильтр Собеля для выявления границ.
                    v = ns[:, -qs * win_h:]
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
                        if (y.max() - y.mean() > setting.Thr) and (y.mean() < setting.MeanThr):
                            t = librosa.frames_to_time(win_w * i + np.argmax(y))
                            print(f'root path file: {self.FileName}, channel number {n}, name of defect: click,'
                                  f' time mark: {s[1] + t}, {s[1] + t}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted(self):
        """
        Получение дефектов muted.
        """

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Muted.Sep)

        # загрузка настроек
        setting = self.Settings.Muted

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    # нормализованная спектрограмма
                    xh = self.generate_h_spectre(x)

                    ns = xh.T
                    w, h = ns.shape[0], ns.shape[1]
                    weights = np.array([range(h)] * w)
                    nsw = ns * weights
                    y = np.array([sum(nsw[i]) / (sum(ns[i]) + 1e-10) for i in range(w)])
                    p = 100.0 * y.mean() / h

                    # Принимаем решение о глухой записи, по порогу среднего значения ортоцентра.
                    if p < setting.Thr:
                        print(f'root path file: {self.FileName}, channel number {n}, name of defect: muted,'
                              f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_muted2(self):

        """
        Получение дефектов muted2.
        """

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Muted2.Sep)

        # загрузка настроек
        setting = self.Settings.Muted2

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    # кратковременное преобразование Фурье
                    xdb = self.generate_spectre(x)

                    # порог db, ниже которых детектируем пустоту
                    lim_db = int((int(xdb.max()) - int(xdb.min())) / 100 * setting.PercentageOfLimDb) + int(xdb.min())

                    # обнулить фреймы с тишиной
                    xdb = self.get_silence2(x=x, limx=setting.Muted2Silence, xdb=xdb)

                    # нашли пустоты во всей матрице - все что ниже пороговой амплитуды
                    xdb = xdb <= lim_db

                    # отсекаем нижню часть спектрограммы
                    val_void_in_song = int(xdb.shape[0] / 100 * setting.PercentNotVoid)
                    xdb = xdb[val_void_in_song:]

                    # процент погрешности
                    val_of_error = int(xdb.shape[0] / 100 * setting.PercentageOfError)

                    # отсортировать и срезать погрешность
                    xdb.sort(axis=0)  # сортировка по столбцам
                    xdb = xdb[val_of_error:]

                    # проверяем наличие пустот в верхней части с учетом ввычета погрешности
                    void_frame = []
                    for nf in range(xdb.shape[1]):
                        if all(xdb.T[nf]):
                            void_frame.append(nf)

                    # высчитываем процент глухих фреймов
                    lim_frame = int(setting.LimPercentFrame * xdb.T.shape[0] / 100)

                    # вывод резудьтата
                    # если глухих фреймов больше лимита, то запись глухая
                    if len(void_frame) >= lim_frame:
                        print(f'root path file: {self.FileName}, channel number {n}, name of defect: muted2,'
                              f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_echo(self):

        """
        Получение маркеров дефекта echo.
        """

        # Сильное эхо детектируется по значению автокорреляционной фунции по короткому окну.
        # Если использовать только данный признак, то возникает слишком много ложных срабатываний
        # на музыке, для которой характерно повторение звуков (темп, барабаны и т.д.).
        # Для отсечения музыки вычисляется глобальное значение автокорреляционной функции
        # для темпограммы - сохранение данной величины на высоком уровне свидетельствует
        # о сохранении темпа в записи.
        # Для скорости отсечение по глобальной автокорреляции темпограммы делаем по одному
        # каналу.

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Echo.Sep)

        # загрузка настроек
        setting = self.Settings.Echo

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    # Ориентируемся по каналу 0 для определения подходящей темпограммы.
                    # Для этого не требуется построение спектрограмм.
                    oenv = librosa.onset.onset_strength(y=x, sr=self.SampleRate)
                    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.SampleRate)
                    acg = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
                    acg = librosa.util.normalize(acg)
                    if (acg[:len(acg) // 2].mean()) > setting.GlobNormCorrThr:
                        return

                    # Для анализа эхо не строим никакие спектрограммы.
                    ln = int(self.SampleRate * setting.LocCorrWin)
                    parts = len(x) // ln
                    for i in range(parts):
                        yp = x[i * ln: (i + 1) * ln]
                        ac = librosa.autocorrelate(yp)
                        ac = ac[ln // 5:]
                        if max(ac) > setting.LocCorrThr:
                            print(f'root path file: {self.FileName}, channel number {n}, name of defect: echo,'
                                  f' time mark: {s[1] + i * (ln / self.SampleRate)}, '
                                  f'{s[1] + (i + 1) * (ln / self.SampleRate)}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_asnc(self):

        """
        Получение дефектов asnc.
        """

        # Дефект рассинхронизации каналов выполняется практически бесплатно на
        # основе анализа коэффициента корреляции Пирсона.
        # На основании данного коэффициента могут быть приняты решения:
        #   - если коэффициент близок к 1.0, то имеет место дефект ложного стерео
        #     (мы это не фиксируем, так как таковы практически все аудиозаписи);
        #   - если коэффициент слишком низкий и положительный, то имеет место
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

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Asnc.Sep)

        # загрузка настроек
        setting = self.Settings.Asnc

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                c = sp.stats.pearsonr(s[0][0], s[0][1])

                if c[0] < setting.Thr:
                    # Вместо номера канала ставим (-1),
                    # так как дефект не относится к какому-то одному каналу.
                    print(f'root path file: {self.FileName}, channel number {-1}, name of defect: asnc,'
                          f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_diff(self):

        """
        Получение дефектов diff.
        """

        # Определение дефекта слишком значимого расхождения каналов
        # по среднему значению силы сигнала на спектрограмме.
        # Помогает отследить, например, резкое выпадение звука на одном из каналов,
        # рассинхронизацию и другие значительные отклонения в звучании каналов.
        # Работает с нормализованной спектрограммой.

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Diff.Sep)

        # загрузка настроек
        setting = self.Settings.Diff
        # print('diff')
        # обрабатываем посегментно
        for s in segs:

            # нормализованная спектрограмма
            ch0 = self.generate_h_spectre(s[0][0])
            ch1 = self.generate_h_spectre(s[0][1])

            ch0 = ch0.T
            ch1 = ch1.T

            m0 = np.array([vi.mean() for vi in ch0])
            m1 = np.array([vi.mean() for vi in ch1])

            d = np.abs(m0 - m1)
            r = sp.ndimage.minimum_filter1d(d, setting.WidthMin)

            if r.max() > setting.Thr:
                # Вместо номера канала ставим (-1),
                # так как дефект не относится к какому-то одному каналу.
                print(f'root path file: {self.FileName}, channel number {-1}, name of defect: diff,'
                      f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_hum(self):

        """
        Получение дефектов hum.
        """

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Hum.Sep)

        # загрузка настроек
        setting = self.Settings.Hum

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    # нормализованная спектрограмма
                    xh = self.generate_h_spectre(x)

                    # Отрезаем совсем нижние частоты.
                    h = xh[int((setting.LoIgnore / 100.0) * len(xh)):880]
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

                    if d.max() > setting.Thr:
                        print(f'root path file: {self.FileName}, channel number {n}, name of defect: hum,'
                              f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_dense(self):

        """
        Получение дефектов dense.
        """

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Dense.Sep)

        # загрузка настроек
        setting = self.Settings.Dense

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    # нормализованная спектрограмма
                    xh = self.generate_h_spectre(x)

                    # задаем процент погрешности
                    percent = setting.Percent

                    # выбираем рабочий диапозон частот
                    xh = xh[setting.MinHz:setting.MaxHz]

                    # смещаем частоты вправо сортировкой
                    xh.sort(axis=1)

                    # условие нахождения звукового события
                    # исследуем только последние проценты
                    xht = xh.T[0:int(xh.shape[1] / 100 * percent)]
                    xh = xht.T

                    # проверсяем каждую строку окна на наличие частот
                    fil = []
                    flag = 0
                    width = setting.Width
                    for nh, i in enumerate(xh):

                        # если в строке есть частоты
                        if i.max() != 0:

                            fil.append(1)

                        # если в строке нет частот
                        else:

                            fil.append(0)

                        # если подряд есть частоты
                        if nh > width - 1 and sum(fil[nh - width:nh]) == width:
                            # пердположительно это не то что мы ищим
                            flag = 1
                            break

                    # если это то что мы искали и там встрачались частоты
                    if flag == 0 and sum(fil) > 0:
                        # условие выполненно, записать фреймы начала и конца
                        print(f'root path file: {self.FileName}, channel number {n}, name of defect: dense,'
                              f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_overload(self):
        """
        Получение дефектов overload.
        """

        # Определение дефекта saturation (перегрузка).
        # Выполняется поиск участков, в которых превышен порог по энергии звука,
        # полученной по спектрограмме амплидуд.

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Overload.Sep)

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    a_spec = self.generate_a_spectre(x)

                    r = librosa.feature.rms(S=a_spec)[0]

                    rs = scipy.ndimage.uniform_filter1d(r, size=self.Settings.Overload.FilterWidth)

                    m = [rsn for rsn, rsi in enumerate(rs) if rsi > self.Settings.Overload.PowerThr]

                    res = wi_utils.diff_signal(rs, m)

                    # проверка начилия фреймов для обработки
                    if len(res) > 1:

                        # формирование интервалов
                        intervals = wi_utils.markers_val_intervals(res)

                        for interval in intervals:
                            print(f'root path file: {self.FileName}, channel number {n}, name of defect: overload,'
                                  f' time mark: {s[1] + librosa.frames_to_time(interval[0], sr=self.SampleRate)}, '
                                  f'{s[1] + librosa.frames_to_time(interval[1], sr=self.SampleRate)}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_loud(self):

        """
        Получение дефектов loud.
        """

        # Метод анализа субъективной громкости программ и истинного пикового уровня сигналов.
        # алгоритм данного метода основан на материалах:РЕКОМЕНДАЦИЯ МСЭ-R BS.1770-4 (10/2015)
        # Алгоритмы измерения громкости звуковых программ и истинного пикового уровня звукового сигнала

        if self.SampleRate != 48000:
            # Для SampleRate, отличного от 48000 данный алгоритм по стандарту неприменим.
            # Просто игнорируем его применение.
            print('не та частота')
            return

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Loud.Sep)

        # обрабатываем посегментно
        for s in segs:

            # Достаем максимум звука из записи.
            levs = wi_utils.get_volume_level_BS_1770_4(s[0], self.SampleRate)

            # Гарантированно низкое значение.
            max_lev = -100.0

            if len(levs) > 0:
                max_lev = max(levs)

            if max_lev > self.Settings.Loud.Thr:
                print(f'root path file: {self.FileName}, channel number {-1}, name of defect: loud,'
                      f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects_dbl(self):

        """
        Получение дефектов dbl.

        """

        # Определение дефекта наличия цифровой копии в участке.
        # Если в участве присутствует цифровая копия того же самого фрагмента, то
        # значения амплитуд в этих скопированных фрагментов совпадают в точности
        # (то есть вещественные значения совпадают по отношению точного равенства "==").
        # Для минимизации числа проверок анализируются только часть амплитуд
        # (количество самых больших амплитуд).
        # В отсортированном массиве самых высоких амплитуд ищется общее количество совпадений
        # соседних пар - при слишком большом количестве таких пар принимается решение
        # о наличии дефекта - цифровой копии.
        #
        # Warning.
        # - При выборе слишком длинного размера сегментации существует риск, что высокие
        #   амплитуды цифровых дублей не попадут в "хит-парад" амплитуд всего фрагмента.
        #   В этом случае дефект будет не виден.
        # - При выборе слишком короткого размера сегментации существует риск, что цифровые
        #   копии будут попадать в разные сегменты - дефект будет утерян.
        # - Задирание порога дублей может привести к срабатыванию дефекта на цифровое эхо
        #   (проблем это доставить не должно, так как такое срабатывание возможно на очень
        #   сильном эхо, на записях с большой долей тишины, по сути на эхо от голоса в
        #   студийных условиях).

        # получаем список сегментов для обработки + время каждого сегмента
        segs = self.separator(self.Settings.Dbl.Sep)

        dim = self.Settings.Dbl.Top

        # обрабатываем посегментно
        for s in segs:

            if self.FlagChannel == 2:

                # обрабатываем каждый канал сегмента
                for n, x in enumerate(s[0]):

                    # Сортируем массив амплитуд и отрезаем верхнюю часть, остальное не понадобится.
                    x.sort()
                    x = x[-dim:]

                    # Считаем количество точных дублей, которые принимаем за индикатор.
                    inds = [x[i] == x[i + 1] for i in range(dim - 1)]
                    part = wi_utils.predicated_part(inds, lambda ind: ind)

                    # Фиксация дефекта при превышении порога числа дублей.
                    if part > self.Settings.Dbl.Thr:
                        print(f'root path file: {self.FileName}, channel number {n}, name of defect: dbl,'
                              f' time mark: {s[1]}, {s[2]}')

    # ----------------------------------------------------------------------------------------------

    def get_defects(self, defects_names):
        """
        Получение списка дефектов по списку имен дефектов.

        :param defects_names: Список имен дефектов.
        """

        m = {'click'  : self.get_defects_click,
             'muted'  : self.get_defects_muted,
             'muted2' : self.get_defects_muted2,
             'echo'   : self.get_defects_echo,
             'asnc'   : self.get_defects_asnc,
             'diff'   : self.get_defects_diff,
             'hum'    : self.get_defects_hum,
             'dense'  : self.get_defects_dense,
             'overload': self.get_defects_overload,
             'loud'   : self.get_defects_loud,
             'dbl'    : self.get_defects_dbl}

        for dn in defects_names:
            fun = m.get(dn)
            if fun:
                fun()


#===================================================================================================


def run(segment):

    defects_names = [
        'click',
        'muted',
        'muted2',
        'echo',
        'asnc',
        'diff',
        'hum',
        'dense',
        'overload',
        'loud',
        'dbl'
    ]

    if segment:

        wav = WAV(segment)

        if wav.is_ok():
            wav.get_defects(defects_names)
