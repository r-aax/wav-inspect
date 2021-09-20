"""
Вспомогательные функии.
"""

import matplotlib.pyplot as plt
import operator
import numpy as np
import cv2
import scipy
import math


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


def array_orthocenter(a):
    """
    Ортоцентр массива.

    :param a: Массив.

    :return: Ортоцентр.
    """

    ln = len(a)

    return sum([a[i] * i for i in range(ln)]) / sum(a)


# ==================================================================================================


def array_weight_quartile(a):
    """
    Квартиль веса массива.

    :param a: Массив.

    :return: Квартиль веса массива.
    """

    ln = len(a)
    ln4 = ln // 4
    q = np.array([sum(a[:ln4]), sum(a[ln4:2 * ln4]), sum(a[2 * ln4:3 * ln4]), sum(a[3 * ln4:])])

    return np.argmax(q)


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


def markers_true_intervals(a):
    """
    Получение интервалов из списка маркеров.

    :param a: Список маркеров.

    :return: Список интервалов.
    """

    intervals = []
    begin = -1

    for i in range(len(a)):

        if a[i]:

            # Если интервал не инициирован, то инициируем его.
            # Если интервал инициирован, то игнорируем.
            if begin == -1:
                begin = i

            # Нужно еще проверить на последнее значение.
            if i == len(a) - 1:
                intervals.append((begin, i))

        else:

            # Если интервал открыт, то закрываем его, добавляем в список
            # и сбрасываем координату его начала.
            if begin >= 0:
                intervals.append((begin, i - 1))
                begin = -1

    return intervals


# ==================================================================================================


def operator_prewitt_gx(w=1.0):
    """
    Получение оператора Превитта Gx.

    :param w: Вес оператора.

    :return: Оператор Превитта Gx.
    """

    return np.array([[-w, -w, -w],
                     [0.0, 0.0, 0.0],
                     [w, w, w]])


# ==================================================================================================


def operator_prewitt_gy(w=1.0):
    """
    Получение оператора Превитта Gy.

    :param w: Вес оператора.

    :return: Оператор Превитта Gy.
    """

    return np.array([[-w, 0.0, w],
                     [-w, 0.0, w],
                     [-w, 0.0, w]])


# ==================================================================================================


def operator_sobel_gx(w=1.0):
    """
    Получение оператора Собеля Gx.

    :param w: Вес оператора.

    :return: Оператор Собеля Gx.
    """

    return np.array([[-w, -2.0 * w, -w],
                     [0.0, 0.0, 0.0],
                     [w, 2.0 * w, w]])


# ==================================================================================================


def operator_sobel_gy(w=1.0):
    """
    Получение оператора Собеля Gy.

    :param w: Вес оператора.

    :return: Оператор Собеля Gy.
    """

    return np.array([[-w, 0.0, w],
                     [-2.0 * w, 0.0, 2.0 * w],
                     [-w, 0.0, w]])


# ==================================================================================================


def apply_filter_2d(src, op):
    """
    Применение 2D фильтра к изображению.

    :param src: Изображение.
    :param op:  Оператор.

    :return: Изображение после применения фильтра.
    """

    # Для удобства мы работаем с транспонированным изображением спектра сигнала,
    # чтобы первым измерением было время.
    # Перед применением фильтра, оператор фильтра тоже нужно транспонировать.

    return cv2.filter2D(src, -1, op.transpose())


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


def show_map(m, figsize=(20, 8)):
    """
    Демонстрация карты.

    :param m:       Карта.
    :param figsize: Размер картинки.
    """

    fig, ax = plt.subplots()

    ax.pcolormesh(m.transpose(), cmap='Greys')

    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    plt.show()


# ==================================================================================================

def get_volume_level_BS_1770_4(sample=None, samplerate=None, tg=0.4, t_itr=10):
    '''
    :param tg: Размер стробления (сек).
    :param t_itr: Количество строблений на участке Т.

    :return: Список громкости звука на участках длиной Т.
    '''

    # определение уровня громкости
    # https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-R.pdf
    # https://tech.ebu.ch/docs/tech/Tech3344-2011-RUS.pdf

    # t = round(t_itr * 0.1 + tg - 0.1, 1) # 1,3 сек - длина Т участка
    t = 1.3

    x = sample
    sr = samplerate

    # К-фильтр частот
    # коэф-ты при sr = 48000:
    a = [1.0, -1.69065929318241, 0.73248077421585]
    b = [1.53512485958697, -2.69169618940638, 1.19839281085285]

    x = scipy.signal.lfilter(b, a, x, axis=-1, zi=None)

    # К-фильтр частот
    # коэф-ты при sr = 48000:
    a = [1.0, -1.99004745483398, 0.99007225036621]
    b = [1.0, -2.0, 1.0]

    x = scipy.signal.lfilter(b, a, x, axis=-1, zi=None)

    # среднеквадратичное значение

    # Энергия j-го стробирующего блока по каналам
    zij = [[], []]

    # дважды отфильтрованный сигнал разбиваем на два канала
    for zni, zi in enumerate(x):

        # канал разбиваем на Т-интерваллы (без хвоста)

        step_seg = int(sr * t)
        segments = [zi[i:i + step_seg] for i in range(0, len(zi), int(step_seg))]
        if len(segments[-1]) < step_seg:
            segments = segments[:-1]
        segments = np.array(segments)

        # Т-интерваллы (каждый) разбиваем на Тg-интерваллы с перекрытием в 75%
        for t_int in segments:

            step_seg = int(sr * tg)
            segments_tg = [t_int[i:i + step_seg] for i in range(0, t_itr * int(step_seg / 4), int(step_seg / 4))]

            # вычисляем энергию каждого Tg-интервала
            for tg_int in segments_tg:
                zj = (1 / len(tg_int)) * sum(tg_int * tg_int)

                zij[zni].append(zj)

    g = [1, 1, 1, 1.41, 1.41]  # весовые коэф-ты каналов

    # Стробированная громкость в интервале измерения T
    lkg = []

    for tj in range(len(segments)):

        sumij = []

        for ti in range(len(x)):
            sumij.append(g[ti] * (sum(zij[ti][tj * t_itr:tj * t_itr + t_itr]) / t_itr))

        lkg.append(-0.691 + 10 * math.log10(sum(sumij)))

    return lkg

# ==================================================================================================

def markers_val_intervals(list_vals):
    '''
    Формирует интервалы величин из списка list_vals

    :param list_vals:  Список возрастающих величин (номера семплов или фреймов)
    :return: список интервалов
    '''
    # формирование интервалов
    # проверка данных
    if len(list_vals) > 1:

        intervals = []

        for n, i in enumerate(list_vals):

            if not intervals:
                intervals.append([i])

            elif len(intervals[-1]) == 2 and i == list_vals[n - 1] + 1 and n != len(list_vals) - 1:
                intervals.append([list_vals[n - 1]])

            elif i == list_vals[n - 1] + 1 and len(intervals[-1]) == 1 and n != len(list_vals) - 1:
                continue

            elif i != list_vals[n - 1] + 1 and len(intervals[-1]) == 1:
                intervals[-1].append(list_vals[n - 1])

            elif i == list_vals[n - 1] + 1 and len(intervals[-1]) == 1 and n == len(list_vals) - 1:
                intervals[-1].append(i)

        return intervals

# ==================================================================================================

def diff_signal(signal, m):
    '''
    проверка производной сигнала на выбранных участках

    :param signal: исходный сигнал
    :param m: индексы для выборочной проверки
    :return: отфильтрованный список индексов (отсеяны индексы не пошедшие проверку)
    '''

    if signal == [] or m == []:
        return []

    # проверка наклона графика сигнала
    res = []
    for i in m:
        if i + 1 >= len(signal):
            a = signal[i]
            b = signal[i - 1]
        elif i - 1 < 0:
            a = signal[i + 1]
            b = signal[i]
        else:
            a = signal[i + 1]
            b = signal[i - 1]
        if abs(a - b) / 2 < 0.01:
            res.append(i)

    return res

# ==================================================================================================

if __name__ == '__main__':
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

    # array_orthocenter
    assert array_orthocenter([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) == 3.0

    # predicated_count
    assert predicated_count([0, 0, 0, 1, 1, 1], lambda e: e > 0.5) == 3
    assert predicated_count([1, 'a', 2, 'b', 3], lambda e: type(e) is str) == 2

    # markers_true_intervals
    assert markers_true_intervals([]) == []
    assert markers_true_intervals([1, 1, 1]) == [(0, 2)]
    assert markers_true_intervals([0, 1, 1, 1, 0]) == [(1, 3)]
    assert markers_true_intervals([1, 1, 0, 0, 1, 1]) == [(0, 1), (4, 5)]
    assert markers_true_intervals([0, 0, 1, 1, 0, 0, 1, 0]) == [(2, 3), (6, 6)]

# ==================================================================================================
