"""
Вспомогательные функии.
"""

import matplotlib.pyplot as plt
import operator
import pocketsphinx
import numpy as np
import cv2


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
    q = np.array([sum(a[:ln4]), sum(a[ln4:2*ln4]), sum(a[2*ln4:3*ln4]), sum(a[3*ln4:])])

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
    Первый и последний маркеры всегда 0 (False).

    :param a: Список маркеров.

    :return: Список интервалов.
    """

    ln = len(a)
    i = 1
    while i < ln:
        if a[i] != a[i - 1]:
            print(i)

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


# ==================================================================================================
