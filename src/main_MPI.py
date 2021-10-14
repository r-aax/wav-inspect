from mpi4py import MPI
import librosa
import numpy as np
import os
import wi_for_MPI

"""
Данный модуль считывает путь к рабочей директории через переменную dir_name.

Для работы необходимо наличие файлов:
    wi_for_MPI.py
    wi_settings.py
    wi_utils.py
в дириктории с исполняемым файлом.

Модуль предназначен для формирования параллельных процессов для исполняющего файла wi_for_MPI.py.
"""

# разбиение на сегменты с пробросом имени файла и времени сегмента
def seg(x, sr, time_lim=30, file_name='Noname', time_start=0, time_end=30):

    res = [] # контайнер для результата сегментации

    step_seg = int(sr * time_lim) # разбиваем на сегменты по time_lim секунд

    # проверка на моно
    # моно
    if len(x.shape) == 1:

        for n, i in enumerate(range(0, len(x), int(step_seg))):
            segment = x[i:i + step_seg]
            # проверка размера сегмента
            # если он меньше 1 секунды - не учитывать
            if len(segment) < sr * 1:
                break
            t0 = time_start + n * time_lim
            t1 = time_start + n * time_lim + time_lim
            if t1 > time_end:
                t1 = time_end
            res += [[segment, sr, file_name, t0, t1]]

        return res

    # многоканальная запись
    else:

        for n, i in enumerate(range(0, len(x[0]), int(step_seg))):
            segment = [[] for ch in range(len(x))]
            for ch in range(len(x)):
                segment[ch] = x[ch][i:i + step_seg]
            segment = np.array(segment)
            # проверка размера сегмента
            # если он меньше 1 секунды - не учитывать
            if len(segment[0]) < sr * 1:
                break
            t0 = time_start + n * time_lim
            t1 = time_start + n * time_lim + time_lim
            if t1 > time_end:
                t1 = time_end
            res += [[segment, sr, file_name, t0, t1]]

        return res


# переменные для MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = comm.Get_name()
print('process %d of %d on %s' % (rank+1, size, name))

dir_name = '../wavs/origin'

# если процессов больше 1, то файлы сегментируются и распределяются по процессам
if __name__ == '__main__' and size > 1:

    # предел времени для сегментации записи, измеряется в секундах
    time_lim = 30

    # флаг для анализа корректных файлов
    flag_for_audio_files = 0

    if rank == 0:

        # список для рассылки в прочие процессы
        array_to_share = []

        # список для сегментов
        segments_to_share = []

        # чтение директории с файлами
        files = os.listdir(dir_name)

        # чтение файлов
        for name in files:

            # проверка формата файла
            if name.endswith(('.wav', '.mp3')):

                # флаг меняется, если был найден хоть один корректный файл
                flag_for_audio_files = 1

                # определение полного пути и имени файла
                file_name = os.path.join(dir_name, name)

                # чтение файла
                x, sr = librosa.load(file_name, mono=False, sr=None)

                # проверка: нужно ли файл сегментировать
                duration = librosa.get_duration(y=x, sr=sr)

                # временные граници цифрового сигнала
                time_start = 0
                time_end = duration

                if duration > time_lim:

                    # сегментация и добавление сегментов записи
                    segments = seg(x=x, sr=sr, time_lim=time_lim, file_name=file_name, time_start=time_start, time_end=time_end)
                    for i in segments:
                        segments_to_share += [i]

                else:

                    # добавление не сегментируемой записи
                    segments = [x, sr, file_name, time_start, time_end]
                    segments_to_share += [segments]

            else:

                # Файлы для обработки не обнаружены
                if flag_for_audio_files == 0:
                    print('Файлы .wav и .mp3 не были обнаружены!')

        # итоговое кол-во сегментов для анализа
        len_segs = len(segments_to_share)

        # создание пустых ячеек для распределения
        # кол-во ячеек равно кол-ву процессов
        for i in range(size):
            array_to_share.append([])

        # заполнение пустых ячеек сегментами записей
        coin = 0
        need_elems = 0
        while need_elems != len_segs:

            array_to_share[coin].append(segments_to_share[need_elems])

            coin += 1
            if coin >= size:
                coin = 0

            need_elems += 1

    else:
        array_to_share = None

    # распределение сегментов по процессам из процесса 0
    recvbuf = comm.scatter(array_to_share, root=0)

    # анализ сегмента
    # распаковываем все сегменты в данном процессе
    for i in recvbuf:

        try:
            wi_for_MPI.run(i)
        except Exception as e:
            print(i[2:])
            print(f"Error! {e}")


# если один процесс, то все файлы просто поочередно подаются на анализ
elif __name__ == '__main__' and size == 1:

    # флаг для корректных файлов для анализа
    flag_for_audio_files = 0

    # чтение директории с файлами
    files = os.listdir(dir_name)

    # чтение файлов
    for name in files:

        # проверка формата файла
        if name.endswith(('.wav', '.mp3')):

            # флаг меняется, если был найден хоть один корректный файл
            flag_for_audio_files = 1

            # определение полного пути и имени файла
            file_name = os.path.join(dir_name, name)

            # чтение файла
            x, sr = librosa.load(file_name, mono=False, sr=None)

            # дилительность цифрового сигнала
            duration = librosa.get_duration(y=x, sr=sr)

            # временные граници цифрового сигнала
            time_start = 0
            time_end = duration

            # формирование пакета данных
            segments = [x, sr, file_name, time_start, time_end]

            # анализ файла
            try:
                wi_for_MPI.run(segments)
            except Exception as e:
                print(segments[2:])
                print(f"Ошибка! {e}")

        else:

            # Файлы для обработки не обнаружены
            if flag_for_audio_files == 0:
                print('Файлы .wav и .mp3 не были обнаружены!')
