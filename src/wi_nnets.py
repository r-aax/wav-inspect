"""
Реализация работы с нейронными сетями.
"""

import time
import random
import numpy as np
import keras
import keras.utils
import keras.utils.np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import RMSprop
import wi
import wi_utils
import wi_settings


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

            wav = wi.WAV('{0}/{1}'.format(directory, file), self.Settings)

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
        all_xs, all_ys = wi_utils.unzip(all_data)
        all_xs = np.array(all_xs)
        shp = all_xs.shape
        all_xs = all_xs.reshape((shp[0], shp[1] * shp[2]))
        all_xs = all_xs.astype('float32')
        all_ys = keras.utils.np_utils.to_categorical(all_ys, 2)

        print('init_data_muted : shuffle : {0}'.format(time.time() - t0))

        # Позиция для разделения данных на обучающую и тестовую выборки.
        p = int(len(all_xs) * self.Settings.Muted.TrainCasesPart)
        self.XTrain, self.XTest = wi_utils.split(all_xs, p)
        self.YTrain, self.YTest = wi_utils.split(all_ys, p)

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


"""
Код для определения дефекта muted с помощью нейросети (не используется).

Настройки.
class DefectMutedSettings:
    Настройки дефекта muted.

    def __init__(self,
                 case_width,
                 case_learn_step,
                 train_cases_part,
                 case_pred_step,
                 category_detect_limits,
                 part_for_decision):
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

        self.CaseWidth = case_width
        self.CaseLearnStep = case_learn_step
        self.TrainCasesPart = train_cases_part
        self.CasePredStep = case_pred_step
        self.CategoryDetectLimits = category_detect_limits
        self.PartForDecision = part_for_decision

        # Грузим нейронку, если она есть.
        if os.path.isfile('nnets/muted.h5'):
            self.NNet = keras.models.load_model('nnets/muted.h5')

Инициализация настроек.
defect_muted_settings = DefectMutedSettings(case_width=16,
                                            case_learn_step=10,
                                            train_cases_part=0.8,
                                            case_pred_step=16,
                                            category_detect_limits=(0.45, 0.55),
                                            part_for_decision=0.9)
    
Метод класса Channel.
def get_defects_muted(self):
    Получение дефектов muted.

    :return: Список дефектов muted.

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
"""


# ==================================================================================================


if __name__ == '__main__':

    nnet_name = 'muted'

    nn = NNetTrainer(nnet_name, wi_settings.defect_settings)
    nn.init_data()
    nn.init_model()
    nn.fit()
    nn.save()


# ==================================================================================================
