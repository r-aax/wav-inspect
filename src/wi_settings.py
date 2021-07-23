"""
Определение настроек для работы с дефектами.
"""

# import os
# import keras


# ==================================================================================================


class DefectSnapSettings:
    """
    Настройки дефекта snap.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 limits_before_sort,
                 limits_after_sort,
                 min_power_lo_threshold,
                 half_snap_len,
                 diff_min_max_powers_hi_threshold):
        """
        Конструктор настроек для дефекта snap.

        :param limits_before_sort:               Границы, по которым обрубаются массивы
                                                 силы звука до сортировки.
        :param limits_after_sort:                Границы, по которым обрубаются массивы
                                                 силы звука после сортировки.
        :param min_power_lo_threshold:           Минимальное отслеживаемое значение скачка
                                                 минимальной силы звука.
        :param half_snap_len:                    Половинная длина щелчка
                                                 (чем меньше она, тем более резкую скейку ловим).
        :param diff_min_max_powers_hi_threshold: Максимально допустимая разница в значениях
                                                 максимума и минимума силы звука (определяет
                                                 степень постоянства силы в массиве).
        """

        self.LimitsBeforeSort = limits_before_sort
        self.LimitsAfterSort = limits_after_sort
        self.MinPowerLoThreshold = min_power_lo_threshold
        self.HalfSnapLen = half_snap_len
        self.DiffMinMaxPowersHiThreshold = diff_min_max_powers_hi_threshold

# ==================================================================================================


class DefectSnap2Settings:
    """
    Настройки дефекта snap2.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 freq_block_width,
                 hi_threshold,
                 lo_threshold,
                 half_snap_len):
        """
        Конструктор.

        :param freq_block_width: Ширина блока частот, по которым ищется максимум.
        :param hi_threshold:     Верхняя граница нормализованнной силы сигнала
                                 (после оператора выявления границ),
                                 выше которой считается всплеск.
        :param lo_threshold:     Нижняя граница нормализованной силы сигнала
                                 (после оператора выявления границ),
                                 ниже которой считается тишина.
        :param half_snap_len:    Половина длины щелчка.
        """

        self.FreqBlockWidth = freq_block_width
        self.HiThreshold = hi_threshold
        self.LoThreshold = lo_threshold
        self.HalfSnapLen = half_snap_len

# ==================================================================================================


"""
Функционал с нейронками выключен.
class DefectMutedSettings:
    Настройки дефекта muted.

    # ----------------------------------------------------------------------------------------------

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
"""

# ==================================================================================================


class DefectCometSettings:
    """
    Настройки дефекта comet.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 signal_threshold,
                 orth_quartile_threshold):
        """
        Конструктор.

        :param signal_threshold:        Порок максимального сигнала, необходимого для
                                        поиска дефекта (с учетом нормирования).
        :param orth_quartile_threshold: Порог ортоцентра с учетом квартиля сигнала для
                                        принятия решения о дефекте.
        """

        self.SignalThreshold = signal_threshold
        self.OrthQuartileThreshold = orth_quartile_threshold

# ==================================================================================================


class DefectsSettings:
    """
    Настройки дефектов.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 limits_db,
                 snap,
                 snap2,
                 # muted,
                 comet):
        """
        Конструктор настроек для всех дефектов.

        :param limits_db: Лимиты по силе (за пределами лимитов вообще
                          не учитываем сигнал).
        :param snap:      Настройки дефекта snap.
        :param snap2:     Настройки дефекта snap2.
        :param muted:     Настройки дефекта muted.
        :param comet:     Настройки дефекта comet.
        """

        self.LimitsDb = limits_db
        self.Snap = snap
        self.Snap2 = snap2
        # self.Muted = muted
        self.Comet = comet

# ==================================================================================================


# Определение настроек по умолчанию.

defect_snap_settings = DefectSnapSettings(limits_before_sort=(0.7, 0.95),
                                          limits_after_sort=(0.25, 0.75),
                                          min_power_lo_threshold=5.0,
                                          half_snap_len=2,
                                          diff_min_max_powers_hi_threshold=5.0)

defect_snap2_settings = DefectSnap2Settings(freq_block_width=16,
                                            hi_threshold=0.5,
                                            lo_threshold=0.01,
                                            half_snap_len=2)

"""
defect_muted_settings = DefectMutedSettings(case_width=16,
                                            case_learn_step=10,
                                            train_cases_part=0.8,
                                            case_pred_step=16,
                                            category_detect_limits=(0.45, 0.55),
                                            part_for_decision=0.9)
"""

defect_comet_settings = DefectCometSettings(signal_threshold=0.75,
                                            orth_quartile_threshold=800)

defects_settings = DefectsSettings(limits_db=(-50.0, 50.0),
                                   snap=defect_snap_settings,
                                   snap2=defect_snap2_settings,
                                   # muted=defect_muted_settings,
                                   comet=defect_comet_settings)

# ==================================================================================================
