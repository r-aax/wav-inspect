"""
Определение настроек для работы с дефектами.
"""


# ==================================================================================================


class DefectClickSettings:
    """
    Настройки дефекта click.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 limits_before_sort,
                 limits_after_sort,
                 min_power_lo_threshold,
                 half_click_len,
                 diff_min_max_powers_hi_threshold):
        """
        Конструктор настроек для дефекта snap.

        :param limits_before_sort:               Границы, по которым обрубаются массивы
                                                 силы звука до сортировки.
        :param limits_after_sort:                Границы, по которым обрубаются массивы
                                                 силы звука после сортировки.
        :param min_power_lo_threshold:           Минимальное отслеживаемое значение скачка
                                                 минимальной силы звука.
        :param half_click_len:                   Половинная длина щелчка
                                                 (чем меньше она, тем более резкую скейку ловим).
        :param diff_min_max_powers_hi_threshold: Максимально допустимая разница в значениях
                                                 максимума и минимума силы звука (определяет
                                                 степень постоянства силы в массиве).
        """

        self.LimitsBeforeSort = limits_before_sort
        self.LimitsAfterSort = limits_after_sort
        self.MinPowerLoThreshold = min_power_lo_threshold
        self.HalfClickLen = half_click_len
        self.DiffMinMaxPowersHiThreshold = diff_min_max_powers_hi_threshold

# ==================================================================================================


class DefectClick2Settings:
    """
    Настройки дефекта snap2.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 freq_block_width,
                 hi_threshold,
                 lo_threshold,
                 half_click_len):
        """
        Конструктор.

        :param freq_block_width: Ширина блока частот, по которым ищется максимум.
        :param hi_threshold:     Верхняя граница нормализованнной силы сигнала
                                 (после оператора выявления границ),
                                 выше которой считается всплеск.
        :param lo_threshold:     Нижняя граница нормализованной силы сигнала
                                 (после оператора выявления границ),
                                 ниже которой считается тишина.
        :param half_click_len:   Половина длины щелчка.
        """

        self.FreqBlockWidth = freq_block_width
        self.HiThreshold = hi_threshold
        self.LoThreshold = lo_threshold
        self.HalfClickLen = half_click_len

# ==================================================================================================


class DefectMutedSettings:
    """
    Настройки дефекта muted.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 orthocenter_threshold):
        """
        Конструктор.

        :param orthocenter_threshold: Порог среднего значения ортоцентра записи.
        """

        self.OrthocenterThreshold = orthocenter_threshold

# ==================================================================================================


class DefectMuted2Settings:
    """
    Настройки дефекта muted2.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 percentage_of_lim_db,
                 percent_not_void,
                 percentage_of_error,
                 lim_percent_frame,
                 muted2_silence):

        """
        Конструктор.

        :param percentage_of_lim_db: порог обнаружения отсутствия частот.
        :param percent_not_void: процент фрейма, который не исследуется.
        :param percentage_of_error: процент погрешности для детектирования глухого фрейма.
        :param lim_percent_frame: процент порога наличия глухих фреймов.
        :param hop_length: ширина окна преобразования Фурье.
        """
        self.PercentageOfLimDb = percentage_of_lim_db
        self.PercentNotVoid = percent_not_void
        self.PercentageOfError = percentage_of_error
        self.LimPercentFrame = lim_percent_frame
        self.Muted2Silence = muted2_silence

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
                 click,
                 click2,
                 muted,
                 muted2,
                 comet):
        """
        Конструктор настроек для всех дефектов.

        :param limits_db: Лимиты по силе (за пределами лимитов вообще
                          не учитываем сигнал).
        :param click:     Настройки дефекта click.
        :param click2:    Настройки дефекта click2.
        :param muted:     Настройки дефекта muted.
        :param muted2:    Настройки дефекта muted2.
        :param comet:     Настройки дефекта comet.
        """

        self.LimitsDb = limits_db
        self.Click = click
        self.Click2 = click2
        self.Muted = muted
        self.Muted2 = muted2
        self.Comet = comet

# ==================================================================================================


# Определение настроек по умолчанию.

defect_click_settings = DefectClickSettings(limits_before_sort=(0.7, 0.95),
                                            limits_after_sort=(0.25, 0.75),
                                            min_power_lo_threshold=5.0,
                                            half_click_len=2,
                                            diff_min_max_powers_hi_threshold=5.0)

defect_click2_settings = DefectClick2Settings(freq_block_width=16,
                                              hi_threshold=0.5,
                                              lo_threshold=0.01,
                                              half_click_len=2)

defect_muted_settings = DefectMutedSettings(orthocenter_threshold=75)

defect_muted2_settings = DefectMuted2Settings(percentage_of_lim_db=10,
                                              percent_not_void=10,
                                              percentage_of_error=10,
                                              lim_percent_frame=65,
                                              muted2_silence=0.005)

defect_comet_settings = DefectCometSettings(signal_threshold=0.75,
                                            orth_quartile_threshold=800)

defects_settings = DefectsSettings(limits_db=(-50.0, 50.0),
                                   click=defect_click_settings,
                                   click2=defect_click2_settings,
                                   muted=defect_muted_settings,
                                   muted2=defect_muted2_settings,
                                   comet=defect_comet_settings)

# ==================================================================================================
