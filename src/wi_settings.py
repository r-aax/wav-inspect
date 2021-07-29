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
    Настройки дефекта click2.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 freq_block_height,
                 detect_window_width,
                 threshold,
                 mean_threshold):
        """
        Конструктор.

        :param freq_block_height:   Высота блока частот, по которым ищется максимум.
        :param detect_window_width: Ширина окна на спектрограмме,
                                    внутри которого определяется щелчок.
        :param threshold:           Порог детектирования щелчка (разница между максимальным
                                    и средним значением в окне).
        :param mean_threshold:      Среднее значение сигнала в окне, выше которого
                                    щелчок не определяется.
        """

        self.FreqBlockHeight = freq_block_height
        self.DetectWindowWidth = detect_window_width
        self.Threshold = threshold
        self.MeanThreshold = mean_threshold

# ==================================================================================================


class DefectDeafSettings:
    """
    Настройки дефекта deaf.
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


class DefectDeaf2Settings:
    """
    Настройки дефекта deaf2.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 percentage_of_lim_db,
                 percent_not_void,
                 percentage_of_error,
                 lim_percent_frame,
                 deaf2_silence):

        """
        Конструктор.

        :param percentage_of_lim_db: Порог обнаружения отсутствия частот.
        :param percent_not_void:     Процент фрейма, который не исследуется.
        :param percentage_of_error:  Процент погрешности для детектирования глухого фрейма.
        :param lim_percent_frame:    Процент порога наличия глухих фреймов.
        :param deaf2_silence:        Порог преобразования глухого сигнала в тишину.
        """
        self.PercentageOfLimDb = percentage_of_lim_db
        self.PercentNotVoid = percent_not_void
        self.PercentageOfError = percentage_of_error
        self.LimPercentFrame = lim_percent_frame
        self.Deaf2Silence = deaf2_silence

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
                 deaf,
                 deaf2,
                 comet):
        """
        Конструктор настроек для всех дефектов.

        :param limits_db: Лимиты по силе (за пределами лимитов вообще
                          не учитываем сигнал).
        :param click:     Настройки дефекта click.
        :param click2:    Настройки дефекта click2.
        :param deaf:      Настройки дефекта deaf.
        :param deaf2:     Настройки дефекта deaf2.
        :param comet:     Настройки дефекта comet.
        """

        self.LimitsDb = limits_db
        self.Click = click
        self.Click2 = click2
        self.Deaf = deaf
        self.Deaf2 = deaf2
        self.Comet = comet

# ==================================================================================================


# Определение настроек по умолчанию.

defect_click_settings = DefectClickSettings(limits_before_sort=(0.7, 0.95),
                                            limits_after_sort=(0.25, 0.75),
                                            min_power_lo_threshold=5.0,
                                            half_click_len=2,
                                            diff_min_max_powers_hi_threshold=5.0)

defect_click2_settings = DefectClick2Settings(freq_block_height=16,
                                              detect_window_width=32,
                                              threshold=0.6,
                                              mean_threshold=0.1)

defect_deaf_settings = DefectDeafSettings(orthocenter_threshold=75)

defect_deaf2_settings = DefectDeaf2Settings(percentage_of_lim_db=10,
                                            percent_not_void=10,
                                            percentage_of_error=10,
                                            lim_percent_frame=65,
                                            deaf2_silence=0.005)

defect_comet_settings = DefectCometSettings(signal_threshold=0.75,
                                            orth_quartile_threshold=800)

defects_settings = DefectsSettings(limits_db=(-50.0, 50.0),
                                   click=defect_click_settings,
                                   click2=defect_click2_settings,
                                   deaf=defect_deaf_settings,
                                   deaf2=defect_deaf2_settings,
                                   comet=defect_comet_settings)

# ==================================================================================================
