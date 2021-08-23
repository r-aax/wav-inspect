"""
Определение настроек для работы с дефектами.
"""


# ==================================================================================================


class DefectClickSettings:
    """
    Настройки дефекта click2.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 quartiles,
                 win_h,
                 win_w,
                 thr,
                 mean_thr):
        """
        Конструктор.

        :param quartiles: Количество квартилей (вертикальных блоков),
                          по которым ищется максимум сигнала.
        :param win_h:     Высота блока частот, по которым ищется максимум.
        :param win_w:     Ширина окна на спектрограмме,
                          внутри которого определяется щелчок.
        :param thr:       Порог детектирования щелчка, разница между максимальным
                          и средним значением в окне
                          (порог применяется к нормализованным значениям, поэтому
                          измеряется в относительных величинах, максимальное значение 1).
        :param mean_thr:  Среднее значение сигнала в окне, выше которого щелчок не определяется
                          (порог применяется к нормализованным значениям, поэтому
                          измеряется в относительных величинах, максимальное значение 1).
        """

        self.Quartiles = quartiles
        self.WinH = win_h
        self.WinW = win_w
        self.Thr = thr
        self.MeanThr = mean_thr

# ==================================================================================================


class DefectDeafSettings:
    """
    Настройки дефекта deaf.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 thr):
        """
        Конструктор.

        :param thr: Порог среднего значения ортоцентра записи (в процентах).
        """

        self.Thr = thr

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


class DefectEchoSettings:
    """
    Настройки дефекта echo.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 loc_corr_win,
                 loc_corr_thr,
                 glob_norm_corr_thr):

        """
        Конструктор.

        :param loc_corr_win:       Длина окна локальной автокорреляции (секунды).
        :param loc_corr_thr:       Порог локальной корреляции, выше которого детектируем эхо.
        :param glob_norm_corr_thr: Порог нормализованной глобальной корреляции темпограммы
                                   (значение берется как среднее хвоста функции) выше которого
                                   это не ищется (думаем, что это музыкальный фрагмент).
        """

        self.LocCorrWin = loc_corr_win
        self.LocCorrThr = loc_corr_thr
        self.GlobNormCorrThr = glob_norm_corr_thr

# ==================================================================================================


class DefectAsncSettings:
    """
    Настройки дефекта asnc.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 thr):
        """
        Конструктор.

        :param thr: Порог рассинхронизации каналов.
        """

        self.Thr = thr

# ==================================================================================================


class DefectDiffSettings:
    """
    Настройки дефекта diff.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 width_min,
                 thr):
        """
        Конструктор.

        :param width: Ширина окна для взятия минимума разницы.
        :param thr:   Порог определения дефекта.
        """

        self.WidthMin = width_min
        self.Thr = thr

# ==================================================================================================


class DefectHumSettings:
    """
    Настройки дефекта hum.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 lo_ignore,
                 thr):
        """
        Конструктор.

        :param lo_ignore: Нижняя граница игнорирования частот (в процентах).
        :param thr:       Порог скачка разницы оригинального и сглаженного отношений квантилей.
        """

        self.LoIgnore = lo_ignore
        self.Thr = thr

# ==================================================================================================


class DefectSaturSettings:
    """
    Настройки дефекта satur.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 filter_width,
                 power_thr):
        """
        Конструктор.

        :param filter_width: Ширина фильтра.
        :param power_thr:    Порог сигнала.
        """

        self.FilterWidth = filter_width
        self.PowerThr = power_thr

# ==================================================================================================


class DefectsSettings:
    """
    Настройки дефектов.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 limits_db,
                 click,
                 deaf,
                 deaf2,
                 echo,
                 asnc,
                 diff,
                 hum,
                 satur):
        """
        Конструктор настроек для всех дефектов.

        :param limits_db: Лимиты по силе (за пределами лимитов вообще
                          не учитываем сигнал).
        :param click:     Настройки дефекта click2.
        :param deaf:      Настройки дефекта deaf.
        :param deaf2:     Настройки дефекта deaf2.
        :param comet:     Настройки дефекта comet.
        :param echo:      Настройки дефекта echo.
        :param asnc:      Настрйоки дефекта asnc.
        :param diff:      Настройки дефекта diff.
        :param hum:       Настройки дефекта hum.
        :param satur:     Настройки дефекта satur.
        """

        self.LimitsDb = limits_db
        self.Click = click
        self.Deaf = deaf
        self.Deaf2 = deaf2
        self.Echo = echo
        self.Asnc = asnc
        self.Diff = diff
        self.Hum = hum
        self.Satur = satur

# ==================================================================================================


# Определение настроек по умолчанию.

defect_click_settings = DefectClickSettings(quartiles=4,
                                            win_h=16,
                                            win_w=32,
                                            thr=0.6,
                                            mean_thr=0.1)

defect_deaf_settings = DefectDeafSettings(thr=7.0)

defect_deaf2_settings = DefectDeaf2Settings(percentage_of_lim_db=10,
                                            percent_not_void=10,
                                            percentage_of_error=10,
                                            lim_percent_frame=65,
                                            deaf2_silence=0.005)

defect_echo_settings = DefectEchoSettings(loc_corr_win=2.0,
                                          loc_corr_thr=100.0,
                                          glob_norm_corr_thr=0.5)

defect_asnc_settings = DefectAsncSettings(thr=0.4)

defect_diff_settings = DefectDiffSettings(width_min=10,
                                          thr=0.1)

defect_hum_settings = DefectHumSettings(lo_ignore=10.0,
                                        thr=0.5)

defect_satur_settings = DefectSaturSettings(filter_width=32,
                                            power_thr=0.2)

defects_settings = DefectsSettings(limits_db=(-50.0, 50.0),
                                   click=defect_click_settings,
                                   deaf=defect_deaf_settings,
                                   deaf2=defect_deaf2_settings,
                                   echo=defect_echo_settings,
                                   asnc=defect_asnc_settings,
                                   diff=defect_diff_settings,
                                   hum=defect_hum_settings,
                                   satur=defect_satur_settings)

# ==================================================================================================
