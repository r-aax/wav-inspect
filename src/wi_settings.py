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
                 sep,
                 quartiles,
                 win_h,
                 win_w,
                 thr,
                 mean_thr):
        """
        Конструктор.

        :param sep:       Параметры разделения (размер фрагмента и хвоста в секундах).
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

        self.Sep = sep
        self.Quartiles = quartiles
        self.WinH = win_h
        self.WinW = win_w
        self.Thr = thr
        self.MeanThr = mean_thr

# ==================================================================================================


class DefectMutedSettings:
    """
    Настройки дефекта muted.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 thr):
        """
        Конструктор.

        :param sep: Параметры разделения (размер фрагмента и хвоста в секундах).
        :param thr: Порог среднего значения ортоцентра записи (в процентах).
        """

        self.Sep = sep
        self.Thr = thr

# ==================================================================================================


class DefectMuted2Settings:
    """
    Настройки дефекта muted2.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 percentage_of_lim_db,
                 percent_not_void,
                 percentage_of_error,
                 lim_percent_frame,
                 muted2_silence):

        """
        Конструктор.

        :param sep:                  Параметры разделения (размер фрагмента и хвоста в секундах).
        :param percentage_of_lim_db: Порог обнаружения отсутствия частот.
        :param percent_not_void:     Процент фрейма, который не исследуется.
        :param percentage_of_error:  Процент погрешности для детектирования глухого фрейма.
        :param lim_percent_frame:    Процент порога наличия глухих фреймов.
        :param muted2_silence:       Порог преобразования глухого сигнала в тишину.
        """

        self.Sep = sep
        self.PercentageOfLimDb = percentage_of_lim_db
        self.PercentNotVoid = percent_not_void
        self.PercentageOfError = percentage_of_error
        self.LimPercentFrame = lim_percent_frame
        self.Muted2Silence = muted2_silence

# ==================================================================================================


class DefectEchoSettings:
    """
    Настройки дефекта echo.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 loc_corr_win,
                 loc_corr_thr,
                 glob_norm_corr_thr):

        """
        Конструктор.

        :param sep:                Параметры разделения (размер фрагмента и хвоста в секундах).
        :param loc_corr_win:       Длина окна локальной автокорреляции (секунды).
        :param loc_corr_thr:       Порог локальной корреляции, выше которого детектируем эхо.
        :param glob_norm_corr_thr: Порог нормализованной глобальной корреляции темпограммы
                                   (значение берется как среднее хвоста функции) выше которого
                                   это не ищется (думаем, что это музыкальный фрагмент).
        """

        self.Sep = sep
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
                 sep,
                 thr):
        """
        Конструктор.

        :param sep: Параметры разделения (размер фрагмента и хвоста в секундах).
        :param thr: Порог рассинхронизации каналов.
        """

        self.Sep = sep
        self.Thr = thr

# ==================================================================================================


class DefectDiffSettings:
    """
    Настройки дефекта diff.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 width_min,
                 thr):
        """
        Конструктор.

        :param sep:   Параметры разделения (размер фрагмента и хвоста в секундах).
        :param width: Ширина окна для взятия минимума разницы.
        :param thr:   Порог определения дефекта.
        """

        self.Sep = sep
        self.WidthMin = width_min
        self.Thr = thr

# ==================================================================================================


class DefectHumSettings:
    """
    Настройки дефекта hum.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 lo_ignore,
                 thr):
        """
        Конструктор.

        :param sep:       Параметры разделения (размер фрагмента и хвоста в секундах).
        :param lo_ignore: Нижняя граница игнорирования частот (в процентах).
        :param thr:       Порог скачка разницы оригинального и сглаженного отношений квантилей.
        """

        self.Sep = sep
        self.LoIgnore = lo_ignore
        self.Thr = thr

# ==================================================================================================


class DefectDenseSettings:
    """
    Настройки дефекта dense.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 percent,
                 width,
                 min_hz,
                 max_hz):
        """
        Конструктор.

        :param sep:      Параметры разделения (размер фрагмента и хвоста в секундах).
        :param percent:  Процент погрешности для детектирования.
        :param width:    Допустимая ширина для детектируемой частоты.
        :param min_hz:   Минимальный порог рабочих частот (в высоте фрейма).
        :param max_hz:   Максимальный порог рабочих частот (в высоте фрейма).
        """

        self.Sep = sep
        self.Percent = percent
        self.Width = width
        self.MinHz = min_hz
        self.MaxHz = max_hz


# ==================================================================================================


class DefectOverLoadSettings:
    """
    Настройки дефекта over_load.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 filter_width,
                 power_thr):
        """
        Конструктор.

        :param sep:          Параметры разделения (размер фрагмента и хвоста в секундах).
        :param filter_width: Ширина фильтра.
        :param power_thr:    Порог сигнала.
        """

        self.Sep = sep
        self.FilterWidth = filter_width
        self.PowerThr = power_thr

# ==================================================================================================


class DefectLoudSettings:
    """
    Настройки дефекта loud.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 thr):
        """
        Конструктор.

        :param sep: Параметры разделения (размер фрагмента и хвоста в секундах).
        :param thr: Порог уровня громкости.
        """

        self.Sep = sep
        self.Thr = thr

# ==================================================================================================


class DefectDblSettings:
    """
    Настройки дефекта dbl.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 sep,
                 top,
                 thr):
        """
        Конструктор.

        :param sep: Параметры разделения (размер фрагмента и хвоста в секундах).
        :param top: Количество анализируемых верхних амплитуд.
        :param thr: Порог по дублям среди верхних амплитуд
                    (если слишком много точных совпадений, то считаем, что
                    имеет место ошибка цифрового дубля).
        """

        self.Sep = sep
        self.Top = top
        self.Thr = thr

# ==================================================================================================


class DefectsSettings:
    """
    Настройки дефектов.
    """

    # ----------------------------------------------------------------------------------------------

    def __init__(self,
                 limits_db,
                 click,
                 muted,
                 muted2,
                 echo,
                 asnc,
                 diff,
                 hum,
                 dense,
                 over_load,
                 loud,
                 dbl):
        """
        Конструктор настроек для всех дефектов.

        :param limits_db: Лимиты по силе (за пределами лимитов вообще
                          не учитываем сигнал).
        :param click:     Настройки дефекта click2.
        :param muted:     Настройки дефекта muted.
        :param muted2:    Настройки дефекта muted2.
        :param comet:     Настройки дефекта comet.
        :param echo:      Настройки дефекта echo.
        :param asnc:      Настрйоки дефекта asnc.
        :param diff:      Настройки дефекта diff.
        :param hum:       Настройки дефекта hum.
        :param dense:     Настройки дефекта dense.
        :param over_load:     Настройки дефекта over_load.
        :param loud:      Настройки дефекта loud.
        :param dbl:       Найтройки дефекта dbl.
        """

        self.LimitsDb = limits_db
        self.Click = click
        self.Muted = muted
        self.Muted2 = muted2
        self.Echo = echo
        self.Asnc = asnc
        self.Diff = diff
        self.Hum = hum
        self.Dense = dense
        self.OverLoad = over_load
        self.Loud = loud
        self.Dbl = dbl

# ==================================================================================================


# Определение настроек по умолчанию.

defect_click_settings = DefectClickSettings(sep=(20.0, 1.0),
                                            quartiles=4,
                                            win_h=16,
                                            win_w=32,
                                            thr=0.6,
                                            mean_thr=0.1)

defect_muted_settings = DefectMutedSettings(sep=(10.0, 2.0),
                                            thr=7.0)

defect_muted2_settings = DefectMuted2Settings(sep=(10.0, 2.0),
                                              percentage_of_lim_db=10,
                                              percent_not_void=10,
                                              percentage_of_error=10,
                                              lim_percent_frame=65,
                                              muted2_silence=0.005)

defect_echo_settings = DefectEchoSettings(sep=(10.0, 2.0),
                                          loc_corr_win=2.0,
                                          loc_corr_thr=100.0,
                                          glob_norm_corr_thr=0.5)

defect_asnc_settings = DefectAsncSettings(sep=(10.0, 5.0),
                                          thr=0.4)

defect_diff_settings = DefectDiffSettings(sep=(5.0, 5.0),
                                          width_min=10,
                                          thr=0.1)

defect_hum_settings = DefectHumSettings(sep=(10.0, 2.0),
                                        lo_ignore=10.0,
                                        thr=0.51)

defect_dense_settings = DefectDenseSettings(sep=(30.0, 0.0),
                                            percent=3,
                                            width=4,
                                            min_hz=500,
                                            max_hz=900)

defect_over_load_settings = DefectOverLoadSettings(sep=(10.0, 2.0),
                                            filter_width=16,
                                            power_thr=0.2)

defect_loud_settings = DefectLoudSettings(sep=(10.0, 1.0),
                                          thr=-12.0)

defect_dbl_settings = DefectDblSettings(sep=(30.0, 4.0),
                                        top=100,
                                        thr=0.05)

defects_settings = DefectsSettings(limits_db=(-50.0, 50.0),
                                   click=defect_click_settings,
                                   muted=defect_muted_settings,
                                   muted2=defect_muted2_settings,
                                   echo=defect_echo_settings,
                                   asnc=defect_asnc_settings,
                                   diff=defect_diff_settings,
                                   hum=defect_hum_settings,
                                   dense=defect_dense_settings,
                                   over_load=defect_over_load_settings,
                                   loud=defect_loud_settings,
                                   dbl=defect_dbl_settings)

# ==================================================================================================
