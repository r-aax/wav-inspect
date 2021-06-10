"""
WAV inspect functionality.
"""

import os
import pathlib
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


# ==================================================================================================


def indices_slice_array(ar_len, start, part_len, step):
    """
    Get indices for array slicing.
    :param ar_len: array length
    :param start: start position
    :param part_len: single part length
    :param step: step between parts
    :return: array of first indices
    """

    idx = [(i, i + part_len)
           for i in range(start, ar_len - part_len + 1)
           if (i - start) % step == 0]

    return idx

# --------------------------------------------------------------------------------------------------


def show_plt(data, sr):
    """
    Show plot.
    :param data: data for plotting
    :param sr: sample rate
    """

    # Create figure and plot graphs onto it.
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(data, sr=sr, x_axis='time', y_axis='hz', cmap='turbo')
    plt.colorbar(format='%+02.0f dB')


# ==================================================================================================


class WAV:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Init WAV.
        """

        # Name of file.
        self.fn = None

        # Array of amplitudes.
        self.y = None

        # Sample rate.
        self.sr = None

        # Duration.
        self.dur = None

        # Spectres of two channels.
        self.sp = None

        pass

    # ----------------------------------------------------------------------------------------------

    def load(self, filename):
        """
        Load WAV file.
        :param filename:
        :return: True - if loading is completed, False - if loading faults.
        """

        # First of all, check file extension.
        if pathlib.Path(filename).suffix != '.wav':
            return False

        # Load file.
        self.fn = filename
        try:
            self.y, self.sr = librosa.load(filename, sr=None, mono=False)
        except BaseException:
            # If there is some problem with file, just ignore it.
            return False

        # Calculate duration.
        self.dur = librosa.get_duration(y=self.y, sr=self.sr)

        return True

    # ----------------------------------------------------------------------------------------------

    def summary(self):
        """
        Print summary.
        """

        print('WAV audio record: fn       = {0}'.format(self.fn))
        print('                  y.shape  = {0}'.format(self.y.shape))
        print('                  sr       = {0}'.format(self.sr))
        print('                  dur      = {0}'.format(self.dur))

        if self.sp is not None:
            print('                  sp.shape = {0}'.format(self.sp.shape))

    # ----------------------------------------------------------------------------------------------

    def get_spectres(self):
        """
        Get spectres.
        """

        get_spectre = lambda d: librosa.amplitude_to_db(abs(librosa.stft(d)))

        self.sp = np.array([get_spectre(d) for d in self.y])

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Translate time position to specpos.
        :param tx: time position
        :return: specpos
        """

        return int(tx * (self.sp.shape[-1] / self.dur))

    # ----------------------------------------------------------------------------------------------

    def specpos_to_time(self, specpos):
        """
        Translate position in spectre to time.
        :param specpos: position in spectre
        :return: time point
        """

        return specpos * (self.dur / self.sp.shape[-1])

# ==================================================================================================


class DataFactory:

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def load_wav_chunks(filename, start, part_len, offset, is_mono=True):
        """
        Load wav file, convert it into mono mode and split.
        :param filename: name of file
        :param start: start position for splitting (seconds)
        :param part_len: part length (seconds)
        :param offset: offset between adjacent parts (seconds)
        :param is_mono: flag for convert file to mono
        :return:
        """

        d, sr = DataFactory.load_wav(filename)

        if is_mono:
            d = librosa.to_mono(d)
            return split_array(d,
                               int(sr * start),
                               int(sr * part_len),
                               int(sr * offset))
        else:
            raise Exception('only mono is supported')


# ==================================================================================================


class DefectWrongSide:

    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def load_data(defect_files, no_defect_files):
        """
        Load data for 'Wrong Side' defect.
        :param defect_files: list of files with defect
        :param no_defect_files: list of files without defect
        :return:
        """

        print('DefectWrongSide :    defect files - {0}'.format(defect_files))
        print('                  no defect files - {0}'.format(no_defect_files))

        start, part_length, offset = 0.0, 1.0, 0.05
        part_of_test_data = 0.2

        # Load defect files.
        defect_x = [DataFactory.load_wav_mono_chunks(df, start, part_length, offset) for df in
                    defect_files]
        defect_x = np.concatenate(defect_x, axis=0)
        defect_y = np.ones(defect_x.shape[0])
        defect_data = list(zip(defect_x, defect_y))

        # Load no defect files.
        no_defect_x = [DataFactory.load_wav_mono_chunks(df, start, part_length, offset) for df in
                       no_defect_files]
        no_defect_x = np.concatenate(no_defect_x, axis=0)
        no_defect_y = np.zeros(no_defect_x.shape[0])
        no_defect_data = list(zip(no_defect_x, no_defect_y))

        # Join all data.
        data = [*defect_data, *no_defect_data]

        # Shuffle and split back.
        random.shuffle(data)
        x, y = unzip(data)
        x = np.array([librosa.amplitude_to_db(np.abs(librosa.stft(xi, n_fft=1024))) for xi in x])

        # Blade index.
        bi = int(len(x) * part_of_test_data)

        return (np.array(x[bi:]), np.array(y[bi:])), (np.array(x[:bi]), np.array(y[:bi]))


# ==================================================================================================


def test_load_wavs(dir):
    """
    Load all wavs and print information.
    :param dir: dir
    """

    ld = os.listdir(dir)

    # Just load wavs.
    print('=== Just load wav file. ===')
    for fn in ld:
        w = WAV()
        if w.load(dir + '/' + fn):
            w.get_spectres()
            w.summary()


# --------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # Tests.
    # indices_slice_array
    assert indices_slice_array(3, 0, 2, 1) == [(0, 2), (1, 3)]
    assert indices_slice_array(10, 3, 3, 2) == [(3, 6), (5, 8), (7, 10)]

    # test_load_wavs('wavs/origin')

    pass

# ==================================================================================================
