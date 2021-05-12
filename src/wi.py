"""
WAV inspect functionality.
"""

import os
import pathlib
import random
import librosa
import numpy as np

# ==================================================================================================


def split_array(a, start, part_len, offset):
    """
    Split array into parts.
    :param a: array
    :param start: start position
    :param part_len: length of each part
    :param offset: offset between parts
    :return: array of parts (two dimensional numpy array)
    """

    # Get set of indices.
    ln = len(a)
    idx = filter(lambda i: ((i - start) % offset == 0) and (i + part_len <= ln),
                 range(ln))

    # Get array slices.
    r = np.array([a[i: i + part_len] for i in idx])

    return r

# ==================================================================================================


class WAV:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Init WAV.
        """

        # Name of file.
        self.fn = None

        # Array of data.
        self.y = None

        # Sample rate.
        self.sr = None

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

        return True

    # ----------------------------------------------------------------------------------------------

    def summary(self):
        """
        Print summary.
        """

        print('WAV audio record: fn      = {0}'.format(self.fn))
        print('                  y.shape = {0}'.format(self.y.shape))
        print('                  sr      = {0}'.format(self.sr))

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
        defect_x = [DataFactory.load_wav_mono_chunks(df, start, part_length, offset) for df in defect_files]
        defect_x = np.concatenate(defect_x, axis=0)
        defect_y = np.ones(defect_x.shape[0])
        defect_data = list(zip(defect_x, defect_y))

        # Load no defect files.
        no_defect_x = [DataFactory.load_wav_mono_chunks(df, start, part_length, offset) for df in no_defect_files]
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
            w.summary()

# --------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    test_load_wavs('wavs/origin')

    pass

# ==================================================================================================
