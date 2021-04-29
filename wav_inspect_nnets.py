"""
WAV inspect NNets.
"""
import random

import librosa
import numpy as np


def unzip(lot):
    """
    Unzip list of tuples.
    :param lot: list of tuples
    :return: tuple of lists
    """

    return np.array([x for (x, y) in lot]), np.array([y for (x, y) in lot])


class ArraysManipulator:

    @staticmethod
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
        idx = filter(lambda i: ((i - start) % offset == 0) and (i + part_len <= ln), range(ln))

        # Get array slices.
        r = np.array([a[i: i + part_len] for i in idx])

        return r


class DataFactory:

    @staticmethod
    def load_wav(filename):
        """
        Load wav file.
        :param filename: name of file
        :return: loaded data and sample rate
        """

        # Always load with native sample rate and in stereo mode.
        return librosa.load(path=filename,
                            sr=None,
                            mono=False)

    @staticmethod
    def load_wav_mono_chunks(filename, start, part_len, offset):
        """
        Load wav file, convert it into mono mode and split.
        :param filename: name of file
        :param start: start position for splitting (seconds)
        :param part_len: part length (seconds)
        :param offset: offset between adjacent parts (seconds)
        :return:
        """

        d, sr = DataFactory.load_wav(filename)
        d = librosa.to_mono(d)
        return ArraysManipulator.split_array(d,
                                             int(sr * start),
                                             int(sr * part_len),
                                             int(sr * offset))


class DefectWrongSide:

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

if __name__ == '__main__':
    pass
