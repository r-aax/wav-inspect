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


# ==================================================================================================


class WAV:

    # ----------------------------------------------------------------------------------------------

    def __init__(self, filename=None):
        """
        Init WAV.
        :param filename: name of file
        """

        # Name of file.
        self.FileName = None

        # Array of amplitudes.
        self.Y = None

        # Sample rate.
        self.SampleRate = None

        # Duration.
        self.Duration = None

        # Spectres of two channels.
        self.Spectres = None

        # If filename if given - load it.
        if filename is not None:
            self.load(filename)

    # ----------------------------------------------------------------------------------------------

    def load(self, filename):
        """
        Load WAV file.
        :param filename: name of file
        :return: True - if loading is completed, False - if loading faults.
        """

        # First of all, check file extension.
        if pathlib.Path(filename).suffix != '.wav':
            return False

        # Load file.
        self.FileName = filename
        try:
            self.Y, self.SampleRate = librosa.load(filename, sr=None, mono=False)
        except BaseException:
            # If there is some problem with file, just ignore it.
            return False

        # Calculate duration.
        self.Duration = librosa.get_duration(y=self.Y, sr=self.SampleRate)

        return True

    # ----------------------------------------------------------------------------------------------

    def summary(self):
        """
        Print summary.
        """

        print('WAV audio record: FileName       = {0}'.format(self.FileName))
        print('                  Y.shape        = {0}'.format(self.Y.shape))
        print('                  SampleRate     = {0}'.format(self.SampleRate))
        print('                  Duration       = {0:.3f} s'.format(self.Duration))

        if self.Spectres is not None:
            print('                  Spectres.shape = {0}'.format(self.Spectres.shape))

    # ----------------------------------------------------------------------------------------------

    def generate_spectres(self):
        """
        Generate spectres.
        """

        generate_spectre = lambda d: librosa.amplitude_to_db(abs(librosa.stft(d, n_fft=2048)))

        self.Spectres = np.array([generate_spectre(d) for d in self.Y])

    # ----------------------------------------------------------------------------------------------

    def time_to_specpos(self, tx):
        """
        Translate time position to specpos.
        :param tx: time position
        :return: specpos
        """

        return int(tx * (self.Spectres.shape[-1] / self.Duration))

    # ----------------------------------------------------------------------------------------------

    def specpos_to_time(self, specpos):
        """
        Translate position in spectre to time.
        :param specpos: position in spectre
        :return: time point
        """

        return specpos * (self.Duration / self.Spectres.shape[-1])

    # ----------------------------------------------------------------------------------------------

    def show_wave(self, idx, figsize=(20, 8)):
        """
        Show wave of the sound.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Create figure and plot graph on it.
        plt.figure(figsize=figsize)
        librosa.display.waveplot(self.Y[idx], sr=self.SampleRate)

    # ----------------------------------------------------------------------------------------------

    def show_spectre(self, idx, figsize=(20, 8)):
        """
        Show spectre.
        :param idx: spectre index
        :param figsize: figure size
        """

        # Create figure and plot graphs onto it.
        plt.figure(figsize=figsize)
        librosa.display.specshow(self.Spectres[idx],
                                 sr=self.SampleRate,
                                 x_axis='time', y_axis='hz', cmap='turbo')
        plt.colorbar(format='%+02.0f dB')

# ==================================================================================================


if __name__ == '__main__':

    # Tests.
    # indices_slice_array
    assert indices_slice_array(3, 0, 2, 1) == [(0, 2), (1, 3)]
    assert indices_slice_array(10, 3, 3, 2) == [(3, 6), (5, 8), (7, 10)]


# ==================================================================================================
