"""
WAV inspect functionality.
"""

import os
import pathlib
import random
import operator
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ==================================================================================================


def zipwith(a, b, f):
    """
    Zip two lists with given function.
    :param a: first list
    :param b: second list
    :param f: zip function
    """

    return [f(ai, bi) for (ai, bi) in zip(a, b)]


# --------------------------------------------------------------------------------------------------


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


def show_graph(data, figsize=(20, 8),
               style='r', linewidth=2.0,
               title='title', xlabel='', ylabel='',
               show_grid='true'):
    """
    Show data on graph.
    :param data: data, may be array of datas
    :param figsize: figure size
    :param style: line style, may be array of styles
    :param linewidth: line width, may be array of linewidths
    :param title: title
    :param xlabel: X label
    :param ylabel: Y label
    :param show_grid: flag for show grid
    """

    # Code for examples:
    # https://pythonru.com/biblioteki/pyplot-uroki

    # Style:
    # colors - 'b', 'g', 'r', 'y'.
    # markers - '*', '^', 's'.
    # line types - '--', '-.'.

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(show_grid)

    # Plot all.
    # Is the single data set is given, the style is just a string.
    if type(style) is str:

        # Single data set.
        plt.plot(data, style, linewidth=linewidth)

    elif type(style) is list:

        # Multiple datasets.
        for i in range(len(style)):
            plt.plot(data[i], style[i], linewidth=linewidth[i])

    plt.show()


# --------------------------------------------------------------------------------------------------


def min_without_some(ar, part=0.03):
    """
    Minimum value from array with ignoring part of values.
    :param ar: array
    :param part: part for ignoring
    :return: minimum values with ignoring 'part' of array
    """

    i = int(len(ar) * part)
    cp = ar.copy()
    cp.sort()

    return cp[i]

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
        # 3-dimensional array:
        #   (channels count) * (Y lines) * (X lines)
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

        # Check for filename.
        if not os.path.isfile(filename):
            # print('No such file ({0}).'.format(filename))
            return False

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

    # ----------------------------------------------------------------------------------------------

    def show_spectral_centroid(self, idx, figsize=(20, 8)):
        """
        Show spectral centroid.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Code source:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Calculate centroid.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y[idx],
                                                               sr=self.SampleRate)[0]

        # Normalize data.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Create figure and plot on it.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y[idx], sr=self.SampleRate, alpha=0.4)
        plt.plot(t, normalize(spectral_centroids), color='b')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_rolloff(self, idx, figsize=(20, 8)):
        """
        Show spectral rolloff.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Code source:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Calculate centroid.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y[idx],
                                                               sr=self.SampleRate)[0]

        # Calculate rolloff.
        spectral_rolloff = librosa.feature.spectral_rolloff(self.Y[idx] + 0.01,
                                                            sr=self.SampleRate)[0]

        # Normalize data.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Create figure and plot.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(self.Y[idx], sr=self.SampleRate, alpha=0.4)
        plt.plot(t, normalize(spectral_rolloff), color='r')

    # ----------------------------------------------------------------------------------------------

    def show_spectral_bandwidth(self, idx, figsize=(20, 8)):
        """
        Show spectral bandwidth.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        # Code source:
        # https://nuancesprog-ru.turbopages.org/nuancesprog.ru/s/p/6713/

        # Calculate centroid.
        spectral_centroids = librosa.feature.spectral_centroid(self.Y[idx],
                                                               sr=self.SampleRate)[0]

        # Construct bandwidth.
        x = self.Y[idx]
        sr = self.SampleRate
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr, p=4)[0]

        # Normalize data.
        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Create figure and plot on it.
        plt.figure(figsize=figsize)
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(x, sr=sr, alpha=0.4)
        plt.plot(t, normalize(spectral_bandwidth_2), color='r')
        plt.plot(t, normalize(spectral_bandwidth_3), color='g')
        plt.plot(t, normalize(spectral_bandwidth_4), color='y')
        plt.legend(('p = 2', 'p = 3', 'p = 4'))

    # ----------------------------------------------------------------------------------------------

    def normalize_spectre_value(self, idx):
        """
        Value for normalize spectre to shift minimum power to zero db.
        :param idx: indxes of amplitudes array
        :return: normalize spectre value
        """

        return -self.Spectres[idx].min()

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_total_power(self, idx, figsize=(20, 8)):
        """
        Show graph spectre total power.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        m = self.Spectres[idx].transpose()
        d = [sum(mi) for mi in m]
        show_graph(d, figsize=figsize, title='Spectre Total Power')

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_min_max_power(self, idx, figsize=(20, 8)):
        """
        Show graph spectre minimum and maximum power.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        m = self.Spectres[idx].transpose()
        d_min = [min_without_some(mi) for mi in m]
        d_max = [max(mi) for mi in m]
        show_graph([d_min, d_max], figsize=figsize,
                   title='Spectre Min/Max Power',
                   style=['b', 'r'], linewidth=[2.0, 2.0])

    # ----------------------------------------------------------------------------------------------

    def show_graph_spectre_total_power_with_high_accent(self, idx, figsize=(20, 8)):
        """
        Show graph spectre total power when high frequences are taken with big weights.
        :param idx: index of amplitudes array
        :param figsize: figure size
        """

        n = self.normalize_spectre_value(idx)
        m = self.Spectres[idx].transpose()

        # Weights.
        w = [i * i for i in range(len(m[0]))]

        # Form data for plot.
        d = [sum(zipwith(c, w, lambda ci, wi: (ci + n) * wi))
             for c in m]

        show_graph(d, figsize=figsize, title='Spectre Total Power With High Accent')

# ==================================================================================================


if __name__ == '__main__':

    # Unit tests.

    # indices_slice_array
    assert indices_slice_array(3, 0, 2, 1) == [(0, 2), (1, 3)]
    assert indices_slice_array(10, 3, 3, 2) == [(3, 6), (5, 8), (7, 10)]

    # min_without_part
    assert min_without_some([2, 1, 3, 5, 2], 0.0) == 1

    # Main test.

    test = 'wavs/origin/0001.wav'
    wav = WAV(test)
    wav.generate_spectres()
    wav.summary()

# ==================================================================================================
