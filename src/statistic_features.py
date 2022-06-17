# built-in
import os

# third-party
import numpy as np
from scipy.stats import skew, kurtosis

def signal_stats(signal=None, hist=True, names=False):
    """Compute statistical metrics describing the signal.
        Parameters
        ----------
        signal : array
            Input signal.

        Returns
        -------
        mean : float
            Mean of the signal.
        median : float
            Median of the signal.
        var : float
            Signal variance (unbiased).
        std : float
            Standard signal deviation (unbiased).
        abs_dev : float
            Absolute signal deviation.
        kurtosis : float
            Signal kurtosis (unbiased).
        skewness : float
            Signal skewness (unbiased).
        iqr : float
            Interquartile Range.
        meanadev : float
            Mean absolute deviation.
        medadev : float
            Median absolute deviation.
        rms : float
            Root Mean Square.
        _hist : list
            Histogram.

        References
        ----------
        TSFEL library: https://github.com/fraunhoferportugal/tsfel
        Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

        """

    # check inputs
    # if signal is None or signal == []:
    #     print("Signal is empty.")

    if names:
        return ['mean', 'median', 'var', 'std', 'abs_dev', 'kurtosis', 'skewness', 'iqr', 'rms']

    # ensure numpy
    signal = np.array(signal)
    if len(signal) == 0:
        return 0., 0., 0., 0., 0., 0. ,0. ,0. ,0. 
    else:
        mean = np.mean(signal)
        median = np.median(signal)
        var = np.var(signal)
        std = np.std(signal)
        abs_dev = np.sum(np.abs(signal - median))
        kurtosis_ = kurtosis(signal)
        skewness = skew(signal)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        rms = np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))


        return np.hstack((mean, median, var, std, abs_dev, kurtosis_, skewness, iqr, rms))

