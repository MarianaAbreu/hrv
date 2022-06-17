# built-in
import json
import os

# third-party
import numpy as np

# local
from biosppy import utils
from biosppy import tools as st


def spectral_roll(f, ff, cum_ff, TH):
    """ Frequency so TH % of the signal energy is below that value.

    Parameters
    ----------
    sign : ndarray
        Signal from which spectral slope is computed.
    fs : int
        Sampling frequency of the signal.
    Returns
    -------
    roll_off : float
        Spectral roll-off.

    References
    ----------
    TSFEL library: https://github.com/fraunhoferportugal/tsfel
    Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

    """
    output = None
    try:
        value = TH*(np.sum(ff))
        for i in range(len(ff)):
            if cum_ff[i] >= value:
                output = f[i]
                break
    except:
        output = None
    args = (output,)
    names = ('spectral_roll',)
    return utils.ReturnTuple(args, names)


def signal_spectral(signal, FS):
    """Compute spectral metrics describing the signal.
        Parameters
        ----------
        signal : array
            Input signal.
        FS : float
            Sampling frequency

        Returns
        -------
        spectral_maxpeaks : int
            Number of peaks in the spectrum signal.

        spect_var : float
            Amount of the variation of the spectrum across time.

        curve_distance : float
            Euclidean distance between the cumulative sum of the signal spectrum and evenly spaced numbers across the signal lenght.

        spectral_roll_off : float
            Frequency so 95% of the signal energy is below that value.

        spectral_roll_on : float
            Frequency so 5% of the signal energy is below that value.

        spectral_dec : float
            Amount of decreasing in the spectral amplitude.

        spectral_slope : float
            Amount of decreasing in the spectral amplitude.

        spectral_centroid : float
            Centroid of the signal spectrum.

        spectral_spread : float
            Variance of the signal spectrum i.e. how it spreads around its mean value.

        spectral_kurtosis : float
            Kurtosis of the signal spectrum i.e. describes the flatness of the spectrum distribution.

        spectral_skewness : float
            Skewness of the signal spectrum i.e. describes the asymmetry of the spectrum distribution.

        max_frequency : float
            Maximum frequency of the signal spectrum maximum amplitude.

        fundamental_frequency : float
            Fundamental frequency of the signal.

        max_power_spectrum : float
            Spectrum maximum value.

        mean_power_spectrum : float
            Spectrum mean value.

        spectral_skewness : float
            Spectrum Skewness.

        spectral_kurtosis : float
            Spectrum Kurtosis.

        spectral_hist_ : list
            Histogram of the signal spectrum.

        References
        ----------
        TSFEL library: https://github.com/fraunhoferportugal/tsfel
        Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

        """
    # check inputs
    if signal is None or signal == []:
        print("Signal is empty.")

    # ensure numpy
    signal = np.array(signal)
    # f, spectrum = st.welch_spectrum(signal, sampling_rate=FS)
    spectrum_signal = np.abs(np.fft.fft(signal, FS))**2
    spectrum = spectrum_signal[:len(spectrum_signal)//2]
    f = np.linspace(0, FS/2, len(spectrum))
    cum_ff = np.cumsum(spectrum)
    spect_diff = np.diff(spectrum)
    #energy, _ = st.signal_energy(spectrum, f)[:]

    args, names = [], []

    try:
        spectral_maxpeaks = np.sum([1 for nd in range(len(spect_diff[:-1])) if (spect_diff[nd+1]<0 and spect_diff[nd]>0)])
    except:
        spectral_maxpeaks = None

    #try:
    #         spect_var = np.convolve(energy)
    #         spect_var /= np.max(np.abs(spect_var))
    #     except:
    #         spect_var = None

    try:
        # this value is too large
        curve_distance = np.linalg.norm(np.linspace(0, cum_ff[-1], len(cum_ff)) - cum_ff)
        # divide by the total sum of the signal (last value of the cum sum)
        curve_distance_small = curve_distance/cum_ff[-1]
    except:
        curve_distance = None
        curve_distance_small = None

    # spectral_roll_off
    try:
        spectral_roll_off = f[np.argwhere(cum_ff/cum_ff[-1]>= 0.95)[0]]
    except:
        spectral_roll_off = None

    try:
        spectral_dec = (1/np.sum(spectrum)) * np.sum((spectrum[:] - spectrum[1])/np.linspace(1, len(spectrum), len(spectrum),1))
    except:
        spectral_dec = None

    sum_f = np.sum(f)
    len_f = len(f)
    try:
        spectral_slope = (len_f * np.dot(f, spectrum) - sum_f * np.sum(spectrum)) / (len_f * np.dot(f, f) - sum_f ** 2)
    except:
        spectral_slope = None

    sum_spectrum = np.sum(spectrum)
    norm_spectrum = spectrum / sum_spectrum
    # spectral_centroid
    try:
        spectral_centroid = np.dot(f, norm_spectrum)
    except:
        spectral_centroid = None

    # spectral_spread
    try:
        spectral_spread = np.dot(((f - spectral_centroid) ** 2), norm_spectrum)
    except:
        spectral_spread = None

    try:
        spectral_kurtosis = np.sum(((f - spectral_centroid) ** 4) * norm_spectrum) / (spectral_spread**2)
    except:
        spectral_kurtosis = None

    import scipy.stats as stats

    try:
        spectral_skewness = stats.skew(norm_spectrum)
    except:
        spectral_skewness = None
        
    if dict['max_frequency']['use'] == 'yes':
        # max_frequency
        try:
            max_frequency = f[np.where(cum_ff > cum_ff[-1]*0.95)[0][0]]
        except:
            max_frequency = None
        args += [max_frequency]
        names += ['max_frequency']

    if dict['fundamental_frequency']['use'] == 'yes':
        # fundamental_frequency
        try:
            fundamental_frequency = f[np.where(cum_ff > cum_ff[-1]*0.5)[0][0]]
        except:
            fundamental_frequency = None
        args += [fundamental_frequency]
        names += ['fundamental_frequency']

    # if dict['max_power_spectrum']['use'] == 'yes':
    #     # max_power_spectrum
    #     try:
    #         max_power_spectrum = np.max(spectrum)
    #     except:
    #         max_power_spectrum = None
    #     args += max_power_spectrum
    #     names += 'max_power_spectrum'

    # if dict['mean_power_spectrum']['use'] == 'yes':
    #     # mean_power_spectrum
    #     try:
    #         mean_power_spectrum = np.mean(spectrum)
    #     except:
    #         mean_power_spectrum = None
    #     args += mean_power_spectrum
    #     names += 'mean_power_spectrum'
    #
    # if dict['spectral_skewness']['use'] == 'yes':
    #     try:
    #         spectral_skewness = np.mean(spectrum)
    #     except:
    #         spectral_skewness = None
    #     args += spectral_skewness
    #     names += 'spectral_skewness'
    #
    # if dict['spectral_kurtosis']['use'] == 'yes':
    #     try:
    #         spectral_kurtosis = np.mean(spectrum)
    #     except:
    #         spectral_kurtosis = None
    #     args += spectral_kurtosis
    #     names += 'spectral_kurtosis'

    # if dict['spectral_hist_']['use'] == 'yes':
    #     # histogram
    #     try:
    #         _hist = list(np.histogram(spectrum, bins=int(np.sqrt(len(spectrum))), density=True)[0])
    #     except:
    #         if len(signal) > 1:
    #             _hist = [None] * int(np.sqrt(len(signal)))
    #         else:
    #             _hist = [None]
    #     args += [i for i in _hist]
    #     names += ['spectral_hist_' + str(i) for i in range(len(_hist))]

    #return utils.ReturnTuple(tuple(args), tuple(names))
    return args, names
