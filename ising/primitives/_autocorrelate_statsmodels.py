# Code in this file is based on the BSD 3-Clause "New" or "Revised" licensed
# statsmodels project: https://www.statsmodels.org/stable/index.html
# The code has been modified. The modified code is released on the same license
# as the original code.

# NOTE: Only the main path is working.
# Don't touch any of the boolean parameters or you will get in trouble!

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Num

from ising.utils.types import assert_concrete

TMissing = Literal["none", "raise", "conservative", "drop"]


def acf(
    x: Num[Array, "a"],
    adjusted: bool = False,
    nlags: int | None = None,
    qstat: bool = False,
    fft: bool = True,
    alpha: float | None = None,
    bartlett_confint: bool = True,
    missing: TMissing = "none",
):
    """
    Calculate the autocorrelation function.

    Parameters
    ----------
    x : array_like
       The time series data.
    adjusted : bool, default False
       If True, then denominators for autocovariance are n-k, otherwise n.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1). The returned value
        includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
    qstat : bool, default False
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, default True
        If True, computes the ACF via FFT.
    alpha : scalar, default None
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett"s formula.
    bartlett_confint : bool, default True
        Confidence intervals for ACF values are generally placed at 2
        standard errors around r_k. The formula used for standard error
        depends upon the situation. If the autocorrelations are being used
        to test for randomness of residuals as part of the ARIMA routine,
        the standard errors are determined assuming the residuals are white
        noise. The approximate formula for any lag is that standard error
        of each r_k = 1/sqrt(N). See section 9.4 of [2] for more details on
        the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2
        in [3].
        For the ACF of raw data, the standard error at a lag k is
        found as if the right model was an MA(k-1). This allows the possible
        interpretation that if all autocorrelations past a certain lag are
        within the limits, the model might be an MA of order defined by the
        last significant autocorrelation. In this case, a moving average
        model is assumed for the data and the standard errors for the
        confidence intervals should be generated using Bartlett's formula.
        For more details on Bartlett formula result, see section 7.2 in [2].
    missing : str, default "none"
        A string in ["none", "raise", "conservative", "drop"] specifying how
        the NaNs are to be treated. "none" performs no checks. "raise" raises
        an exception if NaN values are found. "drop" removes the missing
        observations and then estimates the autocovariances treating the
        non-missing as contiguous. "conservative" computes the autocovariance
        using nan-ops so that nans are removed when computing the mean
        and cross-products that are used to estimate the autocovariance.
        When using "conservative", n is set to the number of non-missing
        observations.

    Returns
    -------
    acf : ndarray
        The autocorrelation function for lags 0, 1, ..., nlags. Shape
        (nlags+1,).
    confint : ndarray, optional
        Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). Returned if alpha is not None.
    qstat : ndarray, optional
        The Ljung-Box Q-Statistic for lags 1, 2, ..., nlags (excludes lag
        zero). Returned if q_stat is True.
    pvalues : ndarray, optional
        The p-values associated with the Q-statistics for lags 1, 2, ...,
        nlags (excludes lag zero). Returned if q_stat is True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    For very long time series it is recommended to use fft convolution instead.
    When fft is False uses a simple, direct estimator of the autocovariances
    that only computes the first nlag + 1 values. This can be much faster when
    the time series is long and only a small number of autocovariances are
    needed.

    If adjusted is true, the denominator for the autocovariance is adjusted
    for the loss of data.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
       and amplitude modulation. Sankhya: The Indian Journal of
       Statistics, Series A, pp.383-392.
    .. [2] Brockwell and Davis, 1987. Time Series Theory and Methods
    .. [3] Brockwell and Davis, 2010. Introduction to Time Series and
       Forecasting, 2nd edition.
    """
    assert x.ndim == 1
    assert_concrete(nlags, "nlags")

    # TODO: should this shrink for missing="drop" and NaNs in x?
    nobs = x.size

    if nlags is None:
        # print(jnp.floor(10 * jnp.log10(nobs)).astype(int))
        _nlag_candidate = int(10 * np.log10(nobs))
        _nlags = min(_nlag_candidate, nobs - 1)

    else:
        _nlags = nlags

    avf = acovf(x, adjusted=adjusted, demean=True, fft=fft, missing=missing)
    acf = avf[: _nlags + 1] / avf[0]
    if not (qstat or alpha):
        return acf
    _alpha = alpha if alpha is not None else 0.05
    if bartlett_confint:
        varacf = np.ones_like(acf) / nobs
        varacf[0] = 0
        varacf[1] = 1.0 / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
    else:
        varacf = 1.0 / len(x)
    interval = stats.norm.ppf(1 - _alpha / 2.0) * np.sqrt(varacf)
    confint = np.array(lzip(acf - interval, acf + interval))
    if not qstat:
        return acf, confint
    qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0
    if alpha is not None:
        return acf, confint, qstat, pvalue
    else:
        return acf, qstat, pvalue


def acovf(
    x: Num[Array, "a"],
    adjusted: bool = False,
    demean: bool = True,
    fft: bool = True,
    missing: TMissing = "none",
    nlag: int | None = None,
):
    """
    Estimate autocovariances.

    Parameters
    ----------
    x : array_like
        Time series data. Must be 1d.
    adjusted : bool, default False
        If True, then denominators is n-k, otherwise n.
    demean : bool, default True
        If True, then subtract the mean x from each element of x.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    missing : str, default "none"
        A string in ["none", "raise", "conservative", "drop"] specifying how
        the NaNs are to be treated. "none" performs no checks. "raise" raises
        an exception if NaN values are found. "drop" removes the missing
        observations and then estimates the autocovariances treating the
        non-missing as contiguous. "conservative" computes the autocovariance
        using nan-ops so that nans are removed when computing the mean
        and cross-products that are used to estimate the autocovariance.
        When using "conservative", n is set to the number of non-missing
        observations.
    nlag : {int, None}, default None
        Limit the number of autocovariances returned.  Size of returned
        array is nlag + 1.  Setting nlag when fft is False uses a simple,
        direct estimator of the autocovariances that only computes the first
        nlag + 1 values. This can be much faster when the time series is long
        and only a small number of autocovariances are needed.

    Returns
    -------
    ndarray
        The estimated autocovariances.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
           and amplitude modulation. Sankhya: The Indian Journal of
           Statistics, Series A, pp.383-392.
    """

    notmask_bool = ~jnp.isnan(x)  # bool
    if missing == "raise" and (~notmask_bool).any():
        raise MissingDataError("NaNs were encountered in the data")

    # missing ∈ ["none", "conservative", "drop"]
    if missing == "none":
        pass
    elif missing == "conservative":
        x = x.at[~notmask_bool].set(0)
    elif missing == "drop":
        x = x[notmask_bool]  # copies non-missing
    else:
        raise ValueError("Bad missing parameter!")

    notmask_int = notmask_bool.astype(int)  # int

    if demean:
        # whether "drop" or "conservative":
        xo = x - x.sum() / notmask_int.sum()
        if missing == "conservative":
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    lag_len = nlag
    if nlag is None:
        lag_len = n - 1
    elif nlag > n - 1:
        raise ValueError("nlag must be smaller than nobs - 1")

    if not fft and nlag is not None:
        acov = jnp.empty(lag_len + 1)
        acov[0] = xo.dot(xo)
        for i in range(lag_len):
            acov[i + 1] = xo[i + 1 :].dot(xo[: -(i + 1)])
        if adjusted:
            divisor = jnp.empty(lag_len + 1, dtype=jnp.int64)
            divisor[0] = notmask_int.sum()
            for i in range(lag_len):
                divisor[i + 1] = notmask_int[i + 1 :].dot(notmask_int[: -(i + 1)])
            divisor[divisor == 0] = 1
            acov /= divisor
        else:  # biased, missing data but npt "drop"
            acov /= notmask_int.sum()
        return acov

    if adjusted and missing == "conservative":
        d = jnp.correlate(notmask_int, notmask_int, "full")
        d[d == 0] = 1
    elif adjusted:
        xi = jnp.arange(1, n + 1)
        d = jnp.hstack((xi, xi[:-1][::-1]))
    else:
        # biased and NaNs given and ("drop" or "conservative")
        d = notmask_int.sum() * jnp.ones(2 * n - 1)

    if fft:
        nobs = len(xo)
        n = _next_regular(2 * nobs + 1)
        Frf = jnp.fft.fft(xo, n=n)
        acov = jnp.fft.ifft(Frf * jnp.conjugate(Frf))[:nobs] / d[nobs - 1 :]
        acov = acov.real
    else:
        acov = jnp.correlate(xo, xo, "full")[n - 1 :] / d[n - 1 :]

    if nlag is not None:
        # Copy to allow gc of full array rather than view
        return acov[: lag_len + 1].copy()
    return acov


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            p2 = 2 ** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
