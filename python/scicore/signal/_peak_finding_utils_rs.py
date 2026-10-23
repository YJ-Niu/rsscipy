"""Python wrapper for peak-finding functions in the unified Rust extension.

This module replaces the original `_peak_finding_utils.pyx`. The Rust module
returns plain Python lists; this wrapper converts them to rsnumpy arrays.
"""

import rsnumpy as np

from scicore._scicore_rs import signal as _rust_signal
_rust_local_maxima_1d = _rust_signal._local_maxima_1d
_rust_select_by_peak_distance = _rust_signal._select_by_peak_distance
_rust_peak_prominences = _rust_signal._peak_prominences
_rust_peak_widths = _rust_signal._peak_widths
PeakPropertyWarning = _rust_signal.PeakPropertyWarning

__all__ = [
    '_local_maxima_1d',
    '_select_by_peak_distance',
    '_peak_prominences',
    '_peak_widths',
    'PeakPropertyWarning',
]


def _local_maxima_1d(x):
    """Find local maxima in a 1-D array.

    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    left_edges : ndarray
        Indices of edges to the left of local maxima in `x`.
    right_edges : ndarray
        Indices of edges to the right of local maxima in `x`.
    """
    midpoints, left_edges, right_edges = _rust_local_maxima_1d(x)
    return (
        np.array(midpoints, dtype=np.intp),
        np.array(left_edges, dtype=np.intp),
        np.array(right_edges, dtype=np.intp),
    )


def _select_by_peak_distance(peaks, priority, distance):
    """Evaluate which peaks fulfill the distance condition.

    Returns
    -------
    keep : ndarray[bool]
        A boolean mask evaluating to true where `peaks` fulfill the distance
        condition.
    """
    keep = _rust_select_by_peak_distance(peaks, priority, distance)
    return np.array(keep, dtype=np.bool_)


def _peak_prominences(x, peaks, wlen):
    """Calculate the prominence of each peak in a signal.

    Returns
    -------
    prominences : ndarray
        The calculated prominences for each peak in `peaks`.
    left_bases, right_bases : ndarray
        The peaks' bases as indices in `x` to the left and right of each peak.
    """
    prominences, left_bases, right_bases = _rust_peak_prominences(x, peaks, wlen)
    return (
        np.array(prominences, dtype=np.float64),
        np.array(left_bases, dtype=np.intp),
        np.array(right_bases, dtype=np.intp),
    )


def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    """Calculate the width of each peak at a relative height.

    Returns
    -------
    widths : ndarray
        The widths for each peak in samples.
    width_heights : ndarray
        The height of the contour lines at which the `widths` where evaluated.
    left_ips, right_ips : ndarray
        Interpolated positions of left and right intersection points of a
        horizontal line at the respective evaluation height.
    """
    widths, width_heights, left_ips, right_ips = _rust_peak_widths(
        x, peaks, rel_height, prominences, left_bases, right_bases
    )
    return (
        np.array(widths, dtype=np.float64),
        np.array(width_heights, dtype=np.float64),
        np.array(left_ips, dtype=np.float64),
        np.array(right_ips, dtype=np.float64),
    )
