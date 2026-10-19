py3.extension_module('_sigtools',
  [
    '_firfilter.cc',
    '_sigtoolsmodule.cc',
    '_correlate_nd.cc'
  ],
  dependencies: np_dep,
  link_args: version_link_args,
  include_directories: ['../_build_utils/src'],
  install: true,
  subdir: 'rsscipy/signal'
)

py3.install_sources([
    '__init__.py',
    '_support_alternative_backends.py',
    '_signal_api.py',
    '_delegators.py',
    '_arraytools.py',
    '_spline_filters.py',
    '_czt.py',
    '_filter_design.py',
    '_fir_filter_design.py',
    '_lti_conversion.py',
    '_ltisys.py',
    '_max_len_seq.py',
    '_max_len_seq_inner.py',
    '_peak_finding.py',
    '_polyutils.py',
    '_savitzky_golay.py',
    '_short_time_fft.py',
    '_sosfilt.py',
    '_spline.py',
    '_medfilt2d.py',
    '_lfilter.py',
    '_signaltools.py',
    '_spectral_py.py',
    '_spline.pyi',
    '_upfirdn_apply.py',
    '_upfirdn.py',
    '_waveforms.py',
    '_wavelets.py',
    '_whittaker.py',
  ],
  subdir: 'rsscipy/signal'
)

subdir('windows')
subdir('tests')
