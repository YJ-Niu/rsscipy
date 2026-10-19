py3.extension_module('_sigtools',
  [
    '_firfilter.cc',
    '_sigtoolsmodule.cc',
    '_medianfilter.cc',
    '_lfilter.cc',
    '_correlate_nd.cc'
  ],
  dependencies: np_dep,
  link_args: version_link_args,
  include_directories: ['../_build_utils/src'],
  install: true,
  subdir: 'rsscipy/signal'
)

pyx_files = [
  ['_upfirdn_apply', '_upfirdn_apply.pyx']
]

foreach pyx_file: pyx_files
  py3.extension_module(pyx_file[0],
    cython_gen.process(pyx_file[1]),
    c_args: cython_c_args,
    dependencies: np_dep,
    link_args: version_link_args,
    install: true,
    subdir: 'rsscipy/signal'
  )
endforeach

# Rust extension modules built by `tools/build_rust.py` at configure time.
install_data(
  [
    '_peak_finding_utils_rs.so',
    '_sosfilt_rs.so',
    '_max_len_seq_inner_rs.so',
  ],
  install_dir: scipy_dir / 'signal',
)

py3.extension_module('_spline',
  [
    '_splinemodule.cc',
  ],
  dependencies: np_dep,
  link_args: version_link_args,
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
    '_signaltools.py',
    '_spectral_py.py',
    '_spline.pyi',
    '_upfirdn.py',
    '_waveforms.py',
    '_wavelets.py',
    '_whittaker.py',
  ],
  subdir: 'rsscipy/signal'
)

subdir('windows')
subdir('tests')
