py3.extension_module('convolve',
  cython_gen.process('convolve.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'rsscipy/fftpack'
)


python_sources = [
  '__init__.py',
  '_basic.py',
  '_helper.py',
  '_pseudo_diffs.py',
  '_realtransforms.py',
]


py3.install_sources(
  python_sources,
  subdir: 'rsscipy/fftpack'
)

subdir('tests')
