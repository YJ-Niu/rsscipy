#![recursion_limit = "256"]

use pyo3::prelude::*;

mod array_utils;
mod ccallback;
mod column_grouping;
mod convolve2d;
mod correlate_nd;
mod fblas;
mod fftpack_convolve;
mod flapack;
mod fpumode;
mod givens_elimination;
mod graph_tools;
mod lfilter;
mod max_len_seq_inner;
mod median_filter_2d;
mod ndimage_label;
mod peak_finding_utils;
mod remez;
mod reordering;
mod rgi;
mod sosfilt;
mod spline;
mod upfirdn_apply;
mod vq;

#[pymodule]
fn _rsscipy_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();

    let lib = PyModule::new(py, "_lib")?;
    ccallback::register(&lib)?;
    fpumode::register(&lib)?;
    module.add("_lib", lib)?;

    let linalg = PyModule::new(py, "linalg")?;
    array_utils::register(&linalg)?;
    fblas::register(&linalg)?;
    flapack::register(&linalg)?;
    givens_elimination::register(&linalg)?;
    module.add("linalg", linalg)?;

    let signal = PyModule::new(py, "signal")?;
    convolve2d::register(&signal)?;
    correlate_nd::register(&signal)?;
    lfilter::register(&signal)?;
    max_len_seq_inner::register(&signal)?;
    median_filter_2d::register(&signal)?;
    peak_finding_utils::register(&signal)?;
    remez::register(&signal)?;
    sosfilt::register(&signal)?;
    spline::register(&signal)?;
    upfirdn_apply::register(&signal)?;
    module.add("signal", signal)?;

    let fftpack = PyModule::new(py, "fftpack")?;
    fftpack_convolve::register(&fftpack)?;
    module.add("fftpack", fftpack)?;

    let cluster = PyModule::new(py, "cluster")?;
    vq::register(&cluster)?;
    module.add("cluster", cluster)?;

    let interpolate = PyModule::new(py, "interpolate")?;
    rgi::register(&interpolate)?;
    module.add("interpolate", interpolate)?;

    let csgraph = PyModule::new(py, "csgraph")?;
    reordering::register(&csgraph)?;
    graph_tools::register(&csgraph)?;
    module.add("csgraph", csgraph)?;

    let optimize = PyModule::new(py, "optimize")?;
    column_grouping::register(&optimize)?;
    module.add("optimize", optimize)?;

    let ndimage = PyModule::new(py, "ndimage")?;
    ndimage_label::register(&ndimage)?;
    module.add("ndimage", ndimage)?;

    Ok(())
}
