#![recursion_limit = "256"]

use pyo3::prelude::*;

mod ccallback;
mod fpumode;
mod array_utils;
mod fblas;
mod flapack;
mod lfilter;
mod max_len_seq_inner;
mod median_filter_2d;
mod peak_finding_utils;
mod sosfilt;
mod spline;
mod upfirdn_apply;

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
    module.add("linalg", linalg)?;

    let signal = PyModule::new(py, "signal")?;
    lfilter::register(&signal)?;
    max_len_seq_inner::register(&signal)?;
    median_filter_2d::register(&signal)?;
    peak_finding_utils::register(&signal)?;
    sosfilt::register(&signal)?;
    spline::register(&signal)?;
    upfirdn_apply::register(&signal)?;
    module.add("signal", signal)?;

    Ok(())
}
