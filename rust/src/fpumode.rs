//! Rust port of `scipy/_lib/_fpumode.c`.
//!
//! Reports the FPU control word on x86 where it is meaningful; every other
//! architecture (notably aarch64) returns `None`, exactly like the C original.

use pyo3::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn fpu_control_word() -> Option<u64> {
    let mut cw: u16 = 0;
    unsafe {
        std::arch::asm!("fstcw word ptr [{}]", in(reg) &mut cw, options(nostack));
    }
    Some(cw as u64)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn fpu_control_word() -> Option<u64> {
    None
}

/// Get the current FPU control word, in a platform-dependent format.
/// Returns None if not implemented on current platform.
#[pyfunction]
fn get_fpu_mode(_py: Python<'_>) -> Option<u64> {
    fpu_control_word()
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_fpu_mode, m)?)?;
    Ok(())
}
