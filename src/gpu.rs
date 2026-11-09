//! Which GPU whisper.cpp would transcribe on, and whether to use it when `transcribe.gpu` is
//! not set.
//!
//! An integrated GPU shares the CPU's memory and power budget, and on a desktop CPU it is often
//! slower than the CPU itself. So with the setting left unset, kute uses the GPU only when it
//! is a discrete one; `kute doctor --bench` measures and can set it either way.

use std::ffi::CStr;
use whisper_rs::whisper_rs_sys as sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Discrete,
    Integrated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    /// What the backend calls it, such as "AMD Radeon RX 7800 XT (RADV NAVI32)".
    pub name: String,
    pub kind: Kind,
}

/// The device whisper.cpp picks when told to use the GPU: the first GPU, discrete or
/// integrated, in ggml's list (which puts discrete ones first). None in a CPU-only build, or
/// when the backend finds no device.
pub fn device() -> Option<Device> {
    // SAFETY: ggml's device registry is a lazily built, thread-safe singleton; the device
    // handles and their strings live as long as the process.
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let dev = sys::ggml_backend_dev_get(i);
            let kind = match sys::ggml_backend_dev_type(dev) {
                sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => Kind::Discrete,
                sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU => Kind::Integrated,
                _ => continue,
            };
            let description = sys::ggml_backend_dev_description(dev);
            let name = if description.is_null() {
                sys::ggml_backend_dev_name(dev)
            } else {
                description
            };
            let name = if name.is_null() {
                "unnamed GPU".to_string()
            } else {
                CStr::from_ptr(name).to_string_lossy().into_owned()
            };
            return Some(Device { name, kind });
        }
    }
    None
}

/// Whether to transcribe on the GPU: the setting when there is one, else only on a discrete GPU.
pub fn use_gpu(setting: Option<bool>, device: Option<&Device>) -> bool {
    setting.unwrap_or_else(|| device.is_some_and(|d| d.kind == Kind::Discrete))
}

/// Why an unset `transcribe.gpu` leaves the GPU out, for the daemon's log and `kute doctor`.
pub fn integrated_note(setting: Option<bool>, device: Option<&Device>) -> Option<String> {
    match (setting, device) {
        (None, Some(d)) if d.kind == Kind::Integrated => Some(format!(
            "the only GPU is integrated ({}), so transcription uses the CPU; \
             `kute doctor --bench` checks which is faster",
            d.name
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(kind: Kind) -> Device {
        Device {
            name: "GPU".into(),
            kind,
        }
    }

    #[test]
    fn unset_uses_only_a_discrete_gpu() {
        assert!(use_gpu(None, Some(&dev(Kind::Discrete))));
        assert!(!use_gpu(None, Some(&dev(Kind::Integrated))));
        assert!(!use_gpu(None, None));
    }

    #[test]
    fn the_setting_wins() {
        assert!(use_gpu(Some(true), Some(&dev(Kind::Integrated))));
        assert!(!use_gpu(Some(false), Some(&dev(Kind::Discrete))));
    }

    #[test]
    fn explains_only_the_integrated_default() {
        let igpu = dev(Kind::Integrated);
        assert!(integrated_note(None, Some(&igpu)).is_some());
        assert_eq!(integrated_note(Some(true), Some(&igpu)), None);
        assert_eq!(integrated_note(None, Some(&dev(Kind::Discrete))), None);
        assert_eq!(integrated_note(None, None), None);
    }

    #[test]
    fn a_cpu_only_build_finds_no_gpu() {
        if cfg!(not(any(feature = "vulkan", feature = "cuda"))) {
            assert_eq!(device(), None);
        }
    }
}
