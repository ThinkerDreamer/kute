//! Finding programs on a `PATH`, the way a shell does.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Where the program `name` is on `path`, if anywhere. Relative entries are skipped: they
/// mean nothing to a service.
pub fn find(name: impl AsRef<OsStr>, path: &OsStr) -> Option<PathBuf> {
    let name = name.as_ref();
    std::env::split_paths(path)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(name))
        .find(|p| is_executable(p))
}

/// A file someone may run: one without the execute bit doesn't hide a later one on `PATH`.
pub fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}
