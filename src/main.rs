use std::process::ExitCode;

fn main() -> ExitCode {
    match kute::cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kute: {e:#}");
            ExitCode::FAILURE
        }
    }
}
