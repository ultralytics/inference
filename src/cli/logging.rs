// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use std::sync::atomic::{AtomicBool, Ordering};

/// Global verbosity flag.
static VERBOSE: AtomicBool = AtomicBool::new(true);

/// Set the global verbosity flag.
pub fn set_verbose(verbose: bool) {
    VERBOSE.store(verbose, Ordering::Relaxed);
}

/// Check if verbose output is enabled.
pub fn is_verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

/// Macro for standard info messages.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        println!("{}", format!($($arg)*));
    }
}

/// Macro for warning messages.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        use colored::Colorize;
        eprintln!("{} {}", "WARNING ⚠️".yellow().bold(), format!($($arg)*));
    }
}

/// Macro for error messages.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        use colored::Colorize;
        eprintln!("{} {}", "Error:".red().bold(), format!($($arg)*));
    }
}

/// Macro for success messages.
#[macro_export]
macro_rules! success {
    ($($arg:tt)*) => {
        use colored::Colorize;
        println!("{} {}", "✅".green(), format!($($arg)*));
    }
}

/// Macro for verbose messages.
#[macro_export]
macro_rules! verbose {
    ($($arg:tt)*) => {
        if $crate::cli::logging::is_verbose() {
            println!("{}", format!($($arg)*));
        }
    }
}

/// Macro for section headers.
#[macro_export]
macro_rules! section {
    ($($arg:tt)*) => {
        use colored::Colorize;
        if $crate::cli::logging::is_verbose() {
            println!();
            println!("{}", format!($($arg)*).cyan().bold());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbosity_toggle() {
        // Default is true
        set_verbose(true);
        assert!(is_verbose());

        set_verbose(false);
        assert!(!is_verbose());

        set_verbose(true);
        assert!(is_verbose());
    }
}
