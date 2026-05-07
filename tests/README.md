<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Tests Directory (`tests/`)

This directory contains Rust integration tests for the Ultralytics inference crate. Keeping tests close to user-facing
behaviors helps ensure changes remain reliable and well-documented.

## 🧪 Overview

- Uses standard [Cargo tests](https://doc.rust-lang.org/cargo/guide/tests.html) with Rust’s built-in test harness.
- Organized to mirror the structure of the source code directory for easy navigation and reference.
- Tests should be comprehensive, covering user-facing behavior, error handling, and edge cases.

## 🚀 Running Tests

Run the full suite from the project root:

```bash
cargo test
```

To generate code coverage locally (Linux recommended):

```bash
cargo llvm-cov --all-features --workspace --html
```

## ✨ Contributing

We love contributions! If you find an issue or have an idea for improving the tests, please open an issue or submit a pull
request. See our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for details.
