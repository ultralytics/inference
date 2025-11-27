<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Documentation Directory (`docs/`)

This directory contains the documentation for the Ultralytics Inference Rust library and CLI.

## 📖 Overview

The Ultralytics Inference project provides high-performance YOLO model inference through a Rust library and CLI application. This documentation directory will contain:

- **API Documentation:** Comprehensive API reference for the Rust library (generated via `cargo doc`)
- **Usage Guides:** Step-by-step tutorials for using the library and CLI
- **Architecture:** Technical details about the inference engine implementation
- **Examples:** Code samples demonstrating various use cases

## 🚀 Building Documentation

To generate and view the API documentation locally:

1. **Generate Rust Documentation:**

    ```bash
    cargo doc --no-deps --open
    ```

    This generates HTML documentation from the code comments and opens it in your browser.

2. **Documentation with Private Items:**

    ```bash
    cargo doc --no-deps --document-private-items --open
    ```

    This includes documentation for private modules and functions.

3. **Test Documentation Examples:**

    ```bash
    cargo test --doc
    ```

    This runs all code examples in the documentation to ensure they compile correctly.

## 📚 Documentation Structure

Future documentation will include:

- **Getting Started:** Installation, setup, and basic usage
- **Library API:** Detailed API reference for all public types and functions
- **CLI Reference:** Command-line interface documentation
- **Python Bindings:** Guide for using the library from Python (when available)
- **Performance:** Benchmarks and optimization tips
- **Examples:** Real-world usage examples and best practices

## 🙌 Contributing

Contributions to improve the documentation are welcome! Whether it's fixing typos, clarifying explanations, adding examples, or improving API docs, your help is valuable. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details.
