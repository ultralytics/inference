<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Tests Directory (`tests/`)

This directory houses the tests for our project, providing two primary options for running tests: using **pytest** or **unittest**. Ensuring robust testing is crucial for maintaining code quality and reliability.

## 🧪 Overview

- Contains test scripts written in [Python](https://www.python.org/).
- Organized to mirror the structure of the source code directory for easy navigation and reference.
- Tests should be comprehensive, covering a wide range of scenarios to ensure functionality and prevent regressions. Adhering to [software testing best practices](https://martinfowler.com/testing/) is encouraged.

## 🚀 Running Tests

You can execute the test suite using either `pytest` or Python's built-in `unittest` module.

### Option 1: Using `pytest` (Recommended)

`pytest` is a popular testing framework offering powerful features and enhanced output readability.

1.  **Install pytest:**
    ```bash
    pip install pytest
    ```
    Find more details on the [pytest PyPI page](https://pypi.org/project/pytest/).

2.  **Run Tests:** Execute the following command from the project's root directory:
    ```bash
    python -m pytest tests -v
    ```
    Refer to the [pytest usage documentation](https://docs.pytest.org/en/stable/how-to/usage.html) for more command-line options.

### Option 2: Using `unittest` (No Additional Dependencies)

If you prefer to avoid installing external dependencies, you can use Python's standard `unittest` framework.

1.  **Run Tests:** Execute the following command from the project's root directory:
    ```bash
    python -m unittest discover tests -v
    ```
    Learn more about the [unittest command-line interface](https://docs.python.org/3/library/unittest.html#command-line-interface).

### Notes

- While `pytest` is recommended for its advanced features, [unittest](https://docs.python.org/3/library/unittest.html) is built into Python, offering a lightweight alternative without needing extra installations.
- Regularly running these tests helps maintain the [software quality](https://en.wikipedia.org/wiki/Software_quality) and reliability of the project.

## ✨ Contributing

We love contributions! If you find an issue or have an idea for improving the tests, please feel free to create an issue or submit a pull request. See our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details on how to get started.
