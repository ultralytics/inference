<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Source Code Directory (`src/` or `your_package_name/`)

Welcome to the core of the project! This directory houses the essential Python source code that powers our application. Whether structured within a `src/` layout or a directly named package folder, this is where the main logic resides.

## 📂 Structure

Our project follows a clean and organized structure to promote maintainability and scalability:

- **Root Organization**: Python [modules and sub-packages](https://docs.python.org/3/tutorial/modules.html) are organized directly within this root source directory.
- **Dedicated Folders**: Each significant feature or component typically resides in its own dedicated folder (sub-package) or file (module).
- **Configuration**: The [`pyproject.toml`](https://packaging.python.org/en/latest/specifications/pyproject-toml/) file, located at the project's root (outside this source directory), defines packaging, dependencies, and project metadata according to [PEP 621](https://peps.python.org/pep-0621/). You can find details on Ultralytics configuration options in our [docs](https://docs.ultralytics.com/usage/cfg/).

## 📝 Guidelines

To ensure code quality, consistency, and collaboration, please adhere to the following guidelines:

- **Documentation**: Write clear, concise docstrings and comments. Consider using tools like [Sphinx](https://www.sphinx-doc.org/en/master/) or [MkDocs](https://www.mkdocs.org/) for generating comprehensive documentation, similar to the main [Ultralytics Docs](https://docs.ultralytics.com/).
- **Coding Standards**: Follow established Python best practices, primarily [PEP 8](https://peps.python.org/pep-0008/), and align with Ultralytics' internal coding conventions and [Code of Conduct](https://docs.ultralytics.com/help/code-of-conduct/).
- **Modularity**: Design components to be modular and reusable. Introduce new modules or packages logically as the project evolves.
- **Testing**: Implement thorough unit and integration tests using frameworks like [`pytest`](https://docs.pytest.org/en/stable/). Rigorous testing is crucial for verifying functionality, performance, and preventing regressions. Explore [continuous integration (CI)](https://www.ultralytics.com/glossary/continuous-integration-ci) practices.
- **Dependencies**: Manage dependencies explicitly via `pyproject.toml`. Avoid unnecessary dependencies.

We encourage contributions! If you have suggestions or improvements, please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) and feel free to open an issue or pull request on the [Ultralytics GitHub](https://github.com/ultralytics).
