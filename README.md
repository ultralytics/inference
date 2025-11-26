<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🛠 Ultralytics Python Project Template

Welcome to the Ultralytics Python Project Template! This repository provides a standardized foundation for initiating Python projects at [Ultralytics](https://www.ultralytics.com/). It incorporates best practices in project structure, configuration, and essential tooling to streamline development. By using this template, Ultralytics developers can ensure consistency, maintain high quality standards, and accelerate the setup process for new Python-based software. Explore our [Ultralytics Solutions](https://www.ultralytics.com/solutions) to see how we apply these standards in real-world applications.

[![Template CI](https://github.com/ultralytics/template/actions/workflows/ci.yml/badge.svg)](https://github.com/ultralytics/template/actions/workflows/ci.yml)
[![Ultralytics Actions](https://github.com/ultralytics/template/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/template/actions/workflows/format.yml)
[![codecov](https://codecov.io/gh/ultralytics/template/graph/badge.svg?token=K9IunpFzjS)](https://codecov.io/gh/ultralytics/template)

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 🗂️ Repository Structure

This template is meticulously organized for intuitive navigation and a clear understanding of project components. Familiarize yourself with the [Python project structure best practices](https://realpython.com/python-application-layouts/) to make the most of this layout.

- `src/` or `your_package_name/`: Contains the core source code of your Python package, organized into modules. Using a `src` layout is a common practice detailed in [Python packaging guides](https://packaging.python.org/en/latest/tutorials/packaging-projects/#configuring-metadata).
- `tests/`: Dedicated directory for unit tests and integration tests, crucial for implementing [continuous testing](https://docs.ultralytics.com/help/CI/) practices. Consider using frameworks like [pytest](https://docs.pytest.org/en/stable/) for writing tests.
- `docs/`: (Optional) Houses project documentation. Tools like [MkDocs](https://www.mkdocs.org/) can be used to generate comprehensive documentation from this directory.
- `pyproject.toml`: The standard configuration file for Python projects, detailing dependencies, build system requirements, formatting rules, and packaging information as specified by [PEP 518](https://peps.python.org/pep-0518/) and subsequent PEPs.
- `.gitignore`: Configured to exclude unnecessary files (like `*.pyc` or virtual environment directories) from [Git](https://git-scm.com/) tracking.
- `LICENSE`: Specifies the open-source license (defaulting to AGPL-3.0) under which the project is released.
- `.github/workflows/`: Contains [GitHub Actions](https://docs.github.com/en/actions) workflows for automating Continuous Integration and Continuous Deployment (CI/CD) processes. Learn more about [CI/CD concepts](https://www.redhat.com/en/topics/devops/what-is-ci-cd).
- `.pre-commit-config.yaml`: (Optional) Configuration for [pre-commit hooks](https://pre-commit.com/) to automatically check and enforce code quality standards before commits.
- `Dockerfile`: (Optional) Defines instructions for building a [Docker](https://www.docker.com/) container image, enabling [containerization](https://www.ultralytics.com/glossary/containerization) of the project environment for consistent deployment.
- `environment.yml`: (Optional, for Conda users) Manages dependencies for [Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```plaintext
your-project/
│
├── your_package_name/          # Or src/ for src-layout
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   └── ...
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_module1.py
│   └── ...
│
├── docs/                       # Documentation files (optional)
│   └── ...
│
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml
|       └── format.yml
│
├── .gitignore                  # Git ignore rules
├── .pre-commit-config.yaml     # Pre-commit hook config (optional)
├── Dockerfile                  # Docker configuration (optional)
├── environment.yml             # Conda environment config (optional)
├── LICENSE                     # Project license file
├── pyproject.toml              # Project configuration and dependencies
└── README.md                   # This file
```

### 📦 Source Code Directory (`src/` or `your_package_name/`)

The `src/` or `your_package_name/` directory is the heart of your project, containing the Python code that constitutes your package. Adopting a structured layout promotes clean imports and simplifies testing and packaging.

### 🧪 Testing Directory (`tests/`)

The `tests/` directory is crucial for ensuring code reliability and robustness. It should contain comprehensive unit and integration tests covering various aspects of your package. Effective testing is a cornerstone of quality software development.

### 📚 Documentation Directory (`docs/`)

For projects requiring detailed documentation beyond the README, the `docs/` directory is the designated space. Utilizing tools like [Sphinx](https://www.sphinx-doc.org/en/master/) allows for the generation of professional, high-quality documentation from reStructuredText or Markdown files. Check out the [Ultralytics Docs](https://docs.ultralytics.com/) for an example.

## ✨ Starting a New Project

Kickstart your new Python project using this template with these steps:

1.  **Create Your Repository**: Use this template on GitHub by clicking the "Use this template" button to generate a new repository for your project. Learn more about [creating a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).
2.  **Customize**: Tailor the template files (`pyproject.toml`, `README.md`, `.github/workflows/*.yml`, etc.) to match your specific project's name, goals, and requirements.
3.  **Develop**: Begin adding your source code into the `your_package_name/` (or `src/`) directory and write corresponding tests in the `tests/` directory.
4.  **Document**: Update this `README.md` thoroughly and, if needed, populate the `docs/` directory with more extensive documentation.
5.  **Integrate**: Leverage the pre-configured GitHub Actions for automated testing, linting, and other [CI/CD](https://www.ultralytics.com/glossary/continuous-integration-ci) processes to maintain code quality.

## 🔧 Utilizing the Template

For Ultralytics team members and external contributors:

- Clone the newly created repository based on this template to start working on the project locally.
- Ensure the `README.md` is updated to accurately reflect the project's purpose, usage, and specifics.
- Remove or modify optional components (like `Dockerfile`, `environment.yml`) based on the project's deployment and dependency management strategy.

With this template, Ultralytics aims to foster a culture of excellence and uniformity in Python software development, ensuring every project starts on a solid foundation aligned with industry standards and organizational best practices. For insights into managing ML projects, explore our [MLOps guide](https://www.ultralytics.com/glossary/machine-learning-operations-mlops).

## 💡 Contribute

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's reporting bugs, suggesting features, or submitting code changes, your involvement is crucial.

- **Reporting Issues**: Encounter a bug? Please report it on [GitHub Issues](https://github.com/ultralytics/template/issues).
- **Feature Requests**: Have an idea for improvement? Share it via [GitHub Issues](https://github.com/ultralytics/template/issues).
- **Pull Requests**: Want to contribute code? Please read our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) first, then submit a Pull Request.
- **Feedback**: Share your thoughts and experiences by participating in our official [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey).

A heartfelt thank you 🙏 goes out to all our contributors! Your efforts help make Ultralytics tools better for everyone.

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## 📄 License

Ultralytics offers two licensing options to accommodate diverse needs:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts passionate about open collaboration and knowledge sharing. This [OSI-approved](https://opensource.org/license/agpl-v3) open-source license promotes transparency and community involvement. See the [LICENSE](LICENSE) file for details.
- **Enterprise License**: Designed for commercial applications, this license permits the seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the copyleft requirements of AGPL-3.0. For commercial use cases, please inquire about an [Ultralytics Enterprise License](https://www.ultralytics.com/license).

## 📮 Contact

For bug reports or feature suggestions related to this template or other Ultralytics projects, please use [GitHub Issues](https://github.com/ultralytics/template/issues). For general questions, discussions, and community support, join our [Discord](https://discord.com/invite/ultralytics) server!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
