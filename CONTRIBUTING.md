# Contributing to Curator

Welcome to the Curator project! We value your contributions and are excited to have you collaborate with us. This document outlines the guidelines for contributing to ensure a smooth and productive process.

## Table of Contents

1. [Setting Up the Project](#setting-up-the-project)
2. [How to Contribute](#how-to-contribute)
3. [Submitting Changes](#submitting-changes)
4. [Issue Reporting](#issue-reporting)
5. [Pull Request Guidelines](#pull-request-guidelines)

---

## Setting Up the Project

1. Ensure you have the following tools installed:
   - Git
   - Any other dependencies listed in the repository's documentation

2. Install dependencies:
   ```bash
   make install
   ```

3. Run tests to verify your setup:
   ```bash
   make test
   ```

4. Activate the virtual environment:
```bash
$(poetry env activate)
```

## How to Contribute

1. **Fork the Repository**: Start by forking the repository to your GitHub account.
2. **Clone Your Fork**: Clone your fork to your local machine using:
   ```bash
   git clone https://github.com/<username>/curator.git
   ```
3. **Create an Issue**: Before starting any work, create an issue in the repository to discuss your proposed changes or enhancements. This helps maintainers guide your contributions effectively.
4. **Create a Branch**: Create a feature branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make Changes**: Implement your changes following the guidelines.
6. **Add Tests**: Ensure your changes work as expected, add corresponding tests and pass all tests.
7. **Commit Your Changes**: Write clear and concise commit messages:
   ```bash
   git commit -m "feat: added new feature"
   ```
8. **Push Your Changes**: Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Submit a Pull Request**: Open a pull request (PR) to the main repository.


## Submitting Changes

1. Ensure your changes align with the project's purpose and coding standards.
2. Include relevant tests for your changes.
3. Update documentation if necessary.
4. Ensure your code passes linting and formatting checks.
   ```bash
   make lint
   make check
   ```
5. Submit a PR with a descriptive title and detailed description.

## Issue Reporting

When reporting an issue, include:

1. A clear and descriptive title.
2. Steps to reproduce the issue.
3. Expected behavior vs. actual behavior.
4. Screenshots or logs, if applicable.
5. Environment details (e.g., OS, Python version).

## Pull Request Guidelines

1. Reference related issues in your PR description (e.g., "Closes #42").
2. Ensure your PR has descriptive and concise commits.
3. Keep your PR focused on a single change to make reviews easier.
4. Ensure your changes are rebased on the latest `main` branch:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
5. Address any feedback from maintainers promptly.

---

Thank you for contributing to Curator! Your support and collaboration make this project better for everyone. 😊
