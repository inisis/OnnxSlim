# Contributing to OnnxSlim

Thank you for your interest in contributing to OnnxSlim! This document provides guidelines and instructions to help you get started.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an open and welcoming environment.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, ONNX version, etc.)

### Suggesting Features

We welcome feature suggestions! Please create an issue with:

- Clear description of the feature
- Rationale for the feature
- Potential implementation approach (optional)

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests if available
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Coding Standards

- Follow the existing code style
- Write clear, commented, and testable code
- Keep commits focused and atomic
- Write meaningful commit messages
- Use type hints where appropriate

## Development Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies

```bash
# Setup commands
git clone https://github.com/username/OnnxSlim.git
cd OnnxSlim
pip install -e . # Install package in development mode
```

## Testing

Please ensure your code passes all tests:

```bash
# Test commands
pytest tests/test_onnxslim.py
```

## Documentation

- Update documentation for any changed functionality
- Document new features thoroughly
- Use clear and concise language
- Add docstrings to new functions and classes

## Review Process

1. Maintainers will review your PR
2. Changes may be requested
3. Once approved, your PR will be merged

## License

By contributing, you agree that your contributions will be licensed under the project's license.

Thank you for contributing to OnnxSlim!
