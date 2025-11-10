# Python Development Guidelines for AI Agents

This document outlines the standards and best practices for developing Python-based AI agents in this project. Following these guidelines will ensure consistency, maintainability, and reliability across the codebase.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Code Style](#code-style)
3. [Type Hints](#type-hints)
4. [Error Handling](#error-handling)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Performance](#performance)
8. [Security](#security)
9. [Version Control](#version-control)
10. [Dependencies](#dependencies)
11. [AI-Specific Guidelines](#ai-specific-guidelines)

## Project Structure

```
project/
├── src/
│   └── module_name/
│       ├── __init__.py
│       ├── models/           # Model definitions
│       ├── data/             # Data processing
│       ├── config/           # Configuration
│       ├── utils/            # Utility functions
│       └── tests/            # Unit tests
├── tests/                    # Integration tests
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── .github/                  # GitHub workflows
├── .gitignore
├── pyproject.toml            # Project metadata and dependencies
├── README.md
└── AGENTS_DEV.md             # This document
```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation
- Maximum line length of 88 characters (Black formatter default)
- Use double quotes for strings
- Sort imports using `isort` with the following order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports

## Type Hints

- Use Python type hints for all function signatures and class attributes
- Use `typing` module for complex types
- Use `Optional[T]` instead of `Union[T, None]`
- Use `@dataclass` for data containers
- Use `TypedDict` for dictionary type hints
- Use `@final` for classes not meant to be subclassed

## Error Handling

- Create custom exception classes for different error types
- Use exception chaining with `raise ... from ...`
- Include meaningful error messages
- Log errors with appropriate severity levels
- Use context managers for resource management

## Testing

- Write unit tests for all non-trivial functions
- Use `pytest` as the testing framework
- Follow the Arrange-Act-Assert pattern
- Use fixtures for common test setup
- Aim for at least 80% code coverage
- Mark slow tests with `@pytest.mark.slow`
- Use property-based testing for complex logic

## Documentation

- Use Google-style docstrings
- Document all public functions, classes, and modules
- Include examples in docstrings
- Keep README.md up to date
- Document design decisions in ADRs (Architecture Decision Records)
- Use type hints as a form of documentation

## Performance

- Use generators for large datasets
- Cache expensive function calls with `@functools.cache` or `@functools.lru_cache`
- Use `asyncio` for I/O-bound operations
- Use `multiprocessing` for CPU-bound operations
- Profile before optimizing
- Use `__slots__` for memory-efficient classes

## Security

- Never hardcode secrets in code
- Use environment variables for configuration
- Validate all inputs
- Use parameterized queries for database access
- Keep dependencies updated
- Use `bandit` for security scanning

## Version Control

- Write clear, concise commit messages
- Use feature branches
- Open pull requests for code review
- Squash and merge PRs
- Tag releases with semantic versioning
- Keep the main branch always deployable

## Dependencies

- Use `poetry` for dependency management
- Pin all dependencies with exact versions
- Separate development and production dependencies
- Document all dependencies in `pyproject.toml`
- Use `pre-commit` hooks for code quality

## AI-Specific Guidelines

### Model Development

- Separate model architecture from training logic
- Use configuration files for hyperparameters
- Implement checkpointing for long-running training
- Log training metrics
- Use TensorBoard or similar for visualization

### Data Processing

- Use datasets library for data loading
- Implement data versioning
- Use data loaders with batching
- Implement data augmentation when applicable
- Cache processed data when possible

### Inference

- Implement batching for inference
- Use ONNX or TorchScript for production deployment
- Implement proper error handling for inference
- Add input validation
- Log inference metrics

### Monitoring

- Log model predictions (be mindful of privacy)
- Monitor model drift
- Track model performance metrics
- Set up alerts for anomalies

## Code Review Checklist

- [ ] Code follows the style guide
- [ ] Type hints are used consistently
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] Error handling is appropriate
- [ ] Performance considerations are addressed
- [ ] Security best practices are followed
- [ ] Dependencies are properly managed

## Development Workflow

1. Create a feature branch from `main`
2. Implement your changes
3. Write tests
4. Run linters and tests
5. Update documentation
6. Open a pull request
7. Address review comments
8. Get approval
9. Merge to `main`
10. Deploy

## Tools

- **Code Formatter**: Black
- **Import Sorter**: isort
- **Linter**: flake8, pylint
- **Type Checker**: mypy
- **Testing**: pytest
- **Documentation**: Sphinx, MkDocs
- **CI/CD**: GitHub Actions
- **Dependency Management**: Poetry
- **Notebooks**: Jupyter, nbconvert

## Best Practices

- Keep functions small and focused
- Follow the Single Responsibility Principle
- Write pure functions when possible
- Avoid global state
- Use dependency injection
- Write self-documenting code
- Keep the codebase clean and organized
- Refactor regularly
- Learn and apply design patterns appropriately
- Write code for humans first, computers second

## Performance Optimization

1. **Measure first**: Use profiling tools to identify bottlenecks
2. **Optimize algorithms**: Choose the right data structures and algorithms
3. **Use built-ins**: They're usually faster than custom implementations
4. **Leverage libraries**: Use optimized libraries like NumPy and pandas
5. **Parallelize**: Use concurrent.futures or asyncio for I/O-bound tasks
6. **Cache results**: For expensive computations with the same inputs
7. **Minimize memory usage**: Use generators and iterators for large datasets

## Security Considerations

- **Input Validation**: Validate all inputs, especially from untrusted sources
- **Authentication**: Implement proper authentication and authorization
- **Secrets Management**: Use environment variables or secret management services
- **Dependencies**: Keep dependencies updated and audit them regularly
- **Logging**: Log security-relevant events
- **Error Handling**: Don't expose sensitive information in error messages
- **Rate Limiting**: Implement rate limiting for APIs
- **HTTPS**: Always use HTTPS for network communication

## Documentation Standards

### Module-Level Docstrings

```python
"""
Short description of the module.

Longer description that may include:
- Purpose of the module
- Key classes and functions
- Usage examples

Example:
    >>> from module import function
    >>> result = function()
"""
```

### Function Docstrings

```python
def function(param1: str, param2: int = 42) -> bool:
    """Short description of the function.

    Longer description with more details about the function's behavior,
    edge cases, and any side effects.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When something goes wrong.

    Example:
        >>> function("test", 123)
        True
    """
```

### Class Docstrings

```python
class ExampleClass:
    """Short description of the class.

    Longer description of the class, including its purpose and behavior.
    Document any public attributes and methods.

    Attributes:
        attr1: Description of attribute 1.
        attr2: Description of attribute 2.
    """

    def __init__(self, param: str):
        """Initialize the class with the given parameters.

        Args:
            param: Description of the parameter.
        """
        self.attr1 = param
        self.attr2 = 0
```

## Code Examples

### Good Example

```python
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class User:
    """Represents a user in the system."""
    
    user_id: str
    username: str
    email: str
    is_active: bool = True


def filter_active_users(users: List[User]) -> List[User]:
    """Filter a list of users to return only active ones.
    
    Args:
        users: List of User objects to filter.
        
    Returns:
        List of active User objects.
        
    Example:
        >>> users = [User("1", "test", "test@example.com")]
        >>> filter_active_users(users)
        [User(user_id='1', username='test', email='test@example.com', is_active=True)]
    """
    return [user for user in users if user.is_active]
```

### Bad Example

```python
# No type hints
# No docstrings
# Poor variable names

def f(u, x):
    r = []
    for i in u:
        if i[3]:
            r.append(i)
    return r
```

## Conclusion

Following these guidelines will help maintain a high-quality, maintainable, and secure codebase. Always prioritize readability and maintainability over clever or overly optimized code. When in doubt, refer to the Zen of Python:

```
>>> import this
```
