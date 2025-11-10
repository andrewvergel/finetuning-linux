# Python Code Review Agent

This document defines the automated code review process and guidelines for Python code in this repository. The goal is to maintain high code quality, security, and performance across all contributions.

## Table of Contents
1. [Review Checklist](#review-checklist)
2. [Automated Checks](#automated-checks)
3. [Security Review](#security-review)
4. [Performance Considerations](#performance-considerations)
5. [Code Style](#code-style)
6. [Documentation Standards](#documentation-standards)
7. [Testing Requirements](#testing-requirements)
8. [Common Issues](#common-issues)
9. [Pull Request Template](#pull-request-template)
10. [Review Process](#review-process)

## Review Checklist

### Code Quality
- [ ] Code follows PEP 8 style guide
- [ ] Type hints are used consistently
- [ ] Functions are small and focused (≤30 lines)
- [ ] No code duplication (DRY principle)
- [ ] Proper error handling and logging
- [ ] No commented-out code
- [ ] No debugging statements (e.g., print, pdb)
- [ ] No sensitive data in code
- [ ] Environment variables used for configuration

### Security
- [ ] Input validation for all external inputs
- [ ] No hardcoded secrets or credentials
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (output encoding)
- [ ] CSRF protection for web endpoints
- [ ] Authentication and authorization checks
- [ ] Secure password handling (hashing, no plaintext)
- [ ] Rate limiting for APIs

### Performance
- [ ] Efficient algorithms and data structures
- [ ] N+1 query problems avoided
- [ ] Proper use of caching
- [ ] Batch processing for large datasets
- [ ] Memory-efficient operations
- [ ] Asynchronous operations for I/O-bound tasks
- [ ] Proper resource cleanup (context managers)

### Documentation
- [ ] Module-level docstrings
- [ ] Function/method docstrings
- [ ] Type hints for all function signatures
- [ ] Complex logic is commented
- [ ] README and other docs are updated
- [ ] Public API is clearly documented

### Testing
- [ ] Unit tests for new functionality
- [ ] Test coverage ≥80%
- [ ] Edge cases are tested
- [ ] Mock external dependencies
- [ ] Tests are fast and independent
- [ ] Integration tests for critical paths

## Automated Checks

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
    - id: black
      language_version: python3.9

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    - id: isort
      name: isort (python)
      args: ["--profile", "black"]

-   repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
    - id: flake8
      additional_dependencies: [flake8-bugbear==23.7.10]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
    - id: mypy
      additional_dependencies: [types-requests, types-python-dateutil]
```

### CI/CD Pipeline
```yaml
# .github/workflows/ci.yml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --max-complexity=10 --max-line-length=88 --statistics
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
    - name: Security check
      run: |
        pip install bandit
        bandit -r . -x ./tests
```

## Security Review

### OWASP Top 10 Checks
1. **Injection**
   - [ ] No raw SQL queries
   - [ ] ORM used correctly
   - [ ] Input validation in place

2. **Broken Authentication**
   - [ ] Secure password storage (bcrypt/Argon2)
   - [ ] Session management
   - [ ] Multi-factor authentication where needed

3. **Sensitive Data Exposure**
   - [ ] No secrets in code
   - [ ] Encryption in transit (HTTPS)
   - [ ] Secure storage of sensitive data

4. **XML External Entities (XXE)**
   - [ ] Disable XML external entity processing
   - [ ] Use safe parsers

5. **Broken Access Control**
   - [ ] Role-based access control
   - [ ] Authorization checks
   - [ ] Principle of least privilege

6. **Security Misconfiguration**
   - [ ] Secure defaults
   - [ ] Error handling without stack traces
   - [ ] Security headers for web apps

7. **Cross-Site Scripting (XSS)**
   - [ ] Output encoding
   - [ ] Content Security Policy (CSP)
   - [ ] Input validation

8. **Insecure Deserialization**
   - [ ] Safe deserialization
   - [ ] Input validation

9. **Using Components with Known Vulnerabilities**
   - [ ] Dependencies up to date
   - [ ] Security scanning of dependencies

10. **Insufficient Logging & Monitoring**
    - [ ] Security events logged
    - [ ] Logs protected from tampering
    - [ ] Monitoring for suspicious activities

## Performance Considerations

### Database
- [ ] Indexes on frequently queried columns
- [ ] N+1 query problems solved
- [ ] Query optimization
- [ ] Connection pooling

### Memory Management
- [ ] No memory leaks
- [ ] Efficient data structures
- [ ] Generator expressions for large datasets
- [ ] Proper resource cleanup

### Concurrency
- [ ] Thread safety considered
- [ ] Asynchronous operations where appropriate
- [ ] Proper locking mechanisms

## Code Style

### Naming Conventions
- Variables and functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_private_member`

### Imports
- Grouped in this order:
  1. Standard library
  2. Third-party libraries
  3. Local application
- One import per line
- No wildcard imports (`from module import *`)

### Formatting
- Maximum line length: 88 characters
- 4 spaces for indentation
- No tabs
- Blank lines to separate logical sections
- Spaces around operators and after commas

## Documentation Standards

### Module Docstring
```python
"""
Short description.

Longer description with details about the module's purpose,
functionality, and usage examples.

Example:
    >>> from module import function
    >>> result = function()
"""
```

### Function/Method Docstring
```python
def function(param1: str, param2: int = 42) -> bool:
    """Short description.
    
    Longer description with details about the function's behavior,
    edge cases, and any side effects.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When something goes wrong.

    Example:
        >>> function("test", 123)
        True
    """
```

## Testing Requirements

### Unit Tests
- Test one thing per test
- Use descriptive test names
- Test edge cases
- Mock external dependencies
- Fast execution

### Integration Tests
- Test component interactions
- Use test databases
- Clean up after tests
- Test error conditions

### Performance Tests
- Benchmark critical paths
- Monitor memory usage
- Identify bottlenecks

## Common Issues

### Anti-patterns to Avoid
- God objects
- Deep inheritance
- Global state
- Magic numbers
- Over-engineering
- Premature optimization

### Code Smells
- Long parameter lists
- Deeply nested code
- Too many responsibilities in a single function
- Inconsistent naming
- Dead code
- Duplicate code

## Pull Request Template

```markdown
## Description

[Description of the changes made]

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

## Related Issues

Fixes #

## Additional Context

[Any additional information that would be helpful for reviewers]
```

## Review Process

1. **Initial Review**
   - Automated checks pass
   - Code follows style guidelines
   - Basic functionality works

2. **In-depth Review**
   - Architecture and design
   - Security considerations
   - Performance implications
   - Test coverage

3. **Final Check**
   - Documentation is complete
   - No commented-out code
   - No debugging artifacts
   - All tests pass

4. **Approval**
   - At least one approval required
   - All comments addressed
   - Conflicts resolved

## Review Comments

Use the following prefixes for review comments:

- `[STYLE]` - Code style issues
- `[SECURITY]` - Security vulnerabilities
- `[PERF]` - Performance concerns
- `[TEST]` - Testing issues
- `[DOCS]` - Documentation problems
- `[BUG]` - Functional bugs
- `[NIT]` - Minor, non-blocking issues

Example:
```
[STYLE] Consider renaming this variable to be more descriptive
[SECURITY] This input should be validated to prevent SQL injection
[PERF] This operation could be optimized by using a set instead of a list
```

## Continuous Improvement

- Regularly update dependencies
- Refactor legacy code when possible
- Learn from production issues
- Stay updated with security advisories
- Share knowledge with the team
