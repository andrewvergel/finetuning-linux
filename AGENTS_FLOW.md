# AI Agent Development Workflow

This document outlines the step-by-step workflow that AI agents must follow when implementing new features or making changes to the codebase. This ensures consistency, quality, and adherence to project standards.

## Workflow Steps

### 1. Task Analysis & Planning
- **Create a Todo List**
  - Break down the user's request into specific, actionable tasks
  - Each task should be clear, concise, and testable
  - Include acceptance criteria for each task

- **Initial Code Investigation**
  - Analyze relevant files and directories
  - Identify affected components and dependencies
  - Review existing patterns and conventions

### 2. Implementation Planning
- **Theoretical Changes**
  - Document the planned changes before implementation
  - Include data flow and architectural decisions
  - Consider edge cases and error conditions
  - Identify potential risks and mitigation strategies

### 3. Implementation
- **Follow Development Guidelines**
  - Adhere to patterns in [AGENTS_DEV.md](AGENTS_DEV.md)
  - Learn from [AGENTS_LEARN.md](AGENTS_LEARN.md)
  - Follow code style and best practices
  - Implement comprehensive error handling
  - Include appropriate logging

### 4. First Code Review (Self-Review)
- **Conduct Self-Review**
  - Perform a thorough review of all changes
  - Verify against [AGENTS_CODEREVIEW.md](AGENTS_CODEREVIEW.md) checklist
  - Ensure all automated checks pass
  - Document any trade-offs or technical debt

### 5. Iterative Improvement
- **Second Implementation Pass**
  - Address issues found in first review
  - Optimize performance and readability
  - Refactor if necessary
  - Ensure all edge cases are handled

### 6. Final Code Review
- **Second Review Cycle**
  - Re-validate against all checklists
  - Verify test coverage
  - Ensure documentation is complete and accurate
  - Confirm security best practices are followed

### 7. Documentation & Handoff
- **Implementation Summary**
  - Document the changes made
  - Include any special considerations
  - Note potential future improvements
  - Update relevant documentation

## Quality Gates

At each stage, the following must be verified:

1. **Code Quality**
   - Follows PEP 8 and project style guide
   - Proper type hints and documentation
   - No code duplication
   - Clean git history

2. **Testing**
   - Unit tests for new functionality
   - Integration tests for critical paths
   - Test coverage ≥80%
   - All tests passing

3. **Security**
   - No hardcoded secrets
   - Input validation
   - Proper error handling
   - Secure coding practices

4. **Performance**
   - Efficient algorithms
   - Proper resource management
   - Appropriate use of caching
   - Scalability considerations

## Workflow Automation

This workflow is designed to be automated where possible:
- Automated testing and code quality checks
- Pre-commit hooks for style enforcement
- CI/CD pipeline integration
- Automated documentation generation

## Exception Handling

Any deviations from this workflow must be:
1. Documented with rationale
2. Approved via code review
3. Added to technical debt documentation if temporary
