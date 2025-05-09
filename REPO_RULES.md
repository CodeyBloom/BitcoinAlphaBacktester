# Repository Rules: Test-Driven Development and Functional Programming

This document outlines our project's principles and practices for Test-Driven Development (TDD) and Functional Programming concepts from "Grokking Simplicity." These rules serve as our guiding framework for all development work.

## Test-Driven Development (TDD)

### Core Principles

1. **Test First, Code Later**
   - Write tests before implementing any functionality
   - Tests should initially fail for the right reasons
   - Implement minimum code required to make tests pass

2. **Red-Green-Refactor Cycle**
   - **Red**: Write a failing test that defines the expected behavior
   - **Green**: Write the minimum code necessary to pass the test
   - **Refactor**: Clean up the code while keeping tests passing

3. **Test Coverage Requirements**
   - Aim for 100% test coverage for core modules
   - Minimum acceptable coverage is 80% for any module
   - Track coverage metrics regularly using the `--cov` flag

### Testing Standards

1. **Test Structure**
   - Use descriptive test names with format: `test_[function]_[scenario]_[expected outcome]`
   - Each test should verify a single behavior or edge case
   - Include clear error messages when assertions fail

2. **Testing Scope**
   - Unit tests for functions and methods in isolation
   - Integration tests for component interactions
   - End-to-end tests for critical user workflows

3. **Test Data Management**
   - Use fixtures for common test data and setup
   - Generate test data programmatically when possible
   - Avoid dependencies on external services in tests (use mocks)

4. **Error and Edge Cases**
   - Test both the "happy path" and error conditions
   - Include boundary testing for numerical inputs
   - Test timeout or retry logic where applicable

5. **Mocking Guidelines**
   - Mock external dependencies (APIs, time, databases)
   - Use the appropriate level of mocking (function, class, module)
   - Ensure mocks accurately represent actual component behavior

## Functional Programming Principles (from Grokking Simplicity)

### Code Organization

1. **Actions vs. Calculations Separation**
   - **Actions**: Functions with side effects (I/O, state change)
   - **Calculations**: Pure functions with no side effects
   - Clearly document function types in docstrings

2. **Function Classification**
   - Comment sections to separate function types:
     ```python
     # ===== UTILITY FUNCTIONS (PURE) =====
     # ===== API INTERACTION FUNCTIONS (ACTIONS) =====
     ```

3. **Documentation Requirements**
   - Specify input/output types in docstrings
   - Document side effects for action functions
   - Include examples for complex functions

### Functional Design Patterns

1. **Data Immutability**
   - Treat data as immutable wherever possible
   - Create new data structures rather than mutating existing ones
   - Use defensive copies when receiving mutable data

2. **Function Composition**
   - Build complex operations from smaller, reusable functions
   - Use pipelines of transformations for data processing
   - Favor composition over inheritance

3. **Higher-Order Functions**
   - Use functions that operate on other functions
   - Apply map, filter, reduce paradigms for data transformations
   - Pass behavior as parameters when appropriate

4. **Error Handling**
   - Use return values to indicate errors when possible
   - Apply "Railway-oriented programming" for error propagation
   - Avoid exceptions for control flow

### Code Quality Standards

1. **Function Purity**
   - Pure functions should:
     - Have no side effects
     - Return the same output for the same input
     - Not depend on global state
   - Document any non-obvious side effects

2. **Function Complexity**
   - Keep functions short and focused (< 20 lines preferred)
   - Single Responsibility Principle for functions
   - No more than 3 parameters for functions when possible

3. **Testing for Purity**
   - Pure functions should be easy to test
   - If a function is hard to test, it likely needs refactoring
   - Use test doubles only for action functions, not calculations

## Development Workflow

1. **Pre-Commit Requirements**
   - All tests must pass
   - Meet minimum code coverage requirements
   - Run linting checks (if applicable)

2. **Code Review Criteria**
   - Verify proper TDD approach was followed
   - Check that functional programming principles are applied
   - Ensure tests are comprehensive and meaningful

3. **Continuous Improvement**
   - Regularly refactor for clarity and simplicity
   - Identify and eliminate code smells
   - Refine tests to improve quality, not just coverage

By following these guidelines, we ensure our codebase remains maintainable, testable, and aligned with functional programming principles for maximum reliability.