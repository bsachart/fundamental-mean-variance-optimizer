# Portfolio Optimizer: Development Philosophy & Rules

This project follows strict quality standards to ensure a robust, maintainable, and "deep" architectural foundation.

## 1. John Ousterhout's Principles
Derived from *"A Philosophy of Software Design"*, we prioritize complexity management:

- **Modules Should Be Deep**: Interfaces must be simple, while the internal logic is powerful. Avoid "shallow" modules that just wrap simple tasks.
- **Information Hiding**: Internal implementation details (e.g., optimization solvers, matrix math) must not leak into the UI or high-level callers.
- **Define Errors Out of Existence**: Design APIs that handle edge cases gracefully or don't allow them, rather than forcing callers to manage complex exception trees.
- **Complexity is Incremental**: Maintain high standards for every small change to prevent cumulative complexity ("death by a thousand cuts").

## 2. Code Quality & Type Safety
- **Python Typehints**: All function signatures **must** include typehints for arguments and return values. This is non-negotiable for project stability.
- **Clarity vs. Cleverness**: Write code that is easy to reason about. Complex math should be well-commented with links to relevant theory if necessary.

## 3. Bug Prevention & Verification
- **Testing is Mandatory**: Every core logic change (Returns, Risk, Optimization) must be accompanied by automated tests.
- **Test Co-location**: Tests are **co-located with their corresponding modules**, not placed in a centralized `tests/` directory.  
  - Each Python file should have its tests defined in a neighboring `test_<module>.py` file.
  - Tests should live in the same package as the code they validate to maximize locality, readability, and refactor safety.
- **Logic Verification**: Optimization constraints and mathematical properties (e.g., positive semi-definite covariance) must be verified through automated tests to prevent silent regressions.

## 4. GEMINI Interaction
When suggesting changes, Antigravity/Gemini must:
- Reference these rules and ensure new code complies with them.
- Proactively run the full test suite after every modification.
- Respect the co-located testing structure when adding or updating tests.
- Maintain the "terminal-style" focused UI aesthetics established in the project.
