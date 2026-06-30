## CRITICAL

These rules MUST be followed in every code generation, modification, or refactoring task.
If a request conflicts with these rules, the rules take priority.

---
alwaysApply: true
description: |
This document defines the development, implementation, experimentation, and validation rules
for a Python AI research project. Its purpose is to guarantee algorithmic correctness,
reproducibility, scientific rigor, architectural clarity, and efficient collection of
research-relevant results. These rules are intended for both human contributors and
AI-assisted coding systems such as Cursor or Antigravity.
---------------------------------------------------------

# AI Research Project Rules + Enforcement Prompts

## 1. Scope

These rules apply to all code related to:

* algorithm implementation
* experiment execution
* evaluation and metrics
* result collection and summarization
* configuration management
* testing and validation
* refactoring of research-critical code

The project is intended for artificial intelligence research and scientific publication. Any code that influences experiments, metrics, ablations, baselines, training, inference, or evaluation must be treated as **scientifically sensitive code**.

---

## 2. Core Objectives

The system must always optimize for the following priorities, in this order:

1. **Correctness of algorithmic implementation**
2. **Experimental reproducibility**
3. **Scientific reliability and traceability**
4. **Architectural clarity and maintainability**
5. **Concise and research-useful outputs**
6. **Performance, only after correctness is secured**

When a tradeoff exists, prefer correctness, verifiability, and clarity over speed, convenience, or unnecessary abstraction.

---

## 3. General Development Rules

### 3.1 Readability and Maintainability

* Code must always be readable, understandable, and maintainable.
* Every change must preserve architectural clarity, operational robustness, and scientific reliability.
* The user-facing interface must remain clear, consistent, and as concise as possible.
* Avoid parameter pollution.
* Prefer simple implementations over clever but opaque ones.

### 3.2 Small and Controlled Changes

* Changes must always be as small as possible and limited to the relevant parts of the code.
* Every intervention must minimize impact on existing behavior not directly involved in the requested change.
* Any change requiring widespread edits must be split into small, self-contained, verifiable, and logically coherent parts.
* Broad refactors not justified by the current task must be avoided.
* Changes must always be proposed before they are applied, when the workflow supports proposal-before-edit behavior.

### 3.3 Single Responsibility

* Functions must be short and perform a single responsibility.
* Each module must have explicit responsibilities and clear boundaries.
* Each function must have understandable inputs and outputs without requiring unnecessary inference.

---

## 4. Architecture Rules

* The system must always follow a clear and well-separated architecture.
* Modules must be grouped by functionality, not by temporary convenience.
* Different features must also be separated at the architectural level.
* Avoid implicit coupling, unnecessary cross-cutting dependencies, and duplicated logic across different parts of the project.
* Shared utilities must remain genuinely shared and not become dumping grounds for unrelated logic.
* Research code, experimental orchestration, evaluation code, and result reporting must remain conceptually distinct.

### Recommended high-level separation

* `algorithms/`: core implementations of algorithms and models
* `experiments/`: experiment runners and orchestration
* `configs/`: experiment and system configuration
* `datasets/`: dataset access, loading, preprocessing interfaces
* `metrics/`: evaluation metrics and analysis utilities
* `tests/`: unit, regression, and integration tests
* `results/`: structured outputs and summaries
* `utils/`: minimal, reusable, non-domain-specific helpers

---

## 5. Algorithm Implementation Rules

### 5.1 Fidelity to Source

* Every algorithm must prioritize **correctness over performance**.
* Implementations must strictly follow the referenced paper, theorem, specification, or theoretical model.
* Any deviation, approximation, or engineering simplification must be explicitly documented.
* Algorithm files should include a clear source reference in comments or docstrings.

Example:

```python
# Source: Author et al., "Paper Title", Year
# Notes: This implementation follows Section 3.2 with the exception of...
```

### 5.2 Explicit Assumptions

* Avoid implicit behaviors.
* Make assumptions explicit in code, comments, and docstrings.
* Shapes, dimensions, domains, expected ranges, and invariants must be clear where relevant.

### 5.3 Determinism and Reproducibility

* Use fixed random seeds unless controlled stochasticity is part of the experiment.
* Provide seed control through configuration, never through hidden constants.
* Randomness sources must be centralized and documented.

### 5.4 Numerical Stability

* Guard against overflow, underflow, division by zero, unstable normalization, and invalid operations.
* Prefer numerically stable formulations whenever applicable.
* Validate outputs for NaNs, infs, or invalid ranges in critical algorithmic paths.

### 5.5 Verifiability

* Algorithms must be testable on small, known, interpretable cases.
* Intermediate states should be inspectable when needed for debugging or validation.
* Research-critical transformations must be transparent and auditable.

---

## 6. Scientific Reliability Rules

* This project is used for scientific research and publication; algorithmic implementations are experimentally sensitive.
* Every algorithmic implementation must be carefully controlled to avoid errors that compromise data collection, analysis, or publications.
* Data transformations, metrics, selection criteria, seeds, and experimental flows must be clear, verifiable, and unambiguous.
* When a change affects experimental or evaluation code, prioritize correctness, traceability, and reproducibility over implementation convenience.
* Never introduce hidden heuristics, undocumented filtering, silent fallback logic, or implicit metric changes in research-critical paths.

### Mandatory scientific safeguards

* Experimental assumptions must be explicit.
* Evaluation criteria must be deterministic unless explicitly designed otherwise.
* Data splits, seeds, preprocessing, and metric definitions must be reproducible.
* Result files must preserve enough metadata to trace how they were produced.

---

## 7. Experimentation Rules

### 7.1 Configuration-Driven Execution

* No hardcoded experimental parameters in runners or algorithm code unless they are mathematically intrinsic constants.
* All experiment settings must be defined through configuration files, clear constructor parameters, or centralized configuration structures.
* Configuration should be centralized when appropriate and kept consistent across modules.

### 7.2 One Change at a Time

* Each experiment should test one hypothesis or one meaningful variation at a time.
* When comparing experimental results, change only the relevant variable unless the experiment explicitly studies interactions.

### 7.3 Baselines and Comparisons

* Baseline implementations or reference settings should be preserved and kept comparable.
* Changes to baselines must be clearly declared and justified.

### 7.4 Traceability

Each experiment must be traceable to:

* code version or git commit hash
* full configuration snapshot
* dataset or split version
* seed information
* metric definitions used for evaluation

### 7.5 No Hidden Experimental Logic

* Do not hide data filtering, evaluation thresholds, seed choices, or selection rules inside helper code without documentation.
* Avoid convenience shortcuts that make results harder to interpret scientifically.

---

## 8. Results, Metrics, and Output Rules

### 8.1 Research-Relevant Metrics Only

* Collect only metrics relevant to the research question.
* Avoid redundant, noisy, or non-actionable outputs.
* Prefer concise summaries over large unstructured dumps.

### 8.2 Structured Outputs

* Results must be saved in structured formats such as JSON, CSV, or similarly machine-readable formats.
* Result records must be compact but sufficiently informative.

Example:

```json
{
  "experiment_id": "exp_001",
  "algorithm": "AlgorithmName",
  "config": "configs/exp_001.yaml",
  "seed": 42,
  "commit": "abc1234",
  "metrics": {
    "accuracy": 0.91,
    "f1": 0.88,
    "loss": 0.23
  }
}
```

### 8.3 Aggregated Reporting

When applicable, include:

* mean
* standard deviation
* confidence interval or equivalent uncertainty summary
* number of runs

### 8.4 No Result Pollution

* Do not store unnecessary intermediate artifacts by default.
* Verbose diagnostics must be opt-in.
* Avoid producing outputs that do not directly help research interpretation, replication, or debugging.

### 8.5 Concision

* Outputs intended for paper writing or experiment review must be concise, comparison-friendly, and easy to scan.
* Prefer structured tables, summarized metrics, and compact experiment metadata.

---

## 9. Parameters and Configuration Rules

* All input parameters must be well documented.
* Parameters must not be duplicated across multiple parts of the codebase.
* Default values must be explicit, justified, and safe.
* Avoid overloaded function signatures or non-essential parameters.
* Minimize the number of required parameters while preserving correctness and clarity.
* Related parameters may be grouped only when grouping improves clarity rather than hiding meaning.

### Configuration principles

* Centralize configuration when appropriate.
* Keep naming consistent across modules.
* Ensure experimental parameters are easy to inspect, compare, and serialize.
* Avoid magic numbers unless mathematically required and documented.

---

## 10. Error Handling and Logging Rules

* Errors must always be checked and handled explicitly.
* Failure conditions must produce clear, contextualized messages useful for diagnosis.
* Log messages must be understandable, specific, and designed to quickly expose potential issues.
* Do not suppress exceptions without an explicit technical reason.
* Critical paths must validate inputs, preconditions, intermediate states, and outputs.

### Logging principles

* Logging must be minimal but informative.
* Avoid debug noise in standard experiment execution.
* Prefer structured logging where possible.
* Research-critical warnings must never be silently ignored.

## 10.1 Model Selection for Coding Tasks

* For tasks involving code writing, code completion, code refactoring, algorithm implementation, or other Codex-style software engineering activities, the preferred model must be **gpt-5.3-codex**.
* This preference applies in particular to tasks requiring:
  * multi-file code generation
  * structured refactoring
  * implementation of research-critical logic
  * test creation
  * debugging of non-trivial code paths
* If another model is used due to system constraints, this must not relax any of the correctness, reproducibility, traceability, or validation rules defined in this document.

## 10.2 Mandatory Terminal Logging

* Code that performs execution steps, long-running operations, experiment orchestration, training, evaluation, data processing, or other non-trivial workflows must always provide informative terminal logging.
* Terminal logs must make the execution state observable and diagnosable without inspecting internal code.
* Logging must always include, where applicable:
  * current phase or step
  * relevant input/output identifiers
  * start and end of critical operations
  * warnings and failure conditions
* Log messages must be concise, explicit, and useful for diagnosing progress and failures.
* Silent execution is forbidden for long-running or research-critical tasks unless explicitly justified.

## 10.3 Progress Visibility for Long Tasks

* Any task expected to take a non-trivial amount of time must expose measurable progress in the terminal.
* Progress reporting should, whenever possible, indicate:
  * current step out of total steps
  * percentage of completion
  * current item / batch / epoch / experiment being processed
  * estimated remaining phases when a reliable estimate is possible
* For loops or pipelines over many items, progress must be periodically updated so that execution does not appear stalled.
* For especially long tasks, logs should clearly distinguish:
  * initialization
  * active processing
  * checkpoint / intermediate milestones
  * completion or failure
* Prefer simple and robust progress indicators over complex logging systems.

### Example expectations

```python
print("[1/5] Loading dataset...")
print("[2/5] Preprocessing samples...")
print(f"[3/5] Training epoch {epoch}/{max_epochs}...")
print(f"[4/5] Evaluating batch {batch_idx + 1}/{num_batches}...")
print("[5/5] Saving results...")
```
---

## 11. API and Interface Rules

* Public interfaces must be simple, stable, and easy to use correctly.
* Each API must expose only what is genuinely necessary to the caller.
* Names, outputs, and user-facing messages must be concise and unambiguous.
* Interfaces should reduce the risk of invalid experimental setup.
* When a safer or more constrained interface prevents research mistakes, prefer the safer interface.

---

## 12. Code Style and Documentation Rules

* Follow PEP 8 unless a project-specific rule overrides formatting.
* Comments and docstrings must clarify intent, constraints, assumptions, and scientific implications.
* Do not write comments that merely restate obvious code.
* Prefer targeted reuse over premature abstraction.
* Reduce logical duplication when it improves clarity and consistency.

### Code typing

Always generate Python code with strict typing.

Requirements:

* Every function must have type hints for all parameters and return values.
* Use modern Python typing (PEP 484, PEP 585, PEP 604).
* Do not use implicit Any.
* Explicitly type all variables when not obvious.
* Use typing constructs such as TypedDict, Protocol, dataclass, and generics where appropriate.
* Ensure the code is compatible with static type checkers (mypy/pyright) with no errors.

Code that does not follow these rules is considered invalid.

### Documentation minimum for research-critical code

Each important module or class should make clear:

* purpose
* inputs and outputs
* assumptions
* important invariants
* experimental relevance where applicable

---

## 13. Testing and Validation Rules

### 13.1 Mandatory Testing

* Every algorithm must have unit tests.
* Test against small known examples, edge cases, and failure cases.
* Changes affecting scientific behavior must include regression protection where appropriate.

### 13.2 Edge Cases

Explicitly test, when applicable:

* empty inputs
* boundary conditions
* extreme values
* invalid shapes or invalid parameter ranges
* deterministic behavior under fixed seeds
* numerical stability issues

### 13.3 Regression and Reproducibility Tests

* Ensure changes do not alter expected results unintentionally.
* If output changes are expected, document why.
* For research-critical code, verify that metrics and experiment outputs remain consistent with expectations.

### 13.4 Validation Before Trust

No algorithmic implementation should be considered reliable until:

* it is compared against theory, paper logic, or a known reference
* it passes tests on controlled cases
* its outputs are inspected for plausibility on at least one interpretable scenario

---

## 14. Dependency Rules

* Introduced libraries must, whenever possible, be standard, actively maintained, and not deprecated.
* Before adding a new dependency, verify whether the requirement can be solved with the standard library or existing dependencies.
* Avoid marginal, obsolete, poorly maintained, or unnecessary libraries, especially in research-critical components.
* New dependencies must not reduce reproducibility, portability, or clarity without strong justification.

---

## 15. Strict Anti-Patterns

The following are strictly forbidden unless explicitly justified and documented:

* hidden parameters
* hardcoded experiment values
* undocumented seed handling
* unverified algorithm implementations
* silent changes to metrics or evaluation logic
* mixing multiple experimental variables without purpose
* storing excessive or irrelevant result data
* broad refactors unrelated to the task
* implicit fallback behavior in research-critical code
* exception suppression without technical justification
* unnecessary abstraction that obscures algorithmic meaning

---

## 16. Decision Criterion

When in doubt, prefer the simplest solution that preserves:

* correctness
* verifiability
* architectural clarity
* reproducibility
* scientific reliability

If a solution is shorter but harder to verify, prefer the more explicit one.
If a solution is faster but scientifically riskier, prefer the safer one.

---

# Enforcement Prompts for AI Coding Systems

These prompts are intended to guide AI assistants before generating, modifying, or refactoring code.

## 17. Global Enforcement Prompt

Use this as the general persistent rule block for AI coding systems:

```text
You are working on a Python AI research project used for implementing, testing, and evaluating algorithms for scientific publication.

Your highest priorities are:
1. algorithmic correctness
2. reproducibility
3. scientific reliability
4. architectural clarity
5. concise and research-useful outputs

You must:
- prefer correctness over speed or convenience
- keep changes minimal, local, and easy to verify
- avoid hidden assumptions, hidden parameters, and silent behavior changes
- preserve clean architecture and explicit module responsibilities
- keep experimental logic, metrics, seeds, and evaluation criteria fully traceable
- avoid hardcoded experiment settings unless mathematically intrinsic
- ensure outputs are structured, concise, and useful for research
- validate edge cases, invariants, and numerical stability in research-critical code
- document any deviation from papers, references, or expected theory

When modifying code:
- propose the intended change clearly
- explain what files/functions are affected
- avoid broad refactors unless strictly necessary
- preserve backwards-compatible behavior unless a change is explicitly required
- highlight any scientific risk introduced by the modification

When implementing algorithms:
- follow the source paper/specification faithfully
- make assumptions explicit
- prefer readable and verifiable implementations over compact clever code
- verify dimensions, parameter semantics, deterministic behavior, and metric correctness

When producing experiment outputs:
- collect only research-relevant data
- avoid raw verbose dumps unless explicitly requested
- produce structured summaries suitable for comparison and publication workflows
```

---

## 18. Enforcement Prompt for Algorithm Implementation

```text
Before implementing or changing any algorithm, verify the following:
- What paper, theorem, or reference defines the algorithm?
- What assumptions, invariants, tensor shapes, and parameter meanings must remain true?
- What parts of the implementation are numerically sensitive?
- What behavior must be deterministic?
- What tests can validate correctness on small controlled examples?

Do not implement algorithmic logic if the expected behavior is ambiguous. Prefer explicit, testable, scientifically traceable code.
```

---

## 19. Enforcement Prompt for Experimental Code

```text
This code affects experiments and may influence scientific conclusions.
Treat it as scientifically sensitive.

Before changing anything, verify:
- whether the change modifies metrics, seeds, preprocessing, selection criteria, or evaluation flow
- whether the change breaks reproducibility or traceability
- whether the change introduces hidden logic or convenience shortcuts
- whether all parameters remain explicit and centralized
- whether the resulting outputs remain concise, structured, and research-useful

If there is any risk of changing scientific meaning, state it explicitly.
```

---

## 20. Enforcement Prompt for Refactoring

```text
Refactor only what is necessary for the current task.
Keep behavior unchanged unless a behavior change is explicitly requested.

Before refactoring, check:
- whether the refactor alters algorithmic behavior
- whether it changes parameter flow or defaults
- whether it affects reproducibility
- whether it makes scientific code harder to inspect
- whether the same result can be achieved with a smaller change

Prefer local simplification over broad redesign.
```

---

## 21. Enforcement Prompt for Testing

```text
For every research-critical code change, identify the minimum validation set required before considering the change acceptable.

Check for:
- unit tests on controlled examples
- edge cases and invalid inputs
- deterministic behavior under fixed seeds
- regression risks
- numerical stability issues
- metric correctness

Do not claim a change is safe unless its behavior is verifiable.
```

---

## 22. Enforcement Prompt for Results and Reporting

```text
All experiment outputs must be concise, structured, and useful for scientific analysis.

Before adding logs, reports, or stored artifacts, verify:
- whether the output is directly useful for research interpretation
- whether it duplicates existing information
- whether it is machine-readable
- whether it includes enough metadata for traceability
- whether it avoids unnecessary verbosity

Prefer compact result summaries over raw dumps.
```

---

## 23. Operational Review Checklist

Before accepting any implementation or modification, verify:

* [ ] The change is minimal and localized.
* [ ] The architecture remains clean and coherent.
* [ ] No hidden parameters or silent behaviors were introduced.
* [ ] Algorithmic logic is explicit and faithful to the source.
* [ ] Seeds, metrics, preprocessing, and evaluation rules remain traceable.
* [ ] Research outputs remain concise and structured.
* [ ] Tests cover the relevant controlled cases.
* [ ] Edge cases and numerical risks were considered.
* [ ] Any scientific risk or deviation is explicitly documented.

---

## 24. Model Selection for Coding Tasks

* For tasks involving code writing, code completion, code refactoring, algorithm implementation, or other Codex-style software engineering activities, the preferred model must be **gpt-5.3-codex**.
* This preference applies in particular to tasks requiring:
  * multi-file code generation
  * structured refactoring
  * implementation of research-critical logic
  * test creation
  * debugging of non-trivial code paths
* If another model is used due to system constraints, this must not relax any of the correctness, reproducibility, traceability, or validation rules defined in this document.

---

## 25. Final Principle

> The goal is not to produce more code or more outputs, but to produce **reliable, verifiable, and scientifically valid results**.

> Clarity > Cleverness
> Correctness > Performance
> Reproducibility > Convenience
> Insight > Volume
