# Computer Science Evaluation Rubric

## Task

Score ONE answer to ONE Computer Science question from 0–4.

The question metadata establishes the relevant domain, intent, difficulty, programming/system context, and expected reasoning style.

Accept different valid algorithms, implementations, architectures, data structures, or debugging approaches when they satisfy the stated requirements.

Do not require a particular programming style or solution structure.

Do not reward unnecessary length, jargon, design-pattern naming, or citations.

## Critical Error

If `critical_computer_science_error` applies:

**score = 0**

A critical Computer Science error is a materially false central programming, algorithmic, systems, database, security, quantitative, machine-learning, or design claim that fundamentally invalidates the answer or reverses the substantive conclusion.

Examples:
- Claims an algorithm is correct despite a clear counterexample central to the question.
- Makes fundamentally incorrect time-complexity reasoning that changes the algorithm-selection conclusion.
- Claims a nonterminating function always terminates.
- Introduces a central off-by-one or indexing error that makes the proposed implementation incorrect.
- Uses fundamentally incorrect data-structure semantics.
- Ignores a race condition in a concurrency solution that depends on unsafe shared state.
- Claims a non-atomic sequence is atomic without required synchronization.
- Recommends inconsistent lock ordering that introduces deadlock.
- Uses fundamentally incorrect transaction or isolation reasoning.
- Claims retries are harmless for a non-idempotent operation when duplicates materially alter state.
- Treats distributed communication as perfectly reliable when the problem explicitly includes partial failure.
- Uses fundamentally incorrect network-layer reasoning that invalidates the diagnosis.
- Confuses authentication with authorization in a central security decision.
- Recommends plaintext password storage or equivalently unsafe credential handling.
- Claims encryption alone establishes integrity when tamper detection is required.
- Recommends a mitigation that clearly fails to address the supplied vulnerability.
- Uses test data during training while claiming unbiased generalization performance.
- Makes a major quantitative error that changes system capacity, complexity, or performance conclusions.
- Proposes a system architecture that cannot satisfy a stated critical requirement because of a central design error.
- Asserts that passing tests prove program correctness when the task explicitly exposes untested states.

Do **not** classify the following as critical by themselves:
- Minor syntax errors that do not obscure the intended correct solution.
- Small constant-factor mistakes.
- Minor naming issues.
- Reasonable alternative algorithms or architectures.
- Different valid language idioms.
- Minor omissions.
- Non-central performance imperfections.
- Reasonable implementation assumptions that are clearly stated.
- Stylistic disagreements.

## Score Anchors

### 4

The answer is technically accurate, directly addresses the question, applies the appropriate Computer Science principles, reasons correctly about code or systems behavior, handles important edge cases, performs relevant calculations correctly, and reaches a sound conclusion.

For programming questions, the solution is correct under the stated language semantics.

For algorithms, correctness and complexity are handled appropriately.

For systems questions, the answer recognizes relevant failure modes and trade-offs.

For security questions, the proposed mitigation addresses the actual threat.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor implementation flaw, limited reasoning gap, non-central complexity error, or insufficient discussion of an edge case.

The central technical conclusion remains correct.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete reasoning, significant implementation flaws, incorrect complexity or systems interpretation, weak assumptions, or incomplete handling of material edge cases.

### 1

The answer contains a substantial Computer Science misunderstanding, inappropriate algorithm or architecture, major non-critical code error, poor debugging reasoning, or badly incomplete analysis, but does not meet the critical-error threshold.

### 0

Critical Computer Science error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question.

A database question need not include ML reasoning. An algorithm problem need not discuss distributed systems. A conceptual question need not contain code.

Conditional criteria should not penalize an answer for omitting irrelevant material. However, when a conditional issue is necessary for correctness, failure on that issue should affect the score.

## Evaluation Criteria

| Criterion | Weight | Conditional | Definition |
|---|---:|:---:|---|
| `relevance` | 0.05 | No | Directly addresses the question actually asked, including supplied code, data, constraints, and requested deliverables, without substituting a nearby but easier Computer Science problem. |
| `technical_accuracy` | 0.05 | No | Uses correct Computer Science facts and semantics. Central claims about program behavior, algorithms, systems, data, security, or machine learning must be materially true under the stated assumptions and must not rely on jargon in place of mechanism. |
| `programming_correctness` | 0.05 | Yes | When code or implementation behavior is relevant, traces state and control flow accurately, respects the specified language semantics, terminates as claimed, preserves intended state, and distinguishes a minor syntax defect from logically incorrect code. |
| `algorithmic_reasoning` | 0.05 | Yes | When an algorithm is required, selects or analyzes an approach whose preconditions match the problem, explains why it is correct, identifies invalid greedy or recursive assumptions, and gives counterexamples when correctness fails. |
| `complexity_analysis` | 0.05 | Yes | When performance analysis is relevant, states time and space costs that match the actual algorithm or code, distinguishes upper from tight bounds and worst from average or amortized behavior, and does not infer practical runtime from asymptotics alone. |
| `data_structure_reasoning` | 0.05 | Yes | When data structures matter, reasons about operation costs, invariants, memory, representation, edge cases, and workload fit. Naming a structure without showing why it satisfies the required operations is insufficient. |
| `systems_reasoning` | 0.05 | Yes | When operating systems, architecture, cloud, performance, or systems design is relevant, models state, resources, locality, latency, throughput, failure, recovery, and operational constraints rather than assuming idealized components. |
| `concurrency_reasoning` | 0.05 | Yes | When concurrent execution is relevant, identifies shared mutable state, atomicity and visibility requirements, possible interleavings, lock and condition invariants, contention, deadlock or starvation risks, and does not treat timing delays as synchronization. |
| `database_reasoning` | 0.05 | Yes | When databases are relevant, preserves schema and integrity semantics, reasons correctly about joins, NULLs, keys, normalization, indexes, plans, transactions and isolation, and distinguishes logical correctness from physical optimization. |
| `networking_and_distributed_reasoning` | 0.05 | Yes | When networking or distributed systems are relevant, distinguishes layers and protocol semantics and accounts for latency, loss, retries, duplicate delivery, stale replicas, clock disagreement, partitions and partial failure without assuming instant global state. |
| `security_reasoning` | 0.05 | Yes | When security is relevant, identifies assets, threats, trust boundaries and required properties; distinguishes authentication, authorization, confidentiality and integrity; and proposes mitigations that directly address the supplied threat model. |
| `software_engineering_reasoning` | 0.05 | Yes | When engineering process or design is relevant, reasons about requirements, interfaces, compatibility, maintainability, testing, rollout, observability, dependencies, and operational complexity rather than offering style preferences or design-pattern names alone. |
| `debugging_and_root_cause_analysis` | 0.05 | Yes | When diagnosing a failure, uses supplied symptoms, traces, code, and measurements to identify a supported root cause, explains the causal mechanism, proposes a targeted correction and verification step, and avoids speculative bug lists when evidence supports a narrower diagnosis. |
| `quantitative_correctness` | 0.05 | Yes | When calculations are required, selects the right quantity or computational model, uses consistent units and assumptions, performs arithmetic correctly, and interprets the result without allowing a major numerical error to reverse the conclusion. |
| `assumptions_and_edge_cases` | 0.05 | Yes | When material, states necessary assumptions and handles cases such as empty or singleton input, duplicates, overflow, null values, disconnected or cyclic structures, concurrent access, failed network operations, retries, or unavailable components. |
| `tradeoff_and_design_reasoning` | 0.05 | Yes | When multiple valid designs exist, connects choices to explicit requirements and compares correctness, complexity, performance, availability, consistency, durability, security, maintainability, and operational cost without rewarding complexity for its own sake. |
| `evidence_and_uncertainty` | 0.05 | Yes | When evidence is incomplete, distinguishes observed facts from hypotheses, states what additional measurement or documentation would resolve uncertainty, and avoids claiming implementation-specific behavior or a causal diagnosis without support. |
| `completeness` | 0.05 | No | Covers the material parts of the requested task, including explanation, correction, comparison, proof, calculation, failure mode, or trade-off where explicitly requested, without penalizing omission of irrelevant criteria. |
| `consistency` | 0.05 | No | The answer's code, examples, calculations, complexity claims, assumptions, and conclusion agree with one another; it does not propose a fix that contradicts its diagnosis or a design that violates a stated requirement. |
| `clarity` | 0.05 | No | Explains reasoning in a precise, inspectable way: important states, invariants, causal steps, assumptions, and trade-offs are understandable without unnecessary jargon, verbosity, or unsupported citations. |

## Programming Answers

Strong programming answers should, where appropriate:

- follow stated language semantics
- produce correct behavior
- handle relevant edge cases
- avoid unintended mutation
- terminate correctly
- use suitable data structures
- explain material implementation choices

Do not require exact reference-solution syntax.

## Algorithm Answers

Strong algorithm answers should, where appropriate:

- identify the algorithm
- explain why it is correct
- state relevant invariants
- analyze time complexity
- analyze space complexity
- identify important edge cases
- compare reasonable alternatives

Do not reward the correct algorithm name without correct application.

## Debugging Answers

Strong debugging answers should:

1. use the observed symptoms
2. identify the likely root cause
3. explain why that cause produces the symptoms
4. propose a targeted fix
5. consider regression risks or relevant edge cases

Do not reward unfocused lists of possible bugs when a more precise diagnosis is supported.

## Systems Answers

Strong systems answers should identify, where relevant:

- workload
- scale
- performance requirements
- state
- storage
- consistency
- concurrency
- partial failure
- observability
- recovery
- security
- operational trade-offs

Do not require distributed complexity when a single-node solution satisfies the requirements.

## Security Answers

Strong security answers should:

- identify the asset
- identify the threat
- identify the trust boundary
- explain the vulnerability
- propose a mitigation
- explain why the mitigation addresses the threat

Do not reward vague advice such as “use encryption” when the actual issue is authorization or input validation.

## Machine Learning Answers

Strong ML answers should, where relevant:

- separate training, validation, and test data
- avoid leakage
- select suitable metrics
- distinguish model fit from generalization
- recognize distribution shift
- interpret results cautiously
- avoid causal claims from predictive associations alone

## Important Evaluation Principles

### Correct output is not enough

A solution that produces the expected sample result through invalid logic should not receive full credit.

### Correctness before optimization

An incorrect fast solution is not superior to a correct slower solution unless the question explicitly permits approximation.

### Complexity must match the code

Do not reward complexity claims that are inconsistent with the actual implementation.

### Edge cases matter

Strong answers should consider material cases such as empty input, singleton input, duplicates, overflow, null values, disconnected graphs, cyclic graphs, concurrent access, and failed network operations when relevant.

### Language semantics matter

Do not assume behavior is universal across programming languages.

### Concurrency requires interleaving reasoning

Code that appears correct sequentially may fail under concurrent execution; reason about interleavings, atomicity, visibility, and synchronization.

### Distributed systems can partially fail

Strong answers should consider timeouts, retries, duplicates, stale data, partitions, and unavailable nodes when relevant.

### Security depends on threat models

A security control should address the actual attack or failure mode.

### Authentication is not authorization

Verifying identity does not by itself determine permitted actions.

### Tests are evidence, not proof

A passing test suite does not guarantee correctness.

### Benchmarks and metrics need context

Performance measurements depend on workload, input distribution, hardware, concurrency, caching, and environment.

### Prediction is not causation

Machine-learning predictions and feature associations do not by themselves establish causal relationships.

### Simpler systems can be better

Do not reward unnecessary architectural complexity.

### Abstraction is not indirection for its own sake

Strong designs use abstraction to manage complexity, not merely to introduce layers.

### Failure modes matter

Strong system designs explain what happens when components fail.

### No jargon worship

Technical vocabulary does not compensate for invalid reasoning.

### No citation worship

Citations do not compensate for incorrect code, algorithms, calculations, or systems reasoning.
