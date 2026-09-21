# Software Engineering & Programming Evaluation Rubric

## Task

Score ONE answer to ONE Software Engineering & Programming question from 0–4.

The question metadata establishes the relevant domain, intent, difficulty, programming context, and expected reasoning style.

Accept different valid implementations, designs, APIs, schemas, architectures, testing strategies, and debugging approaches when they satisfy the stated requirements.

Do not require a particular programming style, framework, or architecture unless specified. Do not reward unnecessary length, jargon, design-pattern naming, or citations.

## Critical Error

If `critical_software_engineering_programming_error` applies, **score = 0**. Otherwise evaluate the applicable criteria.

A critical error is a materially false central claim that fundamentally invalidates the answer or would create a severe software failure. Minor syntax issues, naming preferences, modest inefficiency, reasonable alternative designs, small omissions, non-central refactoring preferences, different valid implementation idioms, and explicitly stated reasonable assumptions are not critical by themselves.

Representative critical examples include: destructive migration steps presented as safe; ignoring a race that breaks a central invariant; claiming a non-atomic operation is atomic; creating a deadlock as the fix; retrying a non-idempotent payment as if harmless; assuming at-least-once delivery means exactly-once execution; omitting required server-side authorization while claiming security; plaintext-equivalent password storage; major capacity arithmetic that invalidates the design; a breaking API change presented as compatible; or an incident diagnosis that would materially worsen the outage.

## Score Anchors

### 4
The answer is technically accurate, directly addresses the question, correctly reasons about program or system behavior, satisfies the stated requirements, handles material edge cases and failure modes, and reaches a sound implementation or engineering conclusion.

For code questions, the implementation is correct under the stated language semantics. For debugging questions, it identifies the root cause and proposes an appropriate fix. For design questions, it recognizes important requirements, trade-offs, and failure modes. For production questions, it accounts for reliability, compatibility, data integrity, and operational behavior where relevant.

### 3
The answer is substantially correct and useful but contains one meaningful omission, minor implementation flaw, limited reasoning gap, incomplete edge-case treatment, or modest design weakness. The central implementation or engineering conclusion remains correct.

### 2
The answer is partly correct or directionally useful but has important omissions, significant implementation problems, weak debugging evidence, incomplete test reasoning, poor handling of edge cases, or an underdeveloped architecture. Meaningful correction would be required before confidently using the solution.

### 1
The answer contains a substantial programming or software-engineering misunderstanding, major non-critical code error, inappropriate design, weak root-cause analysis, or badly incomplete reasoning, but does not meet the critical-error threshold.

### 0
Critical Software Engineering & Programming error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question. A local algorithm problem need not discuss deployment; a database-schema question need not discuss frontend design; a unit-testing question need not discuss distributed systems. Conditional criteria should not penalize answers for omitting irrelevant material. However, if a conditional issue is necessary for correctness, failure on that issue should affect the score.

## Criteria

### Relevance (`relevance`)
- Weight: 0.05
- Applicability: Generally applicable
- Definition: Directly addresses the question actually asked, prioritizes material requirements, and avoids unrelated framework, pattern, or jargon discussion.

### Technical Accuracy (`technical_accuracy`)
- Weight: 0.05
- Applicability: Generally applicable
- Definition: States technically correct software-engineering facts and conclusions under the stated language, runtime, database, API, or system assumptions. Material factual errors reduce credit even when prose is polished.

### Programming Correctness (`programming_correctness`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When code or pseudocode is required, the implementation satisfies the specification beyond the sample case, terminates as required, preserves invariants, respects language semantics, and does not introduce material bugs. Minor syntax slips may be non-critical when intent is unambiguous.

### Code Reasoning (`code_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When code is supplied, accurately traces control flow, evaluation order, mutation, scope, references/values, exceptions, side effects, resource lifetime, and execution order rather than inferring intended behavior from names.

### Debugging and Root-Cause Analysis (`debugging_and_root_cause_analysis`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When diagnosis is requested, uses the supplied evidence to identify the most likely causal mechanism, distinguishes cause from symptom, proposes a targeted fix, and gives relevant verification steps rather than an unfocused list of possibilities.

### Implementation Quality (`implementation_quality`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When implementation is requested, chooses appropriate algorithms/data structures, handles material boundaries and errors, and produces understandable maintainable code without unnecessary abstraction or complexity.

### Testing Reasoning (`testing_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When testing is relevant, selects appropriate test levels and boundaries, tests behavior and invariants, handles nondeterminism, identifies missing integration coverage, and does not equate line coverage or a passing suite with correctness.

### Software Design Reasoning (`software_design_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When design is relevant, assigns cohesive responsibilities, manages dependencies and state, selects abstractions that reduce rather than add complexity, and accepts simple designs when they satisfy requirements.

### API and Contract Reasoning (`api_and_contract_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When interfaces or APIs are relevant, specifies clear request/response or call semantics, validation, error behavior, compatibility, null/optional handling, idempotency, and versioning as required by the scenario.

### Database and State Reasoning (`database_and_state_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When persistent data is relevant, preserves keys, constraints, relationships, transaction boundaries, state invariants, join/cardinality semantics, migration safety, and concurrency behavior. A schema or query that permits required data corruption cannot receive full credit.

### Concurrency and Async Reasoning (`concurrency_and_async_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When overlapping execution is possible, identifies shared state, atomicity/visibility needs, relevant interleavings, synchronization, cancellation, bounded concurrency, and deadlock risks. Sequentially correct reasoning is insufficient if concurrency changes behavior.

### Distributed and Reliability Reasoning (`distributed_and_reliability_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When networked or asynchronous components are involved, accounts for partial failure, ambiguous timeouts, retries, duplicate delivery, ordering, stale state, idempotency, backoff, recovery, and overload interactions as applicable.

### Security Reasoning (`security_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When security is relevant, identifies the real trust boundary and threat, distinguishes authentication from authorization, validates untrusted input at server/system boundaries, protects secrets/credentials, and applies least privilege and safe defaults.

### Performance and Scalability Reasoning (`performance_and_scalability_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When performance is relevant, reasons from workload and measurements, distinguishes latency/throughput and average/tail behavior, identifies the actual limiting resource, and evaluates algorithmic, I/O, contention, cache, and capacity effects before optimizing.

### Edge Cases and Failure Modes (`edge_cases_and_failure_modes`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When material, considers empty/missing/duplicate/malformed input, invalid state, retries, concurrent operations, partial failure, large workloads, and other boundary conditions that can change correctness.

### Assumptions and Constraints (`assumptions_and_constraints`)
- Weight: 0.05
- Applicability: Conditional
- Definition: Identifies and respects stated constraints, makes necessary assumptions explicit, recognizes missing information when it affects correctness, and does not invent requirements or environmental guarantees.

### Trade-off and Maintainability Reasoning (`tradeoff_and_maintainability_reasoning`)
- Weight: 0.05
- Applicability: Conditional
- Definition: When alternatives exist, connects the recommendation to requirements and trade-offs among simplicity, maintainability, latency, consistency, availability, testability, security, cost, scalability, and operational complexity instead of rewarding buzzwords.

### Completeness (`completeness`)
- Weight: 0.05
- Applicability: Generally applicable
- Definition: Covers the material parts of the task at the level needed to support a reliable conclusion, including required fix/verification/design elements, without requiring irrelevant detail.

### Consistency (`consistency`)
- Weight: 0.05
- Applicability: Generally applicable
- Definition: Maintains internally consistent claims, assumptions, state transitions, code behavior, and recommendations; does not contradict earlier reasoning or propose a fix that violates the stated design.

### Clarity (`clarity`)
- Weight: 0.05
- Applicability: Generally applicable
- Definition: Explains the reasoning and conclusion clearly enough to review and implement, distinguishes facts from assumptions, and uses precise terminology without unnecessary jargon or verbosity.

## Code Answers

Strong code answers should satisfy the specification, respect language semantics, handle material edge cases, avoid unintended state changes, use appropriate data structures, handle errors correctly, remain understandable and maintainable, and avoid unnecessary complexity. Do not require exact reference-solution syntax; equivalent correct implementations receive equal credit.

## Debugging Answers

Strong debugging answers should: (1) use the supplied evidence, (2) identify the likely root cause, (3) explain the failure mechanism, (4) propose a targeted fix, (5) explain how to verify the fix, and (6) consider relevant regression risks. A list of possible causes is not equivalent to root-cause analysis.

## Testing Answers

Strong testing answers choose an appropriate testing level, verify important behavior and boundaries, isolate dependencies appropriately, recognize integration risks, avoid brittle implementation-specific assertions, and address nondeterminism where relevant. Do not reward test quantity without test quality.

## Design Answers

Strong design answers identify requirements, constraints, state, ownership, interfaces, failure modes, maintainability, testing, deployment implications, and operational complexity only where relevant. Do not require elaborate architecture when a simple design satisfies the requirements.

## API Answers

Strong API answers consider request semantics, validation, response semantics, errors, compatibility, idempotency, pagination, authentication, authorization, and versioning where relevant. Do not evaluate API quality solely by conformity to a named architectural style.

## Database Answers

Strong database answers consider data integrity, keys, constraints, relationships, concurrency, transactions, migration safety, query behavior, and indexing where relevant. A correct-looking schema that cannot preserve required invariants should not receive full credit.

## Concurrency Answers

Strong concurrency answers identify shared state and atomicity requirements, reason about possible interleavings, select correct synchronization, avoid deadlock, and consider visibility where relevant. Repeatedly passing a test does not prove concurrency correctness.

## Distributed-System Answers

Strong distributed-system answers consider partial failure, retries, timeouts, duplicate delivery, idempotency, ordering, stale state, compatibility, and recovery where relevant. Do not assume exactly-once execution without an explicit mechanism that establishes the required semantics.

## Production-System Answers

Strong production-system answers distinguish immediate mitigation, diagnosis, root cause, remediation, and prevention. During an active failure, restoring service may reasonably precede complete root-cause certainty.

## Evaluation Principles

### Correctness comes first
Code that is elegant or fast but incorrect should not receive high credit.

### Sample success is not proof
Passing the supplied example does not establish general correctness.

### Edge cases matter
Strong answers consider material cases such as empty input, duplicate input, missing values, invalid state, concurrent requests, retries, partial failure, large inputs, and malformed inputs when relevant.

### Root cause matters
When root-cause analysis is requested, fixing a symptom without identifying the underlying failure should receive less credit.

### Tests are evidence
A passing test suite does not prove that the implementation is correct. Tests should verify meaningful behavior and invariants rather than merely execute lines.

### Concurrency changes reasoning
Sequentially valid code can be incorrect under overlapping execution. Answers should reason about shared state, atomicity, visibility, and interleavings where relevant.

### Distributed calls can have ambiguous outcomes
A timeout does not necessarily mean the remote operation failed. Remote side effects may have happened even when the response was lost.

### Idempotency matters
Retry safety requires reasoning about repeated effects, not merely repeated requests.

### Database integrity matters
Application correctness depends on preserving critical invariants across concurrent operations, transactions, and migrations.

### Backward compatibility matters
Software systems often run multiple application and schema versions simultaneously during rollout. Safe changes account for version overlap.

### Performance depends on workload
Do not infer performance solely from source-code appearance. Workload, contention, I/O, data shape, and latency distributions matter.

### Measure before optimizing
Strong performance answers use relevant measurements or identify the evidence needed to locate the bottleneck before recommending optimization.

### Security is part of correctness
A function that returns the correct data to an unauthorized user is not correct.

### Simple designs can be better
Do not reward unnecessary services, layers, abstractions, queues, or frameworks. A simpler design should receive full credit when it meets the requirements.

### Abstraction should reduce complexity
Indirection without a meaningful abstraction does not improve design.

### Production behavior matters
When relevant, software correctness includes deployment, configuration, failure, recovery, observability, and compatibility.

### Maintainability is contextual
Not every duplication or long function requires immediate refactoring. Evaluate the cost and benefit in context.

### No jargon worship
Using terms such as SOLID, microservices, REST, event-driven, clean architecture, DevOps, observability, idempotency, or design patterns without correct mechanism-level reasoning should not receive high credit.

### No citation worship
Citations do not compensate for incorrect code or engineering reasoning.
