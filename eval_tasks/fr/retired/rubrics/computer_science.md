# Computer Science Evaluation Rubric

## Task
Score ONE answer to ONE Computer Science question from 0–4.

The question metadata establishes the relevant technical domain, intent, difficulty, assumptions, and requirements.

Accept different valid approaches and implementations where appropriate.

Do not require a particular wording or structure.

Do not reward unnecessary length, jargon, or citations.

## Critical Error
If `critical_technical_error` applies:

**score = 0**

Otherwise evaluate the applicable criteria.

A critical technical error is a materially false central technical claim, algorithm, calculation, security mechanism, or system explanation that fundamentally misleads the user.

## Score Anchors

### 4
The answer is technically correct, directly addresses the problem, uses appropriate concepts and mechanisms, handles important assumptions and edge cases, and provides sound reasoning. For design questions, it presents a viable solution and accurately explains important trade-offs.

### 3
The answer is substantially correct and useful but has one meaningful omission, minor technical imprecision, limited reasoning gap, or insufficient treatment of an important edge case.

### 2
The answer is partly correct or directionally useful but contains important omissions, incomplete reasoning, weak assumptions, or an incomplete understanding of the system/problem.

### 1
The answer contains a substantial misunderstanding, inappropriate approach, or weak technical reasoning, but does not meet the critical-error threshold.

### 0
Critical technical error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Evaluation Principles

- **Correctness over verbosity:** Do not reward long answers. A concise technically correct answer can receive 4.
- **Reasoning over jargon:** Sophisticated terminology does not increase the score if the underlying reasoning is incorrect.
- **Valid alternatives:** Computer Science often has multiple valid solutions. Do not penalize a valid alternative algorithm, architecture, database, programming language, design pattern, concurrency strategy, or system design if it satisfies the stated requirements and its trade-offs are correctly explained.
- **Assumptions matter:** Evaluate whether the answer recognizes material assumptions such as input constraints, failure model, consistency requirements, workload, hardware characteristics, concurrency model, security model, data distribution, and latency requirements.
- **Security:** Do not require security analysis where it is irrelevant. When security is relevant, evaluate whether the answer correctly identifies the threat, mechanism, mitigation, and remaining limitations.
- **Debugging:** Distinguish symptom, hypothesis, root cause, evidence, and remediation. Naming a plausible symptom alone is not full-credit debugging.
- **System design:** Evaluate requirements, constraints, components, data flow, failure modes, scalability, consistency, latency, and trade-offs. There need not be one correct architecture.

## Criteria

Each criterion has equal weight (0.05). Conditional criteria apply only when relevant to the question.

### relevance — Relevance
Whether the answer directly addresses the question's stated task, constraints, and intent. Generally applies to every question.

### technical_accuracy — Technical Accuracy
Whether factual, conceptual, algorithmic, system, security, and implementation claims are technically correct. Generally applies to every question.

### reasoning — Reasoning
Whether the answer provides sound reasoning that connects premises, mechanisms, and conclusions. Generally applies to every question.

### conceptual_precision — Conceptual Precision
Whether closely related concepts are distinguished accurately and terminology is used precisely. Generally applies to every question.

### algorithmic_correctness — Algorithmic Correctness
Whether proposed or analyzed algorithms are correct for the stated problem and assumptions. Apply when relevant to the question type.

### complexity_analysis — Complexity Analysis
Whether time, space, amortized, asymptotic, or resource-complexity claims are correctly analyzed when relevant. Apply when relevant to the question type.

### system_model_selection — System Model Selection
Whether the answer selects and applies an appropriate system, concurrency, networking, database, or distributed-systems model. Apply when relevant to the question type.

### assumptions — Assumptions
Whether material assumptions, constraints, workload characteristics, and failure models are identified and used correctly. Apply when relevant to the question type.

### quantitative_correctness — Quantitative Correctness
Whether calculations, numerical estimates, equations, and quantitative conclusions are correct when required. Apply when relevant to the question type.

### implementation_correctness — Implementation Correctness
Whether implementation-level details, execution behavior, synchronization, APIs, or code-related reasoning are correct when relevant. Apply when relevant to the question type.

### architecture_and_tradeoffs — Architecture And Tradeoffs
Whether design choices satisfy requirements and constraints and accurately explain meaningful trade-offs. Apply when relevant to the question type.

### failure_mode_analysis — Failure Mode Analysis
Whether relevant failure modes, edge cases, partial failures, and recovery behavior are correctly identified. Apply when relevant to the question type.

### security_and_safety — Security And Safety
Whether security threats, mechanisms, mitigations, and limitations are handled correctly when security is relevant. Apply when relevant to the question type.

### performance_analysis — Performance Analysis
Whether bottlenecks, latency, throughput, resource utilization, caching, scaling, and performance mechanisms are analyzed correctly when relevant. Apply when relevant to the question type.

### causal_debugging — Causal Debugging
Whether the answer distinguishes symptoms, hypotheses, evidence, root causes, and remediation rather than merely naming plausible causes. Apply when relevant to the question type.

### empirical_interpretation — Empirical Interpretation
Whether measurements, benchmark results, logs, traces, and observed behavior are interpreted cautiously and causally when relevant. Apply when relevant to the question type.

### completeness — Completeness
Whether the answer covers the material aspects needed to answer the question without important omissions. Generally applies to every question.

### consistency — Consistency
Whether the answer's claims are internally consistent and do not contradict its own assumptions or conclusions. Generally applies to every question.

### clarity — Clarity
Whether the answer is understandable, well organized, and unambiguous without relying on unnecessary jargon. Generally applies to every question.

### actionability — Actionability
Whether recommendations or debugging/design steps are sufficiently concrete to be useful when the question calls for them. Apply when relevant to the question type.

## Critical Technical Error Examples

The following are examples of critical errors when central to the question:

- a fundamentally incorrect complexity claim
- reversing the behavior of a core networking/protocol mechanism
- an algorithm that cannot satisfy a central requirement
- a fundamental transaction-isolation misunderstanding that changes the conclusion
- claiming a concurrency mechanism prevents a race when it does not
- a fundamentally incorrect security defense
- a major quantitative error that changes the substantive result
- claiming a distributed system guarantees a property it cannot guarantee under the stated assumptions
- describing a core ML mechanism in the fundamentally wrong direction

Do **not** classify as critical: minor terminology mistakes, minor omissions, small arithmetic errors that do not change the conclusion, reasonable alternative implementations, stylistic differences, different but valid architectural choices, or minor code syntax mistakes when the underlying reasoning is correct.
