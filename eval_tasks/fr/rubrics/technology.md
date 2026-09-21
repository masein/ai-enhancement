# Technology Evaluation Rubric

## Task

Score **ONE answer to ONE Technology question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, technology context, and assumptions.

Accept different valid technology selections, architectures, troubleshooting approaches, or migration strategies when they are supported by the stated requirements and evidence. Do not require a particular wording or structure. Do not reward unnecessary length, jargon, brand-name dropping, specification dumping, or citations.

This benchmark evaluates technology understanding, systems reasoning, compatibility, trade-offs, and practical judgment rather than resemblance to a reference answer.

## Critical Error

If `critical_technology_error` applies:

**score = 0**

A critical technology error is a materially false central technology, system, quantitative, compatibility, performance, reliability, or evidence claim that fundamentally invalidates the response or reverses its substantive conclusion.

Typical critical errors include fundamentally confusing RAM with persistent storage; treating Wi-Fi as identical to internet access; confusing bandwidth with latency where the distinction determines the answer; a major unit or throughput error that reverses the conclusion; claiming cloud synchronization automatically guarantees backup; assuming more CPU cores or higher clock speed always produce proportional or universal performance gains; claiming more megapixels necessarily guarantee better images; assuming wireless implies internet connectivity; materially incorrect battery energy/power reasoning; ignoring a central compatibility mismatch; treating technical connection as full interoperability; treating automation as synonymous with AI; treating a prototype as proof of commercial scalability; treating a single benchmark as universal proof; ignoring an explicit bottleneck and recommending an unrelated upgrade; equating replication/synchronization with independent backup; fabricating required product capabilities; or asserting certainty when the supplied information is clearly insufficient.

Do **not** classify harmless rounding, minor terminology slips, small omissions, defensible architecture alternatives, reasonable selection disagreements, minor specification imprecision, or uncertainty that does not materially change the conclusion as critical.

## Evaluation Criteria

All 20 criteria have equal nominal weight (`0.05`). Conditional criteria are applied when relevant to the question.

1. **Relevance** — Directly addresses the stated technology problem.
2. **Technology Accuracy** — Makes materially correct claims about technologies and system behavior.
3. **Conceptual Precision** — Correctly distinguishes easily confused concepts.
4. **Systems Understanding** — Identifies relevant components, layers, dependencies, and interactions.
5. **Quantitative Correctness** *(conditional)* — Correct calculations, units, relationships, and interpretations.
6. **Hardware Reasoning** *(conditional)* — Correct reasoning about hardware components and constraints.
7. **Software and Platform Reasoning** *(conditional)* — Correct reasoning about operating systems, applications, platforms, versions, drivers, dependencies, and updates.
8. **Connectivity and Network Reasoning** *(conditional)* — Correct reasoning about bandwidth, latency, loss, signal, coverage, congestion, and network layers.
9. **Cloud and Distributed-Technology Reasoning** *(conditional)* — Correctly handles cloud, edge, storage, synchronization, replication, backup, locality, and remote-service dependencies.
10. **Compatibility and Interoperability** *(conditional)* — Checks relevant physical, protocol, format, version, API, identity, and semantic compatibility.
11. **Performance and Bottleneck Reasoning** *(conditional)* — Finds the actual limiting component or dependency and distinguishes peak specifications from real performance.
12. **Reliability and Lifecycle Reasoning** *(conditional)* — Considers failure, redundancy, maintenance, support, migration, replacement, and end-of-life.
13. **Technology Selection and Trade-offs** *(conditional)* — Connects choices to requirements and trade-offs rather than assuming a universal best.
14. **Assumptions** *(conditional)* — Makes material assumptions explicit and avoids inventing missing facts.
15. **Evidence and Uncertainty** *(conditional)* — Distinguishes evidence, inference, uncertainty, prototype results, and benchmark limitations.
16. **Troubleshooting and Failure Analysis** *(conditional)* — Prioritizes useful diagnostic checks and separates symptoms from causes.
17. **Completeness** — Covers material requirements without irrelevant specification dumping.
18. **Consistency** — Maintains internally consistent assumptions, calculations, and conclusions.
19. **Clarity** — Explains mechanisms clearly and appropriately.
20. **Practical Technology Judgment** *(conditional)* — Balances real-world performance, usability, maintainability, reliability, privacy/security, ecosystem fit, cost conceptually, and lifecycle.

## Important Evaluation Principles

### Specifications are not performance
Peak specifications do not automatically predict real-world performance. Workload, software, thermal conditions, bottlenecks, configuration, and measurement method matter.

### More is not always better
More cores, memory, pixels, bandwidth, sensors, or features do not automatically improve the user outcome.

### Bottlenecks matter
System performance is constrained by the limiting component or dependency. Improving a non-bottleneck component may have little effect.

### Bandwidth is not latency
High bandwidth does not guarantee low delay, and low latency does not imply high capacity.

### Wi-Fi is not the internet
A device may have excellent local wireless connectivity while the external internet connection or remote service is slow or unavailable.

### RAM is not storage
Working memory and persistent storage serve different functions and have different performance characteristics.

### Cloud is not automatic backup
Synchronization, replication, storage, version history, and independent backup are different concepts.

### Compatibility is not interoperability
Two technologies may connect technically while still failing to exchange the right information or function correctly together.

### Automation is not AI
Deterministic, scheduled, or event-driven automation can operate without machine learning or artificial intelligence.

### Digital is not automatically accurate
Sensors, algorithms, and digital systems can contain measurement error, software defects, calibration problems, or incorrect assumptions.

### Local vs cloud involves trade-offs
Local processing may improve latency, privacy, and offline operation. Cloud processing may improve scale, centralized management, and compute capacity. Neither is universally preferable.

### Technology maturity matters
A successful research demonstration does not automatically establish commercial reliability, affordability, manufacturability, ecosystem readiness, or scalability.

### Reliability is not repairability
A device may fail rarely yet be difficult or expensive to repair.

### Integrated vs modular design involves trade-offs
Integration may improve size, performance, or efficiency while reducing repairability or upgradeability.

### Standards help but do not guarantee interoperability
Versions, optional features, implementation defects, profiles, and semantic differences can still create incompatibility.

### Technology adoption is contextual
Technical superiority alone does not guarantee adoption. Compatibility, ecosystem, cost, training, workflow fit, switching costs, and support matter.

### Technology decisions are lifecycle decisions
Acquisition cost is only one component. Maintenance, updates, energy, migration, support, compatibility, and replacement also matter.

### Privacy and security are not identical
A system may be secure against unauthorized access while still collecting more personal information than users expect or need.

### Multiple valid technologies
Many technology problems have several defensible solutions. Evaluate fit to requirements and trade-offs rather than requiring one preferred platform.

### No jargon worship
Using fashionable technology terms without explaining mechanisms should not receive a high score.

### No specification worship
Do not select technology solely because one headline specification is higher.

### No brand worship
Brand reputation does not substitute for analysis of requirements and capabilities.

### No citation worship
Citations do not compensate for incorrect technology reasoning.

## Score Anchors

### 4
The answer is technically accurate, directly addresses the question, identifies the relevant technological mechanisms, and reaches a well-supported conclusion.

Where relevant, it correctly handles hardware/software interactions, device constraints, connectivity, bandwidth and latency, cloud/local trade-offs, compatibility, interoperability, performance, bottlenecks, reliability, privacy/security considerations, lifecycle, and uncertainty.

For troubleshooting questions, it identifies plausible failure layers and prioritizes useful checks rather than guessing.

For technology-selection questions, it connects the recommendation to explicit requirements and trade-offs.

For quantitative questions, calculations, units, and interpretations are correct.

For emerging-technology questions, it distinguishes demonstrated capability from maturity, scalability, and commercial feasibility.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3
The answer is substantially correct and useful but contains one meaningful omission, minor technical imprecision, limited reasoning gap, incomplete qualification, or insufficient consideration of an important secondary factor. The main conclusion remains correct.

### 2
The answer is partly correct or directionally useful but has important omissions, weak technology reasoning, incomplete compatibility analysis, questionable assumptions, poor bottleneck identification, or insufficient trade-off analysis. It demonstrates meaningful understanding but requires substantial improvement.

### 1
The answer contains a substantial technology, quantitative, compatibility, performance, or systems misunderstanding, but does not meet the critical-error threshold. The response provides limited useful understanding.

### 0
Critical technology error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.
