# Systems & Cybersecurity Evaluation Rubric

## Task

Score **ONE answer to ONE Systems & Cybersecurity question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, acuity, architecture context, and assumptions.

Accept different valid system architectures, security controls, diagnostic approaches, and incident-response strategies where they are supported by the stated evidence and constraints.

Do not require a particular wording or structure. Do not reward unnecessary length, jargon, vendor-name dropping, certification terminology, or citations. The benchmark evaluates systems understanding, defensive security reasoning, reliability, risk analysis, and practical judgment rather than resemblance to a reference answer.

## Critical Error

If `critical_systems_cybersecurity_error` applies:

**score = 0**

A critical Systems & Cybersecurity error is a materially false central systems, networking, security, cryptographic, reliability, quantitative, or incident-response claim that fundamentally invalidates the answer or creates a materially unsafe conclusion.

Examples include:

- fundamentally confusing authentication and authorization in a way that changes the security conclusion
- recommending blindly retrying a non-idempotent operation where duplicate execution causes material harm
- claiming replication is equivalent to backup as the central recovery strategy
- fundamentally incorrect routing or networking reasoning that invalidates the diagnosis
- claiming a digital signature provides confidentiality
- treating hashing and reversible encryption as equivalent
- claiming strong encryption makes access control unnecessary
- claiming a firewall alone secures an application with a known authorization flaw
- fundamentally misinterpreting vulnerability severity as complete risk without considering exposure or impact
- a major availability calculation error that reverses the architecture conclusion
- recommending deletion or destruction of critical evidence during an active incident without justification
- treating absence of alerts as proof no compromise occurred
- claiming MFA makes account compromise impossible
- recommending a single shared privileged credential as a secure architecture
- fundamentally misinterpreting RTO and RPO
- treating a snapshot as a complete and independently recoverable backup without qualification
- ignoring a clearly stated correlated failure that invalidates redundancy assumptions
- fundamentally incorrect containment reasoning that materially worsens an incident
- fabricating logs, system state, or security evidence required to reach a conclusion
- providing an unsafe recommendation based on an invalid security assumption

Do **not** classify as critical merely because of minor arithmetic errors that do not change the conclusion, harmless approximation differences, minor terminology mistakes, reasonable architecture differences, defensible alternative controls, reasonable risk-prioritization differences, legitimate incident-response trade-offs, small omissions, reasonable uncertainty, or minor imprecision without material systems or security consequence.

## Score Anchors

### 4

The answer is technically accurate, directly addresses the question, identifies the relevant systems or security mechanisms, and reaches a well-supported conclusion.

Where relevant, it correctly handles architecture, dependencies, concurrency, network flows, failure domains, retries, identity, authorization, cryptography concepts, trust boundaries, vulnerabilities, risk, monitoring, incident response, recovery, and uncertainty.

For systems-diagnosis questions, it uses evidence to isolate plausible failure points rather than guessing.

For security questions, it distinguishes threat, vulnerability, exposure, control, and risk.

For incident questions, it appropriately prioritizes containment, continuity, evidence, and recovery.

For quantitative questions, calculations, units, and assumptions are correct.

For architecture questions, it identifies relevant trade-offs among reliability, security, performance, cost, usability, and operational complexity.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor technical imprecision, limited diagnostic gap, incomplete qualification, or insufficient consideration of an important secondary factor.

The main conclusion remains correct and safe.

### 2

The answer is partly correct or directionally useful but has important omissions, weak systems reasoning, incomplete security analysis, questionable assumptions, poor risk interpretation, or incomplete failure diagnosis.

It demonstrates meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial systems, networking, security, reliability, cryptographic, or incident-response misunderstanding, but does not meet the critical-error threshold.

The response provides limited useful understanding.

### 0

Critical Systems & Cybersecurity error, no answer, off-topic answer, unrelated fabrication, unsafe operational content, or equivalent fundamental failure.

## Evaluation Criteria

All 20 criteria have equal weight: **0.05**. Conditional criteria are scored only when relevant to the question.

| Criterion | Conditional | Operational definition |
|---|---:|---|
| `relevance` — Relevance | No | Directly addresses the question asked, focuses on the stated system or security problem, and avoids tangents that do not help resolve the task. |
| `systems_accuracy` — Systems Accuracy | No | Uses materially correct systems concepts, including processes, memory, storage, concurrency, performance, operating systems, dependencies, and failure behavior when those concepts are relevant. |
| `cybersecurity_accuracy` — Cybersecurity Accuracy | No | Uses materially correct defensive-security concepts and does not confuse authentication, authorization, threats, vulnerabilities, risk, cryptographic functions, controls, alerts, incidents, or recovery concepts. |
| `conceptual_precision` — Conceptual Precision | No | Makes distinctions that matter to the conclusion, uses terminology consistently, and explains mechanisms rather than substituting jargon for reasoning. |
| `systems_reasoning` — Systems Reasoning | No | Reasons across interacting components, dependencies, resource constraints, state, timing, and failure modes instead of analyzing components in isolation when system interactions matter. |
| `quantitative_correctness` — Quantitative Correctness | Yes | When quantities are involved, selects the relevant metric, preserves units, performs calculations correctly, states assumptions, and interprets the result in context. |
| `network_reasoning` — Network Reasoning | Yes | When networking is relevant, correctly reasons about layers, routing, addressing, transport behavior, DNS, segmentation, latency, loss, stateful controls, and network paths without relying on irrelevant trivia. |
| `distributed_systems_reasoning` — Distributed Systems Reasoning | Yes | When distributed behavior is relevant, correctly handles partial failure, consistency, availability, retries, idempotency, replication, ordering, queues, partitions, quorums, and correlated dependencies. |
| `identity_and_access_reasoning` — Identity and Access Reasoning | Yes | When identity or access is relevant, correctly distinguishes authentication from authorization and applies least privilege, credential lifecycle, service identity, session, role, and deprovisioning concepts. |
| `security_architecture_reasoning` — Security Architecture Reasoning | Yes | When architecture is relevant, identifies trust boundaries, attack surfaces, layered controls, shared dependencies, fail-safe defaults, network paths, and recovery boundaries, and explains trade-offs among valid designs. |
| `threat_and_risk_reasoning` — Threat and Risk Reasoning | Yes | When risk is relevant, distinguishes threat, vulnerability, exposure, likelihood, impact, control, and residual risk, and prioritizes using context rather than severity labels alone. |
| `incident_and_detection_reasoning` — Incident and Detection Reasoning | Yes | When incidents or detections are relevant, distinguishes alerts from confirmed compromise, uses evidence for triage, balances containment and continuity, preserves evidence, and reasons through recovery and lessons learned. |
| `reliability_and_resilience_reasoning` — Reliability and Resilience Reasoning | Yes | When reliability is relevant, correctly reasons about availability, durability, independent redundancy, failover, graceful degradation, backups, RTO/RPO, recovery testing, dependencies, and correlated failure. |
| `assumptions` — Assumptions | Yes | Identifies material assumptions, avoids silently assuming independence, trust, perfect visibility, or guaranteed behavior, and explains how conclusions depend on uncertain or unstated conditions. |
| `evidence_and_uncertainty` — Evidence and Uncertainty | Yes | Uses the evidence supplied, distinguishes facts from hypotheses, avoids fabricated certainty, recognizes incomplete or untrustworthy telemetry, and identifies what additional evidence would change confidence. |
| `failure_analysis_and_debugging` — Failure Analysis and Debugging | Yes | When diagnosis is required, separates symptoms from causes, forms plausible hypotheses, prioritizes discriminating checks, avoids changing many variables at once, and accounts for dependencies and amplification mechanisms. |
| `completeness` — Completeness | No | Covers the material parts of the task needed for a useful answer, including major constraints, mechanisms, risks, and requested calculations or recommendations without unnecessary expansion. |
| `consistency` — Consistency | No | Maintains internally consistent claims, assumptions, units, threat models, architecture statements, and conclusions throughout the answer. |
| `clarity` — Clarity | No | Communicates the reasoning in a structured and understandable way, makes causal relationships explicit, and avoids ambiguity that would change the technical interpretation. |
| `practical_systems_security_judgment` — Practical Systems & Security Judgment | Yes | When a decision is required, balances security, reliability, performance, cost, usability, operational constraints, reversibility, evidence preservation, and recovery implications rather than applying controls mechanically. |

## Important Evaluation Principles

### Systems are composed of interacting components

Do not diagnose one component in isolation when the evidence points to dependency or system-level interactions.

### Partial failure matters

Distributed systems can be partly available, inconsistent, delayed, or degraded rather than simply “up” or “down.”

### Redundancy is not backup

Replication can replicate corruption or deletion. Backups should support recovery from historical failure states.

### Redundancy must be independent

Two redundant components sharing the same power source, credentials, control plane, network, storage, or other failure domain may not provide meaningful independence.

### Retries can make failures worse

Retries can increase load, duplicate side effects, and amplify outages. Idempotency, retry limits, backoff, and backpressure matter.

### Latency is not throughput

A system can have high throughput and poor latency, or low throughput and good latency.

### Authentication is not authorization

Knowing who a user is does not determine what that user should be allowed to do.

### Encryption is not complete security

Encryption does not replace authorization, endpoint security, key management, secure configuration, or monitoring.

### Hashing is not encryption

Hashes are generally one-way integrity or derivation mechanisms, not reversible confidentiality mechanisms.

### Severity is not risk

Security risk depends on exposure, exploitability, asset value, impact, likelihood, and controls, not only technical severity.

### Prevention is not detection

Preventive controls can fail. Detection and response remain necessary.

### Alerts are not incidents

An alert is evidence requiring investigation, not automatic proof of compromise.

### Absence of evidence is not evidence of absence

A lack of logs or detections may reflect insufficient visibility or missing telemetry.

### Least privilege

Accounts, services, and systems should have only the access needed for their intended function.

### Defense in depth

Security should not depend on a single control. Strong answers explain what each layer prevents, detects, contains, or helps recover from.

### Trust boundaries matter

Strong answers identify where data, identity, authority, or control crosses from one trust domain to another.

### Backups must be restorable

Backup existence does not establish recovery readiness. Restore testing, credential access, integrity, dependencies, and recovery procedures matter.

### Shared responsibility

Using cloud or managed infrastructure does not remove customer responsibilities for identity, configuration, data, application behavior, monitoring, and access control.

### Security is contextual

The appropriate control depends on assets, threat model, exposure, cost, usability, operational requirements, and recovery implications.

### Human review is not automatically effective

Human oversight can fail because of fatigue, lack of expertise, poor tooling, missing context, or unclear escalation.

### Multiple valid architectures

Systems and security problems often have several defensible solutions. Evaluate reasoning, constraints, risks, and trade-offs rather than requiring one predetermined design.

### No jargon worship

Using systems or security terminology without correct reasoning should not receive a high score.

### No compliance worship

Meeting a checklist or standard does not itself prove that a system is secure or resilient.

### No tool worship

Deploying a firewall, SIEM, EDR, scanner, or backup product does not by itself establish effective security, monitoring, or recovery.

### No citation worship

Citations do not compensate for incorrect systems or cybersecurity reasoning.

## Defensive Safety Boundary

This benchmark evaluates defensive, analytical, architectural, monitoring, mitigation, incident-response, secure-configuration, risk, recovery, and evidence-interpretation skills.

Do not reward answers that introduce unnecessary harmful operational detail such as weaponized exploit code, malware, ransomware-development instructions, credential-stealing procedures, stealth or persistence techniques, evasion techniques, destructive commands, or unauthorized-access procedures.

When a question concerns a vulnerability, the desired reasoning is about cause, exposure, evidence, mitigation, prioritization, containment, or recovery.

## Applying Conditional Criteria

Apply a conditional criterion only when the question materially invokes that capability. For example, do not penalize a purely conceptual authentication answer for omitting a quantitative calculation, and do not require distributed-systems analysis where no distributed behavior is relevant.

Broad criteria such as relevance, systems accuracy, cybersecurity accuracy, conceptual precision, systems reasoning, completeness, consistency, and clarity generally apply across the benchmark.

## Judgment and Uncertainty

Do not force certainty when the prompt is underdetermined. Strong answers state what is known, what is inferred, what remains uncertain, and what additional evidence would discriminate between plausible explanations.

Do not assume logs are complete or trustworthy, redundancy is independent, scans are ground truth, alerts prove compromise, or backups are recoverable unless the prompt supports those assumptions.
