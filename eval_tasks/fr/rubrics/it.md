# IT Evaluation Rubric

## Task

Score **ONE answer to ONE IT question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, operational context, acuity, and expected reasoning style. Accept different valid troubleshooting approaches, configurations, architectures, recovery strategies, and operational decisions where they satisfy the stated requirements.

Do not require a particular vendor, command, architecture, or troubleshooting sequence unless the question specifies one. Do not reward unnecessary length, jargon, command dumping, or citations.

## Critical Error

If `critical_it_error` applies:

**score = 0**

A critical IT error is a materially false central technical, operational, security, networking, recovery, access-control, or quantitative claim that fundamentally invalidates the answer or would create severe operational, security, availability, or data-loss risk.

Representative critical errors include:

- deleting, overwriting, or destructively reconfiguring the only viable production copy of data
- treating RAID, replication, or a VM snapshot as a complete substitute for independent backup
- claiming an untested backup guarantees recoverability
- fundamentally incorrect RPO/RTO reasoning that invalidates a recovery design
- broadly and indefinitely disabling a major security control as the principal fix
- granting excessive administrative access to solve a routine permission problem
- confusing authentication with authorization in the central access conclusion
- fundamentally incorrect subnetting or routing reasoning that reverses the diagnosis
- creating an addressing conflict or routing loop as the proposed fix
- ignoring a compromised privileged identity during an incident
- treating an actively compromised endpoint as safe solely because visible malware was removed
- exposing protected services through materially incorrect firewall reasoning
- deploying a high-risk, untested production change with no viable rollback where failure would be severe
- misunderstanding cloud shared responsibility so severely that critical customer responsibilities are left unaddressed
- asserting that redundancy eliminates the need for backup or disaster recovery
- sequencing recovery so that downstream services are restored before indispensable upstream dependencies
- making a major capacity error that makes the proposed design infeasible
- claiming service restoration without verification when verification is central to the task
- treating repeated rebooting as proof of root cause
- making a central claim contradicted by the supplied logs, configuration, topology, or timeline

Do **not** classify as critical merely because of minor syntax mistakes, small arithmetic errors that do not change the conclusion, minor terminology errors, reasonable alternative troubleshooting orders, different valid architectures, small omissions, reasonable vendor-specific differences, cautious uncertainty, or non-central assumptions that are clearly stated.

## Score Anchors

### 4

The answer is technically accurate, directly addresses the question, correctly identifies relevant dependencies and failure domains, uses the available evidence appropriately, proposes a safe and effective solution, and handles important operational risks.

For troubleshooting questions, it narrows the cause logically rather than guessing. For networking questions, it correctly reasons about addressing, routing, DNS, switching, transport, and policy where relevant. For security questions, it protects systems and accounts while addressing the actual incident. For backup/recovery questions, it distinguishes backup, replication, recovery objectives, and restore verification. For design questions, it explains requirements and trade-offs.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor technical imprecision, limited diagnostic gap, or insufficient consideration of a dependency or operational risk. The main conclusion remains correct and safe.

### 2

The answer is partly correct or directionally useful but has important omissions, weak troubleshooting logic, significant technical mistakes, inappropriate assumptions, incomplete dependency analysis, or a solution that requires meaningful correction.

### 1

The answer contains a substantial IT misunderstanding, poor troubleshooting approach, major non-critical configuration error, inappropriate architecture, or badly incomplete reasoning, but does not meet the critical-error threshold.

### 0

Critical IT error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question.

A subnetting question need not discuss disaster recovery. A backup question need not discuss Wi-Fi. A routine endpoint-support question need not contain architecture analysis. Conditional criteria should not penalize answers for omitting irrelevant material.

However, if a conditional issue is essential to correctness or safety, failure on that issue should affect the score.

## Criteria

| Criterion | Applicability | Definition |
|---|---|---|
| `relevance` | General | Directly addresses the question asked, the supplied scope, and the operational objective. Does not substitute generic IT advice for the actual failure mode or decision. |
| `technical_accuracy` | General | Uses correct IT concepts and mechanisms. Central claims about systems, networks, identity, security, storage, cloud, recovery, or operations must be technically sound; major errors weigh far more than minor terminology slips. |
| `troubleshooting_reasoning` | Conditional | When troubleshooting is required, establishes scope, uses evidence, forms plausible hypotheses, chooses discriminating and low-disruption tests, avoids random rebooting or broad configuration changes, and verifies restoration. |
| `systems_administration_reasoning` | Conditional | Correctly reasons about operating-system state, services/processes, users/groups, permissions, filesystems, startup, logs, updates, drivers, resource use, and local versus system-wide configuration when relevant. |
| `networking_reasoning` | Conditional | Correctly reasons about source/destination, addressing, subnet boundaries, gateways, routing, switching, DNS, DHCP, transport, VLANs, firewalls, VPNs, directionality, and where connectivity fails. Major subnetting or routing reversals are severe. |
| `identity_and_access_reasoning` | Conditional | Correctly separates identity, authentication, authorization, effective permissions, roles/groups, synchronization/replication, account state, and device state; applies least privilege and does not treat successful login as universal entitlement. |
| `cloud_and_virtualization_reasoning` | Conditional | Correctly reasons about VMs, hosts, overcommitment, snapshots, containers, cloud service models, zones/regions, scaling, managed services, hybrid dependencies, and shared responsibility without assuming cloud or virtualization automatically provides backup or disaster recovery. |
| `security_reasoning` | Conditional | Identifies the actual security risk, distinguishes detection from confirmed compromise, contains active harm proportionately, protects identities and trust boundaries, preserves relevant evidence where appropriate, and avoids weakening controls as a routine fix. |
| `backup_recovery_and_continuity` | Conditional | Correctly distinguishes backup, replication, snapshots, high availability, RPO, RTO, restore testing, copy independence, dependency-aware recovery sequencing, retention, and business continuity. |
| `monitoring_and_incident_reasoning` | Conditional | Interprets alerts, logs, metrics, baselines, thresholds, health checks, and incident evidence as signals rather than automatic root causes; separates immediate mitigation/service restoration from later root-cause work. |
| `configuration_and_change_reasoning` | Conditional | When configuration or change is relevant, considers intended versus effective state, drift, blast radius, testing, maintenance timing, dependencies, reversibility, rollback feasibility, and post-change validation. |
| `quantitative_correctness` | Conditional | Calculations use correct units, assumptions, magnitude, and interpretation. Small arithmetic slips that do not change the conclusion are minor; major subnet, capacity, availability, backup, licensing, or timing errors that invalidate the design are severe. |
| `root_cause_and_dependency_analysis` | Conditional | Distinguishes symptoms, triggers, contributing factors, shared upstream dependencies, workarounds, and root causes. A correct-looking fix based on an unsupported diagnosis should not receive full credit. |
| `risk_and_business_impact` | Conditional | Recognizes user/business impact, criticality, operational burden, security exposure, downtime, lifecycle or vendor risk, and residual risk; does not equate technical severity with business priority. |
| `solution_safety_and_reversibility` | Conditional | Proposed actions minimize avoidable outage, data loss, privilege expansion, or security weakening; preserve recovery options; and include rollback or staged implementation when material production changes are involved. |
| `assumptions_and_uncertainty` | Conditional | Separates observed facts from hypotheses, identifies missing information that materially affects the answer, avoids speculative certainty, and appropriately recognizes when more evidence is needed. |
| `architecture_and_tradeoff_reasoning` | Conditional | For design decisions, connects requirements and constraints to availability, performance, identity, security, recovery, maintainability, cost where supplied, migration risk, and operational complexity; accepts multiple defensible architectures. |
| `completeness` | General | Covers the material parts needed to answer the question safely and usefully without requiring irrelevant domain discussion. Important omissions affecting correctness, verification, dependencies, or recovery reduce the score. |
| `consistency` | General | The answer's claims, calculations, diagnosis, and proposed actions do not contradict each other or the supplied logs, configuration, topology, or timeline. |
| `clarity` | General | Explains reasoning and actions in a clear, operationally usable way. Jargon, command dumping, or citations do not substitute for an intelligible explanation of mechanism, evidence, dependencies, and trade-offs. |

All 20 criteria have equal nominal weight: **0.05**. Conditional criteria are applied only when relevant to the question.

## Evaluation Principles

### Troubleshoot from evidence

Do not guess when scope, logs, timing, configuration, topology, or comparison data can discriminate among explanations. Prefer tests that eliminate multiple hypotheses with minimal disruption.

### Scope matters

Strong answers should determine whether a problem affects one user, one device, one subnet, one site, one application, one dependency, or the whole organization before proposing broad changes.

### Cause is not symptom

Restarting a service may restore functionality without explaining why it failed. A workaround can be operationally reasonable while still being distinct from root-cause remediation.

### Authentication is not authorization

Successful login establishes identity/authentication state; it does not by itself prove entitlement to a resource. Group/role membership, effective permissions, policy, synchronization, and resource ACLs may still deny access.

### DNS is not the entire network

Name-resolution failure must be distinguished from address assignment, switching, routing, firewall, transport, and application failure. If IP connectivity works and hostname access fails, DNS becomes more plausible; it is not automatically the answer to every connectivity problem.

### Redundancy is not backup

Redundant copies can replicate deletion, corruption, malicious encryption, or configuration mistakes. RAID, clustering, synchronous replication, and multiple availability zones do not by themselves provide historical independent recovery copies.

### Backup is not recovery

A successful backup job is evidence that data was written somewhere, not proof that the organization can restore a consistent service within its objectives. Strong answers consider restore testing, integrity, dependencies, credentials, documentation, and recovery order.

### RPO is not RTO

RPO concerns acceptable data loss measured backward from disruption. RTO concerns acceptable time to restore service. Meeting one objective does not imply meeting the other.

### High availability is not disaster recovery

Local component redundancy may not protect against site-wide failure, common dependencies, logical corruption, operator error, or a region-wide outage.

### Cloud is shared responsibility

Provider-managed infrastructure does not remove all customer responsibility for identity, configuration, data, permissions, backup/recovery, endpoint state, network policy, monitoring, or application architecture where relevant.

### Security is part of operations

Operational fixes should not unnecessarily weaken security controls. Strong answers choose narrow, time-bounded, logged, and reversible tests instead of broadly disabling firewalls, MFA, endpoint controls, or access restrictions.

### Least privilege

Permissions should be sufficient for the task but no broader than necessary. Routine access problems should not be solved by granting broad administrator privileges.

### Changes need rollback thinking

For material production changes, consider blast radius, test evidence, reversibility, rollback feasibility, maintenance windows, communication, dependencies, and post-change validation.

### Dependencies matter

A downstream service may be unable to recover before identity, DNS, network, storage, database, certificate, or other upstream dependencies are available. Hidden shared dependencies can defeat apparent redundancy.

### Monitoring signals require interpretation

An alert means a threshold or condition was observed; it is not automatically the root cause. Strong answers correlate logs, metrics, traces/events, synthetic checks, baselines, and user impact.

### Availability is end-to-end

Individual components can appear healthy while the user-facing service is unavailable because an upstream dependency or integration path is failing.

### Performance requires measurement

Do not optimize from intuition alone. Distinguish utilization from saturation, throughput from latency, averages from short peaks, and local resource health from shared-resource contention.

### Capacity and utilization differ

Moderate average utilization can hide short critical peaks. Headroom, queueing, concurrency, latency, growth, and failure-mode behavior may matter more than a single utilization percentage.

### Incidents and problems differ

Incident management restores or mitigates service. Problem management seeks recurring causes and durable remediation. Root-cause certainty is not required before safe mitigation of an active outage.

### Multiple valid solutions

Infrastructure questions often admit several defensible architectures or operational responses. Evaluate whether the answer satisfies the requirements and explains trade-offs rather than requiring one predetermined design.

### No jargon worship

Terms such as “zero trust,” “hybrid cloud,” “defense in depth,” “high availability,” “observability,” or “automation” should receive credit only when the answer explains the actual mechanism, dependency, control, or trade-off.

### No citation worship

Citations do not compensate for incorrect IT reasoning. An uncited but technically sound answer can score highly when citations were not requested.

## Troubleshooting Answers

Strong troubleshooting answers should generally:

1. establish scope
2. identify relevant recent changes
3. gather useful evidence
4. form plausible hypotheses
5. test the least disruptive and most discriminating hypotheses
6. implement an appropriate fix
7. verify actual user-facing service restoration
8. document or monitor after the fix when appropriate

Do not require a rigid sequence when another evidence-based sequence is more efficient.

## Networking Answers

Where appropriate, strong networking answers identify source and destination, inspect addressing and subnet boundaries, identify gateway/routing behavior, distinguish DNS from IP connectivity, consider switching/VLAN state, evaluate firewall policy and directionality, and identify where the path fails.

Do not reward broad configuration changes before the failure domain is narrowed.

## Identity and Access Answers

Where appropriate, strong answers consider identity, authentication, authorization, group/role membership, effective permissions, synchronization/replication, account state, session/token state, and device state. Do not solve routine access problems by indiscriminately granting administrative privilege.

## Security Answers

Where appropriate, strong security-operational answers identify scope, contain active risk, protect identities, preserve relevant evidence, remediate the actual weakness, restore systems safely, and verify recovery. Do not require exhaustive forensic certainty before containing active harm.

## Backup and Recovery Answers

Where appropriate, strong answers consider the recovery objective, backup frequency, retention, copy independence, immutability or isolation where relevant, restore integrity, dependencies, recovery order, credentials, documentation, and testing.

Do not treat backup-job success alone as proof of recoverability.

## Incident Answers

Strong incident answers distinguish immediate mitigation, service restoration, diagnosis, root cause, permanent remediation, and prevention. During a major active outage, safe mitigation of user impact may reasonably precede full root-cause certainty.

## Architecture Answers

Where relevant, strong design answers identify business requirements, availability, performance, identity, network dependencies, security, recovery, operational complexity, migration implications, vendor/lifecycle risk, and uncertainty. Do not require elaborate infrastructure where a simpler solution satisfies the stated requirements.
