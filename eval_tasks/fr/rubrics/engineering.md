# Engineering Evaluation Rubric

## Task

Score **ONE answer to ONE Engineering question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, engineering context, acuity, and expected reasoning style.

Accept different valid models, derivations, designs, assumptions, and solution methods where they are technically justified. Do not require a particular derivation or design structure.

Do not reward unnecessary length, equation dumping, jargon, software-tool naming, or citations.

## Critical Error

If `critical_engineering_error` applies:

**score = 0**

A critical engineering error is a materially false central physical, mathematical, design, safety, reliability, measurement, or systems claim that fundamentally invalidates the answer or creates a severe risk of engineering failure.

Examples include central violations of conservation laws, major dimensional inconsistency, fundamentally wrong force/moment balance, ignoring a clearly governing buckling mode, treating stiffness as strength, impossible efficiency claims, invalid Bernoulli use that changes the conclusion, unsafe electrical conclusions, proposing unstable control as a fix, violating a process mass balance, ignoring a supplied safety-critical failure mode, assuming redundant elements are independent despite stated common cause, selecting a component that violates a critical requirement, a load-path failure, an assembly-breaking tolerance error, a reliability error that reverses a decision, treating a saturated or uncalibrated sensor as definitive truth, operating outside a supplied safe limit, or a lifecycle-cost error that reverses the selected alternative.

Do **not** classify as critical: minor arithmetic or rounding errors that do not change the engineering conclusion, minor notation differences, reasonable alternative models or designs, reasonable safety-factor differences where not code-mandated, small omissions, minor terminology mistakes, reasonable approximations, or consistently stated alternative conventions.

## Score Anchors

### 4
The answer is technically accurate, directly addresses the question, selects an appropriate engineering model, performs relevant calculations correctly, uses consistent units, handles important boundary conditions and assumptions, identifies relevant failure modes, and reaches a sound engineering conclusion.

For design questions, it satisfies the stated requirements and recognizes major trade-offs.

For safety-critical questions, it appropriately prioritizes safety and avoids uncontrolled failure modes.

For experimental questions, it correctly interprets measurements and uncertainty.

### 3
The answer is substantially correct and useful but contains one meaningful omission, minor engineering imprecision, limited reasoning gap, non-central calculation error, or insufficient discussion of an assumption or trade-off. The central conclusion remains technically sound.

### 2
The answer is partly correct or directionally useful but has important omissions, incomplete engineering reasoning, weak modeling assumptions, significant calculation problems, poor boundary-condition handling, or incomplete failure analysis. Meaningful correction is required.

### 1
The answer contains a substantial engineering misunderstanding, inappropriate model, major non-critical calculation error, weak design reasoning, or badly incomplete analysis, but does not meet the critical-error threshold.

### 0
Critical engineering error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question. A structural problem need not contain process-engineering analysis. A circuit question need not discuss fluid mechanics. A conceptual design question need not include numerical calculation.

Conditional criteria should not penalize an answer for omitting irrelevant material. However, when a conditional issue is essential to correctness, failure on that issue should affect the score.

## Criteria

| ID | Criterion | Weight | Conditional | Evaluation focus |
|---|---|---:|:---:|---|
| `relevance` | Relevance | 0.05 | No | Addresses the engineering question actually asked, prioritizes supplied requirements and evidence, and avoids unrelated theory, jargon, or generic advice. A technically correct discussion that solves a different problem should score poorly. |
| `engineering_accuracy` | Engineering Accuracy | 0.05 | No | Uses sound engineering principles and preserves the central physical, design, safety, reliability, and systems relationships. It distinguishes quantities such as load versus stress, strength versus stiffness, pressure versus force, power versus energy, and reliability versus availability. |
| `physical_and_mathematical_correctness` | Physical and Mathematical Correctness | 0.05 | No | Respects conservation laws, signs, directions, dimensions, physical limits, and mathematically valid reasoning. A numerically plausible result based on invalid physics, impossible efficiency, or incompatible units is not correct. |
| `quantitative_correctness` | Quantitative Correctness | 0.05 | Yes | When calculation is required, selects correct quantities and units, performs arithmetic and algebra correctly, applies conversions consistently, and interprets magnitude and sign. Minor arithmetic slips that do not change the engineering conclusion are less serious than model or physics errors. |
| `model_selection` | Model Selection | 0.05 | Yes | Chooses a governing model appropriate to the regime, purpose, and available data, and recognizes when common models such as Bernoulli, Euler buckling, ideal circuits, linear elasticity, lumped capacitance, or linear control assumptions are invalid or incomplete. |
| `assumptions_and_boundary_conditions` | Assumptions and Boundary Conditions | 0.05 | Yes | States material assumptions, selects defensible system boundaries, and applies correct constraints, supports, inlet/outlet conditions, initial conditions, and operating limits. Reasonable approximations are acceptable when justified; invented facts or a wrong governing boundary condition are not. |
| `mechanics_and_structural_reasoning` | Mechanics and Structural Reasoning | 0.05 | Yes | When relevant, correctly handles free-body diagrams, forces, moments, stress, strain, deformation, stiffness, stability, load paths, fatigue, fracture, buckling, and serviceability, including the governing failure mode rather than only nominal stress. |
| `thermal_fluid_reasoning` | Thermal-Fluid Reasoning | 0.05 | Yes | When relevant, correctly uses fluid, thermodynamic, and heat-transfer principles, including continuity, pressure/head distinctions, losses, pumps, energy balances, state/property logic, thermal resistance, and physical performance limits. |
| `electrical_and_control_reasoning` | Electrical and Control Reasoning | 0.05 | Yes | When relevant, correctly reasons about voltage, current, power, impedance, circuit laws, transients, electronics, sampling, feedback, stability, sensors, actuators, saturation, protection, and timing rather than relying on symbol or terminology recall. |
| `process_and_mass_energy_reasoning` | Process and Mass-Energy Reasoning | 0.05 | Yes | When relevant, preserves total and component mass balances, energy balances, accumulation, recycle/purge logic, reaction and separation distinctions, residence time, process dynamics, and the difference between steady state and equilibrium. |
| `design_reasoning` | Design Reasoning | 0.05 | Yes | When relevant, identifies requirements, constraints, loads, operating conditions, material/component limits, manufacturability, maintenance, safety, reliability, and verification needs. Valid alternative designs receive full credit when technically justified and compliant with requirements. |
| `failure_and_root_cause_reasoning` | Failure and Root-Cause Reasoning | 0.05 | Yes | When relevant, separates symptoms, failed parts, contributing factors, initiating events, and underlying causes; identifies plausible mechanisms; seeks discriminating evidence; and proposes corrective action that prevents recurrence rather than random part replacement. |
| `safety_and_risk_reasoning` | Safety and Risk Reasoning | 0.05 | Yes | When relevant, identifies hazards, consequences, critical limits, safeguards, safe states, human factors, and failure of safeguards. It prioritizes prevention and does not assume that a safety factor, redundancy, alarm, or interlock alone guarantees safety. |
| `reliability_and_maintainability` | Reliability and Maintainability | 0.05 | Yes | When relevant, distinguishes reliability, availability, maintainability, MTBF, MTTR, preventive maintenance, and condition-based maintenance; models series/parallel behavior correctly; and recognizes dependence and common-cause failure in redundant systems. |
| `systems_and_interface_reasoning` | Systems and Interface Reasoning | 0.05 | Yes | When relevant, identifies subsystem interfaces, shared resources, dependencies, integration constraints, emergent behavior, failure propagation, verification/validation boundaries, and cases where local component optimization degrades overall system performance. |
| `experimental_and_measurement_reasoning` | Experimental and Measurement Reasoning | 0.05 | Yes | When relevant, distinguishes accuracy, precision, resolution, bias, drift, noise, calibration, uncertainty, sampling, repeatability, and representativeness. It does not treat a saturated, uncalibrated, or out-of-envelope measurement or simulation as definitive truth. |
| `tradeoff_and_optimization_reasoning` | Trade-Off and Optimization Reasoning | 0.05 | Yes | When relevant, distinguishes feasibility, optimality, robustness, and safety; recognizes competing objectives and uncertainty; and explains why a mathematically optimal component or nominal design may be impractical or inferior at system level. |
| `completeness` | Completeness | 0.05 | No | Covers the material steps needed to support the conclusion, including important constraints, assumptions, checks, failure modes, or evidence requested by the prompt, without requiring irrelevant exhaustive treatment. |
| `consistency` | Consistency | 0.05 | No | Maintains internally consistent assumptions, conventions, units, signs, numerical values, and conclusions throughout the answer. A stated convention may differ from another valid convention if it is applied consistently. |
| `clarity` | Clarity | 0.05 | No | Communicates the engineering reasoning in a precise, organized, and auditable way so that equations, assumptions, intermediate logic, and conclusions can be followed. Unnecessary length, equation dumping, software names, jargon, or citations do not increase credit. |

## Important Evaluation Principles

### Physics constrains engineering
Engineering answers must respect relevant physical laws.

### Units matter
Quantitative results must be dimensionally consistent.

### Model validity matters
A mathematically correct calculation can still be wrong if the model is inappropriate.

### Boundary conditions matter
Incorrect constraints or boundary conditions can invalidate otherwise correct calculations.

### Strength is not stiffness
These properties must not be treated as interchangeable.

### Power is not energy
Engineering answers should preserve this distinction.

### Reliability is not availability
Reliability, maintainability, and availability measure different aspects of system performance.

### Redundancy does not guarantee safety
Redundant components may share common failure causes.

### Nominal performance is not robust performance
Strong designs should account for relevant variation and uncertainty.

### Failure modes matter
The governing failure mode may not be the most obvious one.

### Safety precedes optimization
A design should not be optimized for performance while violating critical safety constraints.

### Test data have limits
Validation applies only within the justified operating envelope.

### Simulation is not reality
FEA, CFD, circuit simulation, process simulation, and other numerical models depend on assumptions, inputs, boundary conditions, model or mesh resolution, constitutive models, and validation. A simulated result should not automatically be treated as physical truth.

### System optimization differs from component optimization
Improving one component can worsen overall system behavior.

### Engineering decisions involve trade-offs
Performance, cost, weight, efficiency, safety, maintainability, reliability, environmental impact, and schedule may conflict.

### Multiple valid designs exist
Do not require one predetermined design if multiple solutions satisfy requirements.

### Uncertainty should be recognized
Strong answers should identify material uncertainty without adding irrelevant caveats.

### No jargon worship
Using engineering terms does not compensate for incorrect reasoning.

### No citation worship
Citations do not compensate for incorrect physics, calculations, or design reasoning.

## Quantitative Answers

Strong quantitative answers should, where appropriate:
- define the system;
- select the correct governing relationship;
- state important assumptions;
- use consistent units;
- perform calculations correctly;
- check magnitude;
- test physical plausibility; and
- interpret the result.

Do not require every trivial arithmetic step. A correct number derived from invalid physics should not receive full credit.

## Design Answers

Strong design answers should identify, where relevant:
- requirements;
- constraints;
- operating conditions;
- loads;
- failure modes;
- material or component limits;
- safety;
- reliability;
- manufacturability;
- maintenance; and
- trade-offs.

Do not require unnecessary complexity, and accept technically justified alternative designs that satisfy the stated requirements.

## Failure-Analysis Answers

Strong answers should:
1. describe the observed failure;
2. identify plausible mechanisms;
3. use supplied evidence;
4. distinguish root causes from symptoms;
5. identify discriminating tests; and
6. propose corrective action.

Do not reward a long list of unsupported failure possibilities.

## Safety Answers

Strong answers should:
- identify hazards;
- identify consequences;
- recognize critical limits;
- propose appropriate safeguards;
- consider failure of safeguards; and
- avoid unsafe assumptions.

A design that meets nominal performance but creates unacceptable hazard should not receive high credit.

## Systems Answers

Strong systems answers should identify relevant:
- interfaces;
- dependencies;
- subsystem interactions;
- shared resources;
- failure propagation; and
- trade-offs.

Do not optimize components independently when system interaction is central.

## Experimental Answers

Strong answers should consider:
- what is measured;
- calibration;
- uncertainty;
- controls;
- test conditions;
- repeatability;
- whether test conditions represent service conditions; and
- what conclusions the experiment actually supports.

Do not claim validation beyond the tested or otherwise justified operating envelope.
