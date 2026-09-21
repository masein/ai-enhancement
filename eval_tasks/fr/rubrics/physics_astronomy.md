# Physics & Astronomy Evaluation Rubric

## Task

Score **ONE answer to ONE Physics & Astronomy question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, physical context, and expected reasoning style.

Accept different valid derivations, approximations, coordinate systems, or modeling approaches when they are physically justified. Do not require a particular wording or solution structure. Do not reward unnecessary length, equation dumping, jargon, or citations.

## Critical Error

If `critical_physics_astronomy_error` applies:

**score = 0**

A critical physics/astronomy error is a materially false central physical, mathematical, observational, or astronomical claim that fundamentally invalidates the answer or reverses the substantive conclusion. Minor arithmetic slips, small rounding differences, harmless notation choices, consistent alternative sign conventions, reasonable approximations, and defensible alternative derivations are not critical merely because they differ from a reference solution.

Otherwise evaluate the applicable criteria.

## Score Anchors

### 4

The answer is accurate, directly addresses the question, uses the appropriate physical or astronomical principles, performs relevant calculations correctly, handles units and signs consistently, states important assumptions where needed, and reaches a physically sound conclusion.

For quantitative questions, the method and result are correct to an appropriate precision.

For conceptual questions, the explanation identifies the governing mechanism rather than merely naming a law or formula.

For astronomy questions, the answer correctly distinguishes observation from model-dependent inference and does not claim more than the supplied evidence supports.

For experimental or data questions, it correctly handles uncertainty, systematic effects, and relevant limitations.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor conceptual imprecision, limited reasoning gap, non-central calculation error, insufficient discussion of assumptions, or incomplete qualification.

The central physical conclusion remains correct.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete physical reasoning, a materially weak approximation, an incorrect intermediate interpretation, significant mathematical problems, or incomplete treatment of relevant evidence.

The answer demonstrates some understanding but cannot be considered reliably correct.

### 1

The answer contains a substantial physics or astronomy misunderstanding, inappropriate model, major non-critical calculation error, serious dimensional/sign problem, or badly incomplete reasoning, but does not meet the critical-error threshold.

### 0

Critical physics/astronomy error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question.

A purely conceptual question need not contain numerical calculation. A straightforward mechanics problem need not discuss astronomical accuracy. A question with no experimental uncertainty need not receive artificial uncertainty analysis.

Conditional criteria should not penalize answers for omitting material that the problem does not call for. However, when a conditional issue is essential to solving the problem correctly, failure on that issue should affect the score.

The criteria file assigns equal nominal weight to 20 criteria. In a per-question evaluation, apply only criteria relevant to that question; do not manufacture opportunities to assess a conditional criterion.

## Physical Correctness

The answer must obey the relevant physical laws and use equations only within their valid domains.

A correct-looking formula is not enough: the response should connect the equation to the physical situation, identify the relevant quantities, and respect the assumptions under which the relationship holds.

## Units and Dimensions

Physical equations and numerical results must be dimensionally consistent.

Units should be included when meaningful. A numerically correct-looking result with incompatible dimensions is not fully correct.

## Vectors and Signs

Direction, vector components, coordinate choices, and sign conventions must be handled consistently.

A sign error that reverses the physical conclusion can be serious even when the arithmetic is otherwise correct. Alternative sign conventions are acceptable when stated or used consistently.

## Conservation Laws

Energy, momentum, angular momentum, charge, and other conserved quantities should only be invoked when the relevant conservation conditions apply.

The response should define the system appropriately and should not assume conservation of a subsystem quantity when external forces, torques, work, fluxes, or interactions invalidate that assumption.

## Models and Approximations

Physical models are approximations to reality.

Strong answers should identify important assumptions when they materially affect the conclusion. The evaluator should distinguish between:

- a justified approximation
- an unjustified approximation
- an exact relationship
- a relationship valid only in a limiting regime

Do not require a predetermined model when multiple approximations are defensible. A valid alternative model should receive full credit when its assumptions are clearly stated and physically justified.

## Limiting Cases

Where useful, a strong answer should be consistent with physically sensible limiting cases, for example:

- zero friction
- very large distance
- velocity much smaller than `c`
- weak gravitational field
- very high or low temperature
- small oscillation amplitude
- vanishing coupling
- infinite or zero resistance in an appropriate idealized limit

An answer that produces obviously impossible limiting behavior should be penalized.

## Reference Frames

Frame-dependent quantities must not be treated as invariant.

In mechanics and relativity, the response should identify the relevant reference frame when material. Proper time, proper length, invariant mass, and spacetime intervals should not be confused with frame-dependent coordinate quantities.

## Thermodynamic Reasoning

Heat, work, temperature, internal energy, and entropy must be distinguished correctly.

The first and second laws must not be conflated. State functions such as internal energy and entropy should be distinguished from path-dependent quantities such as heat and work.

A proposed cyclic process that conserves energy can still be impossible because of the second law.

## Quantum Reasoning

Quantum probabilities, probability amplitudes, states, observables, eigenstates, expectation values, and measurement outcomes should be distinguished correctly.

Quantum uncertainty should not be reduced to ordinary apparatus error when intrinsic uncertainty is relevant. A superposition should not be treated as a classical mixture without justification.

Do not penalize an answer for avoiding a specific interpretation of quantum mechanics unless an interpretation is explicitly requested.

## Observational Inference

Astronomical observations often constrain models indirectly.

Strong answers should distinguish:

1. what is measured
2. what is calculated from the measurement
3. what is inferred using a physical model
4. which assumptions enter the inference
5. what remains uncertain or degenerate

Examples include inferring luminosity from flux and distance, mass from orbital dynamics or lensing, exoplanet properties from transits and radial velocities, and cosmological parameters from distance-redshift or standard-ruler observations.

## Astronomy Is Not Catalog Recall

Do not reward obscure memorized object properties.

The benchmark evaluates physical and astronomical reasoning, not recall of arbitrary catalog data. If a numerical result depends on a nonstandard or catalog-specific value, that value should be supplied by the question.

## Experimental Uncertainty

Precision is not accuracy. Random uncertainty is not the same as systematic error.

Statistical significance alone does not establish that a proposed physical interpretation is correct. Repeated measurements can reduce random uncertainty but generally do not remove an unknown common systematic bias.

For uncertainty propagation, accept mathematically equivalent correct methods appropriate to the stated assumptions.

## Evidence and Model Dependence

A result can be observationally robust while its theoretical interpretation remains model-dependent.

Strong answers should not claim more certainty than the evidence supports. Evidence for an additional gravitating component, for example, is not by itself a measurement of that component's microscopic identity.

Upper bounds, non-detections, and model-dependent parameter estimates must not be presented as direct detections unless the question supplies evidence supporting that conclusion.

## Order of Magnitude

Answers should recognize grossly implausible numerical scales.

An algebraically derived result that is many orders of magnitude physically unreasonable should trigger additional scrutiny. When the question requests an estimate, sensible approximations and transparent scaling are more important than spurious precision.

## Multiple Valid Methods

Many physics problems admit several valid derivations.

Do not require a specific method if an alternative method is physically and mathematically correct. Equivalent coordinate systems, energy methods, force methods, invariant methods, numerical estimates, and justified approximations should receive equal credit when they answer the task correctly.

## Underdetermined Questions

If the supplied information is insufficient to determine a unique answer, recognizing that insufficiency is correct.

Do not reward invented assumptions presented as known facts. A strong answer should identify the missing information and, where useful, explain what additional measurement would remove the degeneracy.

## No Formula Worship

Correctly naming or quoting an equation does not compensate for applying it incorrectly.

The response must explain why the relationship is relevant and respect variables, sign conventions, assumptions, and physical mechanism. Equation dumping without a valid connection to the problem should not receive high credit.

## No Citation Worship

Citations do not compensate for incorrect physics or astronomical reasoning.

For most benchmark questions, correct reasoning should be sufficient without external citations unless the question explicitly asks for sources.

## Quantitative Answers

A strong quantitative answer should, where appropriate:

- choose a correct relationship or model
- show enough reasoning to make the method interpretable
- substitute quantities consistently
- preserve units
- handle vectors and signs correctly
- obtain a physically reasonable result
- round sensibly
- interpret the result when interpretation is part of the question

Do not require every trivial arithmetic step. A correct final number produced from fundamentally invalid physics should not receive full credit.

## Conceptual Answers

A strong conceptual answer should identify the mechanism.

Merely restating the phenomenon is insufficient. For example, saying “because of conservation of energy” is incomplete if the response never identifies the relevant energy transfers or explains why the chosen system is closed enough for the conservation statement to apply.

## Derivations

Equivalent derivations should receive equal credit.

Do not require the reference solution's exact sequence of algebraic steps. Judge whether the derivation:

- begins from valid principles
- makes justified assumptions
- follows mathematically
- uses consistent dimensions and definitions
- reaches a consistent result

## Astronomy Answers

For astronomical inference, separate:

1. the observation
2. the physical relationship connecting observation to a quantity
3. assumptions entering the inference
4. the resulting constraint
5. remaining uncertainty or degeneracy where material

Do not demand unnecessary astrophysical detail when the question asks for a simpler inference.

## Experimental Answers

For experiment-design and measurement questions, evaluate whether the answer identifies:

- the quantity being tested
- relevant controls
- major confounders
- calibration issues
- random uncertainty
- systematic uncertainty
- instrumental limitations
- what result would support or disfavor the hypothesis

A sophisticated apparatus description does not compensate for failure to measure the relevant physical quantity.

## Critical Error Examples

Examples that can trigger `critical_physics_astronomy_error` when central to the answer include:

- fundamentally violating conservation of energy or momentum without justification
- reversing a central force or field direction so the conclusion changes
- basing the solution on an equation that is physically inapplicable in the stated regime
- claiming constant-velocity motion requires a net force in the direction of motion
- confusing energy with force or heat with temperature in a way that drives the conclusion
- claiming a cyclic heat engine can exceed the relevant Carnot limit under the stated reservoirs
- making a major dimensional error that invalidates the result
- claiming an ordinary massive object can be accelerated through `c`
- fundamentally misusing simultaneity, time dilation, proper time, or reference frames
- treating intrinsic quantum uncertainty as merely faulty instrumentation
- converting wavefunction amplitudes to probabilities incorrectly in a way that changes the central result
- relying on conservation of a quantity that is not conserved for the chosen system
- predicting the wrong qualitative orbit from a central orbital-mechanics error
- confusing luminosity with apparent brightness in a way that invalidates an inference
- claiming a measured redshift uniquely proves a mechanism when the supplied evidence does not distinguish alternatives
- treating an upper bound or non-detection as a detection
- ignoring a dominant systematic or instrumental limitation so that the experimental or astronomical conclusion is not supported

The critical flag should **not** be used for small rounding differences, minor arithmetic slips that preserve the conclusion, harmless notation differences, reasonable alternative derivations, reasonable approximation choices, legitimate empirically equivalent interpretation differences, or clearly stated defensible modeling assumptions.
