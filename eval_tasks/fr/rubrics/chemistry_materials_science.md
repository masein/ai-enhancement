# Chemistry & Materials Science Evaluation Rubric

## Task

Score **ONE answer to ONE Chemistry & Materials Science question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, chemical/materials context, and expected reasoning style.

Accept different valid mechanisms, derivations, approximations, experimental strategies, processing routes, or material choices where the evidence and constraints allow them. Do not require a particular wording or solution structure. Do not reward unnecessary length, jargon, equation dumping, or citations.

## Critical Error

If `critical_chemistry_materials_error` applies:

**score = 0**

A critical chemistry/materials error is a materially false central chemical, quantitative, mechanistic, thermodynamic, structural, electrochemical, analytical, or materials-science claim that fundamentally invalidates the answer or reverses the substantive conclusion.

Examples include central failures to conserve matter or charge; fundamentally wrong stoichiometry, limiting-reagent, equilibrium, acid-base, redox, electrochemical, phase-diagram, diffusion, corrosion, or property reasoning; treating thermodynamic favorability as proof of rapid kinetics; claiming a catalyst changes the equilibrium constant at fixed temperature; or making a uniquely certain structural/mechanistic claim from plainly non-unique evidence.

Do **not** classify as critical: minor arithmetic or rounding errors that do not change the conclusion, small nomenclature slips, reasonable alternative mechanisms or materials consistent with the evidence, reasonable approximation differences, small omissions, consistent alternative sign conventions, minor spectral-assignment uncertainty, or standard simplified models whose limitations are immaterial.

## Score Anchors

### 4

The answer is accurate, directly addresses the question, applies the appropriate chemical or materials-science principles, performs relevant calculations correctly, uses units consistently, identifies important assumptions when needed, and reaches a chemically and physically sound conclusion.

For chemistry questions, it correctly handles stoichiometry, bonding, equilibrium, thermodynamics, kinetics, mechanism, electrochemistry, or measurement as applicable.

For materials-science questions, it correctly connects structure, processing, properties, and performance where relevant.

For experimental or analytical questions, it correctly interprets what the evidence establishes and recognizes important uncertainty or limitations.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor chemical/materials imprecision, limited reasoning gap, non-central calculation error, or insufficient qualification. The central conclusion remains correct.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete chemical reasoning, weak assumptions, significant quantitative problems, incorrect interpretation of one or more material factors, or incomplete treatment of experimental evidence. It demonstrates meaningful understanding but is not reliably complete or correct.

### 1

The answer contains a substantial chemistry or materials-science misunderstanding, inappropriate model, major non-critical calculation error, serious mechanism error, or badly incomplete reasoning, but does not meet the critical-error threshold.

### 0

Critical chemistry/materials error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question.

A qualitative bonding question need not contain thermodynamic calculations. An organic mechanism question need not discuss materials processing. A straightforward materials-property question need not contain spectroscopy. Conditional criteria should not penalize an answer for omitting material irrelevant to the problem.

However, if a conditional consideration is necessary to solve the question correctly, failure on that issue should affect the score.

## Quantitative Answers

A strong quantitative answer should, where appropriate:

- select the correct chemical or materials model
- establish the relevant relationships
- conserve matter and charge
- substitute quantities correctly
- use consistent units
- calculate correctly
- round sensibly
- check chemical and physical plausibility
- interpret the result when interpretation is requested

Do not require every trivial arithmetic step. A correct final number obtained from fundamentally invalid chemistry should not receive full credit.

## Mechanistic Answers

A strong mechanistic answer should identify, where appropriate:

- relevant species or structural features
- plausible sequence of events
- driving forces
- kinetic limitations
- intermediates
- competing mechanisms
- evidence that supports or discriminates among mechanisms

Merely naming a mechanism is insufficient when explanation is requested.

## Thermodynamic and Equilibrium Answers

Strong answers should distinguish:

- state functions from path-dependent quantities
- equilibrium from rate
- standard-state quantities from actual conditions
- stable from metastable states
- thermodynamic driving force from kinetic accessibility

## Materials-Science Answers

Where relevant, strong answers should reason through:

1. processing history
2. resulting structure or microstructure
3. resulting material properties
4. performance consequences
5. relevant trade-offs or failure mechanisms

Do not require all five steps when the problem only asks about a subset.

## Experimental and Analytical Answers

Strong answers should identify, where applicable:

- what is being measured
- how the method relates to the target quantity
- important calibration issues
- systematic errors
- random errors
- confounding variables
- instrumental limitations
- alternative interpretations
- useful follow-up measurements

A sophisticated instrument description does not compensate for failure to answer the scientific question.

## Important Evaluation Principles

### Conservation
Chemical equations and quantitative analyses must conserve matter and charge.

### Thermodynamics is not kinetics
A process can be thermodynamically favorable yet kinetically slow. Do not equate negative ΔG with a rapid reaction.

### Equilibrium is dynamic
Equilibrium does not mean that reactions have stopped, and it does not necessarily imply equal reactant and product concentrations.

### Catalysts
Catalysts affect kinetics. They do not change the equilibrium constant for the same reaction at the same temperature.

### Strong is not concentrated
Acid/base strength and concentration are distinct concepts.

### Structure matters
Chemical and material behavior depends on structure at multiple levels: electronic, molecular, atomic, crystalline, defect, microstructural, and macroscopic.

### Processing–structure–properties–performance
For materials science, reward correct reasoning through:

**processing → structure → properties → performance**

A response that jumps directly from processing to performance without a plausible structural/property mechanism should receive less credit when that mechanism is central to the question.

### Thermodynamic versus metastable structures
Materials can remain in metastable states because transformation kinetics are slow. Equilibrium phase diagrams alone do not determine every experimentally observed microstructure.

### Mechanical-property distinctions
Do not treat strength, stiffness, hardness, toughness, ductility, or resilience as interchangeable.

### Experimental evidence
A measurement supports only what the method can actually establish. Strong answers distinguish direct observations, quantities inferred from models, proposed mechanisms, and unresolved alternatives.

### Spectroscopy
A spectral feature is evidence, not automatically a unique molecular structure. Use multiple observations when necessary.

### Uncertainty
Strong answers should recognize important experimental, numerical, model, and measurement uncertainty. Do not require irrelevant uncertainty discussion for straightforward textbook calculations.

### Units and dimensions
Quantitative answers must use consistent units and physically meaningful dimensions.

### Approximation
An approximate model may be fully correct when appropriate. Judge whether the approximation is justified, not whether the most sophisticated possible model was used.

### Scale
Strong answers should recognize results that are grossly inconsistent with reasonable chemical or materials scales.

### Multiple valid mechanisms
Where the evidence permits multiple mechanisms, do not require unjustified certainty.

### Underdetermination
If the supplied information cannot uniquely determine composition, mechanism, structure, or property, recognizing that fact is correct.

### No jargon worship
Using chemistry or materials-science terminology without correct causal reasoning should not receive a high score.

### No citation worship
Citations do not compensate for incorrect chemical or materials reasoning.

## Criterion Definitions

The machine-readable criteria file contains 20 equally weighted criteria (0.05 each). Broad criteria such as relevance, chemical accuracy, conceptual precision, completeness, consistency, and clarity generally apply across the benchmark. Domain-specific criteria are conditional and should be applied only when relevant.

A minor arithmetic slip should be distinguished from a stoichiometric or model-selection error. A reasonable approximation should be distinguished from an unjustified equilibrium, ideal-solution, or linear-elastic approximation. A direct observation that supports a mechanism should be distinguished from an inference that exceeds the available evidence.
