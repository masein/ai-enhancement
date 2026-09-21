# Physics & Engineering Evaluation Rubric

## Task
Score ONE answer to ONE Physics & Engineering question from 0–4.

The question metadata establishes the relevant physical/engineering domain, intent, difficulty, assumptions, and constraints.

Accept different valid derivations, models, and engineering approaches where appropriate. Do not require a particular wording or structure. Do not reward unnecessary length, jargon, or citations.

## Critical Error
If `critical_physics_engineering_error` applies:

**score = 0**

A critical physics/engineering error means a materially false central physical or engineering claim, calculation, conservation law, governing equation, or design conclusion that fundamentally misleads the user.

Examples include violating a fundamental conservation law; using a governing equation in a fundamentally invalid way when central; reversing a fundamental physical relationship; a major calculation error that changes the substantive conclusion; claiming an impossible physical result is possible; fundamentally incorrect circuit behavior; fundamentally incorrect stability/control conclusions; or a design that violates a stated physical constraint.

Do not classify minor arithmetic errors, obvious minor unit mistakes, reasonable approximations, clearly stated alternative valid models, different valid engineering designs, minor omissions, or notation differences as critical.

## Score Anchors

### 4
The answer is physically/technically correct, directly addresses the problem, selects an appropriate model, uses equations correctly where needed, handles important assumptions and constraints, and reaches a sound conclusion. Quantitative answers are dimensionally and numerically correct. Engineering answers appropriately address feasibility and relevant trade-offs.

### 3
The answer is substantially correct and useful but contains one meaningful omission, minor imprecision, limited reasoning gap, or insufficient treatment of an important assumption or edge case.

### 2
The answer is partly correct or directionally useful but has important omissions, incomplete modeling, weak assumptions, or incomplete quantitative/physical reasoning.

### 1
The answer contains a substantial misunderstanding, inappropriate physical model, weak engineering reasoning, or major non-critical calculation problem, but does not meet the critical-error threshold.

### 0
Critical physics/engineering error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Important Evaluation Principles

- **Correct physics over formula dumping:** an answer that writes many equations but applies the wrong physical model should not receive a high score.
- **Reasoning over memorization:** reward understanding of mechanisms, assumptions, and consequences.
- **Multiple valid engineering solutions:** engineering questions can have multiple valid solutions. Do not require one predetermined design if multiple approaches satisfy the stated constraints.
- Evaluate feasibility, correctness, assumptions, trade-offs, safety, efficiency, and robustness.
- **Approximation is allowed:** do not penalize a reasonable approximation if its assumptions are appropriate, it remains within the required accuracy, and it does not change the substantive conclusion.
- **Units matter:** for quantitative questions, dimensional consistency is an important part of correctness.
- **Limiting cases matter:** when appropriate, strong answers should behave sensibly in limiting cases.
- **Safety:** for potentially hazardous physical systems, evaluate whether the answer recognizes relevant safety constraints. Do not require safety discussion where it is irrelevant.

## Criteria
- `1` — **relevance**: Directly addresses the stated physics or engineering task and its constraints.
- `2` — **physical_accuracy**: Uses physically correct principles, relationships, and interpretations.
- `3` — **engineering_accuracy**: Uses technically sound engineering reasoning and design practice where applicable.
- `4` — **reasoning**: Shows a coherent chain from the problem to the conclusion rather than unsupported assertions.
- `5` — **conceptual_precision**: Distinguishes closely related concepts and describes mechanisms accurately.
- `6` — **model_selection**: Chooses an appropriate governing model, equation set, or engineering abstraction when the question requires it. *(conditional)*
- `7` — **mathematical_correctness**: Performs algebra, derivations, and mathematical transformations correctly when applicable. *(conditional)*
- `8` — **dimensional_correctness**: Maintains consistent physical dimensions and units in quantitative work. *(conditional)*
- `9` — **assumptions**: Identifies and appropriately uses important modeling assumptions and their limits. *(conditional)*
- `10` — **conservation_and_constraints**: Respects relevant conservation laws, boundary conditions, physical constraints, and feasibility limits. *(conditional)*
- `11` — **quantitative_correctness**: Produces numerically correct results and appropriate numerical interpretation when quantitative work is required. *(conditional)*
- `12` — **limiting_case_and_sanity_checks**: Checks limiting behavior, signs, magnitudes, efficiencies, or other physical plausibility conditions when relevant. *(conditional)*
- `13` — **engineering_tradeoffs**: Recognizes competing objectives and constraints instead of assuming one universally optimal engineering choice. *(conditional)*
- `14` — **system_analysis**: Integrates interacting components, mechanisms, stages, or domains when the problem requires system-level reasoning. *(conditional)*
- `15` — **experimental_interpretation**: Correctly interprets observations, plots, measurements, trends, and model-versus-data discrepancies when applicable. *(conditional)*
- `16` — **uncertainty_and_calibration**: Handles measurement uncertainty, calibration, repeatability, and error sources appropriately when applicable. *(conditional)*
- `17` — **completeness**: Covers the material needed to answer the question without omitting a central required element.
- `18` — **consistency**: Maintains internally consistent definitions, signs, assumptions, equations, and conclusions.
- `19` — **clarity**: Communicates the reasoning and result clearly enough to be checked and used.
- `20` — **actionability**: Provides a usable conclusion, diagnostic step, calculation, or design implication when the question calls for one. *(conditional)*

## Critical Flag
`critical_physics_engineering_error` → **score = 0** when its condition applies.
