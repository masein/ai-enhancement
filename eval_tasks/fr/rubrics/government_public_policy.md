# Government & Public Policy Evaluation Criteria

## Task

Score **ONE answer to ONE Government & Public Policy question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, jurisdictional context, acuity, and expected reasoning style.

Accept different valid policy analyses, program designs, implementation strategies, and institutional interpretations where the supplied facts and objectives allow them. Do not require a particular political ideology, partisan position, policy framework, or wording. Do not reward unnecessary length, jargon, partisan rhetoric, or citations.

## Political Neutrality

The evaluator must not reward or penalize an answer because it supports or opposes a political party, candidate, government, ideology, or policy coalition.

Score factual accuracy, institutional reasoning, evidence, causal reasoning, implementation, quantitative correctness, trade-off analysis, and uncertainty. When policy desirability depends on values, evaluate whether the answer states the value judgment clearly rather than presenting it as an objective empirical fact.

## Critical Error

If `critical_government_public_policy_error` applies:

**score = 0**

A materially false central institutional, quantitative, causal, administrative, regulatory, public-finance, or policy claim fundamentally invalidates the answer or reverses its substantive conclusion. This includes central assertions that contradict rules explicitly supplied in the question, invented powers essential to the analysis, or reasoning failures that convert non-causal, output, or implementation evidence into an unsupported claim of policy success.

Representative critical errors include:
- Fundamentally misidentifying which institution has authority under rules explicitly supplied in the question.
- Inventing a legal or constitutional power central to the answer.
- Confusing legislation with administrative implementation in a way that invalidates the analysis.
- Making a major budget calculation error that reverses the policy conclusion.
- Double-counting major costs or benefits in a cost-benefit analysis.
- Treating transfer payments as new social resource costs without justification when that distinction is central.
- Treating correlation as proof of program impact.
- Interpreting a before/after comparison as causal despite a clearly supplied confounder.
- Ignoring a clearly invalid comparison group in a policy evaluation.
- Claiming a policy achieved its objective solely because administrative outputs increased.
- Confusing program participation with program effectiveness.
- Recommending a policy design that cannot be implemented under explicitly supplied authority or administrative constraints.
- Ignoring a central implementation dependency that makes the proposed plan infeasible.
- Using fundamentally incorrect tax or subsidy incidence reasoning that reverses the distributional conclusion.
- Claiming a policy has no trade-offs when supplied evidence clearly establishes major offsetting effects.
- Fundamentally misinterpreting public-opinion, administrative, or program-evaluation data.
- Treating a successful pilot as definitive proof of nationwide effectiveness despite explicitly supplied nonrepresentativeness.
- Confusing formal eligibility with actual program access when administrative burden is central to the scenario.
- Making a central claim about a jurisdiction's law or institutional procedure that contradicts rules explicitly supplied in the prompt.
- Presenting an ideological preference as if it were an empirically established policy fact when the distinction determines the answer.

Do **not** classify the following as critical by themselves:
- Minor arithmetic errors that do not alter the substantive conclusion.
- Small terminology mistakes.
- Reasonable differences in normative priorities.
- Legitimate disagreement among plausible policy alternatives.
- Reasonable institutional assumptions when clearly stated.
- Minor omissions.
- Appropriate uncertainty.
- Valid alternative causal interpretations where evidence is ambiguous.
- Different defensible implementation strategies.
- Reasonable differences in weighting efficiency, equity, cost, or administrative simplicity.

## Score Anchors

### 4

The answer is accurate, directly addresses the question, applies appropriate institutional and public-policy reasoning, interprets evidence correctly, performs relevant calculations correctly, identifies major assumptions and constraints, recognizes implementation requirements and distributional effects, and reaches a well-supported conclusion.

For policy-design questions, it connects the policy instrument to the underlying problem mechanism. For implementation questions, it recognizes authority, capacity, delivery, coordination, and monitoring. For evaluation questions, it distinguishes association from causal impact and recognizes relevant limitations. For normative questions, it distinguishes empirical findings from value judgments.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor institutional or policy imprecision, limited reasoning gap, non-central quantitative error, or insufficient treatment of implementation, distribution, or uncertainty. The central analysis remains sound.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete policy reasoning, weak causal analysis, significant quantitative problems, unrealistic implementation assumptions, or incomplete treatment of relevant institutions or stakeholders. Meaningful correction would be required.

### 1

The answer contains a substantial governmental or policy misunderstanding, inappropriate analytical framework, major non-critical quantitative error, poor causal reasoning, or badly incomplete analysis, but does not meet the critical-error threshold.

### 0

Critical Government & Public Policy error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question. A program-evaluation question need not discuss electoral systems; a budgeting question need not discuss comparative government; a constitutional-institution question need not contain cost-benefit analysis. Conditional criteria should not penalize answers for omitting irrelevant material. However, when a conditional consideration is necessary for correctness, failure on that issue should affect the score.

## Evaluation Criteria

| ID | Criterion | Conditional | Weight | What it evaluates |
|---|---|---:|---:|---|
| `relevance` | Relevance | No | 0.05 | Addresses the question actually asked, prioritizes the supplied facts and objectives, and avoids drifting into unrelated political commentary, civics trivia, or generic policy exposition. |
| `government_and_policy_accuracy` | Government and Policy Accuracy | No | 0.05 | Uses government and public-policy concepts correctly, including the distinction between intent and effect, adoption and implementation, authority and capacity, inputs/outputs/outcomes, and empirical versus normative claims. Central institutional or policy misstatements weigh more heavily than terminology imperfections. |
| `institutional_reasoning` | Institutional Reasoning | Yes | 0.05 | When institutions are material, correctly identifies relevant branches, levels, veto points, delegation relationships, oversight mechanisms, and interactions. Does not invent powers and distinguishes formal rules from actual implementation. |
| `policy_problem_definition` | Policy Problem Definition | Yes | 0.05 | When problem definition is material, distinguishes the underlying problem from a preferred intervention, identifies affected populations, baseline conditions, causes versus symptoms, measurable objectives, scope, and time horizon. |
| `policy_design_reasoning` | Policy Design Reasoning | Yes | 0.05 | When design is material, connects objectives and causal mechanisms to appropriate instruments, eligibility, benefits or obligations, delivery channels, funding, enforcement, monitoring, incentives, appeals, and administrative burden. Gives full credit to defensible alternative designs rather than requiring one ideological instrument. |
| `implementation_reasoning` | Implementation Reasoning | Yes | 0.05 | When implementation is material, tests whether legal authority, responsible agencies, staffing, budget, technology, procurement, data, delivery channels, communication, coordination, timelines, enforcement, and monitoring are sufficient. Distinguishes implementation failure from failure of the underlying policy theory. |
| `public_administration_reasoning` | Public Administration Reasoning | Yes | 0.05 | When administrative organization is material, reasons correctly about hierarchy, specialization, discretion, frontline practice, civil-service expertise, continuity, incentives, coordination, administrative burden, and the difference between formal responsibility and operational capability. |
| `public_finance_and_budget_reasoning` | Public Finance and Budget Reasoning | Yes | 0.05 | When fiscal issues are material, distinguishes stocks from flows, recurring from one-time costs, annual from lifecycle cost, transfers from real resource costs where relevant, deficits from debt, and statutory from economic incidence. Recognizes opportunity cost, fiscal constraints, financing effects, and double-counting risks. |
| `regulatory_reasoning` | Regulatory Reasoning | Yes | 0.05 | When regulation is material, identifies the regulatory objective, baseline, affected entities, behavioral response, design type, monitoring requirements, enforcement capacity, compliance cost, incentives, competition or innovation effects, and realistic alternatives. Does not equate stricter rules or formal compliance with better outcomes. |
| `quantitative_correctness` | Quantitative Correctness | Yes | 0.05 | When calculations or quantitative interpretations are required, uses correct arithmetic, units, denominators, percentage versus percentage-point changes, discounting, per-capita or per-outcome measures, and uncertainty. A small non-central slip is distinct from a major numerical error that reverses the substantive conclusion. |
| `evidence_and_causal_reasoning` | Evidence and Causal Reasoning | Yes | 0.05 | When empirical claims are material, distinguishes association from causation, identifies the relevant counterfactual, selection bias, confounding, reverse causality, statistical uncertainty, data-quality limits, and external validity. Does not treat before/after change or popularity as proof of effectiveness. |
| `program_evaluation_reasoning` | Program Evaluation Reasoning | Yes | 0.05 | When evaluation is material, distinguishes process, output, outcome, and impact evidence; assesses implementation fidelity, baseline and comparison-group quality, effect magnitude, subgroup heterogeneity, scalability, and whether observed improvement is plausibly attributable to the program. |
| `stakeholder_and_distributional_reasoning` | Stakeholder and Distributional Reasoning | Yes | 0.05 | When distribution is material, identifies beneficiaries, cost bearers, taxpayers, users and non-users, workers, firms, affected communities, geographic groups, and implementation actors. Recognizes that aggregate net benefit or average effects can conceal unequal burdens, access, or subgroup outcomes. |
| `legal_and_jurisdictional_context` | Legal and Jurisdictional Context | Yes | 0.05 | When authority or procedure depends on jurisdiction, applies only the rules supplied or otherwise established for that jurisdiction, makes reasonable qualifications where law is unspecified, and avoids unsupported assertions about constitutional, statutory, administrative, procurement, electoral, or emergency powers. |
| `risk_uncertainty_and_unintended_effects` | Risk, Uncertainty, and Unintended Effects | Yes | 0.05 | When risk or uncertainty is material, distinguishes hazard, likelihood, impact, mitigation, contingency, and residual risk; avoids false precision; considers behavioral responses, gaming, spillovers, displacement, irreversibility, scenario variation, and plausible unintended effects tied to concrete mechanisms. |
| `ethics_and_public_value_reasoning` | Ethics and Public Value Reasoning | Yes | 0.05 | When ethical or public-value issues are material, distinguishes legal compliance from ethical judgment, actual from apparent conflicts, transparency from accountability, technical effectiveness from legitimacy, and empirical evidence from value choices. Does not score partisan or ideological agreement. |
| `assumptions_and_model_selection` | Assumptions and Model Selection | Yes | 0.05 | When analytical choices are material, states important assumptions, selects an appropriate institutional, fiscal, causal, regulatory, or decision framework, tests sensitivity to assumptions where useful, and avoids forcing a framework when the evidence does not support it. |
| `completeness` | Completeness | No | 0.05 | Covers the major considerations necessary for a sound answer at the question's stated difficulty, including material constraints, trade-offs, implementation dependencies, evidence limitations, and requested calculations, without penalizing omission of irrelevant conditional criteria. |
| `consistency` | Consistency | No | 0.05 | Maintains internally consistent facts, assumptions, units, causal claims, institutional roles, and conclusions. Does not contradict supplied rules or use a premise in one part of the answer that is abandoned without explanation elsewhere. |
| `clarity` | Clarity | No | 0.05 | Communicates the analysis in a clear, organized, proportionate way; separates facts, assumptions, empirical inferences, and normative judgments; explains mechanisms rather than relying on jargon; and makes calculations and trade-offs easy to follow. |

## Institutional Answers

Strong institutional answers should, where appropriate, identify the responsible institution, relevant authority, jurisdiction, political versus administrative roles, institutional constraints, and interactions among branches or levels of government. Do not assume institutional rules are universal.

## Policy-Design Answers

Strong answers should identify, where relevant, the policy objective, target population, mechanism, instrument, administrative requirements, behavioral response, fiscal implications, distributional effects, monitoring, risks, and trade-offs. Do not require one predetermined policy instrument.

## Implementation Answers

Strong answers should consider, where relevant, legal authority, responsible agencies, staffing, budget, technology, procurement, delivery channels, coordination, enforcement, communication, monitoring, and timelines. Do not assume policy adoption equals implementation success.

## Evaluation Answers

Strong evaluation answers should identify the outcome and relevant counterfactual, distinguish correlation from causation, recognize selection and confounding, consider data quality and implementation fidelity, recognize external-validity limits, and interpret effect magnitude. Statistical significance alone is not automatically policy importance.

## Budget and Public-Finance Answers

Strong answers should use correct figures and units; identify recurring versus one-time costs; recognize opportunity cost; distinguish transfers from resource costs where relevant; avoid double counting; consider time horizon; and recognize uncertainty and fiscal constraints.

## Stakeholder and Distributional Answers

Strong answers should identify affected groups, beneficiaries, cost bearers, geographic effects, access differences, and implementation effects where material. Aggregate benefits do not imply that every group benefits.

## Crisis Answers

Strong crisis-management answers should: (1) identify the immediate objective, (2) prioritize urgent risks, (3) identify responsible institutions, (4) coordinate essential actors, (5) communicate reliable information, (6) preserve continuity, and (7) plan the transition to recovery. Do not require complete information before every time-critical action.

## Important Evaluation Principles

- Policy intent is not policy effect: evaluate mechanism, implementation, and outcomes rather than stated objective alone.
- Adoption is not implementation: a policy can be legally adopted yet poorly implemented.
- Outputs are not outcomes: processing applications, spending budgets, or delivering services are not automatically the intended social outcome.
- Correlation is not causation: causal claims require a credible counterfactual.
- Evidence has scope: internally valid evidence may have limited external validity.
- Institutions matter: policy effects depend on legal authority, administrative structure, implementation capacity, intergovernmental relationships, and enforcement.
- Jurisdiction matters: governmental authority and procedure are not universal across countries or subnational systems.
- Efficiency is not equity: aggregate gains can coexist with unequal distribution of costs and benefits.
- Popularity is not effectiveness: public support does not establish that a policy achieves its objective.
- Effectiveness is not legitimacy: technically effective interventions can still raise legal, ethical, procedural, or distributional questions.
- Public budgets involve opportunity costs.
- More spending is not automatically better performance.
- More regulation is not automatically more effective regulation.
- More decentralization is not automatically better governance.
- More centralization is not automatically better governance.
- Participation is not automatically representation.
- Administrative burden matters: formal eligibility does not guarantee practical access.
- Metrics affect behavior: agencies may optimize measured output rather than substantive outcomes.
- Crisis decisions differ from routine policymaking: urgency can change sequencing and information needs but not erase oversight and recovery requirements.
- Normative and empirical questions differ.
- Multiple valid policy analyses can exist.
- Political neutrality: evaluate analytical quality rather than support for or opposition to any political actor or ideology.
- No framework worship: analytical frameworks are tools, not answers.
- No jargon worship: vocabulary without valid reasoning should not receive high credit.
- No citation worship: citations do not compensate for incorrect institutional, quantitative, causal, or policy reasoning.

## Political and Normative Agency

The rubric evaluates reasoning rather than ideological agreement. Reasonable differences in normative priorities, policy instrument choice, or institutional design should receive full credit when they are consistent with the prompt, supported by evidence, explicit about assumptions, and attentive to trade-offs. Do not turn a benchmark score into an endorsement of a candidate, party, government, ideology, or ballot choice.

