# Political Science & International Relations Evaluation Rubric

## Task

Score **ONE answer to ONE Political Science & International Relations question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, jurisdiction or institutional context where relevant, time period, and assumptions.

Accept different theoretically defensible explanations when the evidence and stated assumptions reasonably support them. Do not require a particular theoretical school, wording, ideology, or rhetorical structure. Do not reward unnecessary length, political slogans, jargon, citations, or partisan advocacy.

The benchmark evaluates **political-science and international-relations reasoning**, not resemblance to a reference answer or agreement with a political preference.

## Critical Error

If `critical_political_science_international_relations_error` applies:

**score = 0**

A critical error is a materially false central political-science, institutional, electoral, causal, quantitative, comparative, or IR claim that fundamentally invalidates or reverses the answer. It also includes fabricated material facts or rules, partisan endorsement in place of analysis, or an independent future-election prediction when the task calls for descriptive analysis.

Typical critical errors include:
- fundamentally confusing state, government, and regime when the distinction drives the conclusion;
- materially misreading an electoral system supplied in the prompt;
- a major vote-share, seat-share, coalition, or denominator error that reverses the analysis;
- claiming correlation alone proves a political causal relationship;
- treating a poll as an election result or misidentifying the surveyed population;
- ignoring an explicit institutional rule that reverses the outcome;
- claiming a large sample eliminates selection, coverage, or nonresponse bias;
- drawing a decisive individual-level conclusion from aggregate data;
- fundamentally misreading a causal design;
- asserting that an alliance guarantees identical action in every contingency;
- confusing deterrence with compellence;
- treating defensive capability as proof of offensive intent in a security-dilemma problem;
- claiming interdependence necessarily prevents conflict;
- treating realism, liberalism, constructivism, or another framework as universally proven;
- fabricating political facts, polling, rules, treaty provisions, positions, or historical facts required by the task;
- substituting partisan endorsement or voter recommendation for neutral analysis;
- independently predicting a future election winner where the task is descriptive.

Do **not** treat as critical: minor terminology slips, harmless arithmetic errors that do not change the conclusion, defensible theoretical disagreement, small omissions, reasonable approximations, cautious uncertainty, or supported alternative causal explanations where evidence is incomplete.

## Score Anchors

### 4

The answer is politically and analytically accurate, directly addresses the question, identifies the relevant actors, institutions, incentives, mechanisms, and evidence, and reaches a well-supported and appropriately qualified conclusion.

Where relevant, it correctly handles institutions, electoral rules, political behavior, public opinion, comparative context, political economy, causal mechanisms, research design, strategic interaction, international institutions, diplomacy, conflict, interdependence, competing theories, quantitative evidence, and uncertainty.

For comparative questions, it identifies material differences between cases rather than relying on national stereotypes. For causal questions, it distinguishes association from causation and evaluates rival explanations. For election questions, it interprets electoral systems, votes, seats, turnout, or supplied polling without recommending political choices or forecasting winners. For IR questions, it distinguishes capabilities, interests, institutions, beliefs, and strategic constraints. For contested political questions, it presents relevant perspectives and evidence without partisan advocacy.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor conceptual imprecision, limited causal reasoning, incomplete rival explanation, or insufficient qualification. The main analytical conclusion remains well supported.

Examples include one important institutional factor omitted, a correct theory with an incomplete mechanism, a minor quantitative error with no effect on the conclusion, reasonable analysis with insufficient uncertainty discussion, or one credible counterexplanation being underdeveloped.

### 2

The answer is partly correct or directionally useful but has important omissions or reasoning weaknesses. It may identify relevant institutions but explain incentives poorly, use an appropriate theory superficially, interpret data correctly but overstate causality, overlook a material alternative explanation, treat a conditional relationship as too deterministic, or give incomplete comparative analysis.

It demonstrates meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial political-science, comparative, electoral, causal, strategic, quantitative, or IR misunderstanding, but does not meet the critical-error threshold. The response provides limited useful analysis.

### 0

Critical Political Science & International Relations error, no answer, off-topic answer, fabricated evidence, partisan endorsement in place of analysis, or equivalent fundamental failure.

## Evaluation Criteria

| Criterion | Weight | Conditional | Operational standard |
|---|---:|:---:|---|
| `relevance` | 0.05 | No | Addresses the actual task and supplied facts/rules without drifting into unrelated disciplines or advocacy. |
| `political_science_accuracy` | 0.05 | No | Correctly handles core distinctions such as state/government/regime, votes/seats, policy adoption/implementation, and incentives/outcomes. |
| `international_relations_accuracy` | 0.05 | Yes | When relevant, correctly distinguishes capability/intent, deterrence/compellence, alliance/automatic action, norm/law, and interdependence/equal dependence. |
| `conceptual_precision` | 0.05 | No | Applies concepts precisely and does not use labels as substitutes for mechanisms. |
| `institutional_reasoning` | 0.05 | Yes | Identifies formal and informal rules, veto points, delegation, accountability, coordination, and incentive effects. |
| `comparative_reasoning` | 0.05 | Yes | Compares cases on relevant dimensions, avoids stereotypes, and respects scope and transfer limits. |
| `causal_reasoning` | 0.05 | Yes | Distinguishes association from causation and evaluates mechanisms, confounding, selection, reverse causality, and counterfactuals. |
| `political_behavior_and_electoral_reasoning` | 0.05 | Yes | Applies supplied rules and correctly distinguishes turnout, votes, seats, polls, preferences, strategy, and mobilization. |
| `political_economy_reasoning` | 0.05 | Yes | Analyzes distribution, institutions, organized interests, credibility, taxation, regulation, trade, and development as political phenomena. |
| `strategic_and_ir_reasoning` | 0.05 | Yes | Identifies actors, preferences, outside options, information, commitment, signaling, bargaining, gains, and repeated interaction where relevant. |
| `evidence_and_source_evaluation` | 0.05 | Yes | Assesses what evidence establishes, source incentives, methodology, and primary/secondary distinctions. |
| `quantitative_and_data_interpretation` | 0.05 | Yes | Uses correct denominators, populations, periods, formulas, and comparisons; interprets limits of the data. |
| `research_design_reasoning` | 0.05 | Yes | Handles variables, comparison logic, identifying assumptions, measurement, case selection, internal validity, and external validity. |
| `alternative_explanations` | 0.05 | Yes | Identifies credible rivals and evidence that would discriminate among them. |
| `assumptions_and_scope_conditions` | 0.05 | Yes | States material assumptions and where/when conclusions should apply. |
| `uncertainty_and_neutrality` | 0.05 | Yes | Calibrates confidence, separates facts from values, avoids unsupported motives, partisan advocacy, and independent election forecasting. |
| `completeness` | 0.05 | No | Covers the material components needed for the requested level of analysis. |
| `consistency` | 0.05 | No | Keeps facts, concepts, calculations, assumptions, and conclusions internally coherent. |
| `clarity` | 0.05 | No | Presents understandable mechanism-based reasoning without unexplained jargon. |
| `practical_political_analysis` | 0.05 | Yes | Converts concepts into concrete institutional, electoral, policy, diplomatic, or strategic implications while remaining neutral. |

All 20 criteria have equal weight. When a criterion is conditional, apply it only when the question materially implicates that dimension; do not penalize an answer for omitting an inapplicable category.

## Important Evaluation Principles

### Neutrality
Political analysis should inform rather than advocate. Do not reward partisan endorsement, opposition, persuasion, ideological ranking, candidate ranking, or telling a voter how to vote.

### Facts vs values
Distinguish empirical claims from normative judgments. Political science can analyze mechanisms and documented effects without determining which outcome a person should prefer.

### Institutions create incentives
Institutions shape behavior without mechanically determining it. Written rules, veto points, delegation, accountability, party organization, and informal norms alter incentives and constraints.

### Comparative context matters
An institutional effect observed in one political system may not transfer unchanged to another. Differences in complementary institutions, party systems, administrative capacity, social structure, and historical sequence may matter.

### Correlation is not causation
Political associations require causal reasoning, an appropriate research design, or other evidence before strong causal claims are justified. Consider selection, confounding, reverse causality, simultaneity, mechanisms, and counterfactuals.

### Measurement matters
Democracy, corruption, ideology, polarization, state capacity, power, legitimacy, and nationalism can be operationalized in multiple ways. Index values depend on indicator selection, measurement quality, aggregation, and weighting.

### Electoral rules matter
Vote shares do not automatically translate into equivalent seat shares. Apply the supplied district structure, thresholds, majority requirements, or allocation rules.

### Polls are measurements, not outcomes
Polls have sampling uncertainty, coverage and nonresponse risks, weighting assumptions, measurement error, wording effects, and timing limitations. Do not transform descriptive polling into unsupported election forecasts.

### No election forecasting
Do not reward independent predictions of who will win a future election, who is favored, or a candidate's winning probability. It is acceptable to interpret supplied hypothetical or historical polling descriptively.

### Public opinion is contextual
Responses may depend on wording, salience, information, response order, timing, and sample composition.

### Political behavior is multicausal
Voting, turnout, protest, participation, and identification may reflect institutions, mobilization, resources, beliefs, identities, group networks, economic conditions, and strategic context. Avoid one-variable determinism.

### Formal and informal institutions both matter
Constitutions and statutes may not fully describe political practice. Conventions, patronage, bargaining practices, and unwritten norms can shape outcomes without becoming formal law.

### Policy adoption is not implementation
Formal approval does not guarantee administrative execution. Capacity, resources, discretion, incentives, coordination, information, and local implementation can alter outcomes.

### State capacity is not regime type
Democracies and authoritarian systems can each vary widely in taxation, administration, enforcement, territorial control, information capacity, and service delivery.

### Theories are analytical tools
Realism, liberalism, constructivism, institutionalism, political economy, and other approaches are frameworks for explanation. Naming a theory is not enough; the answer must apply its mechanism to the evidence.

### IR explanations operate at several levels
Leader, domestic, state, bureaucratic, and systemic factors can coexist. Do not assume one level always dominates.

### Capabilities are not intentions
Military or economic capabilities constrain and enable behavior but do not by themselves prove goals, motives, or offensive intent.

### Interdependence has two sides
Interdependence can support cooperation while also creating vulnerability, coercive leverage, distributional conflict, or adjustment costs. Mutual dependence need not be symmetrical.

### Conflict is not monocausal
Conflict can involve information problems, commitment problems, bargaining, institutional incentives, economic factors, identity, leaders, external support, and international structure. Distinguish triggers from deeper mechanisms.

### Alternative explanations matter
Strong answers identify credible rival explanations and state what evidence would discriminate among them.

### Scope conditions matter
Political claims should state the populations, institutions, periods, or strategic conditions under which they are expected to apply.

### Uncertainty matters
Politics involves incomplete information, strategic behavior, measurement error, changing conditions, and heterogeneous actors. Confidence should reflect the quality and limits of the evidence.

### No ideology worship
An ideological label does not substitute for explaining institutions, incentives, policies, or mechanisms. Do not reward claims that one political ideology is inherently superior.

### No framework worship
Theory names and jargon do not earn credit without correct mechanism-based application.

### No citation worship
Citations cannot compensate for incorrect political-science, quantitative, institutional, causal, or IR reasoning.
