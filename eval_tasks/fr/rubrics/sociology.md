# Sociology Evaluation Rubric

## Task

Score **ONE answer to ONE Sociology question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, social context, population, time period, and assumptions.

Accept different theoretically defensible sociological explanations when the evidence and stated assumptions reasonably support them. Do not require a particular theoretical school, wording, or essay structure.

Do not reward unnecessary length, jargon, sociologist-name dropping, ideological language, or citations. The benchmark evaluates **sociological reasoning**, not resemblance to a reference answer.

## Critical Error

If `critical_sociology_error` applies: **score = 0**.

**Condition:** A materially false central sociological, methodological, quantitative, demographic, causal, or group-level claim fundamentally invalidates or reverses the answer.

Examples include:
- Fundamentally confuses individual-level and population-level evidence in a way that changes the conclusion.
- Claims an aggregate relationship proves the same relationship at the individual level.
- Claims one anecdotal individual case disproves a population-level social pattern.
- Treats correlation as proof of causation when causality is central to the question.
- Fundamentally misidentifies the unit of analysis.
- Makes a major denominator or rate error that reverses the social comparison.
- Claims a very large convenience sample is necessarily representative.
- Fundamentally confuses reliability with validity.
- Treats social construction as meaning a phenomenon is imaginary or has no real consequences.
- Treats income and wealth as interchangeable where the distinction controls the conclusion.
- Treats deviance and crime as identical.
- Treats recorded crime counts as direct measures of total offending without qualification.
- Treats a disparity as proof of discrimination when the evidence only establishes disparity and unresolved alternatives remain.
- Claims demographic-group membership mechanically determines an individual's behavior.
- Makes a biologically deterministic claim about a socially defined group without evidence.
- Fundamentally misinterprets social mobility.
- Confuses cohort change with individual aging where the distinction controls the explanation.
- Claims network similarity proves social contagion while ignoring stated homophily, shared-environment, or selection alternatives.
- Invents survey results, demographic data, institutional facts, or study findings required for the conclusion.
- Presents a political or moral preference as though it were an empirical sociological finding.

Do **not** classify as critical:
- Minor terminology mistakes.
- Harmless arithmetic errors that do not alter the conclusion.
- Reasonable theoretical disagreements.
- Different defensible sociological interpretations.
- Small omissions.
- Cautious uncertainty.
- Reasonable differences in operationalization.
- Alternative qualitative interpretations supported by evidence.
- Differences in emphasis among plausible mechanisms.
- Minor imprecision that does not materially change the conclusion.

## Score Anchors

### 4

The answer is sociologically accurate, directly addresses the question, identifies the relevant social mechanisms, institutions, group processes, contextual factors, and evidence, and reaches a well-supported and appropriately qualified conclusion.

Where relevant, it correctly handles levels of analysis, structure and agency, institutions, norms and culture, groups and networks, inequality, class, race/ethnicity, gender, organizations, demographics, social change, selection, causality, measurement, research design, and uncertainty.

For causal questions, it distinguishes association from causation and considers credible alternatives. For inequality questions, it distinguishes observed disparity from evidence about the mechanism producing it. For group-level questions, it avoids assuming population averages determine individuals. For methods questions, it evaluates sampling, measurement, validity, design, and scope appropriately. For comparative questions, it identifies material institutional and contextual differences rather than relying on stereotypes.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor conceptual imprecision, limited causal reasoning, incomplete contextualization, or insufficient qualification. The main conclusion remains sound.

Typical reasons for a 3 include omitting one important alternative mechanism, slightly incomplete concept application, a minor quantitative error that does not affect the conclusion, incomplete discussion of selection or measurement, or insufficient but not misleading contextual qualification.

### 2

The answer is partly correct or directionally useful but has important omissions or reasoning weaknesses. It may identify a plausible mechanism but treat it as the only possible cause, recognize structure while ignoring agency, interpret data correctly while overstating causality, use a theory superficially, blur individual and aggregate evidence, or overlook an important institutional or compositional mechanism.

It demonstrates meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial sociological, methodological, demographic, causal, quantitative, or level-of-analysis misunderstanding, but does not meet the critical-error threshold. The response provides limited useful sociological analysis.

### 0

Critical Sociology error, no answer, off-topic answer, fabricated evidence, stereotype presented as fact, or equivalent fundamental failure.

## Evaluation Criteria

### Relevance (`relevance`)

- **Weight:** 0.05
- **Conditional:** false
- **Definition:** Directly answers the question asked, focuses on the stated sociological problem, population, time frame, and assumptions, and avoids tangents that do not help resolve the task.

### Sociological Accuracy (`sociological_accuracy`)

- **Weight:** 0.05
- **Conditional:** false
- **Definition:** Uses sociological concepts, mechanisms, institutions, population reasoning, and methodological claims accurately; does not substitute stereotypes, unsupported group claims, or individual psychology for sociological explanation.

### Conceptual Precision (`conceptual_precision`)

- **Weight:** 0.05
- **Conditional:** false
- **Definition:** Correctly distinguishes concepts that materially differ, including individual versus aggregate evidence, norms versus values, status versus role, household versus family, income versus wealth, deviance versus crime, disparity versus discrimination, and association versus causation.

### Micro–Meso–Macro Reasoning (`micro_meso_macro_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, identifies the level at which an explanation operates and explains cross-level links without treating individual, group, organizational, institutional, and societal processes as interchangeable.

### Structure and Agency Reasoning (`structure_and_agency_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, explains how institutional rules, resources, norms, opportunities, and constraints interact with individual or collective choices, avoiding both structural determinism and unrestricted-choice explanations.

### Institutional Reasoning (`institutional_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, analyzes how formal rules, informal practices, organizations, legitimacy, path dependence, and interactions among institutions shape behavior and distribute resources or opportunities.

### Inequality and Stratification Reasoning (`inequality_and_stratification_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, distinguishes dimensions such as income, wealth, status, power, occupation, education, opportunity, and mobility; identifies plausible mechanisms without treating disparity as automatic proof for or against discrimination.

### Culture and Socialization Reasoning (`culture_and_socialization_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, explains norms, values, symbols, socialization, meaning, identity, sanctions, and cultural variation while recognizing internal diversity, historical change, and the difference between explanation and moral endorsement.

### Group, Network, and Organizational Reasoning (`group_network_and_organizational_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, correctly reasons about groups, strong and weak ties, brokerage, homophily, social capital, hierarchy, bureaucracy, informal organization, diffusion, and organizational inequality without assuming network position or group membership mechanically determines outcomes.

### Demographic and Life-Course Reasoning (`demographic_and_life_course_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, correctly interprets fertility, mortality, migration, age structure, cohorts, transitions, linked lives, and cumulative processes, distinguishing cohort, age, and period explanations.

### Comparative and Contextual Reasoning (`comparative_and_contextual_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, accounts for differences across societies, regions, institutions, historical periods, and category systems; checks measurement equivalence and avoids stereotypes or unjustified transfer of one context's categories to another.

### Causal Reasoning (`causal_reasoning`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, distinguishes association from causation; considers temporal ordering, confounding, selection, reverse causality, mediation, moderation/context, mechanisms, and credible alternative explanations.

### Research Design and Methods (`research_design_and_methods`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, matches method to research question and evaluates surveys, interviews, ethnography, observation, experiments, natural experiments, administrative data, content analysis, comparative designs, longitudinal designs, and mixed methods in terms of what they can and cannot establish.

### Quantitative and Data Interpretation (`quantitative_and_data_interpretation`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, uses correct denominators, units of analysis, populations, rates, proportions, subgroup comparisons, mobility patterns, and supplied quantitative definitions; interprets numerical differences at an appropriate substantive level.

### Evidence and Source Evaluation (`evidence_and_source_evaluation`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, evaluates evidence by design, sampling, measurement, reliability, validity, generalizability, triangulation, and fit to the claim rather than by source prestige, sample size, citation count, or method label alone.

### Assumptions and Uncertainty (`assumptions_and_uncertainty`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, makes material assumptions explicit, identifies underdetermination or scope conditions, avoids fabricated certainty, and states what additional evidence would change confidence among competing explanations.

### Completeness (`completeness`)

- **Weight:** 0.05
- **Conditional:** false
- **Definition:** Addresses the major parts of the prompt and includes the key mechanisms, distinctions, calculations, comparisons, or limitations needed for a substantively adequate answer, while not requiring exhaustive length.

### Consistency (`consistency`)

- **Weight:** 0.05
- **Conditional:** false
- **Definition:** Maintains internally consistent claims, units, definitions, assumptions, and conclusions; later statements do not materially contradict earlier sociological or quantitative reasoning.

### Clarity (`clarity`)

- **Weight:** 0.05
- **Conditional:** false
- **Definition:** Communicates the reasoning in a clear, interpretable way, distinguishes evidence from inference and normative claims from empirical claims, and uses sociological terminology only when it improves precision.

### Practical Sociological Judgment (`practical_sociological_judgment`)

- **Weight:** 0.05
- **Conditional:** true
- **Definition:** When relevant, reaches a proportionate, evidence-sensitive conclusion that reflects multiple plausible mechanisms, contextual variation, heterogeneous groups, and the limits of what the evidence supports.

## Important Evaluation Principles

### Social patterns are not individual destiny

Population-level relationships do not determine every individual's outcome.

### Individual anecdotes do not settle population questions

One exceptional case does not automatically invalidate a social pattern.

### Structure and agency interact

Do not assume either complete structural determination or unrestricted individual choice.

### Levels of analysis matter

Individual, group, organizational, institutional, and societal explanations are not interchangeable.

### Correlation is not causation

Associations may reflect confounding, selection, reverse causality, measurement, or common environments.

### Mechanisms matter

Strong answers explain how a social process could produce the observed pattern.

### Social construction does not mean imaginary

Socially produced categories and institutions can have powerful material consequences.

### Categories require context

Class, race, ethnicity, gender, family, occupation, and other categories can vary across time and place.

### Groups are heterogeneous

Do not assume all members of a social category behave alike.

### Inequality is multidimensional

Income, wealth, status, power, opportunity, education, and social connections are distinct dimensions.

### Disparity is not automatically proof of discrimination

Unequal outcomes require investigation of mechanisms; absence of explicit intent also does not rule out institutional mechanisms.

### Selection matters

People may select into neighborhoods, schools, occupations, relationships, or organizations, complicating contextual inference.

### Aggregation matters

Group composition can create, hide, or reverse aggregate patterns.

### Networks matter

Similarity among connected people can reflect influence, homophily, shared environments, or combinations of these.

### Institutions matter

Formal and informal arrangements shape opportunities, incentives, norms, and outcomes.

### Culture is not destiny

Cultural explanations should specify mechanisms and acknowledge internal variation and change.

### Qualitative and quantitative evidence answer different questions

Neither methodological tradition automatically deserves higher evidentiary status.

### Large samples do not automatically remove bias

Representativeness depends on sampling and participation, not sample size alone.

### Measurement matters

Operational definitions and classification choices can change observed social patterns.

### Historical context matters

Social relationships may depend on institutions and conditions specific to a time and place.

### Comparative context matters

A mechanism observed in one society may not operate identically in another.

### Normative and empirical claims differ

Sociology can explain consequences without deciding which social arrangement people ought to prefer.

### Multiple sociological explanations may be valid

Evaluate mechanisms, evidence, assumptions, context, and competing explanations rather than requiring one theoretical school.

### No theory-name worship

Naming a theorist without correctly applying a relevant mechanism should not receive a high score.

### No jargon worship

Specialized terminology does not substitute for explanation.

### No citation worship

Citations do not compensate for incorrect sociological reasoning.

## Application Notes

- Apply only criteria that are relevant to the prompt when a criterion is marked conditional.
- Do not penalize a concise answer merely for being concise if it fully answers the question.
- Do not require theorist names when the mechanism is correctly explained.
- Do not treat demographic categories as internally homogeneous, biologically deterministic, timeless, or universally defined.
- For inequality questions, distinguish wealth, income, status, power, opportunity, education, and social connections where the distinction matters.
- For causal questions, check temporal ordering, confounding, selection, reverse causality, mechanisms, and scope conditions.
- For methods questions, distinguish sampling from measurement, reliability from validity, representativeness from sample size, and qualitative depth from population prevalence.
- For political-sociology questions, remain descriptive and analytical; do not endorse parties, candidates, ideologies, or political choices and do not predict election outcomes.
- When evidence is underdetermined, calibrated uncertainty is a strength rather than a defect.
