# Education Evaluation Criteria
## Task
Score **ONE answer to ONE Education question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, learner context, educational setting, and assumptions. Accept different valid instructional approaches, curriculum designs, assessment methods, and implementation strategies where supported by the learning goal, evidence, and stated constraints.

Do not require a particular pedagogical theory, wording, or structure. Do not reward unnecessary length, jargon, theorist-name dropping, citations, or fashionable education terminology. The benchmark evaluates **educational reasoning**, not resemblance to a reference answer.

Apply all broadly applicable criteria and only the conditional criteria relevant to the question. All 20 criteria have equal weight (0.05); conditional criteria are omitted from consideration when genuinely irrelevant rather than penalized.

## Critical Error
If `critical_education_error` applies, **score = 0** regardless of performance on the remaining criteria.

A materially false central educational, assessment, quantitative, causal, inclusion, or research claim that fundamentally invalidates the answer or would lead to a materially misleading educational decision.

Examples include:
- Claiming immediate classroom performance necessarily demonstrates durable learning.
- Claiming a reliable test is necessarily valid for every intended use.
- Fundamentally misinterpreting percentile rank as percent correct where the distinction determines the conclusion.
- Claiming correlation between teacher practice and achievement proves the practice caused the achievement difference.
- Treating a large convenience sample as automatically representative.
- Recommending a high-stakes student placement based on an assessment that clearly does not measure the relevant construct.
- Fundamentally misinterpreting formative and summative assessment in a way that invalidates the recommendation.
- Claiming accommodations necessarily make an assessment unfair or lower standards.
- Treating limited proficiency in the language of instruction as proof of low general ability.
- Claiming engagement or enjoyment proves learning occurred.
- Claiming matching teaching to a supposed learning style has established causal benefit.
- Making a major quantitative error that reverses interpretation of achievement, growth, attendance, or intervention results.
- Treating absence of statistical significance as proof that an intervention has no effect.
- Treating one successful classroom as proof a method works universally.
- Ignoring major differential attrition in a study and concluding strong causal effectiveness.
- Claiming a curriculum is effective merely because all intended content was taught.
- Fundamentally misinterpreting growth and attainment.
- Fabricating student data, assessment properties, research findings, curriculum requirements, or educational regulations required for the conclusion.
- Making a central recommendation based on a demographic stereotype about learner ability.
- Asserting certainty where the supplied educational evidence is clearly insufficient.

Do **not** classify as critical:
- Minor arithmetic errors that do not change the conclusion.
- Minor terminology mistakes.
- Reasonable pedagogical differences or defensible curriculum choices.
- Different valid classroom-management approaches or grading-policy choices.
- Legitimate interpretation differences where evidence is mixed.
- Small omissions or reasonable approximations.
- Cautious uncertainty.
- Different evidence-consistent instructional strategies.

## Evaluation Criteria

| ID | Criterion | Operational definition | Weight | Conditional |
|---|---|---|---:|:---:|
| `relevance` | Relevance | Directly addresses the educational question, stated goal, learner or system context, and decision being asked about; avoids tangents that do not help resolve the task. | 0.05 | No |
| `education_accuracy` | Education Accuracy | Makes materially correct claims about teaching, learning, curriculum, assessment, inclusion, educational systems, and research as applicable; avoids unsupported universal claims. | 0.05 | No |
| `conceptual_precision` | Conceptual Precision | Uses educational concepts with the distinctions required by the task, such as learning vs performance, reliability vs validity, growth vs attainment, and correlation vs causation. | 0.05 | No |
| `learning_and_development_reasoning` | Learning and Development Reasoning | When relevant, correctly reasons from prior knowledge, misconceptions, memory, retrieval, spacing, cognitive load, motivation, metacognition, transfer, development, and expertise to instructional implications. | 0.05 | Yes |
| `pedagogical_reasoning` | Pedagogical Reasoning | When relevant, selects or compares instructional approaches based on learning goals, learner knowledge, task complexity, guidance, practice, feedback, and mechanisms rather than slogans or method labels. | 0.05 | Yes |
| `curriculum_and_instructional_design` | Curriculum and Instructional Design | When relevant, evaluates objectives, prerequisites, sequencing, coherence, progression, breadth/depth, modeling, guided practice, independent practice, and alignment across curriculum, instruction, and assessment. | 0.05 | Yes |
| `assessment_and_measurement_reasoning` | Assessment and Measurement Reasoning | When relevant, identifies the intended construct and use; reasons correctly about formative/summative/diagnostic purposes, validity, reliability, accessibility, scoring, rubrics, measurement error, and score interpretation. | 0.05 | Yes |
| `quantitative_correctness` | Quantitative Correctness | When numerical reasoning is required, uses correct arithmetic, denominators, units, comparisons, growth/level distinctions, percentage or percentile interpretation, and does not overstate precision. | 0.05 | Yes |
| `classroom_and_learning_environment_reasoning` | Classroom and Learning Environment Reasoning | When relevant, reasons about routines, expectations, transitions, participation, climate, belonging, behavior support, discussion, questioning, engagement, and instructional time in relation to learning. | 0.05 | Yes |
| `inclusion_and_accessibility_reasoning` | Inclusion and Accessibility Reasoning | When relevant, identifies access barriers and supports participation while preserving the intended learning construct; avoids deficit assumptions and distinguishes accommodations from changed expectations when needed. | 0.05 | Yes |
| `learner_variability_and_context` | Learner Variability and Context | When relevant, accounts for prior knowledge, language, disability, pace, motivation, background, experience, interests, and within-group variability without stereotyping or inferring individual ability from group averages. | 0.05 | Yes |
| `evidence_and_research_reasoning` | Evidence and Research Reasoning | When relevant, matches claims to study design; identifies confounding, selection, attrition, measurement limits, sampling, implementation, replication, and the difference between association and causal evidence. | 0.05 | Yes |
| `educational_data_interpretation` | Educational Data Interpretation | When relevant, interprets educational data using appropriate baselines, denominators, subgroup patterns, missingness, composition, measurement comparability, uncertainty, and growth versus attainment. | 0.05 | Yes |
| `technology_and_media_reasoning` | Technology and Media Reasoning | When relevant, evaluates technology through the learning processes and practices it enables, including accessibility, multimedia design, online interaction, AI accuracy, privacy, bias, oversight, and overreliance. | 0.05 | Yes |
| `system_organization_and_policy_reasoning` | System Organization and Policy Reasoning | When relevant, reasons neutrally about school organization, leadership, education systems, policy mechanisms, incentives, capacity, implementation, accountability, and unintended effects. | 0.05 | Yes |
| `assumptions_and_uncertainty` | Assumptions and Uncertainty | When relevant, identifies key assumptions, missing information, alternative explanations, measurement limits, and uncertainty; calibrates confidence instead of fabricating certainty. | 0.05 | Yes |
| `practical_educational_judgment` | Practical Educational Judgment | When relevant, proposes feasible educational actions that fit the objective, learners, evidence, constraints, implementation capacity, and likely trade-offs; allows multiple defensible approaches. | 0.05 | Yes |
| `completeness` | Completeness | Covers the material considerations needed to answer the question at its stated difficulty without substituting unnecessary breadth or length for substance. | 0.05 | No |
| `consistency` | Consistency | Maintains internally coherent reasoning, terminology, calculations, assumptions, and conclusions; does not contradict earlier claims or apply incompatible standards. | 0.05 | No |
| `clarity` | Clarity | Communicates the reasoning in a clear, organized, interpretable way appropriate to the task; necessary qualifications are explicit and jargon does not substitute for explanation. | 0.05 | No |

## Important Evaluation Principles

### Learning is not immediate performance

Strong short-term performance does not necessarily imply durable retention or transfer.

### Prior knowledge matters

Instruction should account for what learners already know and misunderstand.

### Engagement is not learning

Enjoyment and participation can support learning but are not themselves proof that learning occurred.

### Difficulty is not automatically beneficial

Challenge should support the target learning rather than create irrelevant barriers.

### Guidance depends on learner knowledge

Novices and experienced learners may benefit from different amounts of instructional guidance.

### Retrieval and spacing support retention

Learning should not be evaluated solely by ease during practice.

### Feedback must be usable

Feedback should help learners understand what to improve and how.

### Curriculum, instruction, and assessment should align

A learning objective should be taught and assessed in ways that genuinely represent the intended capability.

### Assessment purpose matters

The same assessment may be suitable for one purpose and unsuitable for another.

### Reliability is not validity

Consistent scores do not establish that an assessment measures the intended construct.

### Percentile is not percent correct

Norm-referenced rankings and raw performance are different quantities.

### Measurement contains error

Small score differences should not automatically be treated as meaningful differences in learning.

### Growth is not attainment

Current performance level and change over time answer different questions.

### Formative assessment is about use

An activity is formative when evidence is used to adapt learning or teaching.

### Inclusion is not simply placement

Access, participation, curriculum, assessment, belonging, and support all matter.

### Accommodations are not automatically lowered expectations

An accommodation may remove an irrelevant barrier while preserving the target construct.

### Learner groups are heterogeneous

Do not infer individual ability or potential from demographic group averages.

### Language proficiency is not general ability

Performance may reflect the language demands of instruction or assessment.

### Technology is not pedagogy

A device, platform, or AI system improves education only through the learning processes and practices it enables.

### Policy adoption is not implementation

A policy or program cannot be evaluated without considering how it was implemented.

### Program activity is not educational impact

Delivering lessons, devices, training, or tutoring does not itself establish learning improvement.

### Correlation is not instructional causation

Observed associations may reflect prior attainment, selection, motivation, teacher assignment, school context, demographics, or other confounding.

### Research design constrains claims

Causal claims require stronger evidence than descriptive claims.

### Statistical significance is not educational significance

Magnitude, context, cost, feasibility, and learning value matter.

### Multiple valid instructional approaches

Evaluate learning objective, evidence, learner needs, implementation, and trade-offs rather than enforcing one pedagogical ideology.

### No framework worship

Naming a framework or theorist is not enough; the answer must explain the relevant mechanism and apply it correctly.

### No technology worship

New technology does not automatically improve education.

### No citation worship

Citations do not compensate for incorrect educational reasoning.

## Score Anchors

### 4

The answer is educationally accurate, directly addresses the question, identifies the relevant learning goal, learner characteristics, instructional mechanisms, assessment considerations, and evidence, and reaches a well-supported and appropriately qualified conclusion.

Where relevant, it correctly handles prior knowledge, memory and retention, instructional guidance, curriculum sequence, practice, feedback, formative and summative assessment, validity, reliability, learner variability, inclusion, accessibility, technology, implementation, research design, data, and uncertainty.

For instructional questions, it explains why the proposed approach fits the learners and objective. For assessment questions, it distinguishes construct, purpose, validity, reliability, and score interpretation. For inclusion questions, it preserves the educational goal while addressing access barriers. For research questions, it matches the strength of the claim to the study design. For quantitative questions, calculations, denominators, and interpretations are correct. Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor educational imprecision, limited reasoning gap, incomplete qualification, or insufficient consideration of an important contextual factor. The main conclusion remains sound.

### 2

The answer is partly correct or directionally useful but has important omissions or reasoning weaknesses. Examples include a plausible method with poor alignment to the learning goal, correct interpretation with weak consideration of prior knowledge, an appropriate assessment with incomplete validity reasoning, a useful intervention with limited implementation analysis, or correct data reading paired with an overstated causal conclusion. It demonstrates meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial educational, pedagogical, assessment, quantitative, inclusion, or research misunderstanding, but does not meet the critical-error threshold. The response provides limited useful educational analysis.

### 0

Critical Education error, no answer, off-topic answer, fabricated evidence, materially discriminatory reasoning, or equivalent fundamental failure.
