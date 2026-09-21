# Biology & Life Sciences Evaluation Rubric

## Task

Score **ONE answer to ONE Biology & Life Sciences question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, biological context, and expected reasoning style. Accept different valid mechanisms, models, hypotheses, or experimental strategies when the supplied evidence permits them. Do not require particular wording or a fixed solution structure. Do not reward unnecessary length, jargon, or citations.

## Critical Error

If `critical_biology_life_sciences_error` applies:

**score = 0**

A critical error is a materially false central biological, genetic, biochemical, evolutionary, ecological, physiological, quantitative, or experimental claim that fundamentally invalidates the answer or reverses the substantive conclusion.

Examples include: fundamentally incorrect replication/transcription/translation reasoning; major inheritance or allele-frequency errors that reverse the conclusion; treating dominance as allele frequency; teleological adaptive-mutation claims; confusing acclimation with evolution; fundamentally wrong phylogenetic interpretation; major enzyme/metabolic errors such as claiming catalysts change equilibrium; unsupported reversal of an ecological relationship; treating correlation as causal proof when central; invalid feedback reasoning; treating technical replicates as independent biological replicates; claiming expression correlation proves gene function; claiming sequence similarity uniquely establishes function; or a major calculation error that changes the biological conclusion.

Do **not** classify as critical: minor arithmetic slips that do not change the biological conclusion, minor terminology errors, small omissions, reasonable approximations, reasonable differences in assumptions, legitimate alternative mechanisms consistent with the data, minor taxonomic imprecision, non-central pathway uncertainty, or appropriate caution about ambiguous evidence.

## Score Anchors

### 4
The answer is accurate, directly addresses the question, applies appropriate biological principles, correctly explains relevant mechanisms, handles quantitative reasoning correctly where needed, identifies important assumptions or alternative explanations where material, and reaches a biologically sound conclusion.

For experimental questions, it correctly distinguishes observation from causal inference and identifies relevant controls or limitations. For evolutionary questions, it reasons in population-level, non-teleological terms. For ecological questions, it recognizes interacting mechanisms, scale, and uncertainty. For molecular and genetic questions, it correctly distinguishes sequence, regulation, expression, function, and phenotype where relevant.

### 3
The answer is substantially correct and useful but contains one meaningful omission, minor biological imprecision, limited reasoning gap, non-central calculation error, insufficient discussion of assumptions, or incomplete qualification. The central biological conclusion remains correct.

### 2
The answer is partly correct or directionally useful but has important omissions, incomplete mechanistic reasoning, weak causal inference, significant quantitative problems, incorrect interpretation of some material evidence, or incomplete treatment of relevant biological context. It demonstrates meaningful understanding but is not reliably complete or correct.

### 1
The answer contains a substantial biological misunderstanding, inappropriate model, major non-critical genetics/evolution/ecology error, serious experimental-design problem, or badly incomplete mechanistic reasoning, but does not meet the critical-error threshold.

### 0
Critical biology/life-sciences error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question. A cell-biology question need not contain ecological reasoning; a phylogenetics question need not discuss physiology; a qualitative mechanism question need not contain quantitative calculations. Conditional criteria should not penalize an answer for omitting material the question does not call for. However, when a conditional issue is essential to solving the question correctly, failure on that issue should affect the score.

## Evaluation Criteria

### Relevance (`relevance`)

Directly addresses the biological question asked, prioritizes information that bears on the requested conclusion, and avoids digressions that do not help solve the task.

### Biological Accuracy (`biological_accuracy`)

Uses correct biological concepts and relationships. Major errors about DNA/RNA/protein processes, inheritance, physiology, ecology, evolution, or experimental interpretation materially reduce credit even if the final conclusion happens to be correct.

### Molecular and Cellular Accuracy (`molecular_and_cellular_accuracy`)

When applicable, accurately distinguishes DNA sequence, transcription, RNA processing, translation, protein abundance, signaling, membrane transport, organelle function, cytoskeletal processes, cell-cycle control, and related cellular mechanisms.

### Genetic Reasoning (`genetic_reasoning`)

When applicable, correctly reasons about segregation, dominance, linkage, recombination, penetrance, expressivity, epistasis, polygenic traits, heritability, genomic variation, and genotype-versus-phenotype relationships.

### Biochemical and Metabolic Reasoning (`biochemical_and_metabolic_reasoning`)

When applicable, correctly reasons about enzyme kinetics, affinity versus catalytic rate, inhibition, thermodynamics, pathway flux, redox/energy coupling, allostery, and metabolic regulation without equating intermediate concentration with pathway flux.

### Evolutionary Reasoning (`evolutionary_reasoning`)

When applicable, explains evolution as population-level change across generations; distinguishes selection, drift, mutation, migration, and recombination; treats fitness as context-dependent reproductive success; and avoids teleological explanations.

### Ecological Reasoning (`ecological_reasoning`)

When applicable, correctly analyzes population, community, and ecosystem processes, including direct and indirect interactions, density dependence, resource limitation, trophic effects, scale, temporal dynamics, resilience, and non-equilibrium behavior.

### Physiological Reasoning (`physiological_reasoning`)

When applicable, correctly explains homeostasis, feedback, signaling, transport, respiration, osmoregulation, endocrine effects, nervous-system function, plant physiology, or comparative physiological responses, distinguishing acclimation from adaptation.

### Quantitative Correctness (`quantitative_correctness`)

When applicable, chooses the appropriate relationship, calculates accurately, uses units and probabilities correctly, interprets the biological meaning of the result, and checks plausibility. A minor arithmetic slip is less serious than using a biologically invalid model.

### Mechanistic Reasoning (`mechanistic_reasoning`)

Explains how the relevant biological entities and processes produce the observed outcome, including direction of effects, material intermediate steps, regulation, and interactions. Merely naming a pathway, gene, process, or concept is insufficient.

### Causal Inference (`causal_inference`)

When applicable, distinguishes correlation from causation, necessity from sufficiency, direct from indirect effects, upstream from downstream consequences, and association from demonstrated function. It avoids causal claims that the design or evidence cannot support.

### Experimental Design (`experimental_design`)

When applicable, identifies suitable hypotheses, manipulations, positive/negative controls, randomization or blinding where relevant, biological replication, confounders, rescue logic, and what conclusions the design can and cannot establish.

### Data Interpretation (`data_interpretation`)

When applicable, accurately distinguishes measurements from inferences, handles normalization and replicate structure appropriately, considers detection limits and batch effects, and avoids treating statistical significance or a single readout as definitive biological proof.

### Model Selection and Assumptions (`model_selection_and_assumptions`)

When applicable, selects an appropriate biological model, states or recognizes its important assumptions, identifies violations, and does not continue using a simplified model when supplied evidence clearly invalidates its assumptions.

### Scale and Systems Reasoning (`scale_and_systems_reasoning`)

When applicable, connects mechanisms across molecular, cellular, tissue, organismal, population, community, or ecosystem levels without assuming that an effect at one level automatically determines outcomes at another.

### Uncertainty and Variability (`uncertainty_and_variability`)

When applicable, recognizes genetic, environmental, developmental, temporal, stochastic, population, species, measurement, and sampling variation; qualifies generalizations; and treats ambiguous or noisy evidence with appropriate caution.

### Evidence and Model Limits (`evidence_and_model_limits`)

When applicable, states what the supplied evidence supports, what remains underdetermined, what assumptions connect observation to inference, and which alternative mechanisms or follow-up evidence could discriminate among explanations.

### Completeness (`completeness`)

Covers the material parts of the task, including required calculations, mechanisms, comparisons, assumptions, controls, caveats, or interpretations, without requiring unnecessary detail unrelated to the prompt.

### Consistency (`consistency`)

Maintains internally compatible claims, calculations, biological assumptions, and conclusions. It should not contradict earlier statements or switch biological models without explanation.

### Clarity (`clarity`)

Communicates the reasoning in a precise, understandable way with correct use of biological terminology. Clear explanation is valued over jargon, excessive length, or citation density.

## Important Evaluation Principles

### Biological mechanism
Strong answers should explain **how** a biological process produces the observed outcome. Naming a pathway, gene, or process is not sufficient by itself.

### Correlation is not causation
Association among genes, traits, species, expression levels, or experimental variables does not automatically establish a causal relationship.

### Genotype is not phenotype
Phenotype can depend on genotype, gene regulation, genetic background, environment, developmental stage, and stochastic processes.

### Gene expression is not gene function
Differential expression alone does not prove that a gene causes an observed phenotype.

### Evolution is population change
Evolutionary explanations should concern changes in populations across generations. Individual acclimation is not evolution.

### No teleology
Do not explain evolutionary change by claiming organisms acquired traits because they "needed" them.

### Fitness is contextual
Evolutionary fitness is reproductive success relative to a particular environment and population context.

### Biology is variable
Biological systems contain genetic, environmental, measurement, stochastic, temporal, developmental, and individual variation. Strong answers should avoid unjustified universal claims.

### Experiments require controls
Strong experimental reasoning should identify controls necessary for causal interpretation.

### Replication matters
Biological and technical replicates are not interchangeable. Pseudoreplication weakens inference about biological variability.

### Observation versus inference
Strong answers should distinguish: (1) what was measured, (2) what was inferred, (3) what assumptions connect the two, and (4) what alternative explanations remain.

### Multiple scales
Biological explanations may operate across molecules, cells, tissues, organisms, populations, communities, and ecosystems. Do not assume a mechanism at one scale automatically explains outcomes at another scale.

### Model assumptions
Models such as Hardy-Weinberg equilibrium, Michaelis-Menten kinetics, exponential growth, logistic growth, diffusion, binding equilibria, and simplified gene-regulatory models rely on assumptions. Strong answers should recognize material assumption violations.

### Phylogenetic reasoning
Tree topology, not left-right ordering or branch rotation, determines evolutionary relationships. Visual distance is not evolutionary distance unless an appropriate scale is defined.

### Statistical evidence
Statistical significance does not by itself establish biological importance or causal mechanism.

### Underdetermination
If evidence cannot uniquely identify a gene, pathway, mechanism, ecological interaction, or evolutionary process, recognizing that ambiguity is correct.

### Multiple valid explanations
Where several mechanisms fit the supplied evidence, do not require unjustified certainty. A valid, well-justified alternative explanation should receive full credit when consistent with the evidence.

### No jargon worship
Biological terminology without accurate mechanism or reasoning should not receive high credit.

### No citation worship
Citations do not compensate for incorrect biological reasoning.

## Quantitative Answers

A strong quantitative answer should, where appropriate:
- choose the correct biological model;
- identify relevant variables;
- perform the calculation correctly;
- use appropriate units;
- interpret the result biologically;
- test whether the result is plausible; and
- acknowledge important model assumptions.

Do not require every trivial arithmetic step. A correct number derived from a biologically invalid model should not receive full credit.

## Mechanistic Answers

A strong mechanistic answer should identify relevant biological entities, direction of effects, material intermediate steps, regulation, causal relationships, and alternative explanations when evidence is incomplete. Merely naming a pathway or biological process is insufficient when explanation is requested.

## Genetics Answers

Strong answers should distinguish genotype from phenotype, dominance from frequency, linkage from causation, penetrance from expressivity, heritability from determinism, and gene association from gene function.

## Evolutionary Answers

Strong answers should reason at the population level, identify sources of heritable variation, distinguish selection/drift/migration/mutation, avoid teleological claims, recognize environmental context, and distinguish adaptation from acclimation.

## Ecological Answers

Strong answers should identify relevant species interactions, recognize direct and indirect effects, consider spatial and temporal scale, avoid assuming equilibrium without justification, and recognize competing mechanisms where relevant.

## Experimental Answers

Strong answers should identify, where applicable, the hypothesis, prediction, manipulation, controls, replication, confounders, measured outcome, uncertainty, alternative explanations, and what conclusions the experiment can and cannot support. Experimental complexity does not compensate for failure to test the actual hypothesis.

## Scientific Judgment

For open-ended questions, do not require a single predetermined answer when multiple mechanisms are compatible with the supplied evidence. Reward answers that distinguish evidence from assumptions, identify biologically plausible alternatives, respect scale and context, recognize uncertainty, avoid unjustified causal claims, explain trade-offs, and identify model limitations.
