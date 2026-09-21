# Data & Information Science Evaluation Rubric

## Task

Score **ONE answer to ONE Data & Information Science question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, data context, and assumptions.

Accept different valid schema designs, database approaches, pipeline architectures, information organizations, retrieval strategies, and governance approaches when they are supported by the stated requirements.

Do not require a particular wording or structure. Do not reward unnecessary length, jargon, software-name dropping, or citations. The benchmark evaluates data understanding, information reasoning, system design, analytical interpretation, and practical judgment rather than resemblance to a reference answer.

## Critical Error

If `critical_data_information_science_error` applies:

**score = 0**

A critical Data & Information Science error is a materially false central data, database, information, quantitative, analytical, retrieval, privacy, or governance claim that fundamentally invalidates the response or reverses its substantive conclusion.

Representative critical errors include:
- treating missing values as zero when doing so materially changes the analysis;
- accepting duplicated totals from a many-to-many join as valid;
- fundamentally misunderstanding dataset grain;
- using a wrong denominator that reverses the conclusion;
- treating a nonunique or unstable field as a guaranteed primary identifier when this corrupts the result;
- combining incompatible units without conversion;
- ignoring a material change in historical definitions;
- claiming schema or type validity proves factual correctness;
- claiming technical compatibility proves semantic equivalence;
- reversing precision and recall in a material retrieval decision;
- claiming metadata or provenance proves the data are correct;
- claiming removal of direct identifiers guarantees anonymity;
- fundamentally misreading a chart or table;
- claiming aggregate association establishes causation;
- recommending pipeline logic that systematically duplicates or loses records;
- claiming a large dataset is necessarily representative;
- treating known provenance as proof of source reliability;
- destructive cleaning based on an invalid assumption; or
- fabricating required data, metadata, lineage, schema, or source information.

Do **not** classify as critical: harmless rounding, minor arithmetic that does not change the conclusion, small terminology mistakes, minor omissions, reasonable uncertainty, defensible alternative schemas or architectures, reasonable taxonomy or ontology differences, different valid visualizations, or alternative cleaning strategies that preserve meaning.

## Evaluation Criteria

All 20 criteria have equal nominal weight **0.05**. Conditional criteria are scored only when the question makes them materially relevant; non-applicable criteria should not be used to penalize an answer.

| ID | Criterion | Weight | Conditional | Operational definition |
|---|---|---:|:---:|---|
| `relevance` | Relevance | 0.05 | No | Directly addresses the question asked, the stated decision or information need, and the supplied data context. Does not substitute a different problem or focus on irrelevant tools, jargon, or side issues. |
| `data_information_accuracy` | Data & Information Science Accuracy | 0.05 | No | Makes materially correct claims about data, information, databases, retrieval, analytics, governance, and related concepts. Preserves the meaning of identifiers, units, null states, grain, time, metadata, and other supplied facts. |
| `conceptual_precision` | Conceptual Precision | 0.05 | No | Uses core distinctions correctly, including data versus metadata, entity versus record, schema versus instance, identifier versus measurement, missing versus zero, lineage versus provenance, and technical versus semantic compatibility when relevant. |
| `data_quality_reasoning` | Data Quality Reasoning | 0.05 | Yes | When quality is material, evaluates relevant dimensions such as accuracy, completeness, consistency, validity, uniqueness, timeliness, plausibility, integrity, and representativeness; recognizes trade-offs and avoids treating quality as a single context-free score. |
| `quantitative_correctness` | Quantitative Correctness | 0.05 | Yes | When calculations or numeric interpretation are required, uses correct arithmetic, denominators, weights, units, rates, counts, distinctness, and multiplicity; avoids false precision and explains what the computed quantity represents. |
| `data_modeling_reasoning` | Data Modeling Reasoning | 0.05 | Yes | When modeling structure is relevant, identifies entities, attributes, relationships, keys, grain, normalization or denormalization trade-offs, and appropriate relational, document, graph, hierarchical, or analytical representations. |
| `database_and_query_reasoning` | Database and Query Reasoning | 0.05 | Yes | When database operations are relevant, correctly reasons about keys, constraints, transactions, filtering, grouping, aggregation, ordering, join types, and cardinality without depending on vendor-specific syntax. |
| `data_integration_reasoning` | Data Integration Reasoning | 0.05 | Yes | When combining sources, checks grain, identifiers, relationship cardinality, units, time alignment, schema compatibility, semantic definitions, duplication risk, and unmatched records; proposes mappings or transformations that preserve meaning. |
| `pipeline_and_system_reasoning` | Pipeline and System Reasoning | 0.05 | Yes | When pipelines or architecture are involved, identifies sources, destinations, dependencies, transformation order, validation, retries, idempotency, latency, failure points, recovery, observability, and downstream effects in proportion to the scenario. |
| `metadata_and_provenance_reasoning` | Metadata and Provenance Reasoning | 0.05 | Yes | When interpretation or traceability depends on context, identifies needed definitions, units, allowed values, ownership, versions, origins, collection methods, transformations, timestamps, lineage, or provenance, while recognizing that provenance does not prove correctness. |
| `information_retrieval_reasoning` | Information Retrieval Reasoning | 0.05 | Yes | When search or retrieval is involved, correctly reasons about information need, relevance, indexing, keyword versus semantic matching, ranking, filtering, precision, recall, top-k behavior, and context-specific trade-offs. |
| `information_organization_reasoning` | Information Organization Reasoning | 0.05 | Yes | When organizing information, selects and explains suitable hierarchies, facets, taxonomies, ontologies, controlled vocabularies, thesauri, tags, labels, or navigation structures, while recognizing overlap, ambiguity, and multiple defensible designs. |
| `data_interpretation` | Data Interpretation | 0.05 | Yes | When interpreting results, identifies the correct unit, population, denominator, comparison, time period, subgroup, and level of aggregation; distinguishes descriptive patterns from causal claims and ties conclusions to what the data actually support. |
| `governance_privacy_and_context` | Governance, Privacy, and Context | 0.05 | Yes | When governance or responsible use is material, considers ownership, stewardship, accountability, access, retention, security, minimization, privacy, re-identification risk, purpose, secondary use, and contextual constraints without inventing jurisdiction-specific law. |
| `assumptions_and_uncertainty` | Assumptions and Uncertainty | 0.05 | Yes | When information is incomplete or ambiguous, states material assumptions, distinguishes observed facts from inferred or estimated values, identifies unknowns and limitations, and avoids unsupported certainty. |
| `error_detection_and_debugging` | Error Detection and Debugging | 0.05 | Yes | When the task involves a failure or suspicious result, identifies plausible mechanisms, uses evidence to localize the error, proposes discriminating checks, and avoids destructive fixes that could hide or worsen the underlying problem. |
| `completeness` | Completeness | 0.05 | No | Covers the important parts of the requested reasoning at an appropriate level of detail. Omissions do not leave a major requirement, trade-off, or consequence unaddressed. |
| `consistency` | Consistency | 0.05 | No | Maintains internally consistent definitions, units, assumptions, calculations, and conclusions. Later claims do not contradict earlier reasoning or silently change the data grain, denominator, or meaning. |
| `clarity` | Clarity | 0.05 | No | Explains the reasoning in a clear, precise, and usable way. Important distinctions, calculations, and recommendations are easy to follow without unnecessary jargon or length. |
| `practical_data_judgment` | Practical Data Judgment | 0.05 | Yes | When design or action is requested, proposes a defensible solution fitted to the use case, risks, scale, latency, and evidence; preserves provenance where uncertainty remains and does not reward unnecessary architectural or tooling complexity. |

## Important Evaluation Principles

### Grain before calculation
Understand what each row or record represents before joining, aggregating, or interpreting data.

### Identifiers are not measurements
A numeric-looking identifier is not automatically a quantitative variable.

### Missing is not zero
Missing, unknown, not applicable, censored, and zero may represent fundamentally different states.

### Technical validity is not factual correctness
Passing schema, type, or format validation does not prove a value is true, plausible, or meaningful.

### Semantic consistency matters
Fields with the same name may have different definitions, and differently named fields may represent the same concept.

### Joins can change multiplicity
Joining data can duplicate or remove observations; relationship cardinality and grain must be checked.

### Data quality is multidimensional
Accuracy, completeness, timeliness, consistency, validity, uniqueness, integrity, plausibility, and representativeness can conflict. Quality is use-case dependent.

### More data is not automatically better
Large biased, stale, duplicated, or poorly measured datasets can remain unsuitable.

### Provenance matters
Origin and transformation history affect interpretation and trust, but provenance does not guarantee correctness.

### Definitions matter
Metrics require explicit definitions, numerators, denominators, units, time periods, populations, and segmentation rules.

### Aggregation can mislead
Aggregates can hide subgroup patterns, outliers, duplication, and changing composition.

### Visualization can mislead
Technically correct values can still be presented with scales, encodings, or aggregation choices that distort interpretation.

### Search depends on information need
Retrieval quality depends on user intent and task; precision, recall, and ranking trade-offs are contextual.

### Metadata enables interpretation
Data without sufficient definitions, units, ownership, provenance, and null meanings can be unusable or easily misinterpreted.

### Reproducibility is not validity
A perfectly repeatable analysis can still be wrong.

### Governance is not bureaucracy for its own sake
Governance should support reliable definitions, accountability, access, quality, privacy, retention, and appropriate use.

### Privacy is not just removing names
Quasi-identifiers and linkable attributes can preserve re-identification risk.

### Descriptive analysis is not causal inference
Patterns and associations in observational data do not automatically establish causes.

### Uncertainty matters
Strong answers identify missing information, ambiguous definitions, measurement limitations, inferred values, and unknowns.

### Multiple valid architectures
Several data-system designs may be defensible; score fit to requirements and trade-offs rather than conformity to one architecture.

### No jargon worship
Terminology without correct mechanism-level reasoning should not receive a high score.

### No tool worship
Naming a database, warehouse, visualization tool, or platform does not by itself establish a good solution.

### No citation worship
Citations do not compensate for incorrect data or information reasoning.


## Score Anchors

### 4

The answer is technically and conceptually accurate, directly addresses the question, identifies the relevant data or information issue, and reaches a well-supported conclusion.

Where relevant, it correctly handles:
- data grain;
- types and representations;
- missingness;
- identifiers;
- joins and relationship cardinality;
- aggregation;
- data quality;
- database relationships and transactions;
- transformations;
- units and time;
- metadata and provenance;
- information retrieval and relevance;
- semantic meaning;
- visualization;
- governance and privacy; and
- uncertainty.

For integration questions, it checks identifiers, grain, units, temporal alignment, and semantic compatibility.

For pipeline questions, it identifies relevant dependencies, failure points, validation, recovery, retry, and observability concerns.

For retrieval questions, it appropriately reasons about user intent, precision, recall, ranking, and relevance.

For quantitative questions, calculations, units, and denominators are correct.

For interpretation questions, it distinguishes what the data show from unsupported causal conclusions.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor data-modeling imprecision, limited reasoning gap, incomplete qualification, or insufficient consideration of an important secondary factor. The main conclusion remains correct.

### 2

The answer is partly correct or directionally useful but has important omissions, weak data-quality reasoning, incomplete integration logic, questionable assumptions, poor treatment of metadata or granularity, or incomplete interpretation. It demonstrates meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial data, database, information-retrieval, quantitative, governance, privacy, or analytical misunderstanding but does not meet the critical-error threshold. The response provides limited useful understanding.

### 0

Critical Data & Information Science error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Scoring Notes

- Apply the critical-error check first.
- Judge only criteria that are relevant to the question and answer.
- Do not infer missing facts that the answer would have needed to state.
- Accept more than one defensible data model, architecture, ontology, taxonomy, retrieval strategy, or governance approach when the reasoning fits the requirements.
- Reward explicit handling of grain, denominator, semantic meaning, uncertainty, and provenance when those issues are material.
- Do not award high scores for fashionable terminology without mechanism-level reasoning.
