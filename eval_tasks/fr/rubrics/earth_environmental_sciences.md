# Earth & Environmental Sciences Evaluation Rubric
## Task
Score **ONE answer to ONE Earth & Environmental Sciences question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, spatial scale, temporal scale, and environmental context. Accept different valid scientific explanations, modeling approaches, calculations, and environmental-management strategies where the evidence and assumptions support them. Do not require a particular wording or structure. Do not reward unnecessary length, jargon, or citations. The benchmark evaluates scientific understanding, Earth-system reasoning, and environmental judgment rather than resemblance to a reference answer.
## Critical Error
If `critical_earth_environmental_sciences_error` applies, **score = 0**. Otherwise evaluate the applicable criteria.

**Condition:** A materially false central geological, physical, chemical, hydrological, ecological, climatic, quantitative, or environmental claim fundamentally invalidates the answer or reverses its substantive conclusion.

**Typical critical examples include:**
- Fundamentally incorrect plate-tectonic reasoning that reverses the inferred setting or process.
- A major geologic-sequence error that invalidates the reconstructed history.
- Claiming S waves propagate normally through Earth's liquid outer core.
- Fundamentally incorrect water-balance reasoning.
- A major mass-balance or energy-balance error that changes the conclusion.
- Claiming correlation alone establishes an environmental cause.
- Fundamentally confusing weather variability with long-term climate change.
- Claiming the greenhouse effect and ozone depletion are the same mechanism.
- Reversing a major climate feedback in a way that changes the conclusion.
- Claiming melting floating sea ice directly raises sea level by the same added-mass mechanism as melting land ice.
- Fundamentally incorrect groundwater-flow reasoning.
- Treating concentration and total pollutant mass as interchangeable in a way that changes the conclusion.
- Fundamentally incorrect interpretation of hazard, exposure, vulnerability, or risk.
- Claiming dilution necessarily destroys pollutant mass.
- Fundamentally misinterpreting the supplied environmental dataset, map, profile, or cross-section.
- Claiming an ecosystem relationship that directly contradicts the supplied evidence.
- A major quantitative error that reverses the environmental conclusion.
- Fabricating environmental observations required for the answer.
- Asserting a unique environmental outcome when the supplied evidence clearly cannot determine one.

**Do not treat the following as automatically critical:**
- Minor arithmetic errors that do not change the conclusion.
- Minor terminology mistakes.
- Harmless rounding differences.
- Legitimate scientific uncertainty.
- Defensible alternative assumptions that are stated.
- Reasonable model differences.
- Small omissions.
- Alternative but scientifically plausible environmental-management strategies.
- Disagreement about values or policy when scientific facts are handled correctly.
- Minor uncertainty in magnitude that does not alter the core mechanism.

## Score Anchors
### 4

The answer is scientifically accurate, directly addresses the question, identifies the relevant Earth or environmental processes, explains mechanisms correctly, and reaches a well-supported conclusion. Where relevant, it correctly handles spatial scale, temporal scale, stocks and flows, mass or energy balance, feedbacks, uncertainty, quantitative calculations, environmental risk, and interacting Earth-system components. For environmental-decision questions, it appropriately considers objectives, constraints, environmental effects, risks, trade-offs, and uncertainty. For underdetermined questions, it correctly identifies what cannot be established and what additional evidence would be needed. Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor scientific imprecision, limited mechanistic gap, incomplete scale consideration, or insufficient qualification. The main scientific conclusion remains correct.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete process reasoning, weak systems analysis, inappropriate assumptions, poor scale handling, or incomplete interpretation. It shows meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial Earth-science, environmental-science, quantitative, causal, or systems misunderstanding, but does not meet the critical-error threshold. The response provides limited useful understanding.

### 0

Critical Earth/environmental sciences error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Evaluation Criteria

| Criterion | Operational definition | Conditional? | Weight |
|---|---|:---:|---:|
| `relevance` | Directly addresses the scientific task posed, focuses on the material variables and relationships, and avoids irrelevant digressions. | No | 0.05 |
| `earth_science_accuracy` | Uses geologic, geophysical, geomorphic, atmospheric, oceanographic, hydrologic, cryospheric, and Earth-system facts and principles correctly where they are relevant; does not confuse core distinctions such as weathering versus erosion, crust versus plate, or land ice versus sea ice. | No | 0.05 |
| `environmental_science_accuracy` | Correctly represents ecological, biogeochemical, pollution, resource, sustainability, and human-environment processes relevant to the prompt, including material pathways, exposure, and environmental trade-offs. | No | 0.05 |
| `systems_reasoning` | Identifies relevant reservoirs, fluxes, interactions, feedbacks, and couplings among Earth-system components rather than treating connected processes as independent. | No | 0.05 |
| `conceptual_precision` | Uses key scientific concepts with the distinctions required by the question, such as stock versus flux, forcing versus feedback, hazard versus risk, concentration versus mass, and observation versus inference. | No | 0.05 |
| `quantitative_correctness` | When calculation is required, selects an appropriate relationship, uses consistent units and signs, performs arithmetic correctly, and interprets the numerical result in context. | Yes | 0.05 |
| `spatial_reasoning` | When spatial relationships matter, correctly interprets direction, geometry, gradients, landscape position, map or cross-section relationships, spatial heterogeneity, or scale-dependent patterns. | Yes | 0.05 |
| `temporal_and_scale_reasoning` | When relevant, distinguishes short-term variability from long-term change, local from regional or global behavior, transient from steady responses, and recognizes lags, residence times, and scale dependence. | Yes | 0.05 |
| `mass_and_energy_balance` | When applicable, conserves mass and energy, distinguishes inputs, outputs, storage, and transformation, and does not infer net change from a gross flux alone. | Yes | 0.05 |
| `causal_reasoning` | When causality is at issue, links causes to outcomes through a plausible mechanism, respects temporal order and confounding, considers multiple causes, and does not treat correlation alone as proof. | Yes | 0.05 |
| `data_interpretation` | When data are supplied or described, extracts the relevant pattern, trend, contrast, or anomaly accurately and avoids claims not supported by the data or their resolution. | Yes | 0.05 |
| `mechanistic_explanation` | When the task asks why or how, explains the physical, chemical, geological, hydrological, ecological, or biogeochemical mechanism rather than merely naming the process. | Yes | 0.05 |
| `assumptions` | When conclusions depend on simplifying conditions or missing information, identifies material assumptions and indicates how changing them could affect the result. | Yes | 0.05 |
| `uncertainty_analysis` | When uncertainty is material, characterizes its source and consequence proportionately, distinguishes uncertainty in magnitude from uncertainty in mechanism, and avoids either false precision or blanket agnosticism. | Yes | 0.05 |
| `risk_and_hazard_reasoning` | When hazards are involved, separates hazard, exposure, vulnerability, and resilience; recognizes compound or transferred risk; and does not equate a physical event automatically with disaster impact. | Yes | 0.05 |
| `evidence_evaluation` | When evidence quality matters, distinguishes direct measurement, proxy, model output, inference, and assumption; evaluates representativeness, sampling, calibration, alternative explanations, and corroborating evidence. | Yes | 0.05 |
| `completeness` | Covers the principal parts of the task and the major variables needed for a defensible answer, while not requiring exhaustive treatment of immaterial details. | No | 0.05 |
| `consistency` | Maintains internal logical, scientific, numerical, and unit consistency across the response and does not contradict its own stated assumptions or conclusions. | No | 0.05 |
| `clarity` | Communicates the reasoning in a coherent, interpretable way with enough structure to follow the scientific argument, without rewarding unnecessary jargon or length. | No | 0.05 |
| `environmental_decision_reasoning` | For decision questions, identifies objectives, constraints, mechanisms, risks, trade-offs, distribution of impacts, time horizons, and uncertainty, while allowing multiple defensible choices when evidence and values do not determine a unique option. | Yes | 0.05 |

## Important Evaluation Principles
### Mechanisms over labels

Naming an Earth or environmental process is not enough. Where the question requires reasoning, the answer should explain the mechanism and apply it correctly.

### Earth-system coupling

Atmosphere, hydrosphere, geosphere, biosphere, and cryosphere interact. Strong answers should recognize the couplings that are material to the prompt.

### Scale matters

Environmental conclusions can change with spatial and temporal scale. Local evidence should not automatically be generalized globally, and short-term variability should not automatically be treated as a long-term trend.

### Stocks and flows

Stored quantities and rates of movement are different. Large gross fluxes can coexist with small net stock changes when opposing fluxes are also large.

### Mass balance

Material cannot disappear from a system without an output, a transformation to another tracked form, or a change in storage.

### Energy balance

Climate and many environmental processes must respect energy conservation; claims about warming, cooling, phase change, or energy systems should be consistent with the relevant energy flows.

### Weather vs climate

A short-term weather event does not by itself establish or refute a long-term climate trend.

### Hazard vs risk

Hazard is not identical to risk. Risk also depends on exposure and vulnerability, and can be changed without changing the physical hazard.

### Forcing vs feedback

A forcing initiates a change in the climate energy balance; a feedback modifies the response to an initial change. They should not be treated as interchangeable.

### Correlation vs causation

Environmental correlation alone does not establish causation. Strong causal claims require mechanism, design, attribution evidence, or clearly stated assumptions.

### Model interpretation

Models are simplified representations rather than exact copies of Earth systems. Model limitations matter, but do not make models useless.

### Uncertainty

Represent uncertainty proportionately. Uncertainty about magnitude does not necessarily imply uncertainty about whether a mechanism exists or what direction it acts.

### Environmental trade-offs

Interventions can create benefits in one dimension and costs in another. Material trade-offs, burden shifting, and displaced impacts should be identified when relevant.

### Multiple valid decisions

Environmental-management questions may have more than one defensible answer. Evaluate evidence, assumptions, objectives, constraints, trade-offs, and uncertainty rather than demanding a predetermined policy choice.

### No jargon worship

Technical terminology without correct Earth-system reasoning should not receive a high score.

### No citation worship

Citations do not compensate for incorrect Earth or environmental science reasoning.

## Application Notes
- Apply only the criteria that are material to the specific prompt, but always consider the broadly applicable criteria.
- Difficulty describes the reasoning complexity of the question, not the amount of terminology in the answer.
- Reasonable rounding receives full credit when it does not alter the conclusion.
- For underdetermined tasks, identifying the limits of inference can be part of a fully correct answer.
- Environmental-management answers should not lose credit merely for selecting a different defensible strategy if the scientific reasoning, assumptions, objectives, and trade-offs are handled well.
- An answer that uses the correct technical label with an incorrect mechanism should not receive a high score.
