# Agriculture Evaluation Rubric

## Task

Score **ONE answer to ONE Agriculture question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, agricultural context, and expected reasoning style. Accept different valid management strategies, calculations, models, or interpretations when the supplied constraints and evidence allow them.

Do not require a particular wording or answer structure. Do not reward unnecessary length, jargon, or citations.

## Critical Error

If `critical_agriculture_error` applies:

**score = 0**

A critical agriculture error is a materially false central agronomic, soil, crop, livestock, water, engineering, economic, quantitative, or experimental claim that fundamentally invalidates the answer or would lead to a seriously misleading conclusion.

Examples include a management-changing fertilizer-rate or irrigation-volume error; fundamentally misunderstanding plant-available water; confusing salinity with sodicity in a way that reverses interpretation; equating total soil nutrient with immediately available nutrient; treating pest presence as proof that intervention is justified; invalid stocking-rate, feed-basis, machinery-capacity, marginal-response, or causal-inference reasoning; or recommending management from incompatible units or a fundamentally invalid diagnosis.

Do **not** classify as critical: small rounding differences, minor arithmetic that does not change the conclusion, minor terminology mistakes, reasonable alternative management strategies, legitimate regional differences, clearly stated reasonable assumptions, small omissions, or defensible uncertainty.

## Score Anchors

### 4

The answer is accurate, directly addresses the question, applies appropriate agricultural principles, correctly performs relevant calculations, identifies important assumptions and constraints, and reaches a sound conclusion.

For crop and soil questions, it correctly handles relevant biological, physical, and chemical mechanisms. For livestock questions, it correctly handles feed, production, management, and biological constraints. For farm decisions, it considers relevant objectives, costs, resource constraints, risks, and trade-offs. For experimental and data questions, it correctly distinguishes observation from causal inference and appropriately interprets variability and uncertainty.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor agricultural imprecision, limited reasoning gap, non-central calculation error, or insufficient qualification. The central conclusion remains sound.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete agronomic reasoning, significant quantitative problems, weak assumptions, incorrect interpretation of some material factors, or incomplete treatment of relevant trade-offs.

### 1

The answer contains a substantial agricultural misunderstanding, inappropriate model, major non-critical calculation error, serious management-reasoning problem, or badly incomplete analysis, but does not meet the critical-error threshold.

### 0

Critical agriculture error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question. A crop physiology question need not contain machinery analysis; a livestock ration question need not discuss irrigation; a conceptual soil question need not contain economic analysis.

Conditional criteria should not penalize an answer for omitting irrelevant material. However, when a conditional consideration is essential to solving the question correctly, failure on that issue should affect the score.

All 20 criteria have equal nominal weight (`0.05`). Conditional criteria apply only when material to the task.

| Criterion | Conditional | Evaluation focus |
|---|---:|---|
| `relevance` | No | Directly answers the agricultural task and uses supplied context. |
| `agronomic_accuracy` | No | Correct agricultural relationships, context, timing, limiting factors, and diminishing returns. |
| `soil_and_nutrient_accuracy` | Yes | Soil physics/chemistry, plant-available water, salinity/sodicity, nutrient forms, balances, and losses. |
| `crop_science_reasoning` | Yes | Establishment, phenology, physiology, genetics, crop competition, yield components, and horticultural mechanisms. |
| `plant_health_reasoning` | Yes | Disease, pests, weeds, thresholds, resistance, diagnosis, and integrated management. |
| `animal_science_reasoning` | Yes | Dry matter, nutrient intake, maintenance, growth, reproduction, grazing, welfare, and environment. |
| `water_and_irrigation_reasoning` | Yes | Root-zone water, ET, irrigation depth/volume/efficiency, drainage, salinity, drought, and timing. |
| `agricultural_engineering_reasoning` | Yes | Machinery capacity, efficiency, power, traction, compaction, timeliness, and operational fit. |
| `farm_economic_reasoning` | Yes | Relevant costs, marginal returns, opportunity cost, break-even, liquidity, risk, and profit versus yield. |
| `systems_and_resource_reasoning` | Yes | Interactions and resource flows across crop, livestock, soil, water, labor, machinery, and finance. |
| `quantitative_correctness` | Yes | Correct equations, units, bases, arithmetic, magnitude, and interpretation. |
| `experimental_reasoning` | Yes | Experimental units, replication, randomization, blocking, confounding, interactions, and practical significance. |
| `data_interpretation` | Yes | Agricultural records, maps, tables, curves, variability, practical significance, and measurement quality. |
| `causal_and_mechanistic_reasoning` | No | Valid mechanisms and separation of association from causation. |
| `assumptions_and_model_selection` | Yes | Appropriate frameworks, assumptions, validity range, hidden limits, and needed information. |
| `risk_and_uncertainty` | Yes | Variability, downside risk, uncertainty, sensitivity, and limits to confidence. |
| `sustainability_and_tradeoff_reasoning` | Yes | Measurable environmental, biological, resource, and economic trade-offs rather than labels. |
| `completeness` | No | Material requested elements are addressed at an appropriate depth. |
| `consistency` | No | Internal consistency of units, assumptions, claims, calculations, and conclusions. |
| `clarity` | No | Clear, efficient, evaluable explanation without jargon or citation padding. |

## Quantitative Answers

A strong quantitative answer should, where appropriate:

- identify the relevant agricultural quantity;
- select the correct equation or relationship;
- use consistent units;
- distinguish bases such as per hectare, per animal, dry matter, as-fed basis, and nutrient mass versus fertilizer-product mass;
- perform calculations correctly;
- check biological and operational plausibility; and
- interpret the result.

Do not require every trivial arithmetic step. A correct numerical result obtained from fundamentally invalid agricultural reasoning should not receive full credit.

## Management Answers

Strong management answers should identify relevant:

- objectives and constraints;
- evidence and assumptions;
- biological mechanisms;
- expected benefits and costs;
- risks and trade-offs;
- opportunity costs;
- uncertainty; and
- information needed before implementation.

Do not require one predetermined management choice where several options are defensible.

## Experimental Answers

Strong experimental answers should identify, where applicable:

- experimental unit;
- treatments and controls;
- replication;
- randomization;
- blocking;
- field variability;
- measured response;
- uncertainty;
- confounding;
- interactions; and
- practical significance.

A statistically significant result does not automatically imply a large or economically meaningful agricultural effect.

## Systems Answers

For whole-farm and sustainability questions, strong answers should identify interactions among relevant components rather than analyzing every enterprise in isolation. Where appropriate, consider crops, livestock, soil, nutrients, water, labor, machinery, finances, risk, and environment. Do not require all components when only some are material.

## Important Evaluation Principles

### Agriculture is context-dependent

Agricultural recommendations often depend on climate, soil, crop, livestock species, growth stage, production system, management objectives, resource constraints, and economic conditions. Do not present context-dependent recommendations as universal rules.

### Yield is not profit

The treatment producing maximum biological yield may not maximize economic return.

### Total nutrient is not available nutrient

Soil nutrient reserves, fertilizer applied, nutrient available to crops, nutrient taken up, and nutrient removed in harvest are distinct quantities.

### More input is not always better

Water, fertilizer, seed, feed, pesticides, machinery, and labor often show diminishing returns or negative effects at excessive levels.

### Timing matters

Agricultural outcomes depend on timing as well as total quantities, including rainfall, irrigation, fertilizer, pest pressure, planting, harvest, and grazing.

### Soil properties interact

Physical, chemical, and biological soil properties should not be treated independently when interactions are important.

### Plant symptoms are often nonspecific

A symptom may have several possible causes. Strong answers should avoid confident diagnosis without enough evidence.

### Pest presence is not automatically economic damage

Intervention decisions should consider pest abundance, crop stage, expected damage, natural enemies, treatment cost, and economic thresholds where relevant.

### Production is not efficiency

Higher total production does not necessarily imply higher nutrient-use efficiency, water-use efficiency, feed efficiency, or profitability.

### Farm systems contain trade-offs

A practice may improve one outcome while worsening another.

### Experiments require replication and control

Field variability can easily produce misleading conclusions. Repeated subsamples are not a substitute for independent treatment replication.

### Association is not causation

Farm records and observational data may identify useful relationships without establishing causal effects.

### Multiple seasons and environments matter

Agricultural performance observed in one location or season may not generalize.

### Risk matters

Average yield or profit alone may not characterize the desirability of a management strategy.

### Sustainability is multidimensional

Do not treat "sustainable" as a single measurable outcome. Consider relevant dimensions such as productivity, soil condition, water use, nutrient losses, resilience, biodiversity, resource efficiency, and economic viability only when material to the question.

### No jargon worship

Agricultural terminology does not substitute for valid reasoning.

### No citation worship

Citations do not compensate for incorrect agronomic reasoning.
