# Architecture & Built Environment Evaluation Rubric

## Task

Score **ONE answer to ONE Architecture & Built Environment question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, building/site context, users, and assumptions.

Accept different valid spatial organizations, structural concepts, environmental strategies, material selections, construction approaches, urban-design responses, and adaptive-reuse strategies when they satisfy the stated requirements and are well justified.

Do not require a particular wording, aesthetic style, or design philosophy. Do not reward unnecessary length, jargon, famous-architect references, or citations. The benchmark evaluates architectural reasoning, technical understanding, built-environment integration, and practical design judgment rather than resemblance to a reference answer.

## Critical Error

If `critical_architecture_built_environment_error` applies:

**score = 0**

A critical error is a materially false central architectural, building-science, structural, construction, accessibility, life-safety, quantitative, or urban-design claim that fundamentally invalidates the answer or creates a materially unsafe conclusion.

Do **not** classify as critical: harmless rounding, minor arithmetic that does not change the conclusion, minor terminology mistakes, small omissions, reasonable assumptions stated openly, legitimate design alternatives, defensible material choices, differences in design philosophy, or minor drawing-interpretation imprecision.

## Score Anchors

### 4

The answer is architecturally and technically accurate, directly addresses the question, identifies the relevant spatial, environmental, structural, material, construction, and/or urban mechanisms, and reaches a well-supported conclusion.

Where relevant, it correctly handles program, adjacency, circulation, site, climate, daylight, ventilation, envelope, structure, materials, building services, accessibility, life safety, constructability, sustainability, urban context, lifecycle effects, and uncertainty.

For design questions, it identifies objectives and constraints and explains material trade-offs. For building-pathology questions, it distinguishes symptoms from likely causes and identifies evidence needed for diagnosis. For quantitative questions, calculations, units, areas, ratios, slopes, or performance metrics are correct. For urban-design questions, it considers both building form and relationships to streets, public space, movement, neighboring uses, and infrastructure.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor architectural or technical imprecision, limited reasoning gap, incomplete qualification, or insufficient consideration of an important secondary factor. The main conclusion remains sound.

### 2

The answer is partly correct or directionally useful but has important omissions, weak spatial reasoning, incomplete environmental or technical analysis, questionable assumptions, poor integration across systems, or insufficient consideration of users and context. It demonstrates meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial architectural, structural, envelope, accessibility, construction, quantitative, environmental, or urban-design misunderstanding, but does not meet the critical-error threshold. The response provides limited useful understanding.

### 0

Critical Architecture & Built Environment error, no answer, off-topic answer, unrelated fabrication, materially unsafe design reasoning, or equivalent fundamental failure.

## Evaluation Criteria

All 20 criteria have equal weight (`0.05`). Conditional criteria are applied only when the question materially invokes them.

| ID | Criterion | Conditional | Operational definition |
|---|---|---:|---|
| `relevance` | Relevance | No | Addresses the actual prompt, stated scenario, users, objectives, constraints, and requested output without drifting into unrelated architectural commentary. |
| `architectural_accuracy` | Architectural accuracy | No | Uses correct architectural concepts and relationships, including plan/section/elevation, gross versus net area, program, circulation, spatial organization, and building-scale implications. |
| `built_environment_accuracy` | Built-environment accuracy | No | Correctly reasons about sites, streets, blocks, public realm, density, connectivity, land-use relationships, urban climate, infrastructure context, and relevant spatial/temporal scales. |
| `spatial_reasoning` | Spatial reasoning | No | Accurately infers, compares, or proposes spatial relationships, routes, adjacencies, separations, sequences, orientations, and vertical/horizontal organization from the information given. |
| `conceptual_precision` | Conceptual precision | No | Keeps distinct concepts distinct—for example daylight vs direct sun, insulation vs thermal mass, airtightness vs ventilation, gravity load path vs lateral stability, density vs height, and intent vs measured performance. |
| `quantitative_correctness` | Quantitative correctness | Yes | When quantities are required, selects the correct denominator and units, performs calculations accurately, states assumptions, and interprets the result without overclaiming. |
| `program_and_circulation_reasoning` | Program and circulation reasoning | Yes | When relevant, translates users, activities, capacity, privacy, servicing, accessibility, public/staff/service flows, and emergency movement into coherent spatial relationships. |
| `site_and_context_reasoning` | Site and context reasoning | Yes | When relevant, integrates orientation, climate, topography, access, noise, views, drainage, vegetation, neighboring uses, hazards, infrastructure, and future change into the design reasoning. |
| `environmental_performance_reasoning` | Environmental performance reasoning | Yes | When relevant, correctly explains daylight, solar gain, shading, ventilation, thermal comfort, passive design, microclimate, energy, and their climate- and use-dependent trade-offs. |
| `structural_and_material_reasoning` | Structural and material reasoning | Yes | When relevant, maintains coherent load paths and stability concepts and evaluates material behavior, span, weight, moisture, fire, durability, constructability, maintenance, and embodied effects at an architectural level. |
| `building_systems_and_envelope_reasoning` | Building systems and envelope reasoning | Yes | When relevant, integrates services with architecture and correctly reasons about heat, air, water, vapor, insulation continuity, airtightness, thermal bridges, drainage, maintainability, and service zones. |
| `accessibility_and_life_safety` | Accessibility and life safety | Yes | When relevant, prioritizes inclusive primary routes and usable spaces and applies sound conceptual reasoning about egress, redundancy, compartmentation, smoke/fire separation, and safety without inventing code requirements. |
| `construction_and_constructability` | Construction and constructability | Yes | When relevant, identifies sequencing, logistics, tolerances, interfaces, movement, waterproofing, maintenance access, temporary dependencies, prefabrication constraints, and realistic buildability. |
| `urban_and_public_realm_reasoning` | Urban and public-realm reasoning | Yes | When relevant, evaluates how buildings and sites shape streets, frontages, public space, movement, permeability, microclimate, mixed use, density, and neighboring conditions. |
| `sustainability_and_lifecycle_reasoning` | Sustainability and lifecycle reasoning | Yes | When relevant, distinguishes operational and embodied effects and considers durability, adaptability, water, ecology, transport context, maintenance, replacement, reuse, resilience, and end-of-life boundaries. |
| `assumptions_and_tradeoffs` | Assumptions and trade-offs | Yes | When the problem is underdetermined or multi-objective, states material assumptions, identifies missing information, compares consequences, and avoids asserting a unique optimum without support. |
| `completeness` | Completeness | No | Covers the major mechanisms and constraints needed to answer the question; omissions are judged by their effect on the conclusion rather than by checklist length. |
| `consistency` | Consistency | No | The reasoning, calculations, spatial relationships, and conclusions do not contradict one another or the supplied scenario. |
| `clarity` | Clarity | No | Explains the mechanism and conclusion in a legible, precise way, using architectural vocabulary only where it adds meaning rather than as a substitute for reasoning. |
| `practical_design_judgment` | Practical design judgment | Yes | When a design decision is requested, proposes workable options that respect users, site, technical systems, safety, construction, maintenance, lifecycle, uncertainty, and legitimate alternative solutions. |

## Important Evaluation Principles

### Architecture integrates multiple systems
A strong answer should not treat spatial planning, structure, envelope, services, circulation, accessibility, and environmental performance as unrelated disciplines when they interact.

### Program before form
Architectural form should respond to users, activities, relationships, site, and constraints. Formal preference alone is not sufficient reasoning.

### Site matters
Orientation, climate, topography, access, noise, views, drainage, neighboring uses, and hazards can materially change the appropriate design strategy.

### Scale matters
A strategy appropriate at room scale may not work at building or urban scale.

### More glazing is not automatically better
Glazing can improve daylight and views while increasing heat loss, solar gain, glare, privacy issues, and facade cost.

### Airtightness is not ventilation
Reducing uncontrolled leakage does not remove the need for deliberate ventilation.

### Insulation is not thermal mass
They affect heat flow and thermal behavior through different mechanisms.

### Structure requires a continuous load path
Loads must reach the ground through a coherent structural system.

### Vertical load capacity is not complete stability
Buildings must also resist lateral actions and other relevant forces.

### Building-envelope performance is about continuity
Insulation, air control, water management, and other layers often fail at junctions and penetrations.

### Drainage matters
Water should normally be managed through reliable drainage paths rather than relying only on perfect surface sealing.

### Accessibility is integral to design
Accessibility should influence circulation, entrances, levels, spaces, wayfinding, and facilities from the outset.

### Compliance is not the same as design quality
A technically compliant solution may still perform poorly for users.

### Density is not height
Similar densities can be achieved with different building forms.

### Sustainability is multidimensional
Operational energy, embodied impact, durability, adaptability, water, transport, ecology, and lifecycle may all matter.

### Reuse is not automatically superior
Adaptive reuse may preserve existing resources but can involve structural, service, accessibility, environmental, or functional constraints.

### Lifecycle thinking matters
Initial construction cost or impact is only part of building performance over time.

### Building performance must be verified
Design intent, simulation, certification, or modeling does not automatically guarantee actual performance.

### Constructability matters
A concept that cannot be reliably built, waterproofed, accessed, maintained, or tolerated dimensionally is incomplete.

### Details matter
Interfaces among systems and materials are frequent points of failure.

### Urban design is relational
Buildings influence streets, public space, neighbors, mobility, climate, and infrastructure.

### Multiple valid designs
Architecture rarely has one mathematically unique optimum. Evaluate whether a proposal satisfies objectives and constraints and explains its trade-offs.

### No aesthetic absolutism
Aesthetic judgment may be relevant, but subjective preference should not replace functional, technical, environmental, or contextual reasoning.

### No jargon worship
Architectural terminology without mechanism or application should not receive a high score.

### No sustainability-label worship
Calling a proposal "green," "resilient," or "sustainable" without demonstrating relevant performance should not receive a high score.

### No citation worship
Citations do not compensate for incorrect architectural or building-science reasoning.

## Benchmark Balance Summary

### Difficulty distribution
- Difficulty 1: **2**
- Difficulty 2: **22**
- Difficulty 3: **38**
- Difficulty 4: **29**
- Difficulty 5: **9**

Total: **100**

### Major-domain distribution
- Architectural Design: **18**
- Site & Environmental Design: **15**
- Building Technology: **27**
- Construction & Delivery: **14**
- Urban & Built Environment: **13**
- Sustainability & Existing Buildings: **9**
- Cross-disciplinary: **4**

### Scoring note
A valid alternative design strategy can receive full credit when it satisfies the stated constraints, maintains life safety and accessibility, and explains its assumptions and trade-offs. Do not penalize an answer merely because it uses a different spatial organization, structural concept, environmental strategy, material system, or urban response from another valid solution.
