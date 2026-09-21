# Design Evaluation Criteria

## Task

Score **ONE answer to ONE Design question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, users, context, constraints, and design objective.

Accept different valid design solutions when they satisfy the stated user needs, constraints, accessibility requirements, and evidence. Do not require a particular visual style, software tool, design framework, process vocabulary, or reference-answer wording.

Do not reward unnecessary length, jargon, framework-name dropping, or citations. The benchmark evaluates **design reasoning**, not resemblance to a reference answer or subjective stylistic preference.

## Critical Error

If `critical_design_error` applies, **score = 0** regardless of the rest of the answer.

**Condition:** A materially false central design, accessibility, interaction, research, usability, safety, product, service, ethics, sustainability, quantitative, or evaluation claim that fundamentally invalidates the answer or would lead to a seriously misleading or harmful design decision. Reasonable stylistic differences, defensible alternative solutions, minor terminology errors, small quantitative errors that do not change the conclusion, and cautious uncertainty are not critical.

Examples of critical errors include:

- Solving the requested feature instead of the explicitly evidenced underlying user problem in a way that invalidates the recommendation.
- Hiding a critical system state where safety, irreversible loss, or consequential action depends on users understanding that state.
- Removing necessary error recovery from a high-consequence workflow.
- Relying exclusively on color for critical information despite an explicit color-vision accessibility requirement.
- Claiming a design is accessible while ignoring an explicitly stated inability to operate it with a required input method.
- Counting a usability task as successful when the moderator completed or directly instructed the decisive interaction.
- Treating one participant's preference as established evidence about all users when that generalization drives the design decision.
- Using a research method that cannot answer the stated research question and presenting its result as decisive evidence.
- Making a major quantitative interpretation error that reverses the conclusion about user performance or task success.
- Treating visual attractiveness as proof of usability despite explicit evidence that users cannot complete the task.
- Removing essential information solely to achieve a minimalist appearance, causing the design to fail its stated task.
- Claiming fewer clicks prove a better workflow when the redesign materially increases errors or cognitive burden.
- Recommending an irreversible destructive action without appropriate warning, recovery, or confirmation in a high-risk context.
- Fundamentally reversing the meanings of affordance and signifier in a way that causes the interaction problem to be misdiagnosed.
- Designing only for an average physical user despite supplied anthropometric evidence showing substantial exclusion.
- Recommending a physical design that clearly violates an explicit safety constraint supplied in the question.
- Treating a high-fidelity prototype as validated implementation when core task structure remains untested.
- Claiming an A/B metric proves overall design superiority when the metric is explicitly misaligned with the user goal.
- Endorsing a deceptive pattern intended to trick users into a choice contrary to their stated interests.
- Fabricating user research findings, usability results, constraints, or accessibility evidence required for the conclusion.

The following are **not** critical by themselves: minor terminology mistakes; reasonable stylistic differences; different valid visual treatments; defensible alternative layouts; reasonable differences in prototype fidelity; legitimate design-system exceptions; minor spacing or typographic issues; small quantitative errors that do not affect the conclusion; cautious uncertainty; alternative design strategies that satisfy the requirements; or aesthetic disagreement where usability and constraints remain satisfied.

## Evaluation Criteria

All 20 criteria have equal weight (**0.05 each; total = 1.00**). Apply conditional criteria only when the question makes them relevant.

| ID | Criterion | Weight | Conditional | Operational definition |
|---|---|---:|:---:|---|
| `relevance` | Relevance | 0.05 | No | Directly addresses the stated Design question, its users, context, objective, and requested task without substituting a different problem or drifting into unrelated implementation, marketing, or aesthetic commentary. |
| `design_accuracy` | Design accuracy | 0.05 | No | Applies sound design reasoning and does not make materially false claims about visual communication, interaction, usability, accessibility, product design, service design, research, evaluation, ethics, or sustainability. |
| `conceptual_precision` | Conceptual precision | 0.05 | No | Uses design concepts precisely and preserves important distinctions such as problem versus solution, goal versus task, affordance versus signifier, legibility versus readability, stated preference versus observed behavior, and prototype versus validated implementation. |
| `problem_framing` | Problem framing | 0.05 | Yes | When relevant, identifies the underlying user or system problem, objectives, stakeholders, constraints, assumptions, unknowns, and success criteria instead of accepting a requested feature as the need by default. |
| `user_and_context_reasoning` | User and context reasoning | 0.05 | Yes | When relevant, reasons from users’ goals, tasks, capabilities, expertise, language, physical variation, environments, devices, interruptions, and other context-of-use factors rather than assuming one average user. |
| `design_research_reasoning` | Design research reasoning | 0.05 | Yes | When relevant, matches research methods to the question, distinguishes what people say from what they do, addresses sampling and moderation limits, and keeps claims proportional to the evidence. |
| `information_architecture_reasoning` | Information architecture reasoning | 0.05 | Yes | When relevant, organizes content around user goals and mental models, uses understandable labels, supports navigation/search/browse and findability, and allows appropriate overlap or multiple schemes rather than forcing an incoherent taxonomy. |
| `visual_communication_reasoning` | Visual communication reasoning | 0.05 | Yes | When relevant, connects hierarchy, alignment, spacing, typography, contrast, color, imagery, grouping, and composition to attention, comprehension, scanning, meaning, legibility, and readability rather than subjective taste alone. |
| `interaction_and_usability_reasoning` | Interaction and usability reasoning | 0.05 | Yes | When relevant, reasons about controls, mappings, states, transitions, feedback, discoverability, learnability, efficiency, error prevention, error recovery, and user control across both happy paths and failure states. |
| `product_form_and_function_reasoning` | Product form and function reasoning | 0.05 | Yes | When relevant, integrates form, physical affordances, ergonomics, user variation, materials, durability, manufacturability, assembly, maintenance, safety, packaging, and lifecycle rather than optimizing appearance in isolation. |
| `service_and_system_reasoning` | Service and system reasoning | 0.05 | Yes | When relevant, reasons across journeys, touchpoints, frontstage and backstage dependencies, handoffs, queues, channels, wayfinding, and failure recovery instead of optimizing one touchpoint at the expense of the end-to-end service. |
| `accessibility_and_inclusive_design` | Accessibility and inclusive design | 0.05 | Yes | When relevant, identifies and removes unnecessary barriers across perception, input, focus, reading order, labels, contrast, color use, cognition, language, device, environment, and physical variation while preserving the core task. |
| `prototyping_and_iteration` | Prototyping and iteration | 0.05 | Yes | When relevant, selects prototype form and fidelity according to what must be learned, recognizes prototype limitations, and proposes evidence-driven iteration rather than assuming higher fidelity or repeated redesign is inherently better. |
| `evaluation_and_testing` | Evaluation and testing | 0.05 | Yes | When relevant, uses representative tasks and appropriate measures such as success, errors, time, assistance, comprehension, findability, and satisfaction, and does not count moderator rescue or a misaligned metric as proof of success. |
| `tradeoff_and_constraint_reasoning` | Trade-off and constraint reasoning | 0.05 | Yes | When relevant, explicitly incorporates constraints and competing objectives, explains trade-offs, avoids universal rules where context matters, and justifies choices against the stated design objective. |
| `sustainability_and_design_ethics` | Sustainability and design ethics | 0.05 | Yes | When relevant, considers user agency, informed choice, privacy, manipulation risk, vulnerable users, lifecycle impacts, durability, repairability, material use, reuse, and end-of-life without rewarding deceptive conversion or one-attribute sustainability claims. |
| `evidence_and_uncertainty` | Evidence and uncertainty | 0.05 | Yes | When relevant, separates observation from inference, identifies missing evidence and uncertainty, avoids fabricated research or unjustified causal claims, and proposes proportionate validation when conclusions remain uncertain. |
| `completeness` | Completeness | 0.05 | No | Covers the material parts of the question and includes the major design factors needed for a useful answer, without requiring exhaustive discussion of irrelevant principles. |
| `consistency` | Consistency | 0.05 | No | Maintains internally coherent recommendations, terminology, assumptions, calculations, and rationale; later claims do not contradict earlier stated goals, constraints, or evidence. |
| `clarity` | Clarity | 0.05 | No | Communicates the design reasoning in a clear, specific, organized, and actionable way, using terminology only where it helps explain mechanisms or decisions. |

## Score Anchors

### 4 — Strong

The answer is design-accurate and directly addresses the problem. It identifies the relevant users, context, objective, constraints, evidence, and trade-offs, and proposes or evaluates a solution that is well justified.

Where relevant, it correctly handles problem framing; user needs; hierarchy; typography; information structure; navigation; interaction; system states; feedback; errors; accessibility; inclusive design; physical form and function; ergonomics; service journeys; research; prototypes; testing; sustainability; ethical implications; and uncertainty.

For visual questions, it connects design choices to communication and perception. For interaction questions, it explains states, actions, feedback, and recovery. For research questions, it matches methods and claims appropriately. For product questions, it considers users, form, function, materials, and constraints. For service questions, it reasons across touchpoints and operational dependencies. For accessibility questions, it removes unnecessary barriers without losing the core task.

Minor stylistic imperfections or harmless quantitative rounding do not reduce a 4.

### 3 — Substantially correct

The answer is substantially correct and useful but contains one meaningful omission, minor design imprecision, incomplete trade-off analysis, insufficient accessibility consideration, or limited evidence. The central recommendation or critique remains sound.

Examples include a secondary edge case omitted; correct hierarchy analysis with one weak implementation detail; a good interaction proposal with incomplete error-state treatment; an appropriate research method with incomplete sampling discussion; or a useful physical design with limited lifecycle discussion.

### 2 — Partly correct

The answer is directionally useful but has important omissions or reasoning weaknesses. It demonstrates meaningful design understanding but requires substantial improvement.

Examples include proposing a plausible design without adequately identifying the user goal; improving visual appearance while overlooking information architecture; identifying usability problems but proposing solutions unsupported by evidence; addressing the happy path while ignoring major failure states; improving visual accessibility while overlooking interaction accessibility; or choosing a research method only partly suited to the question.

### 1 — Major weakness

The answer contains a substantial design, usability, accessibility, research, product, service, visual, or interaction misunderstanding, but does not meet the critical-error threshold. It provides limited useful design reasoning.

### 0 — Fundamental failure

A `critical_design_error`, no answer, an off-topic answer, fabricated research evidence, or an equivalent fundamental failure.

## Important Evaluation Principles

### Design solves problems, not requests
A requested feature or artifact may not be the underlying user need. Distinguish stated solutions from the problem, opportunity, and implementation choice.

### Context matters
A design can work well for one user, task, or environment and poorly for another.

### Users are not averages
Account for meaningful variation in ability, expertise, language, device, physical characteristics, and environment.

### Attractive is not automatically usable
Visual appeal and task effectiveness are related but distinct.

### Simple is not the same as minimal
Removing necessary information or controls can make a design harder to use.

### Fewer clicks are not automatically better
Interaction quality depends on cognitive effort, errors, clarity, reversibility, task frequency, and user goals.

### Hierarchy supports communication
Visual emphasis should reflect information importance and task priority.

### Typography serves reading
Typography should support hierarchy, legibility, and readability rather than stylistic novelty alone.

### Color should not carry critical meaning alone
Use redundant cues when important information must remain understandable without color.

### Interaction requires feedback
Users should understand what happened, current state, and what can happen next.

### Error prevention and recovery both matter
A robust design prevents serious mistakes where possible and supports recovery when errors occur.

### Accessibility is part of design quality
Accessibility is not a late-stage optional add-on.

### Inclusion requires considering variability
Do not design solely around an imagined average user.

### Consistency is valuable but not absolute
Consistency can reduce learning burden, while contextual needs may justify deliberate exceptions.

### Information architecture affects findability
Visual polish cannot compensate for a fundamentally incoherent content structure.

### User statements and user behavior answer different questions
Stated preference is not identical to observed task success or behavior.

### Research methods must match research questions
Interviews, surveys, observation, analytics, and usability testing provide different kinds of evidence.

### A large sample does not eliminate sampling bias
Representativeness depends on recruitment and fit to the target population.

### Prototypes answer questions
Choose prototype form and fidelity according to what must be learned.

### Iteration should be evidence-driven
Repeated redesign without a learning objective is not meaningful iteration.

### Design metrics require alignment
Optimizing the wrong metric can worsen the user experience.

### Design systems are tools, not answers
Reusable components support consistency but do not eliminate contextual design judgment.

### Technology is not design quality
Adding automation, AI, or new technology does not automatically improve an experience.

### Automation needs user control
Consequential automated behavior should provide appropriate visibility, correction, recovery, and oversight.

### Services are end-to-end systems
Optimizing one touchpoint can worsen another or shift costs to users and staff.

### Physical design involves lifecycle trade-offs
Form, materials, durability, manufacturing, repairability, safety, and sustainability interact.

### Sustainability is systemic
A lighter or recyclable product is not automatically more sustainable across its full lifecycle.

### Ethical design respects user agency
Do not reward manipulation merely because it increases conversion or engagement.

### Design decisions involve trade-offs
Several valid solutions may satisfy the same problem differently; evaluate fit, evidence, and constraints.

### No framework worship
Naming a design framework or heuristic does not substitute for applying the mechanism correctly.

### No trend worship
A fashionable design is not necessarily a good design.

### No citation worship
Citations do not compensate for weak design reasoning.

## Domain-Specific Accuracy Notes

### Visual design
Identify hierarchy correctly; distinguish grouping from decoration; use contrast purposefully; recognize alignment and spacing relationships; distinguish legibility from readability; avoid treating preference as objective quality; and connect aesthetic choices to communication purpose.

### Interaction design
Identify relevant state; provide feedback; prevent and recover from errors; preserve user control; use predictable mappings; distinguish affordances from signifiers; consider edge cases; and avoid hidden consequential actions.

### Information architecture
Organize around user goals; use understandable labels; support findability; distinguish navigation from content hierarchy; recognize that multiple organizational schemes may be valid; and avoid organization-centric terminology where user language is clearer.

### Accessibility and inclusive design
Do not rely on one sensory channel; consider keyboard and non-pointer operation where relevant; provide meaningful labels; preserve focus and reading order conceptually; avoid color-only meaning; support adaptable presentation; and preserve the target task while removing unnecessary barriers.

### Product and industrial design
Consider form and function, user variation, ergonomics, materials, manufacturing and assembly constraints, maintenance, safety, packaging, repairability, and lifecycle. Do not optimize appearance in isolation.

### Service and systems design
Consider end-to-end journeys, touchpoints, handoffs, backstage dependencies, cross-channel continuity, queues, wayfinding, and failure recovery. Do not optimize isolated touchpoints at the expense of the overall service.

### Design research
Match method to research question; distinguish stated preference from observed behavior; avoid leading questions; recognize sampling limitations; distinguish qualitative and quantitative claims; identify facilitator effects; avoid overgeneralization; and connect insights to evidence.

### Prototyping
Choose fidelity based on what must be learned; recognize prototype limitations; avoid high-fidelity work before major structural uncertainty is resolved; distinguish prototypes from finished implementation; and iterate based on evidence.

### Design ethics and sustainability
Identify user interests, agency, privacy, manipulation risk, vulnerable users, lifecycle effects, repairability, and systemic trade-offs. Do not reward deceptive patterns or one-attribute sustainability claims.

## Scoring Procedure

1. Read the question metadata and identify the relevant design objective, users, context, constraints, and expected type of reasoning.
2. Check for `critical_design_error`. If present, assign **0**.
3. Apply the broadly relevant criteria and only the conditional criteria implicated by the question.
4. Use the 0–4 anchors to judge the answer as a whole. Equal criterion weights support consistency but do not require mechanical averaging when the answer contains a decisive central error short of the critical threshold.
5. Accept multiple solutions when they are compatible with the evidence and constraints. Score the quality of reasoning, not stylistic preference or framework vocabulary.
