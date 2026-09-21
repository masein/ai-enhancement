# Language & Literature Evaluation Rubric

## Task

Score **ONE answer to ONE Language & Literature question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, textual/linguistic context, and expected reasoning style.

Accept different valid grammatical analyses, literary interpretations, translations, theoretical readings, and editorial judgments where the supplied evidence supports them.

Do not require a particular theoretical school, terminology set, translation, or essay structure unless specified. Do not reward unnecessary length, jargon, famous-author trivia, critic names, or citations.

## Critical Error

If `critical_language_literature_error` applies: **score = 0**.

A critical Language & Literature error is a materially false central linguistic, textual, interpretive, translational, rhetorical, or literary claim that fundamentally invalidates the answer or reverses the substantive conclusion.

Examples include a central structural misparse, reversed entailment, decisive author/narrator or poet/speaker confusion, invented central textual evidence, meaning-reversing mistranslation, decisive historical-semantic error, impossible textual-critical conclusion, or corpus inference invalidated by a supplied confound.

Do **not** classify minor terminology differences, punctuation preferences, reasonable alternative interpretations, legitimate theoretical disagreements, defensible translations, reasonable uncertainty, or non-central omissions as critical.

## Score Anchors

### 4
The answer is linguistically and literarily accurate, directly addresses the question, uses the supplied textual evidence correctly, explains relevant linguistic or literary mechanisms, recognizes ambiguity and context where appropriate, and reaches a well-supported conclusion.

For grammar questions, it correctly analyzes the relevant structure. For close-reading questions, it connects specific textual details to interpretation. For narrative questions, it correctly distinguishes author, narrator, perspective, and character where relevant. For translation questions, it recognizes semantic and stylistic trade-offs. For theoretically open questions, it supports interpretation without presenting one defensible reading as uniquely mandatory.

### 3
The answer is substantially correct and useful but contains one meaningful omission, minor linguistic or literary imprecision, limited evidence use, non-central interpretive gap, or insufficient qualification. The central analysis remains sound.

### 2
The answer is partly correct or directionally useful but has important omissions, weak textual evidence, significant grammatical or interpretive problems, incomplete close reading, or insufficient treatment of ambiguity or context. Meaningful correction would be required.

### 1
The answer contains a substantial linguistic or literary misunderstanding, inappropriate interpretation, major non-critical textual error, or badly incomplete analysis, but does not meet the critical-error threshold.

### 0
Critical Language & Literature error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question. A syntax question need not discuss literary theory; a poem-analysis question need not discuss corpus linguistics; a translation question need not analyze narrative focalization unless relevant. Conditional criteria should not penalize answers for omitting irrelevant material. However, when a conditional issue is essential to correctness, failure on that issue should affect the score.

## Evaluation Criteria

### `relevance` — Relevance (broadly applicable, weight 0.05)
Directly addresses the task actually asked, prioritizes the supplied passage, linguistic data, comparison, or scenario, and avoids substituting unrelated biography, plot trivia, theory labels, or generic commentary.

### `linguistic_accuracy` — Linguistic Accuracy (broadly applicable, weight 0.05)
Uses linguistic facts and distinctions accurately when present, including grammaticality versus prescription, phoneme versus spelling, morphology versus syntax, semantics versus pragmatics, synchronic pattern versus historical origin, and systematic dialect variation. A minor terminology slip is less serious than a structurally false analysis or stigmatizing claim about a language variety.

### `literary_accuracy` — Literary Accuracy (broadly applicable, weight 0.05)
Represents the supplied text faithfully; distinguishes author, narrator, speaker, character, and reader interpretation; preserves chronology, perspective, genre facts, and formal features; and does not invent quotations, scenes, actions, or textual details. A plausible reading is not accurate if it depends on evidence the text does not contain.

### `conceptual_precision` — Conceptual Precision (broadly applicable, weight 0.05)
Makes the relevant distinctions precisely rather than relying on jargon: for example theme versus topic, motif versus isolated image, entailment versus likely inference, implicature versus literal meaning, symbol versus allegory, or evidentiary consistency versus proof.

### `grammar_and_syntax_reasoning` — Grammar and Syntax Reasoning (conditional, weight 0.05)
When relevant, correctly analyzes clauses, phrases, constituency, grammatical relations, argument structure, agreement, tense/aspect, voice, coordination, subordination, negation, attachment, scope, and structural ambiguity. Equivalent analyses should receive credit when they explain the supplied facts; a central misparse is much more serious than a small labeling difference.

### `semantics_pragmatics_and_discourse_reasoning` — Semantics, Pragmatics, and Discourse Reasoning (conditional, weight 0.05)
When relevant, accurately distinguishes encoded sentence meaning from speaker meaning, entailment from probable inference, assertion from presupposition, implicature from literal content, deixis from fixed reference, and sentence-level meaning from discourse coherence, framing, and conversational organization.

### `close_reading` — Close Reading (conditional, weight 0.05)
When relevant, moves from specific textual features—such as diction, syntax, imagery, repetition, sound, lineation, contrast, perspective, or structure—to an explained interpretive effect and broader meaning. Plot summary, paraphrase, or device naming alone is insufficient.

### `textual_evidence_use` — Textual Evidence Use (conditional, weight 0.05)
When the task is evidence-based, identifies or accurately paraphrases the supplied evidence and explains how it supports the claim. Penalize invented quotations, selective quotation that reverses context, vague references to 'the text,' and interpretations whose support is merely asserted rather than demonstrated.

### `narrative_and_genre_reasoning` — Narrative and Genre Reasoning (conditional, weight 0.05)
When relevant, correctly reasons about narrator, focalization, reliability, narrative level, chronology, duration, characterization, setting, genre conventions, and genre subversion. It must not infer authorial belief directly from narrative voice or confuse reader knowledge with character knowledge.

### `poetic_and_formal_analysis` — Poetic and Formal Analysis (conditional, weight 0.05)
When relevant, analyzes speaker, lineation, stanza, syntax, meter/rhythm, rhyme, sound, imagery, figurative language, form, enjambment, caesura, repetition, and formal shifts in relation to meaning. A correct label with no account of function receives limited credit.

### `rhetorical_and_stylistic_reasoning` — Rhetorical and Stylistic Reasoning (conditional, weight 0.05)
When relevant, identifies speaker/writer, audience, purpose, claim, evidence, warrant, framing, diction, syntax, repetition, analogy, emotional appeal, and arrangement, then explains their likely effect. It distinguishes persuasive technique from logical support and does not treat emotional language as automatically fallacious.

### `translation_and_cross_linguistic_reasoning` — Translation and Cross-Linguistic Reasoning (conditional, weight 0.05)
When relevant, evaluates denotation, implication, grammar, idiom, register, tone, syntax, cultural reference, ambiguity, rhythm, and form across source and target texts. Recognizes defensible alternatives and explicit trade-offs; word-for-word correspondence is not treated as automatically superior.

### `comparative_and_contextual_reasoning` — Comparative and Contextual Reasoning (conditional, weight 0.05)
When relevant, compares texts, traditions, media, periods, or contexts on explicit dimensions and integrates historical, social, political, economic, religious, publishing, or cultural context without replacing textual analysis. Similar motifs across cultures are not assumed to mean the same thing.

### `literary_theory_and_interpretive_reasoning` — Literary Theory and Interpretive Reasoning (conditional, weight 0.05)
When relevant, applies a theoretical lens accurately, grounds it in evidence, explains what it illuminates, and recognizes limits or alternatives. Theory-name dropping, unfalsifiable reasoning, or presenting a theoretical interpretation as textual fact should score poorly.

### `textual_history_and_variant_reasoning` — Textual History and Variant Reasoning (conditional, weight 0.05)
When relevant, evaluates manuscripts, variants, chronology, distribution, copying relationships, internal plausibility, scribal processes, provenance, editions, and historical word meanings. It does not assume the oldest surviving manuscript is automatically correct or that one modern edition transparently preserves an original.

### `ambiguity_alternatives_and_uncertainty` — Ambiguity, Alternatives, and Uncertainty (conditional, weight 0.05)
When the evidence permits multiple readings, identifies the source of ambiguity, compares plausible alternatives, states what would resolve them, and calibrates certainty. A valid alternative interpretation or translation should receive full credit when adequately supported; false certainty should not.

### `quantitative_and_corpus_reasoning` — Quantitative and Corpus Reasoning (conditional, weight 0.05)
When relevant, uses counts, normalization, distributions, lexical diversity, collocation, stylometric probabilities, or other quantitative evidence correctly; considers sample size, corpus composition, genre, period, editorial, and register confounds; and does not treat statistical association as self-explanatory proof of meaning or authorship.

### `completeness` — Completeness (broadly applicable, weight 0.05)
Covers the material parts of the task, including requested comparisons, evidence, alternative readings, calculations, or qualifications. Minor omissions are distinguishable from omissions that leave the central reasoning incomplete.

### `consistency` — Consistency (broadly applicable, weight 0.05)
Maintains compatible claims, terminology, readings, calculations, and evidentiary standards throughout the answer; does not contradict its own structural analysis, translation choice, chronology, or stated uncertainty.

### `clarity` — Clarity (broadly applicable, weight 0.05)
Presents reasoning in a clear, readable, appropriately organized way so that claims, evidence, mechanisms, distinctions, and uncertainty can be evaluated. Unnecessary jargon or verbosity does not compensate for unclear analysis.

## Important Evaluation Principles

### Textual evidence matters
Literary interpretation should be grounded in the text actually supplied.

### Interpretation is constrained but not unique
Multiple readings can be valid when adequately supported.

### Author is not narrator
Do not automatically attribute a narrator's statements or beliefs to the historical author.

### Poet is not speaker
The speaker of a poem is a textual voice unless evidence establishes otherwise.

### Summary is not analysis
Retelling what happens does not substitute for explaining how language or form creates meaning.

### Device identification is not analysis
Naming metaphor, irony, enjambment, symbolism, or another device is insufficient without explaining its function.

### Form and meaning interact
Syntax, rhythm, narrative structure, genre, diction, and sound can affect interpretation.

### Grammar is not style preference
A construction may be grammatically valid while differing from a style guide's recommendation.

### Standard language is not inherently linguistically superior
Dialect and nonstandard varieties can have systematic grammar.

### Semantics is not pragmatics
Literal meaning and context-dependent inference should be distinguished.

### Ambiguity should not be forced away
When the text genuinely supports multiple readings, strong answers should acknowledge them.

### Historical language matters
Words and grammatical constructions may change meaning over time.

### Context matters without replacing the text
Historical or biographical context can illuminate literature but should not substitute for textual evidence.

### Biography is not interpretation by itself
Authorial biography does not automatically determine textual meaning.

### Translation involves trade-offs
Meaning, tone, rhythm, register, syntax, and cultural reference may not all transfer perfectly.

### Literal is not automatically accurate
Word-for-word translation can distort meaning or register.

### Literary theory is a lens
A theoretical framework should illuminate evidence rather than predetermine the answer.

### Corpus evidence requires appropriate comparison
Frequency differences may reflect genre, period, text length, editorial practice, or corpus composition.

### Quantitative patterns still require interpretation
A statistically visible textual pattern is not self-explanatory.

### Manuscript age is not the only textual criterion
Older surviving copies are important evidence but are not automatically correct.

### No interpretation by prestige
A famous critic's view should not receive credit merely because it is famous.

### No jargon worship
Technical literary or linguistic vocabulary without correct analysis should not receive high credit.

### No citation worship
Citations do not compensate for inaccurate textual or linguistic reasoning.

## Grammar and Syntax Answers

Strong answers should, where appropriate, identify relevant structure; distinguish clauses and phrases; identify attachment, agreement, dependencies, voice, scope, and ambiguity; and preserve intended meaning when rewriting. Do not require one syntactic theory where several analyses explain the relevant facts.

## Semantic and Pragmatic Answers

Strong answers should distinguish literal content, entailment, presupposition, implicature, contextual inference, and reference where relevant. Do not treat pragmatic inference as encoded literal meaning without justification.

## Close-Reading Answers

Strong answers should: (1) identify specific textual details; (2) explain their linguistic or formal features; (3) connect those features to interpretive effects; and (4) relate them to the larger passage, tension, or theme. Do not reward plot summary instead of analysis.

## Poetry Answers

Strong answers should consider, where relevant, speaker, syntax, lineation, rhythm, meter, rhyme, imagery, sound, figurative language, and formal shifts. Do not reduce poetry analysis to paraphrase, and do not attribute the speaker's claims automatically to the poet.

## Narrative Answers

Strong answers should distinguish author, narrator, character, perspective/focalization, temporal order, narrative level, and reliability where relevant. Do not infer authorial belief directly from narrative voice.

## Rhetorical Answers

Strong answers should identify audience, purpose, claim, evidence, framing, rhetorical choices, and likely effect where relevant. Do not reward merely listing ethos/pathos/logos labels, and do not equate rhetorical effectiveness with logical validity.

## Translation Answers

Strong answers should consider denotation, implication, tone, register, syntax, idiom, ambiguity, cultural reference, and rhythm/form where relevant. Equivalent translations should receive full credit when their trade-offs are justified; literal wording is not automatically more accurate.

## Literary-Theory Answers

Strong answers should apply the chosen framework accurately, identify relevant textual evidence, explain what the framework illuminates, and recognize limits or alternatives. Do not reward theory-name dropping or unfalsifiable readings.

## Textual-Criticism Answers

Strong answers should consider manuscript evidence, chronology, textual distribution, possible copying relationships, internal plausibility, and scribal processes where relevant. Do not assume the oldest surviving manuscript automatically equals the original reading.

## Corpus and Quantitative Answers

Strong answers should normalize appropriately, recognize sample composition, consider genre/period/editorial confounds, interpret frequency in context, and distinguish statistical pattern from literary explanation or proof of authorship.

## Ambiguity and Alternative Readings

When a prompt supports multiple grammatical, semantic, narrative, symbolic, theoretical, editorial, or translational readings, credit any answer that accurately identifies the evidence and explains the reasoning. Penalize false certainty, not legitimate plural interpretation.
