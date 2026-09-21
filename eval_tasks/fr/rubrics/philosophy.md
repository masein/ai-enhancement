# Philosophy Evaluation Rubric

## Task

Score **ONE answer to ONE Philosophy question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, philosophical assumptions, and argument context.

Accept different philosophically defensible positions where the reasoning, premises, evidence, and stated assumptions support them. Do not require a particular philosopher, theoretical school, wording, essay structure, or conclusion where the issue is genuinely contested.

Do not reward unnecessary length, jargon, philosopher-name dropping, quotations, or citations. The benchmark evaluates **philosophical reasoning**, not resemblance to a reference answer.

## Critical Error

If `critical_philosophy_error` applies:

**score = 0**

A critical Philosophy error is a materially false central logical, conceptual, argumentative, epistemological, metaphysical, ethical, or philosophical claim that fundamentally invalidates or reverses the answer.

Typical examples include fundamentally misidentifying an argument's conclusion; materially altering a premise; reversing validity and truth, validity and soundness, or necessary and sufficient conditions; treating an invalid central inference as valid; using a non-qualifying case as a counterexample; treating conceivability as conclusive metaphysical possibility; treating epistemically lucky true belief as automatically knowledge; confusing determinism with fatalism; deriving an ought solely from descriptive premises while denying any bridge premise; claiming disagreement proves relativism; confusing legality with morality; fabricating a premise or supposed consensus; or substituting partisan/religious advocacy for requested philosophical analysis.

Do **not** treat minor terminology errors, defensible philosophical disagreement, reasonable alternative reconstructions, small omissions, differences in intuition, minor notation errors, stylistic differences, cautious uncertainty, or failure to mention weak secondary objections as critical by themselves.

## Evaluation Criteria

All 20 criteria have equal nominal weight `0.05`. Conditional criteria are applied when the question calls for them.

| ID | Name | Operational definition | Weight | Conditional |
|---|---|---|---:|:---:|
| `relevance` | Relevance | Directly addresses the philosophical task actually asked, focusing on the supplied argument, case, distinction, or question rather than substituting a different issue. | 0.05 | No |
| `philosophical_accuracy` | Philosophical Accuracy | States central logical and philosophical claims accurately; does not materially misrepresent supplied positions, arguments, thought experiments, or standard distinctions. | 0.05 | No |
| `conceptual_precision` | Conceptual Precision | Uses key concepts with appropriate precision, including when relevant validity, soundness, necessity, sufficiency, knowledge, justification, identity, modality, causation, obligation, permission, and related distinctions. | 0.05 | No |
| `argument_reconstruction` | Argument Reconstruction | When an argument is supplied or implicit, identifies its conclusion, supporting premises, intermediate conclusions, and structure fairly, preserving the strongest reasonable reading rather than replacing it with a weaker or different argument. | 0.05 | Yes |
| `logical_reasoning` | Logical Reasoning | Correctly evaluates deductive form, conditional reasoning, quantifiers, contradictions, reductio, or other formal/informal inferential structure when these are material to the task. | 0.05 | Yes |
| `premise_and_inference_evaluation` | Premise and Inference Evaluation | Separates questions about premise truth or support from questions about inferential validity or strength, and explains where any argumentative failure actually occurs. | 0.05 | Yes |
| `distinction_making` | Distinction Making | Identifies and applies distinctions needed to avoid conflation, such as truth/validity, validity/soundness, necessary/sufficient, descriptive/normative, explanation/justification, legality/morality, actuality/possibility, or behavior/consciousness. | 0.05 | Yes |
| `counterexample_reasoning` | Counterexample Reasoning | When counterexamples matter, tests universal claims with cases that genuinely satisfy the stated conditions and explains exactly why the target conclusion fails; does not treat mere disagreement or an inapplicable case as a counterexample. | 0.05 | Yes |
| `epistemological_reasoning` | Epistemological Reasoning | When relevant, reasons correctly about knowledge, truth, belief, justification, evidence, testimony, perception, skepticism, disagreement, induction, reliability, and epistemic luck, with calibrated evidential claims. | 0.05 | Yes |
| `metaphysical_reasoning` | Metaphysical Reasoning | When relevant, reasons correctly about identity, persistence, modality, causation, time, freedom, personal identity, or explanation, avoiding conflations such as similarity with identity or determinism with fatalism. | 0.05 | Yes |
| `ethical_and_normative_reasoning` | Ethical and Normative Reasoning | When relevant, identifies the normative principles, rights, duties, consequences, virtues, intentions, permissions, or values at issue; does not derive an ought from descriptive premises without a normative bridge; represents competing ethical frameworks fairly. | 0.05 | Yes |
| `philosophy_of_mind_and_language_reasoning` | Philosophy of Mind and Language Reasoning | When relevant, accurately analyzes consciousness, mental states, functional roles, behavioral evidence, understanding, meaning, reference, context, speech acts, implicature, ambiguity, or vagueness without collapsing distinct concepts. | 0.05 | Yes |
| `philosophy_of_science_reasoning` | Philosophy of Science Reasoning | When relevant, handles confirmation, falsification, auxiliary assumptions, explanation, models, theory choice, observation, realism, instrumentalism, and underdetermination without treating evidence as automatic proof. | 0.05 | Yes |
| `alternative_positions_and_objections` | Alternative Positions and Objections | When the issue is contestable or argumentative, presents important competing positions, objections, or replies charitably and at appropriate strength rather than straw-manning or pretending disagreement has vanished. | 0.05 | Yes |
| `assumptions` | Assumptions | Identifies material explicit or hidden assumptions required for the reasoning, explains their role, and does not silently insert assumptions that change the supplied argument. | 0.05 | Yes |
| `uncertainty_and_qualification` | Uncertainty and Qualification | Matches confidence and conclusions to what the reasoning establishes, marks unresolved or underdetermined issues, and distinguishes evidence, plausibility, possibility, and proof. | 0.05 | Yes |
| `completeness` | Completeness | Covers all material parts of the prompt and includes the premises, distinctions, objections, applications, or requested comparisons needed for a sufficient answer without padding. | 0.05 | No |
| `consistency` | Consistency | Maintains compatible claims, definitions, and standards across the answer; does not rely on mutually inconsistent premises or switch criteria without explanation. | 0.05 | No |
| `clarity` | Clarity | Presents the reasoning in a clear, traceable form with distinctions and inferential steps understandable from the answer itself; jargon, names, quotations, and citations never substitute for explanation. | 0.05 | No |
| `philosophical_judgment` | Philosophical Judgment | When evaluative judgment is required, weighs premises, inferences, counterexamples, alternatives, conceptual costs, and uncertainty proportionately, reaching only conclusions supported by the stated assumptions and argument. | 0.05 | Yes |


## Important Evaluation Principles

- **Arguments must be reconstructed fairly.** Do not reward straw-man criticism; use the strongest reasonable interpretation consistent with the text.
- **Validity is not truth.** A deductively valid argument can have false premises.
- **Validity is not soundness.** Soundness requires validity and true premises.
- **Premise truth and inferential quality are separate.** Locate problems in evidence, premises, inference, or conclusion rather than treating them as interchangeable.
- **Necessary and sufficient conditions are different.** Do not reverse them.
- **Counterexamples must satisfy the stated conditions.** A merely unusual or imaginable case is not automatically a counterexample.
- **Ambiguity matters.** Different lexical, syntactic, scope, or conceptual readings can yield different arguments.
- **Charity matters.** Prefer the strongest reasonable reading, while still allowing criticism.
- **Conceptual possibility is not actuality.** Something's being imaginable or logically consistent does not show that it exists.
- **Conceivability is not automatically metaphysical possibility.** The connection itself requires argument.
- **Knowledge is not mere true belief.** Even justified true belief can remain defective when truth is substantially accidental.
- **Skeptical possibility is not automatically rational defeat.** The mere possibility of error need not eliminate knowledge.
- **Disagreement does not prove relativism.** Disagreement can exist even if there is a fact of the matter.
- **Is does not automatically imply ought.** Normative conclusions need normative premises or bridging principles.
- **Morality and legality differ.** Legal status alone does not determine moral status.
- **Freedom and determinism require definitions.** Do not assume compatibility or incompatibility without stating the relevant conception of freedom.
- **Explanation and justification differ.** Explaining an action does not necessarily justify it.
- **Behavior and consciousness differ.** Behavior may be evidence of consciousness without being definitionally identical to it.
- **Scientific evidence and philosophical interpretation differ.** Empirical success can inform but does not automatically settle realism, explanation, causation, or ontology.
- **Thought experiments are arguments, not verdicts.** They can reveal implications or pressure principles, but do not automatically prove a theory.
- **Intuitions are defeasible.** They may carry evidential weight without being infallible.
- **Philosophical frameworks are tools.** Theory-name dropping does not substitute for reasoning.
- **Genuine disagreement may remain.** Evaluate accuracy, argument quality, assumptions, objections, replies, and conceptual clarity.
- **Political neutrality.** Inform and analyze rather than persuade toward political actors, parties, legislation, ballot choices, or real-world partisan outcomes.
- **Religious neutrality.** Analyze arguments for and against religious positions without advocating belief or disbelief.
- **No jargon worship.** Technical vocabulary without correct application should not receive a high score.
- **No philosopher-name worship.** Historical attribution does not compensate for poor reasoning.
- **No citation worship.** Citations do not compensate for incorrect philosophical reasoning.


## Score Anchors

### 4

The answer is philosophically accurate, directly addresses the question, reconstructs the relevant argument fairly, makes necessary distinctions, evaluates premises and inferences appropriately, considers material objections or alternatives, and reaches a well-supported and appropriately qualified conclusion.

Where relevant, it correctly handles validity, soundness, necessary/sufficient conditions, ambiguity, assumptions, counterexamples, epistemic justification, modality, identity, causation, ethical principles, competing theories, empirical versus conceptual claims, descriptive versus normative claims, and uncertainty.

For contested questions, it represents important alternatives accurately and explains what assumptions drive different conclusions. For thought experiments, it identifies what principle or theory is being tested rather than treating intuition as decisive by itself. For argument-analysis questions, it separates criticism of premises from criticism of inference. For formal questions, the logical structure is correct. Minor stylistic imperfections do not reduce a 4.

### 3

The answer is substantially correct and philosophically useful but contains one meaningful omission, minor conceptual imprecision, incomplete objection, limited qualification, or slight weakness in argument reconstruction. The main reasoning remains sound.

### 2

The answer is partly correct or directionally useful but has important omissions or philosophical weaknesses. It may identify the correct issue but analyze it superficially; state a relevant theory without applying it carefully; give a plausible objection while missing the main argument; miss an important distinction; treat an intuition as more decisive than warranted; or represent one side accurately while oversimplifying another.

It demonstrates meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial logical, conceptual, epistemological, metaphysical, ethical, or argumentative misunderstanding, but does not meet the critical-error threshold. The response provides limited useful philosophical analysis.

### 0

Critical Philosophy error, no answer, off-topic answer, fabricated argument or evidence, partisan or religious advocacy in place of requested analysis, or equivalent fundamental failure.

## Evaluator Guidance

Evaluate the quality of reasoning rather than ideological, religious, or theoretical allegiance. A strong answer may conclude that one argument is stronger under stated assumptions, that the issue remains underdetermined, or that different frameworks yield different outcomes.

For political philosophy, keep evaluation neutral among political positions and assess concepts, arguments, and supplied institutional hypotheticals rather than real candidates, parties, or voting choices. For philosophy of religion, assess the structure and evidential force of arguments without rewarding advocacy for belief or disbelief.

For open-ended questions, attend especially to: issue identification; fair reconstruction; definitions and distinctions; premise versus inference evaluation; hidden assumptions; counterexamples; charitable treatment of alternatives; empirical versus conceptual claims; descriptive versus normative claims; uncertainty; and proportionate conclusions.
