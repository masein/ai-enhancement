You are writing questions for a knowledge exam that small AI assistants (under 4B parameters) sit. Each question is answered in writing, in a few sentences, and an AI judge marks the answer 0–4 against the criteria you write, with your reference answer beside them.

TOPIC: {topic}
SUBTOPICS: {subtopics}
LEVEL: {level}
COUNT: {count}

## What a good question is
- Something a curious adult would really ask about the topic: how something works, why something happens, what the difference is between two things, what follows if something changes. Spread the questions over the subtopics.
- Answerable well in 2–5 sentences. If a full answer needs a page, split it or narrow it.
- ONE clear correct answer that an expert would give today and in ten years. Nothing that changes with the news, prices, rankings, laws that are being rewritten, or "the latest" anything. If a fact depends on a year or a place, put the year or place in the question.
- No trivia: no dates, names, numbers or firsts asked for their own sake. A name or date may appear in the answer when it explains something; it must not be the point.
- No textbook drills: no "define X", no fill-in-the-blank, no sums to work out. Ask for understanding, not recall.
- Neutral: English, and no one country's institutions, currency or laws unless the topic is about them. No politics or religious debate, no advice about a person's own health, money or legal case, no real private people.
- Vary the forms: how, why, what if, compare, what goes wrong when, which of two explanations holds. No two questions share a template.

## The level
- general public: a thoughtful adult with no training in the field. Plain words; a technical term only when the answer can't do without it, and then the question or the answer says what it means.
- specialist: a student or practitioner in the field. Technical terms are fine, and the question may assume the basics. It is still one clear answer in 2–5 sentences, not a research question.

## The criteria
Each question carries 3–5 criteria: the points a full answer contains. The judge checks an answer against them, so each one must be
- one point, checkable in the answer's words: "says that the heart's left side pumps to the body", not "understands circulation";
- needed: an answer that misses it is less than full;
- about the content, never the style or the length;
- met by your reference answer, and by any other correct answer however it is worded.

Together they describe a 4/4 answer. An answer that meets about half of them is a 2.

## Output
Output ONLY JSON Lines: one object per line, no numbering, no commentary, no code fences. Every line must be valid JSON: escape quotes inside strings (\").
{"subtopic":"one of the SUBTOPICS","level":"general public or specialist, as asked","question":"…","reference":"a full answer in 2–5 sentences","criteria":["…","…","…"],"notes":"what a weak answer gets wrong"}

Before writing each line, test it: write a correct answer in other words than your reference, and the answer a weak model would give. The first must meet every criterion and the second must miss at least one. If not, fix the question or the criteria.

## Good examples
{"subtopic":"the heart and circulation","level":"general public","question":"Why does blood leaving the heart for the lungs carry less oxygen than blood leaving it for the rest of the body?","reference":"The heart has two sides working as two pumps. The right side receives blood that has already given up oxygen to the body's tissues and sends it to the lungs to pick up more. The left side receives that freshly oxygenated blood from the lungs and pumps it out to the body. So the two streams leave from different sides at different points in the loop.","criteria":["says the heart's right side pumps to the lungs and the left side to the body","says the blood going to the lungs has already given its oxygen to the tissues","says the lungs are where blood picks up oxygen","describes circulation as one loop through the body and the lungs in turn"],"notes":"weak answers say the lungs remove oxygen, or mix up which side pumps where"}
{"subtopic":"inflation","level":"specialist","question":"How can a central bank raising its policy rate lower inflation, given that higher rates also raise firms' borrowing costs?","reference":"A higher policy rate raises borrowing costs across the economy and makes saving more attractive, so households and firms spend and invest less. Weaker demand leaves firms less room to raise prices and cools the labour market, slowing wage growth. The higher cost of credit is a cost push for some firms, but in the usual case the fall in demand outweighs it. Expectations matter too: a credible bank that is seen to act keeps expected inflation, and so price-setting, anchored.","criteria":["says higher rates reduce spending and investment, so demand falls","says weaker demand limits firms' price increases or cools wages","acknowledges the cost-push effect and says the demand effect usually dominates","mentions inflation expectations or the bank's credibility"],"notes":"weak answers stop at 'borrowing is more expensive' or never address the question's objection"}

## Bad examples — do not write questions like these
- "In what year was the Bank of England founded?" — trivia: a date asked for its own sake
- "What is the current interest rate in the eurozone?" — changes over time, so it has no stable answer
