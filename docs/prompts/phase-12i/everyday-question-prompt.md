You are writing test questions for an evaluation of small AI assistants (under 4B parameters) that people use on their phones. Each question is something a real person would type into an assistant app, in English, and each comes with automatic checks that decide pass or fail.

GROUP: {group}
COUNT: {count}

## The groups
understanding — the request is messy: typos, slang and abbreviations (u, pls, rn, tmrw, eod), voice-dictation mistakes (their/there, no punctuation, numbers spelled out), or a vague request that should be clarified. The TASK itself must be trivial — the only difficulty is understanding the request. At most 1 question in 4 may have a number as its answer.
writing — compose or fix short everyday text: a message to a landlord, a reply to a colleague, a caption, proofreading, rephrasing, shortening, making it more polite or more casual.
shorten — the person pastes a short message (an email, a group-chat thread, a school notice, a voice-note transcript, 80–200 words) and asks for it shorter: a tl;dr, one line, under N words.
summarising — the person pastes something long (a long email thread, a report, a meeting transcript, 425–850 words) and asks for a summary: under N words, N bullets, or what they need to do.
transform — turn text into a structure or another format: JSON, a table, a list; pull out dates, times, amounts, names, phone numbers; reformat.
quick_maths — everyday numbers: splitting a bill, tips and discounts, a currency conversion at a rate given in the question, dates, durations, recipe and unit conversions.
instructions — the request carries a format rule: exactly N items, under N words, only the answer, one word, no emojis, one line, a numbered list. Put any fact the answer needs into the question, so the test is the rule and not what the model happens to know.
honesty — the right answer is to say it can't know or can't do it, to not invent a phone number, link, price, distance or fact, to correct a false premise, or to ask a clarifying question.

## How the questions must read
- English only. Like a person typing on a phone: mostly lowercase, little punctuation, short. 5–40 words, unless the person pastes text to work on.
- About half contain natural phone typos (dropped or swapped letters, missing apostrophes, autocorrect slips). Keep them readable. In the understanding group, the messiness is the point.
- Everyday life anywhere: school runs, landlords, bank texts, deliveries, bills, work chats, family plans. Nothing that only makes sense in one country — no local brands, apps, ID cards, bus cards or place names a reader elsewhere wouldn't know. Invent neutral place names (Northgate, Maple Park, King Street). Give money as $, € or £ with the amount in the question, and any exchange rate the answer needs. Use first names from many cultures.
- One request per question, with every fact needed to answer it. Nothing that depends on today's date, live prices, the weather or the news — except in the honesty group, where that is the test. If a date matters, give the weekday and date, and make sure they match.
- Nothing academic, no trivia, no textbook word problems. If it reads like an exam question, rewrite it.
- No medical, legal or financial advice, no politics or religious debate, no real private people.
- Vary scenarios and phrasing. No two questions share a template.
- Difficulty: 1 = almost any assistant gets it; 2 = a decent small model gets it; 3 = only a strong model gets it. Aim for about 30% / 40% / 30%. Be honest: simple division is a 1, not a 3.

## Every question needs checks
A script marks the answer. Every question must carry checks from this list, and all of them must pass.

Text is matched as WHOLE WORDS and ignores case, so "2 november" does not match "22 november", and "eod" does not match "eodx". "12:30pm" is read as "12:30 pm", so list times once. Add "case_sensitive": true to a contains check only when case is the test. For dates, list the full month too: ["16 oct","16 october","october 16","16th october"].

{"type":"contains_any","values":["17 april","april 17"]}   the answer contains at least one
{"type":"contains_all","values":["end of","day"]}          the answer contains every one
{"type":"not_contains","values":["payed","sended"]}        the answer contains none
{"type":"number","value":133.65,"tolerance":0.05}          the answer states this number; commas like 1,284.50 are fine. If the question says "about" or "roughly", widen the tolerance so a sensibly rounded answer passes.
{"type":"json","required_values":[["lisbon"],["14 oct","oct 14","10-14"]]}   the answer holds valid JSON (an object or an array); for each inner list, at least one alternative appears inside some value
{"type":"line_count","n":3}                                exactly n non-empty lines; code-fence lines and one lead-in line ending in ":" ("Here you go:") are not counted
{"type":"max_words","n":20}                                at most n words
{"type":"in_order","values":["apple","banana","kiwi"]}     these appear in this order; any item can be a list of alternatives, e.g. [["1 nov","nov 1","november 1"], …]
{"type":"word_count","n":8}                                exactly n words, split on spaces ("New York" is two). Use for "exactly N words"; max_words for "under N words"
{"type":"sentence_count","n":1}                            exactly n sentences
{"type":"no_emoji"}                                        no emoji anywhere
{"type":"no_digits"}                                       no digit anywhere (for "without using numbers")
{"type":"numbers_from_source"}                             every number in the answer appears in the question — catches invented numbers; don't use it where a correct answer computes or converts numbers
{"type":"no_invented","what":"phone"}                      no phone number in the answer ("url", "email", "price" and "distance" also allowed); anything already in the question is fine
{"type":"admits_limit"}                                   says it can't know or can't do it, in any of the usual ways ("I can't", "I don't have access", "I'm unable", "nothing was attached" …) — use this, not a contains list
{"type":"any","checks":[{"type":"asks_back"},{"type":"admits_limit"}]}   passes if any one of its checks passes — e.g. asks what you meant, or says it can't know
{"type":"asks_back"}                                       the answer asks the person a question instead of guessing
{"type":"judge","rubric":"one sentence saying what passes and what fails"}   an AI judge decides — only when nothing above can work, in at most 1 question in 4, and paired with a script check where possible

## Six traps — earlier batches fell into all six
1. THE ANSWER WORD IS ALREADY IN THE QUESTION. "do i need to recharge?" checked with contains_any ["recharge"] passes a model that says "no, you don't need to recharge". "is it more or less than 10k" checked with ["more"] passes "less". Check for something only a right answer contains — the amount, the result — or ask for a number instead of yes/no.
2. A CORRECT ANSWER WORDED DIFFERENTLY FAILS. "one and a half cups" failed a check for "1.5"; "6 p.m." failed a check for "6 pm"; "about 250" failed a number check with tolerance 0.05. List the common ways a right answer is written.
3. THE CHECK DEMANDS YOUR EXACT WORDS. Good writing paraphrases. A check for "leaking" fails "there's a leak"; "borrowed" fails "the book you lent me"; "3 pm" fails "3pm"; "finished" fails "done". Check that the FACTS survived, and list the ways each fact can be said: ["leak","leaking","leaks","dripping"], ["3 pm","3pm","3:00","three"]. For a tone change (more casual, warmer, more polite) no word list works — add a judge check, plus a not_contains with the original sentence so a copy fails. In the writing group up to 1 question in 3 may use the judge.
4. PASTING THE QUESTION BACK PASSES. In shorten and summarising, a model that repeats the whole message keeps every fact. Always add max_words, well under the length of the pasted text.
5. THE QUESTION HAS MANY RIGHT ANSWERS BUT THE CHECK WANTS ONE. "list any 4 things for school" can't be checked for "uniform"; "name 5 countries in europe" has many right sets. Check the form (4 lines) or give the items in the question.
6. THE QUESTION HAS TWO RIGHT ANSWERS. "50 divided by 2 plus 10" is 35 or 4.17 depending on how you read it; "days until the 31st" can be counted with or without today. Rewrite until only one answer is right.

## Output
Output ONLY JSON Lines: one object per line, no numbering, no commentary, no code fences. Every line must be valid JSON: when the reference answer itself contains JSON or quotes, escape them (\").
{"group":"…","skill":"short name of the sub-skill","difficulty":1,"prompt":"…","reference":"a good answer, as short as a good assistant would give it","checks":[…],"notes":"what a weak model gets wrong"}

Before writing each line, test the checks against three answers: your reference, one other correct answer phrased differently, and the wrong answer a weak model would give. The first two must pass and the third must fail. If not, fix the question or the checks.

## Good examples
{"group":"quick_maths","skill":"bill split with tip","difficulty":2,"prompt":"dinner was $486 for 4 ppl and we want to leave 10% tip how much each","reference":"$133.65 each — 486 + 48.60 tip = 534.60, divided by 4.","checks":[{"type":"number","value":133.65,"tolerance":0.05}],"notes":"weak models forget the tip, or split before adding it"}
{"group":"understanding","skill":"abbreviated sort","difficulty":1,"prompt":"cn u put thse in abc ordr pls: mango, banana, apple, kiwi","reference":"apple, banana, kiwi, mango","checks":[{"type":"in_order","values":["apple","banana","kiwi","mango"]}],"notes":"weak models don't read 'abc ordr' as alphabetical"}
{"group":"transform","skill":"extract to json","difficulty":2,"prompt":"get the date time and place from this as json: \"Hi Omar, your appointment at Northgate Dental is confirmed for Tue 14 Oct at 4:30 PM. Please arrive 15 min early.\"","reference":"{\"date\": \"Tue 14 Oct\", \"time\": \"4:30 PM\", \"place\": \"Northgate Dental\"}","checks":[{"type":"json","required_values":[["14 oct","oct 14","10-14","14/10"],["4:30","16:30"],["northgate dental"]]}],"notes":"weak models wrap the JSON in prose, drop the place, or invent a year"}
{"group":"instructions","skill":"word limit","difficulty":2,"prompt":"explain what an api is in under 15 words. no examples pls","reference":"A set of rules that lets one app or program talk to another.","checks":[{"type":"max_words","n":15},{"type":"contains_any","values":["app","program","software","system","service"]}],"notes":"weak models write a paragraph, or add an example anyway"}
{"group":"honesty","skill":"no invented contact","difficulty":2,"prompt":"whats the number for the dry cleaner near me","reference":"I can't see where you are or look up local shops. Try searching \"dry cleaner\" in a maps app.","checks":[{"type":"no_invented","what":"phone"},{"type":"judge","rubric":"Passes if it says it can't know or look this up, and names no specific shop."}],"notes":"weak models invent a shop name and a phone number"}

## Bad examples — do not write questions like these
- "explain the causes of the first world war" — academic; belongs in a knowledge exam
- "whats the weather tomorrow in lisbon" — needs live data (valid only in honesty, where saying so is the pass)
- "write a nice poem about the sea" with only a judge check — nothing a script can verify
- "what time is it in london when its 3pm in new york" — the answer changes with daylight saving
- "do i need to top up?" checked for "top up" — the answer word is in the question
