Absolutely. For the **law topic**, I would make the dataset more comprehensive than the medical one because legal QA has several dimensions that medical QA doesn't: **jurisdiction, authority/hierarchy of sources, applicability, procedural posture, deadlines, burden of proof, distinction between legal information and legal advice, and handling uncertainty**.

I also recommend making the 100 questions span different legal tasks rather than having 100 generic "is this legal?" questions.

Below is a proposed **100-question law evaluation dataset**, followed by a **law-specific evaluation rubric** you can use as the judge prompt.

## 1. Law dataset — 100 questions

> 12p.4: the Knowledge exam's report-half questions are not quoted here — they are the test, kept in the server's store and never in the repo. Each is cited by id instead: its retired bank's row and the first 12 characters of its qid.

I've deliberately mixed:

* legal information
* legal reasoning
* contracts
* employment
* consumer law
* criminal law
* civil disputes
* property
* family law
* immigration
* intellectual property
* corporate law
* privacy/data protection
* procedure
* evidence
* taxation
* estate/inheritance
* technology/AI
* practical legal scenarios
* high-stakes legal situations

Because law is jurisdiction-dependent, many questions intentionally test whether the LLM **asks for or recognizes the need for jurisdiction** rather than hallucinating a universal rule.

```json
[
  {
    "id": 1,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #1, qid 1f51c088860f]",
    "intent": "legal_information",
    "domain": "general_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 2,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #2, qid 639978abb65f]",
    "intent": "legal_information",
    "domain": "general_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 3,
    "prompt": "What is the difference between a law, a regulation, and a court decision?",
    "intent": "legal_information",
    "domain": "general_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 4,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #4, qid 1b74883f0a75]",
    "intent": "legal_information",
    "domain": "civil_procedure",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 5,
    "prompt": "What is the difference between a statute and case law?",
    "intent": "legal_information",
    "domain": "legal_system",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 6,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #6, qid 6a427c6305f9]",
    "intent": "legal_information",
    "domain": "legal_system",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 7,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #7, qid bd1a0e2da61a]",
    "intent": "legal_information",
    "domain": "civil_procedure",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 8,
    "prompt": "What is an injunction and when might a court issue one?",
    "intent": "legal_information",
    "domain": "civil_procedure",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 9,
    "prompt": "What does 'liable' mean in a civil lawsuit?",
    "intent": "legal_information",
    "domain": "civil_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 10,
    "prompt": "What is the difference between a legal right and a legal remedy?",
    "intent": "legal_information",
    "domain": "general_law",
    "acuity": "routine",
    "style": "conversational"
  },

  {
    "id": 11,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #11, qid 12a69f6c0083]",
    "intent": "legal_assessment",
    "domain": "contract_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 12,
    "prompt": "A company emailed me a contract and I replied 'I agree.' Is that legally binding?",
    "intent": "legal_assessment",
    "domain": "contract_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 13,
    "prompt": "I paid a contractor a deposit to renovate my house, but they never started the work. Can I demand my money back?",
    "intent": "legal_assessment",
    "domain": "contract_law",
    "acuity": "urgent",
    "style": "context_rich"
  },
  {
    "id": 14,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #14, qid b3ec435af0cd]",
    "intent": "legal_assessment",
    "domain": "employment_contracts",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 15,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #15, qid ca4ba3d61123]",
    "intent": "legal_assessment",
    "domain": "contract_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 16,
    "prompt": "What makes a contract legally enforceable?",
    "intent": "legal_information",
    "domain": "contract_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 17,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #17, qid e34ac0cdb46c]",
    "intent": "legal_information",
    "domain": "contract_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 18,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #18, qid 2b70aaa731b1]",
    "intent": "legal_information",
    "domain": "contract_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 19,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #19, qid a3c015b98b0b]",
    "intent": "legal_assessment",
    "domain": "contract_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 20,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #20, qid 3a0ca2b70bdc]",
    "intent": "legal_information",
    "domain": "contract_law",
    "acuity": "routine",
    "style": "conversational"
  },

  {
    "id": 21,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #21, qid 33bea4d21da4]",
    "intent": "legal_assessment",
    "domain": "employment_law",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 22,
    "prompt": "My employer fired me after I complained about unsafe working conditions. Could that be unlawful retaliation?",
    "intent": "legal_assessment",
    "domain": "employment_law",
    "acuity": "urgent",
    "style": "context_rich"
  },
  {
    "id": 23,
    "prompt": "Can my employer monitor my work email and messages?",
    "intent": "legal_assessment",
    "domain": "employment_privacy",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 24,
    "prompt": "My employer wants me to sign a non-disclosure agreement. What should I look for before signing?",
    "intent": "legal_guidance",
    "domain": "employment_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 25,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #25, qid 051ff704b931]",
    "intent": "legal_assessment",
    "domain": "employment_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 26,
    "prompt": "Can an employer fire someone without giving a reason?",
    "intent": "legal_information",
    "domain": "employment_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 27,
    "prompt": "My employer says I am an independent contractor, but I work fixed hours under their direct supervision. Could I legally be an employee?",
    "intent": "legal_assessment",
    "domain": "employment_law",
    "acuity": "moderate",
    "style": "context_rich"
  },
  {
    "id": 28,
    "prompt": "Can my employer change my salary without my agreement?",
    "intent": "legal_assessment",
    "domain": "employment_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 29,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #29, qid 4c8a537bf5d8]",
    "intent": "legal_information",
    "domain": "employment_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 30,
    "prompt": "I was dismissed yesterday. How quickly do I usually need to challenge an unfair dismissal?",
    "intent": "legal_assessment",
    "domain": "employment_procedure",
    "acuity": "urgent",
    "style": "conversational"
  },

  {
    "id": 31,
    "prompt": "I bought a laptop that stopped working after two weeks. What rights might I have as a consumer?",
    "intent": "legal_assessment",
    "domain": "consumer_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 32,
    "prompt": "A store says 'all sales are final.' Does that always mean I cannot return a defective product?",
    "intent": "legal_assessment",
    "domain": "consumer_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 33,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #33, qid 47fdb00b9781]",
    "intent": "legal_guidance",
    "domain": "consumer_law",
    "acuity": "mild",
    "style": "conversational"
  },
  {
    "id": 34,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #34, qid 60cc7671c270]",
    "intent": "legal_assessment",
    "domain": "consumer_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 35,
    "prompt": "Can a company advertise something as 'free' if I have to provide my credit card details?",
    "intent": "legal_assessment",
    "domain": "consumer_protection",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 36,
    "prompt": "What is an unfair or deceptive business practice?",
    "intent": "legal_information",
    "domain": "consumer_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 37,
    "prompt": "A subscription renewed automatically even though I forgot about it. Can I get a refund?",
    "intent": "legal_assessment",
    "domain": "consumer_law",
    "acuity": "mild",
    "style": "conversational"
  },
  {
    "id": 38,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #38, qid 0a970655bfb5]",
    "intent": "legal_assessment",
    "domain": "consumer_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 39,
    "prompt": "Can a business refuse to refund a service that was never provided?",
    "intent": "legal_assessment",
    "domain": "consumer_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 40,
    "prompt": "What evidence should I keep when I have a dispute with an online seller?",
    "intent": "legal_guidance",
    "domain": "consumer_law",
    "acuity": "routine",
    "style": "conversational"
  },

  {
    "id": 41,
    "prompt": "I was arrested but the police did not tell me why. What rights might I have?",
    "intent": "legal_assessment",
    "domain": "criminal_law",
    "acuity": "emergency",
    "style": "conversational"
  },
  {
    "id": 42,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #42, qid 3a6eb6e187ff]",
    "intent": "legal_guidance",
    "domain": "criminal_procedure",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 43,
    "prompt": "Can police search my home without a warrant?",
    "intent": "legal_information",
    "domain": "criminal_procedure",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 44,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #44, qid e87289ad9d76]",
    "intent": "legal_guidance",
    "domain": "criminal_law",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 45,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #45, qid 3339317517b7]",
    "intent": "legal_information",
    "domain": "criminal_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 46,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #46, qid 64fca83a44f5]",
    "intent": "legal_information",
    "domain": "evidence",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 47,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #47, qid b04ef8072f87]",
    "intent": "legal_assessment",
    "domain": "criminal_procedure",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 48,
    "prompt": "Someone threatened me online and I have screenshots. Should I report it to the police?",
    "intent": "legal_guidance",
    "domain": "criminal_law",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 49,
    "prompt": "What does 'presumption of innocence' mean?",
    "intent": "legal_information",
    "domain": "criminal_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 50,
    "prompt": "I was stopped by police while driving and asked to provide identification. Do I have to comply?",
    "intent": "legal_assessment",
    "domain": "criminal_procedure",
    "acuity": "moderate",
    "style": "conversational"
  },

  {
    "id": 51,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #51, qid e8627657c045]",
    "intent": "legal_assessment",
    "domain": "housing_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 52,
    "prompt": "Can a landlord increase the rent whenever they want?",
    "intent": "legal_information",
    "domain": "housing_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 53,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #53, qid 6677e9e70be5]",
    "intent": "legal_assessment",
    "domain": "housing_law",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 54,
    "prompt": "My landlord wants to evict me because I complained about unsafe conditions. Could that be unlawful?",
    "intent": "legal_assessment",
    "domain": "housing_law",
    "acuity": "urgent",
    "style": "context_rich"
  },
  {
    "id": 55,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #55, qid 32d384680407]",
    "intent": "legal_information",
    "domain": "housing_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 56,
    "prompt": "My tenant stopped paying rent but refuses to leave. Can I change the locks?",
    "intent": "legal_guidance",
    "domain": "landlord_tenant",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 57,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #57, qid 78cc468d14dc]",
    "intent": "legal_information",
    "domain": "housing_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 58,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #58, qid d117637f29d8]",
    "intent": "legal_assessment",
    "domain": "housing_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 59,
    "prompt": "My landlord says I cannot have a pet even though the lease does not mention pets. What should I check?",
    "intent": "legal_assessment",
    "domain": "housing_law",
    "acuity": "mild",
    "style": "conversational"
  },
  {
    "id": 60,
    "prompt": "What evidence should a tenant keep if they believe the landlord is violating the lease?",
    "intent": "legal_guidance",
    "domain": "housing_law",
    "acuity": "routine",
    "style": "conversational"
  },

  {
    "id": 61,
    "prompt": "My spouse and I are separating. How are assets usually divided in a divorce?",
    "intent": "legal_information",
    "domain": "family_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 62,
    "prompt": "My ex refuses to let me see our child even though there is no court order. What can I do?",
    "intent": "legal_assessment",
    "domain": "family_law",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 63,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #63, qid f272147cc8f1]",
    "intent": "legal_information",
    "domain": "family_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 64,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #64, qid 7d2988cc73fe]",
    "intent": "legal_information",
    "domain": "family_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 65,
    "prompt": "My spouse wants me to sign a divorce settlement immediately. Should I sign it without having it reviewed?",
    "intent": "legal_guidance",
    "domain": "family_law",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 66,
    "prompt": "Does cheating automatically affect how property is divided in a divorce?",
    "intent": "legal_information",
    "domain": "family_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 67,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #67, qid 9eede2a7de7f]",
    "intent": "legal_assessment",
    "domain": "family_law",
    "acuity": "urgent",
    "style": "context_rich"
  },
  {
    "id": 68,
    "prompt": "What is the difference between legal custody and physical custody?",
    "intent": "legal_information",
    "domain": "family_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 69,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #69, qid 76376685f442]",
    "intent": "legal_assessment",
    "domain": "family_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 70,
    "prompt": "My partner and I are not married but have lived together for ten years. Do we automatically have the same legal rights as married couples?",
    "intent": "legal_assessment",
    "domain": "family_law",
    "acuity": "moderate",
    "style": "conversational"
  },

  {
    "id": 71,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #71, qid d30547def4c8]",
    "intent": "legal_assessment",
    "domain": "intellectual_property",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 72,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #72, qid 191f7b2e0114]",
    "intent": "legal_assessment",
    "domain": "copyright",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 73,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #73, qid 30b241c5a78d]",
    "intent": "legal_information",
    "domain": "intellectual_property",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 74,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #74, qid e3ba35521157]",
    "intent": "legal_assessment",
    "domain": "intellectual_property",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 75,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #75, qid 97df0498b190]",
    "intent": "legal_assessment",
    "domain": "trademark",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 76,
    "prompt": "Someone is selling a product using a logo very similar to mine. What legal options might I have?",
    "intent": "legal_assessment",
    "domain": "trademark",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 77,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #77, qid 16214e89c68c]",
    "intent": "legal_information",
    "domain": "patent_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 78,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #78, qid 45e52b15cc43]",
    "intent": "legal_assessment",
    "domain": "intellectual_property",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 79,
    "prompt": "Can I use a competitor's trademark in an advertisement comparing our products?",
    "intent": "legal_assessment",
    "domain": "trademark",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 80,
    "prompt": "What should a small software company do to protect its source code and intellectual property?",
    "intent": "legal_guidance",
    "domain": "technology_law",
    "acuity": "routine",
    "style": "context_rich"
  },

  {
    "id": 81,
    "prompt": "I am starting a company with two partners. What agreements should we have before we begin?",
    "intent": "legal_guidance",
    "domain": "corporate_law",
    "acuity": "routine",
    "style": "context_rich"
  },
  {
    "id": 82,
    "prompt": "One of my business partners wants to sell their shares to a stranger. Can they do that?",
    "intent": "legal_assessment",
    "domain": "corporate_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 83,
    "prompt": "What is the difference between a shareholder agreement and a company's articles of association?",
    "intent": "legal_information",
    "domain": "corporate_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 84,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #84, qid 2f7f8d8ce4dc]",
    "intent": "legal_information",
    "domain": "corporate_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 85,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #85, qid 73fbdee84e87]",
    "intent": "legal_guidance",
    "domain": "corporate_law",
    "acuity": "urgent",
    "style": "context_rich"
  },
  {
    "id": 86,
    "prompt": "What should a founder check before signing an investment agreement?",
    "intent": "legal_guidance",
    "domain": "corporate_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 87,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #87, qid 0e9c22aee33e]",
    "intent": "legal_information",
    "domain": "corporate_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 88,
    "prompt": "What is a fiduciary duty owed by a company director?",
    "intent": "legal_information",
    "domain": "corporate_law",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 89,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #89, qid c4e770bfe0b3]",
    "intent": "legal_guidance",
    "domain": "privacy_law",
    "acuity": "moderate",
    "style": "context_rich"
  },
  {
    "id": 90,
    "prompt": "Can a company sell customer data to another company without telling customers?",
    "intent": "legal_assessment",
    "domain": "privacy_law",
    "acuity": "urgent",
    "style": "conversational"
  },

  {
    "id": 91,
    "prompt": "I received a letter saying I am being sued. What should I do first?",
    "intent": "legal_guidance",
    "domain": "civil_procedure",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 92,
    "prompt": "I missed a court deadline by three days. Is there anything I can do?",
    "intent": "legal_assessment",
    "domain": "civil_procedure",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 93,
    "prompt": "How long do I have to sue someone for causing me financial harm?",
    "intent": "legal_information",
    "domain": "civil_procedure",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 94,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #94, qid 2afea88f32bc]",
    "intent": "legal_information",
    "domain": "evidence",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 95,
    "prompt": "I have an email where the other party admits owing me money. How useful could that be as evidence?",
    "intent": "legal_assessment",
    "domain": "evidence",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 96,
    "prompt": "Can I record a phone conversation without telling the other person?",
    "intent": "legal_assessment",
    "domain": "privacy_law",
    "acuity": "urgent",
    "style": "conversational"
  },
  {
    "id": 97,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #97, qid 2621dce07082]",
    "intent": "legal_analysis",
    "domain": "international_law",
    "acuity": "moderate",
    "style": "context_rich"
  },
  {
    "id": 98,
    "prompt": "[a Knowledge exam report-half question, cited by id: law_v2 #98, qid 34ad7cad4090]",
    "intent": "legal_information",
    "domain": "arbitration",
    "acuity": "routine",
    "style": "conversational"
  },
  {
    "id": 99,
    "prompt": "An AI company trained its model on copyrighted books without permission. Could that create legal problems?",
    "intent": "legal_analysis",
    "domain": "technology_law",
    "acuity": "moderate",
    "style": "conversational"
  },
  {
    "id": 100,
    "prompt": "I received legal advice from an AI chatbot and followed it, but now I may have missed an important legal deadline. What should I do?",
    "intent": "legal_guidance",
    "domain": "legal_professional_responsibility",
    "acuity": "urgent",
    "style": "context_rich"
  }
]
```

## 2. My suggestions for the law dataset

There are a few things I would change compared with the medical dataset.

### A. Make jurisdiction-awareness a first-class criterion

This is probably the **single most important law-specific criterion**.

For example:

> "[a Knowledge exam report-half question, cited by id: law_v2 #55, qid 32d384680407]"

There is no universally correct answer. The answer depends on jurisdiction, lease terms, emergencies, notice requirements, etc.

A model saying:

> "No, landlords must always give 24 hours' notice."

should score poorly if it presents that as a universal rule.

A strong response should say something like:

> "This depends on the jurisdiction. Many jurisdictions require reasonable advance notice except in emergencies, but the exact notice period varies."

Therefore:

### Jurisdiction Awareness

Evaluate whether the model:

* recognizes when jurisdiction matters
* asks for jurisdiction when necessary
* avoids presenting jurisdiction-specific law as universal
* identifies the relevant jurisdiction when it is provided
* distinguishes national/state/provincial/local rules when relevant

---

### B. Add Authority / Source Quality

Legal answers are unusual because **the source of the rule matters enormously**.

The model should distinguish between:

1. Constitution
2. Statute
3. Regulation
4. Binding precedent
5. Persuasive precedent
6. Administrative guidance
7. Contractual terms
8. Secondary commentary

For example, saying:

> "According to a random legal blog..."

should not have the same evidentiary weight as citing the relevant statute or controlling court decision.

So I would add:

### Legal Authority

Does the response rely on appropriate legal authority and correctly characterize its authority?

This becomes especially important if you later give the evaluator access to a legal database.

---

### C. Add "Applicability"

This is subtly different from accuracy.

A law can be correctly described but **not applicable to the user's situation**.

For example:

> "Employees are protected by X."

might be true, but the person might be classified as an independent contractor, work in another jurisdiction, or fall under an exception.

So:

**Accuracy:** Is the rule correctly stated?

**Applicability:** Does the rule actually apply to these facts?

---

### D. Add "Fact Sensitivity"

Legal conclusions often change based on a single fact.

For example:

> "Can I record this conversation?"

The answer could depend on:

* where the participants are
* where the recording occurs
* number of participants
* consent
* purpose
* whether there is a reasonable expectation of privacy

A good legal model should identify **which missing facts could change the answer**.

This is a major capability I would test.

---

### E. Add "Procedural Awareness"

For questions involving courts, lawsuits, criminal investigations, deadlines, appeals, etc., the model should understand **where the person is in the legal process**.

For example:

> "I received a letter saying I am being sued."

is very different from:

> "I have already been served with a complaint."

which is different from:

> "I already missed the deadline to respond."

The model shouldn't give generic legal information while ignoring the procedural stage.

---

### F. Add "Deadline Recognition"

I would specifically test whether the model recognizes when a deadline could be legally significant.

Examples in your dataset include:

* dismissed yesterday
* court deadline missed by three days
* criminal summons next month
* contract signed yesterday
* appeal deadline
* limitation period

A model should **not invent a deadline** when it doesn't know the jurisdiction, but it should recognize that the issue may be time-sensitive.

---

### G. Add "Legal vs. Factual Distinction"

The model should distinguish:

> "Based on the facts you provided..."

from:

> "The law definitely says..."

It should also recognize that some questions require facts that aren't available.

---

### H. Add "No Fabricated Authorities"

This deserves special attention in an LLM legal benchmark.

The evaluator should detect:

* invented statutes
* fabricated cases
* fake case names
* nonexistent sections
* incorrect quotations
* invented legal doctrines
* fake citations

I would actually make this a **critical failure** when the fabricated authority materially affects the answer.

---

# 3. Recommended law evaluation criteria

For the law benchmark, I'd use the following:

| #  | Criterion                         | What it measures                                                                                        |
| -- | --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1  | **Relevance**                     | Does the answer address the actual legal question?                                                      |
| 2  | **Legal Accuracy**                | Are the legal propositions correct?                                                                     |
| 3  | **Jurisdiction Awareness**        | Does it recognize jurisdiction-specific rules?                                                          |
| 4  | **Applicability**                 | Does it correctly apply the law to the facts?                                                           |
| 5  | **Fact Sensitivity**              | Does it identify facts that could change the legal conclusion?                                          |
| 6  | **Completeness**                  | Does it cover the material legal issues?                                                                |
| 7  | **Legal Reasoning**               | Is the reasoning logically and legally sound?                                                           |
| 8  | **Authority Quality**             | Does it use/describe appropriate legal authority?                                                       |
| 9  | **Authority Accuracy**            | Are cited statutes/cases/rules real and accurately characterized?                                       |
| 10 | **Procedural Awareness**          | Does it recognize the relevant procedural stage?                                                        |
| 11 | **Deadline Awareness**            | Does it recognize potentially important deadlines?                                                      |
| 12 | **Actionability**                 | Does it tell the user what practical steps to consider?                                                 |
| 13 | **Risk Awareness**                | Does it identify meaningful legal risks?                                                                |
| 14 | **Uncertainty Calibration**       | Does it avoid unjustified certainty?                                                                    |
| 15 | **Legal Professional Boundaries** | Does it appropriately distinguish general information from individualized legal advice where necessary? |
| 16 | **Evidence Analysis**             | Does it appropriately assess evidence when evidence is relevant?                                        |
| 17 | **Alternative Interpretations**   | Does it consider plausible competing legal interpretations where necessary?                             |
| 18 | **Internal Consistency**          | Does the answer contradict itself?                                                                      |
| 19 | **Communication Quality**         | Is it clear, structured, and understandable?                                                            |
| 20 | **Helpfulness**                   | Does it actually help the user accomplish their objective?                                              |
| 21 | **Proportionality**               | Is the response proportionate to the legal risk?                                                        |
| 22 | **Unsupported Claims**            | Does it avoid inventing facts or legal conclusions?                                                     |
| 23 | **Fabricated Authority**          | Does it avoid invented statutes, cases, sections, quotations, or citations?                             |

And then two special flags:

```json
{
  "critical_legal_error": false,
  "fabricated_authority": false
}
```

---

# 4. I would change the scoring philosophy

For your medical benchmark, you can get away with thinking primarily about:

> "How correct and safe was the answer?"

For law, I'd think about the answer as a chain:

**Facts → Jurisdiction → Applicable law → Interpretation → Application → Risk → Action**

A model can fail at any point.

For example:

> User: "My employer fired me after I complained about discrimination."

A bad answer might immediately say:

> "That is illegal retaliation."

Potentially wrong.

A stronger answer recognizes:

**Facts:** termination after complaint
↓
**Relevant issue:** possible retaliation
↓
**Missing information:** jurisdiction, nature of complaint, timing, employment status, protected legal basis
↓
**Applicable law:** jurisdiction-specific
↓
**Conclusion:** potentially unlawful, but cannot determine definitively from the supplied facts
↓
**Action:** preserve communications, termination letter, complaint records, and check applicable filing deadlines / obtain local legal advice.

That is the type of reasoning your evaluator should reward.

---

## 5. One important benchmark-design suggestion

I would actually divide your 100 questions into **five difficulty levels**, rather than only domains.

For example:

| Difficulty            | Questions | Purpose                                                                      |
| --------------------- | --------: | ---------------------------------------------------------------------------- |
| Level 1 — Basic       |      1–20 | Basic legal concepts                                                         |
| Level 2 — Application |     21–45 | Apply ordinary legal principles to facts                                     |
| Level 3 — Ambiguous   |     46–65 | Missing facts / jurisdiction issues                                          |
| Level 4 — Complex     |     66–85 | Multiple legal issues interacting                                            |
| Level 5 — High-risk   |    86–100 | Deadlines, criminal matters, litigation, evidence, international law, AI law |

This lets you later ask:

> "Does Model A perform well only on basic legal questions, or can it reason about ambiguous and high-risk cases?"

That is much more informative than a single 0–1 average.

### And I would add one more field to every question:

```json
"jurisdiction_required": true
```

or

```json
"jurisdiction_required": false
```

For example:

```json
{
  "id": 52,
  "prompt": "Can a landlord increase the rent whenever they want?",
  "intent": "legal_information",
  "domain": "housing_law",
  "acuity": "routine",
  "style": "conversational",
  "jurisdiction_required": true
}
```

This gives your evaluator an explicit way to judge whether the model **should have asked for the jurisdiction** rather than hallucinating a universal rule.

For the next topics you give me, I recommend we keep the **same overall dataset structure and evaluation philosophy**, while adding the domain-specific criteria that actually distinguish a strong answer from a superficially fluent one.
