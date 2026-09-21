# Finance & Accounting Evaluation Rubric

## Task

Score **ONE answer to ONE Finance & Accounting question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, accounting framework where applicable, and financial context.

Accept different valid accounting presentations, valuation approaches, financing strategies, portfolio approaches, forecasts, and analytical methods when they are supported by the stated assumptions and evidence.

Do not require a particular wording, calculation layout, or structure. Do not reward unnecessary length, jargon, formula dumping, market commentary, or citations. The benchmark evaluates financial and accounting reasoning rather than resemblance to a reference answer.

## Critical Error

If `critical_finance_accounting_error` applies:

**score = 0**

A critical error is a materially false central accounting, financial, valuation, investment, risk, audit, or quantitative claim that fundamentally invalidates the answer or reverses its substantive conclusion.

Typical critical failures include violating the accounting equation in a central way; equating revenue with cash receipt or net income with cash flow when timing is central; treating depreciation as current-period cash payment; treating principal repayment as operating expense; reversing a material working-capital sign; a decision-reversing NPV/WACC error; fundamentally confusing enterprise and equity value; using incompatible valuation multiples; reversing the bond price-yield relationship; fundamentally incorrect diversification or leverage logic; equating liquidity with solvency; claiming audit assurance is a guarantee; alleging fraud solely from a reasonable estimate difference; or fabricating required facts or assumptions.

Minor arithmetic errors that do not change the conclusion, harmless rounding, minor terminology mistakes, equivalent accounting presentations, defensible valuation methods, reasonable forecast differences, and well-supported alternative financing or portfolio choices are **not** critical.

## Score Anchors

### 4
The answer is financially and accounting-wise accurate, directly addresses the question, applies appropriate concepts and calculations, and reaches a justified conclusion.

Where relevant, it correctly handles accounting mechanics, financial-statement relationships, cash-flow effects, working capital, profitability, leverage, capital budgeting, discounting, cost of capital, valuation, investment risk, portfolio effects, fixed income, audit/control reasoning, assumptions, and uncertainty.

For transaction questions, statement effects are internally consistent. For valuation questions, cash flows, discount rates, and value definitions are aligned. For investment questions, return is interpreted together with risk and relevant constraints. For forecasting questions, assumptions are internally coherent and connected across statements. For audit questions, evidence and control limitations are interpreted appropriately.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3
The answer is substantially correct and useful but contains one meaningful omission, minor accounting/financial imprecision, limited reasoning gap, incomplete assumption discussion, or insufficient qualification. The main conclusion remains correct.

Examples include a minor intermediate calculation error with no effect on the conclusion, incomplete sensitivity discussion, omission of one secondary statement effect, a reasonable but insufficiently qualified valuation assumption, or incomplete treatment of one relevant risk.

### 2
The answer is partly correct or directionally useful but has important omissions or reasoning weaknesses.

Examples include a correct setup but incomplete solution, partially correct cross-statement analysis, correct valuation arithmetic with weak interpretation, ignoring a material working-capital effect, using a plausible but poorly justified discount rate, incomplete risk analysis, or correct accounting mechanics with materially weak decision analysis.

### 1
The answer contains a substantial finance, accounting, valuation, investment, audit, or quantitative misunderstanding, inappropriate framework, or major non-critical calculation error, but does not meet the critical-error threshold. The response demonstrates limited useful understanding.

### 0
Critical Finance & Accounting error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Evaluation Criteria

All criteria have equal nominal weight **0.05**. Apply conditional criteria only when the question calls for them.

| ID | Criterion | Weight | Conditional | Operational definition |
|---|---|---:|:---:|---|
| `relevance` | Relevance | 0.05 | No | Directly addresses the question asked, uses the supplied facts and constraints, and avoids materially distracting finance or accounting discussion. |
| `accounting_accuracy` | Accounting Accuracy | 0.05 | No | Applies accounting mechanics correctly where relevant, preserves the accounting equation, distinguishes recognition from cash timing, and treats assets, liabilities, equity, revenue, expenses, and estimates consistently. |
| `finance_accuracy` | Finance Accuracy | 0.05 | No | Uses correct finance relationships and definitions, including time value, risk-return, financing, valuation, fixed-income, portfolio, and capital-allocation concepts where relevant. |
| `financial_reasoning` | Financial Reasoning | 0.05 | No | Explains the economic mechanism linking facts to conclusions rather than merely naming formulas, ratios, or jargon. |
| `conceptual_precision` | Conceptual Precision | 0.05 | No | Keeps distinct concepts separate, including profit versus cash, revenue versus collection, enterprise versus equity value, coupon versus yield, liquidity versus solvency, and expected versus realized return. |
| `quantitative_correctness` | Quantitative Correctness | 0.05 | Yes | When calculations are required, selects the relevant quantity and formula, uses correct signs, timing, denominators and units, performs arithmetic accurately, and interprets the result. |
| `financial_statement_reasoning` | Financial Statement Reasoning | 0.05 | Yes | When statements are involved, traces effects consistently across the income statement, balance sheet, cash-flow statement and equity statement, including immediate and subsequent-period effects. |
| `cash_flow_reasoning` | Cash-Flow Reasoning | 0.05 | Yes | When cash flows matter, distinguishes operating, investing and financing effects, treats working-capital changes with correct signs, and separates noncash accounting items from cash movements. |
| `valuation_reasoning` | Valuation Reasoning | 0.05 | Yes | When valuation is involved, aligns cash flows, discount rates and value definitions; distinguishes enterprise and equity value; uses compatible multiples; and interprets terminal assumptions and sensitivity. |
| `capital_budgeting_and_cost_of_capital` | Capital Budgeting and Cost of Capital | 0.05 | Yes | When project appraisal or discount rates are relevant, uses incremental cash flows, appropriate timing, NPV/IRR logic, working capital and terminal effects, and a risk-appropriate cost of capital. |
| `investment_and_portfolio_reasoning` | Investment and Portfolio Reasoning | 0.05 | Yes | When investments are involved, computes and interprets returns correctly, distinguishes expected from realized outcomes, and accounts for weighting, correlation, diversification and concentration. |
| `risk_and_leverage_analysis` | Risk and Leverage Analysis | 0.05 | Yes | When risk or financing structure matters, identifies relevant market, credit, liquidity, currency, interest-rate, concentration, refinancing and leverage channels, including downside amplification and residual risk after hedging. |
| `managerial_accounting_reasoning` | Managerial Accounting Reasoning | 0.05 | Yes | When internal decisions are involved, distinguishes fixed, variable, avoidable, sunk and opportunity costs; uses contribution analysis appropriately; and interprets budgets, variances and allocations in context. |
| `audit_and_control_reasoning` | Audit and Control Reasoning | 0.05 | Yes | When audit or controls are relevant, evaluates assertions, evidence, materiality, segregation, authorization, reconciliation, access and review controls without treating reasonable assurance as a guarantee. |
| `assumptions_and_sensitivity` | Assumptions and Sensitivity | 0.05 | Yes | When results depend on assumptions, identifies key drivers, checks internal and economic plausibility, tests material sensitivities or scenarios, and avoids false precision. |
| `evidence_and_uncertainty` | Evidence and Uncertainty | 0.05 | Yes | When facts are incomplete or judgments are contestable, distinguishes evidence from inference, considers alternative explanations, identifies missing information, and calibrates confidence. |
| `completeness` | Completeness | 0.05 | No | Covers the material components needed to answer the question, including secondary statement, risk, assumption, or decision effects when they are necessary to the conclusion. |
| `consistency` | Consistency | 0.05 | No | Maintains internal consistency across calculations, accounting entries, financial statements, periods, assumptions, terminology, and conclusions. |
| `clarity` | Clarity | 0.05 | No | Communicates the analysis in a clear, organized, proportionate way, showing enough setup and interpretation to make the reasoning auditable without unnecessary jargon or formula dumping. |
| `practical_financial_judgment` | Practical Financial Judgment | 0.05 | Yes | When a decision or open-ended evaluation is required, weighs relevant cash flows, opportunity costs, liquidity, leverage, financing consequences, risk, objectives, constraints and uncertainty without forcing a single answer when alternatives are defensible. |

## Financial Decision-Making Principles

### Profit is not cash
Accounting profit and cash flow differ because of accruals, noncash charges, working capital, investing, and financing.

### Revenue is not cash receipt
Revenue recognition and cash collection may occur at different times.

### Financing is not operating performance
Borrowing can increase cash without increasing profit. Debt repayment can reduce cash without reducing operating profit.

### Historical cost and market value differ
Accounting carrying values do not automatically equal current economic values.

### Ratio interpretation requires context
Interpret ratios relative to prior periods, business model, peers, leverage, accounting choices, and risk where relevant.

### Growth can consume cash
Rapid growth may increase receivables, inventory, capital expenditure, and financing needs.

### NPV measures value creation under stated assumptions
Use incremental cash flows and a risk-appropriate discount rate.

### IRR has limitations
IRR can mislead when projects differ in scale, timing, cash-flow pattern, reinvestment assumptions, or have multiple sign changes.

### Cost of capital reflects risk
Do not automatically apply a company-wide rate to a materially different-risk project.

### Leverage amplifies outcomes
Debt can enhance equity returns in favorable outcomes and magnify downside risk.

### Higher ROE is not automatically better
ROE can rise because of leverage or a reduced equity base rather than stronger operations.

### Diversification depends on correlation
Owning many highly correlated assets may provide limited diversification.

### Market price is not intrinsic value
Observed prices and analytical valuation estimates are distinct concepts.

### Valuation is assumption-dependent
DCF and relative valuation depend materially on growth, margins, reinvestment, risk, discount rates, and terminal conditions.

### Hedging does not necessarily eliminate risk
Hedges may leave basis, counterparty, liquidity, and opportunity-cost risks.

### Audit is not a guarantee
Reasonable assurance does not mean all errors or fraud are impossible.

### Internal controls reduce rather than eliminate risk
Controls can fail, be overridden, or operate incorrectly.

### Accounting judgment is not automatically manipulation
Estimates and policy choices can involve legitimate ranges of judgment.

### Multiple valid financial strategies
Financing, investment, hedging, and capital-allocation decisions may have several defensible approaches. Evaluate assumptions, objectives, constraints, and risks.

### No formula worship
A correct formula applied to irrelevant inputs should not receive a high score.

### No multiple worship
A valuation multiple without comparable fundamentals or numerator-denominator consistency is not sufficient analysis.

### No jargon worship
Financial terminology without correct reasoning should not receive a high score.

### No citation worship
Citations do not compensate for incorrect accounting or financial reasoning.

## Evaluation Notes

- Preserve the accounting equation and cross-statement consistency.
- Distinguish recognition from cash timing, operating performance from financing, and accounting carrying amounts from market values.
- For quantitative work, grade framework selection, setup, calculations, signs, timing, units, and interpretation—not arithmetic alone.
- For valuation, align the cash-flow definition with the discount rate and value definition.
- For open-ended decisions, multiple answers can earn full credit when assumptions, objectives, constraints, and risks are handled well.
- Treat historical evidence, forecasts, scenarios, and sensitivities as different forms of information.
- Do not reward unsupported certainty, formula dumping, jargon, or citations that do not improve the reasoning.
