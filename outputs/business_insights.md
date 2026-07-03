# Business Insights & Recommendations

## Key Insights

1. Overall churn rate is 20.37% across the customer base.

2. Germany customers churn at 32.44%, well above the other markets (France: 16.15%, Spain: 16.67%) - suggesting a market-specific service, pricing, or competitive issue worth investigating.

3. Customers holding 3-4 products churn at 82.71%-100.0%, dramatically higher than the 7-8% churn rate of customers with exactly 2 products. This is very likely a symptom of over-selling or unresolved service issues rather than genuine multi-product loyalty, and should be treated as a red flag, not a cross-sell success.

4. Inactive members make up 48.5% of the base and exit at a substantially higher rate than active members, confirming activity status as one of the strongest retention levers.

5. 1,271 customers (50.84% of the high-value segment) are both high-value and disengaged, representing an estimated $180,160,771 in balance/income exposure and a 30.76% churn rate - the single highest-leverage segment for a targeted retention campaign.

6. The Retention Strength Score cleanly separates risk: Critical Risk customers churn at 53.33% versus 10.22% for Strong-band customers, validating it as a usable early-warning signal for frontline teams.

7. The XGBoost model achieves 85.9% ROC-AUC and 72.0% recall on churners, meaning it correctly flags the majority of customers who go on to leave - suitable for prioritizing outreach lists, not for fully automated decisions.

8. The 51-60 age group shows the highest churn (56.21%), an important segment for tailored messaging and product design.


## Recommendations

- **[High]** Launch a proactive outreach campaign for high-value disengaged customers (1,271 identified) - personal relationship manager contact, tailored offers, and a service health-check within 30 days.

- **[High]** Audit the multi-product (3-4 product) customer journey. The 85.89% churn rate in this group points to bundling, fee stacking, or cross-sell practices that are backfiring - review sales incentives tied to product count.

- **[High]** Investigate Germany market specifically: conduct customer interviews or NPS deep-dive to identify whether the elevated churn is driven by pricing, local competition, service quality, or regulatory friction.

- **[Medium]** Build an automated 're-engagement' trigger for members who go inactive for 60+ days, before they reach the 'inactive member' churn risk profile.

- **[Medium]** Operationalize the Retention Strength Score in CRM so relationship managers can see a live 0-100 score per customer and prioritize Critical Risk / At Risk accounts in their weekly outreach.

- **[Medium]** Use the churn prediction model's monthly scored list to feed targeted retention offers (rate improvements, fee waivers, loyalty perks) to the top-decile predicted-churn customers.

- **[Low]** Monitor model drift quarterly - retrain on fresh data as customer behavior and product mix evolve, and re-validate feature importance rankings.


## Model Summary

- Best model: **XGBoost**

- accuracy: 0.809

- precision: 0.5223

- recall: 0.7199

- f1_score: 0.6054

- roc_auc: 0.8594
