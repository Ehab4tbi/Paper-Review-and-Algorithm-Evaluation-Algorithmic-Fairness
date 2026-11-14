# Task 1 – Paper Review: outline and working notes

> These are notes, not finished text. The assignment requires the summary to be in **your own words**, and copying sentences from the paper gives zero points. Write each section yourself, using these notes and your own reading of the paper. Anything marked [CHECK] must be confirmed in the paper before you use it.

**Paper:** F. Kamiran and T. Calders (2012). *Data preprocessing techniques for classification without discrimination.* Knowledge and Information Systems, 33(1), 1–33.

**Target length:** at most 2 pages (about 900–1,000 words).

---

## 1. Problem and setting (about 120 words)
Notes:
- Historical decisions (hiring, credit, admissions) can contain discrimination, and a classifier trained on them learns and repeats it.
- Goal: learn a classifier that stays accurate but whose predictions do not discriminate.
- Setting: one binary sensitive attribute (a deprived group vs. a favoured group) and a binary class with one desired outcome.
- Assumption: the discrimination is in the training labels, but at prediction time the model should be non-discriminatory. [CHECK how the authors phrase this assumption]

Highlight: introduction / problem statement, and the definition of the sensitive attribute and the desired class.

## 2. Category (about 60 words)
Notes:
- Pre-processing: the data is changed **before** training, and the learning algorithm itself is not changed.
- Benefit: any standard classifier can be trained on the cleaned data afterwards.

Highlight: where the authors say they modify the input data rather than the algorithm.

## 3. Fairness measure targeted (about 120 words)
Notes:
- The paper's "discrimination" = P(desired class | favoured group) − P(desired class | deprived group).
- This is statistical (demographic) parity, the same quantity as DP_Diff in Task 2 (with the sign flipped).
- Measured on the dataset (labels) and on the classifier's predictions.
- Removing the sensitive attribute does not solve the problem, because other attributes correlated with it carry the same information ("redlining"). [CHECK the example the paper uses]

Highlight: the formal definition of discrimination, and the redlining / attribute-removal discussion.

## 4. The algorithm and its alternatives (about 300 words)
**Reweighing (main algorithm):**
- For each combination of group and class: weight = expected probability if group and class were independent ÷ observed probability = P(s)·P(c) / P(s, c).
- Combinations that occur less often than expected (e.g., deprived group with the desired outcome) get a weight above 1; over-represented combinations get a weight below 1.
- Result: in the weighted data, the discrimination measure is exactly 0.
- Labels and features are not changed. The classifier must accept instance weights.
- Own example from Task 2 (Adult): high-income women weight 2.19, high-income men 0.79.

**Other techniques in the paper (briefly):**
- Suppression: a baseline that removes the sensitive attribute and the attributes most correlated with it. [CHECK]
- Massaging: a ranker (e.g., Naive Bayes) scores everyone; the most borderline deprived-group negatives are relabelled positive and the most borderline favoured-group positives relabelled negative, just enough to remove the discrimination.
- Uniform Sampling: randomly duplicate or drop rows in each (group, class) combination until it reaches its expected size.
- Preferential Sampling: the same idea, but duplicate or drop the rows closest to the decision boundary (chosen with a ranker).

Highlight: the Reweighing section including the weight formula, and the short descriptions of Massaging, Uniform Sampling and Preferential Sampling.

## 5. Optimization techniques and special cases (about 120 words)
Notes:
- No iterative optimization: the weights have a closed formula and are computed once in a single pass over the data.
- Special case: if the data has no discrimination, every weight equals 1 and nothing changes.
- Special case: for learners that cannot use weights, Sampling reproduces the effect of the weights by resampling.
- The paper analyses the trade-off between accuracy and discrimination theoretically. [CHECK the exact statement of the theorem / result and summarise it in your own words]

Highlight: the theoretical trade-off section, and any remark on classifiers that do not support weights.

## 6. Experiments and findings (about 100 words)
Notes:
- Datasets: Census Income (Adult), German Credit and others. [CHECK the full list, e.g. Dutch census, Communities and Crime]
- Classifiers tested: [CHECK, e.g. Naive Bayes, decision tree (J48), k-NN]
- Main result: discrimination drops substantially for a small loss in accuracy. [CHECK which technique the authors found best overall]
- Authors' point: accuracy is measured against biased test labels, so some of the "loss" is really the model no longer copying discrimination.

Highlight: the experimental setup and the conclusions of the experiments.

## 7. Strengths (about 100 words)
- Simple and easy to explain.
- Works with any model that accepts sample weights, and does not depend on the model.
- No hyper-parameters to tune.
- Features and labels stay unchanged (unlike Massaging), so the data stays truthful.
- Cheap to compute: one pass over the data.
- Task 2 confirms it: DP_Diff reduced by 49–87 % for about 1 point of accuracy.

## 8. Limitations (about 120 words)
- Handles only one binary sensitive attribute and a binary label (in its original form).
- Targets only group fairness (statistical parity). It does not ensure individual fairness and may conflict with error-based measures such as equalized odds.
- Proxy attributes can still carry the bias: in Task 2, Adult's DIR was still only 0.575.
- Requires a learner that supports instance weights (otherwise use Sampling).
- Weights become large and unstable when a (group, class) combination is very small.
- Parity is guaranteed only in the weighted training data, not in the predictions on new data.
- Statistical parity may not be the right goal when outcome differences between groups are justified. [CHECK whether the authors discuss this, e.g. explainable discrimination]

---

