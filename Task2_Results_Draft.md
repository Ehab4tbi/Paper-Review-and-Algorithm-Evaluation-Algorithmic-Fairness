# Task 2 – Evaluating Reweighing on Adult, German Credit and COMPAS

Code: <GitHub repository link>

## 1. Experimental setup

**Datasets and protected attributes.** Three benchmark datasets were used. Each has a binary protected attribute *s* (1 = privileged) and a binary label *y* (1 = favourable outcome).

| Dataset | Rows | Protected attribute (privileged) | Favourable label | P(y=1) | P(s=1) |
| --- | --- | --- | --- | --- | --- |
| Adult (UCI) | 30,162 | sex (male) | income > 50K | 0.249 | 0.676 |
| German Credit (UCI Statlog) | 1,000 | age (> 25 years) | good credit risk | 0.700 | 0.810 |
| COMPAS (ProPublica, two-year) | 6,172 | race (non-African-American) | no recidivism in two years | 0.545 | 0.486 |

**Pre-processing.** Adult: 2,399 rows with missing values ("?") removed; *education* (duplicate of *education.num*) and *fnlwgt* (a census sampling weight) dropped. German: original categorical file. COMPAS: ProPublica's filtered data, with the COMPAS score itself (*score_factor*) excluded so the model does not inherit the tool's decisions. Categorical features one-hot encoded, numerical features standardised; the encoder is fitted on the training split only.

**Mitigation pipeline.** Reweighing is a pre-processing method, so only the training data is modified:
1. Stratified 70/30 train/test split, stratified jointly on label and protected group.
2. Fit Reweighing on the training split: every (group, label) combination receives the weight W(s, y) = P(s)·P(y) / P(s, y).
3. Train a logistic-regression classifier with these weights as sample weights.
4. Evaluate the predictions on the untouched test split.

The same classifier trained without weights serves as the baseline. Reweighing was implemented directly from the paper's formula. Its correctness was checked on a small example with known weights, and the weighted demographic parity difference of every training split was confirmed to be exactly 0.

**Parameter settings.** Logistic regression (L2, C = 1.0, lbfgs solver, max_iter = 2000); test size 0.30; five random splits (seeds 0, 1, 2, 3, 42), reported as mean ± standard deviation; consistency with k = 5 nearest neighbours in the encoded feature space. The protected attribute is kept as a model input in all experiments.

**Metrics.** DP_Diff = P(ŷ=1 | s=0) − P(ŷ=1 | s=1), where 0 is fair. DIR = P(ŷ=1 | s=0) / P(ŷ=1 | s=1), where 1 is fair and the "80 % rule" requires at least 0.8. Consistency = 1 − mean |ŷᵢ − mean of ŷ over the 5 nearest neighbours of xᵢ|. Predictive performance is reported as accuracy, balanced accuracy and F1. The before-mitigation values are computed on the original labels of the full dataset, before any split or training.

## 2. Results

**Table 1 – Fairness before mitigation (original labels) and after mitigation (test-set predictions, mean over 5 splits).**

| Dataset | Stage | DP_Diff | DIR | Consistency |
| --- | --- | --- | --- | --- |
| Adult | Original labels | −0.200 | 0.362 | 0.839 |
| | Baseline LR | −0.194 ± 0.007 | 0.288 ± 0.011 | 0.957 ± 0.002 |
| | Reweighing + LR | **−0.095 ± 0.006** | **0.575 ± 0.015** | 0.959 ± 0.001 |
| German | Original labels | −0.149 | 0.795 | 0.742 |
| | Baseline LR | −0.159 ± 0.054 | 0.799 ± 0.073 | 0.821 ± 0.020 |
| | Reweighing + LR | **−0.081 ± 0.033** | **0.897 ± 0.045** | 0.819 ± 0.022 |
| COMPAS | Original labels | −0.140 | 0.773 | 0.618 |
| | Baseline LR | −0.292 ± 0.031 | 0.621 ± 0.033 | 0.987 ± 0.002 |
| | Reweighing + LR | **−0.037 ± 0.051** | **0.945 ± 0.076** | 0.986 ± 0.001 |

**Table 2 – Predictive performance on the test set (mean ± std over 5 splits).**

| Dataset | Model | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| Adult | Baseline | 0.848 ± 0.002 | 0.770 ± 0.003 | 0.668 ± 0.005 |
| | Reweighing | 0.842 ± 0.002 | 0.751 ± 0.002 | 0.642 ± 0.002 |
| German | Baseline | 0.759 ± 0.027 | 0.681 ± 0.026 | 0.836 ± 0.021 |
| | Reweighing | 0.755 ± 0.034 | 0.672 ± 0.039 | 0.834 ± 0.024 |
| COMPAS | Baseline | 0.673 ± 0.010 | 0.663 ± 0.010 | 0.719 ± 0.008 |
| | Reweighing | 0.660 ± 0.007 | 0.649 ± 0.008 | 0.712 ± 0.008 |

**Figure 1 – DP_Diff and DIR before mitigation (original labels), for the baseline and after Reweighing. Error bars: std over 5 splits; dashed line: 80 % rule.** *(insert fairness_before_after.png)*

## 3. Discussion

**Bias in the original data.** All three datasets favour the privileged group before any model is trained. Adult is by far the most biased: women receive the favourable label at only 36 % of the rate of men (DIR 0.362). German Credit and COMPAS have smaller gaps (DP_Diff around −0.14 to −0.15), and their DIR values of 0.795 and 0.773 fall just short of the 80 % rule.

**The unmitigated classifier keeps, and can amplify, this bias.** The baseline logistic regression reproduces the label bias on Adult and German, and makes Adult's ratio worse (DIR 0.362 → 0.288). On COMPAS the amplification is strongest: the parity gap doubles from −0.140 in the labels to −0.292 in the predictions. A plausible explanation is that features such as the number of prior offences are distributed differently across races, so the model pushes many African-American defendants across the decision threshold. This shows why measuring fairness only on the data is not enough.

**Effect of Reweighing.** Reweighing reduced the parity gap on every dataset: by 51 % on Adult (−0.194 → −0.095), 49 % on German (−0.159 → −0.081) and 87 % on COMPAS (−0.292 → −0.037). German (DIR 0.897) and COMPAS (DIR 0.945) now satisfy the 80 % rule. Adult improves strongly (DIR 0.288 → 0.575) but still does not pass. The learned weights explain the size of the effect: Adult has the most extreme weight (2.19 for high-income women, against 0.79 for high-income men), which reflects its large initial imbalance. For German and COMPAS all weights lie between 0.71 and 1.21. The weights only make sex and income independent in the training data. Features strongly related to sex, such as *relationship* (husband/wife) and *marital status*, can still pass sex information to the model. This is the "redlining" effect discussed in the paper.

**Fairness–accuracy trade-off.** The cost of mitigation is small: accuracy drops by 0.4 (German), 0.6 (Adult) and 1.3 (COMPAS) percentage points, and balanced accuracy by 0.9 to 1.9 points. On German, the drop is smaller than the variation between splits. Accuracy is measured against the original, biased test labels, so part of this "loss" comes from the model no longer reproducing historical discrimination. Kamiran and Calders make the same argument.

**Consistency.** Consistency barely changes with Reweighing (differences ≤ 0.002). This is expected: Reweighing targets group fairness (statistical parity) and does not explicitly constrain how similar individuals are treated. The large difference between the consistency of the original labels (0.62–0.84) and of the predictions (0.82–0.99) mainly reflects that a linear model gives smooth decisions to nearby points, while real labels are noisy. It should not be read as an improvement in individual fairness caused by the model.

**Stability and limitations.** Results vary more on the smaller datasets: the DIR after Reweighing has a standard deviation of 0.076 on COMPAS, and on German (300 test rows) the fairness metrics vary by 0.03–0.07, so single-split results should be read with care. This evaluation uses one classifier, one binary protected attribute per dataset and only parity-based measures; error-rate measures such as equalized odds may behave differently.

**Conclusion.** Reweighing is simple, needs no tuning and works with any classifier that accepts sample weights. In all three datasets it roughly halved or almost removed the demographic parity gap, for an accuracy loss of at most 1.3 percentage points. It is not a complete solution, as Adult shows: when proxy features carry the protected attribute, reweighing the training data alone does not reach the 80 % threshold.
