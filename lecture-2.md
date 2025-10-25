# Decision Trees — Lecture W4 Cheat Sheet

A crisp, exam-ready summary of Decision Trees, splitting criteria, pruning, cross‑validation, and evaluation, tailored to your slides.

## What is a Decision Tree?

- A supervised learning model that recursively partitions the input space and predicts by majority vote (classification) or average (regression) in leaves.
- Learns human‑readable IF–THEN rules, e.g., IF Outlook = Overcast THEN Play = Yes.

## Core algorithms

- ID3 (Quinlan, 1986): uses Information Gain (entropy) to choose splits; handles categorical features.
- C4.5: successor to ID3; handles continuous features via thresholds, deals with missing values, uses Gain Ratio.
- CART (Breiman et al., 1984/85): uses Gini for classification and MSE for regression; binary splits only; basis for scikit‑learn trees.

## How trees are built (high level)

1. Pick the split that best reduces impurity (highest gain, lowest Gini/MSE).
2. Create a child for each split outcome.
3. Recurse on each child until a stopping rule triggers.

Typical stopping rules (pre‑pruning):

- Max depth, min samples to split, min samples per leaf, min impurity decrease, class purity at node.

## Handling different feature types

- Categorical: multiway split, one‑vs‑rest, or grouped categories.
- Continuous/real‑valued: learn a threshold t and split on x_j ≤ t vs > t.
- Regression: choose splits that minimize node mean squared error (MSE).

## Overfitting and pruning

- Overfitting: tree memorizes noise; high training accuracy, poor generalization.
- Pre‑pruning (early stopping): control growth with hyperparameters (max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease).
- Post‑pruning (cost‑complexity pruning): grow full tree, then trim branches using a penalty α for tree size (ccp_alpha). Select α via validation or cross‑validation.

## Cross‑validation (detect/avoid overfitting)

- Holdout: simple train/test split (e.g., 50/50). Fast, higher bias.
- LOOCV: train on n−1, test on 1; repeats n times. Low bias, high variance, slow.
- Stratified k‑Fold: preserves class ratios in each fold; essential for imbalanced data.
- k‑Fold (often k=10): rotate which fold is test; balances bias/variance.

## Evaluation highlights

- Accuracy can be misleading with class imbalance; prefer Precision/Recall and F1 when positives are rare/costly.
- F1 balances Precision and Recall; useful when you need a single score reflecting both.

## Symbols reference (what each symbol means)

| Symbol                                   | Name                      | Meaning / how it’s used                                     |
| ---------------------------------------- | ------------------------- | ----------------------------------------------------------- |
| $S$                                      | Sample set / node dataset | The set of training examples reaching a node.               |
| $A$                                      | Attribute / feature       | Candidate feature used to split a node.                     |
| $p_i$                                    | Class proportion          | $p_i = \Pr(y=i\mid S)$, fraction of class $i$ in $S$.       |
| $H(S)$                                   | Entropy                   | Impurity/uncertainty of labels in $S$.                      |
| $v$                                      | Attribute value           | A specific value taken by attribute $A$.                    |
| $\mathcal{V}(A)$                         | Value set of $A$          | The set of all possible values that $A$ can take.           |
| $S_v$                                    | Value subset              | Subset of $S$ where $A=v$.                                  |
| $\lvert S \rvert$                        | Cardinality of $S$        | Number of samples in $S$.                                   |
| $IG(S,A)$                                | Information Gain          | Decrease in entropy after splitting $S$ on $A$.             |
| $\text{Gini}(S)$                         | Gini impurity             | CART impurity measure for classification nodes.             |
| $\text{Err}(S)$                          | Misclassification error   | Fraction not in the majority class in $S$.                  |
| $x_j$                                    | Feature $j$               | The $j$‑th input feature used for threshold splits.         |
| $t$                                      | Threshold                 | Numeric cutpoint for splitting a continuous feature.        |
| $n$                                      | Samples count             | Number of samples in a node (often $n=\lvert S \rvert$).    |
| $y_k$                                    | Target value              | The target of the $k$‑th sample in a node (regression).     |
| $\bar{y}$                                | Node mean                 | Mean target value in a node (regression leaf prediction).   |
| $T$                                      | (Sub)tree                 | A decision tree (or subtree) structure.                     |
| $\lvert T \rvert$                        | Tree size                 | Number of leaves (terminal nodes) in $T$.                   |
| $R(T)$                                   | Empirical risk            | Training loss of tree $T$ (e.g., misclassification or SSE). |
| $\alpha$                                 | Complexity weight         | Penalty coefficient in cost‑complexity pruning.             |
| $TP, FP, TN, FN$                         | Confusion counts          | True/False Positives/Negatives used to compute metrics.     |
| $\text{Precision},\ \text{Recall},\ F_1$ | Metrics                   | Derived from $TP,FP,TN,FN$; see formulas above.             |

## Functions and terms reference table

| Term / Formula                                                                                     | Name                      | What it is and how it works                                                                                                                                        |
| -------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $H(S) = -\sum_i p_i \, \log_2 p_i$                                                                 | Entropy                   | Measures label uncertainty/impurity in set $S$. Higher when classes are mixed; 0 when pure. Used by ID3/C4.5 to guide splits.                                      |
| $IG(S, A) = H(S) - \sum_{v\in \mathcal{V}(A)} \frac{\lvert S_v \rvert}{\lvert S \rvert} \, H(S_v)$ | Information Gain          | Reduction in entropy achieved by splitting $S$ on attribute $A$. Choose attribute with max gain as the next split.                                                 |
| $\text{Gini}(S) = 1 - \sum_i p_i^2$                                                                | Gini Impurity             | CART criterion for classification. Probability of incorrect labeling if a label is drawn at random from $S$. Lower is purer.                                       |
| $\text{Err}(S) = 1 - \max_i p_i$                                                                   | Misclassification Error   | Simple impurity measure: fraction not in the majority class. Less sensitive than entropy/Gini; sometimes used for pruning.                                         |
| Split: $x_j \le t$                                                                                 | Threshold split           | For continuous feature $x_j$, pick threshold $t$ that maximizes gain (classification) or minimizes weighted MSE (regression).                                      |
| $\text{MSE} = \frac{1}{n} \sum_{k=1}^{n} (y_k - \bar{y})^2$                                        | Mean Squared Error        | Node impurity for regression trees; choose splits that minimize the (weighted) MSE of children. Leaf predicts mean $\bar{y}$.                                      |
| $R_\alpha(T) = R(T) + \alpha\,\lvert T \rvert$                                                     | Cost‑complexity objective | Post‑pruning: total misclassification (or SSE) plus a penalty on number of leaves $\lvert T \rvert$. Increase $\alpha$ to prune more. Implemented via `ccp_alpha`. |
| $\text{Precision} = \frac{TP}{TP + FP}$                                                            | Precision                 | Of predicted positives, how many are correct. Sensitive to false positives.                                                                                        |
| $\text{Recall} = \frac{TP}{TP + FN}$                                                               | Recall (Sensitivity)      | Of actual positives, how many were found. Sensitive to false negatives.                                                                                            |
| $F_1 = 2\cdot \frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$           | F1 Score                  | Harmonic mean of Precision and Recall; high only when both are high. Prefer over accuracy for imbalanced data.                                                     |
| ID3, C4.5, CART                                                                                    | Tree learners             | Strategies for choosing splits and handling feature types. ID3/C4.5 use entropy (C4.5 adds gain ratio, thresholds); CART uses Gini/MSE and binary splits.          |
| Pre‑pruning hyperparams: max_depth, min_samples_split, min_samples_leaf                            | Early stopping            | Limits tree growth to reduce variance/overfitting. Tune via validation/CV.                                                                                         |
| Post‑pruning: `ccp_alpha`                                                                          | Cost‑complexity pruning   | Grow full tree, prune weakest links by increasing $\alpha$; pick $\alpha$ minimizing validation error.                                                             |
| Random Forest (bagging)                                                                            | Ensemble of trees         | Trains many trees on bootstrap samples and random feature subsets; averages to reduce variance and overfitting.                                                    |
| Holdout, LOOCV, k‑Fold, Stratified k‑Fold                                                          | Cross‑validation schemes  | Protocols for estimating generalization and tuning hyperparameters, especially pruning strength.                                                                   |
| Bias vs. Variance                                                                                  | Error sources             | Bias: error on training (oversimplification). Variance: sensitivity to data (overfitting). Balance via model complexity and ensembles.                             |

---

## Quick build recipe (classification)

- Compute impurity at node: use Entropy (ID3/C4.5) or Gini (CART).
- For each feature:
  - Categorical: evaluate splits per value (or grouped), compute weighted child impurity.
  - Continuous: sort unique values; evaluate midpoints as thresholds.
- Pick the split with maximum impurity reduction (gain) and recurse.
- Stop when a stopping rule fires; assign leaf label by majority class.

## Practical tips

- Prefer Stratified CV when classes are imbalanced; report F1, not just accuracy.
- Shallow trees (lower max_depth) generalize better; prune with validation, not training error.
- For regression trees, also check MAE if outliers dominate; MSE is more sensitive to outliers.
- Random Forest is a strong default: lower variance than a single tree with similar bias.

## scikit‑learn knobs (for quick use)

- Classification: `DecisionTreeClassifier(criterion="gini"|"entropy"|"log_loss", max_depth=..., min_samples_leaf=..., ccp_alpha=...)`
- Regression: `DecisionTreeRegressor(criterion="squared_error", max_depth=..., min_samples_leaf=..., ccp_alpha=...)`
- Post‑pruning path: use `cost_complexity_pruning_path` to pick `ccp_alpha` via CV.

## Tiny mnemonic

- Entropy/IG: “information‑hungry” (choose the most informative split).
- CART/Gini: “fast and binary.”
- Pruning: “grow, then trim with α.”
- CV: “stratify when in doubt.”

---
