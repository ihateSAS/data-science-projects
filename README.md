Data Science Projects
A collection of machine learning and data analysis projects.
---
 Beyond Accuracy: Testing Fairness in Student Depression Prediction
Can a machine learning model predict student depression — and does it work equally well for everyone?
This project builds logistic regression and random forest models on a dataset of ~27,000 students, then slices performance by gender, age, financial stress, and degree level to check for fairness disparities.
Key finding: The model catches 95% of depression cases in high-financial-stress students but only 62% in low-stress students — a disparity that overall accuracy completely hides.
Topics: Machine learning · Fairness · Bias · Mental health · Classification
Links:
 Kaggle Notebook
---
 Can Student Schedules Predict Burnout?
Can a student's daily schedule — sleep, homework, screen time, activities, support — predict whether they feel burned out?
This project builds and compares four ML models (logistic regression, decision tree, random forest, gradient boosting) on 2,000 students, then uses K-means clustering to discover natural "schedule profiles" and compare burnout rates across them.
Key finding: Two students can have identical schedules and very different burnout outcomes depending on the support around them. "Busy but Supported" students burn out far less than "Overloaded Achievers" with the same workload — support is the lever schools can actually pull.
Best model: Logistic Regression — ROC-AUC 0.94 on held-out data
Topics: `classification` · `clustering` · `feature-importance` · `student-wellbeing` · `scikit-learn`
---
 The Mathematics of Piano Difficulty
An ordinal-regression benchmark on the CIPI piano score dataset, with composer-disjoint evaluation and conformal uncertainty.
Automatic difficulty grading is an open MIR problem. The published SOTA (Ramoneda et al., ESWA 2024) reports 39.5% balanced accuracy / 1.1 MSE on 9-class Henle grades using neural backbones over fingering and expressive-performance models. This project asks how much of that gap a correctly designed tabular ordinal model closes — and how much of the published number depends on composer-style leakage across random splits.
Methods: Musicologically grounded features (Lerdahl pitch-space distance, IOI entropy, beat-aligned hand independence). Frank–Hall ordinal GBM with monotonic constraints. CORN MLP for rank-consistent neural ordinal regression (Shi/Cao/Raschka 2023). Leave-one-composer-out CV. Conformal prediction sets at 90% coverage. Label-noise sensitivity via bootstrap.
Key finding: Going from random stratified CV to leave-one-composer-out degrades MAE by ~0.5 grades — published benchmarks likely overstate generalization to unseen composers by a similar margin. SHAP on the median binary head shows `hand_independence` and `max_simultaneous_span` dominating `notes_per_sec`, quantifying the pianist's intuition that a slow Bach fugue outranks a fast Czerny etude.
Topics: `ordinal-regression` · `music21` · `corn-loss` · `monotonic-gbm` · `conformal-prediction` · `loco-cv` · `shap`
Links:
 Kaggle Notebook
---
More projects coming soon.
