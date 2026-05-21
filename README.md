# Data Science Projects

A collection of machine learning and data analysis projects.

---

## Beyond Accuracy: Testing Fairness in Student Depression Prediction

Can a machine learning model predict student depression — and does it work equally well for everyone?

This project builds logistic regression and random forest models on a dataset of ~27,000 students, then slices performance by gender, age, financial stress, and degree level to check for fairness disparities.

**Key finding:** The model catches 95% of depression cases in high-financial-stress students but only 62% in low-stress students — a disparity that overall accuracy completely hides.

**Topics:** Machine learning · Fairness · Bias · Mental health · Classification

**Links:**
- [Kaggle Notebook](https://www.kaggle.com/code/danieljhuang/testing-fairness-in-student-depression)

---

## 🔥 Can Student Schedules Predict Burnout?

Can a student's daily schedule — sleep, homework, screen time, activities, support — predict whether they feel burned out?

This project builds and compares four ML models (logistic regression, decision tree, random forest, gradient boosting) on 2,000 students, then uses K-means clustering to discover natural "schedule profiles" and compare burnout rates across them.

**Key finding:** Two students can have identical schedules and very different burnout outcomes depending on the support around them. "Busy but Supported" students burn out far less than "Overloaded Achievers" with the same workload — support is the lever schools can actually pull.

**Best model:** Logistic Regression — ROC-AUC 0.94 on held-out data

**Topics:** `classification` `clustering` `feature-importance` `student-wellbeing` `scikit-learn`

*_More Projects coming soon.._____
