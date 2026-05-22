Data Science Projects
A collection of machine learning and data analysis projects.

Beyond Accuracy: Testing Fairness in Student Depression Prediction
Can a machine learning model predict student depression and does it work equally well for everyone?
This project builds logistic regression and random forest models on a dataset of ~27,000 students, then slices performance by gender, age, financial stress, and degree level to check for fairness disparities.
Key finding: The model catches 95% of depression cases in high-financial-stress students but only 62% in low-stress students, a disparity that overall accuracy completely hides.
Topics: Machine learning, Fairness, Bias, Mental health, Classification
Links:

Kaggle: https://www.kaggle.com/code/danieljhuang/testing-fairness-in-student-depression


Can Student Schedules Predict Burnout?
Can a student's daily schedule — sleep, homework, screen time, activities, support — predict whether they feel burned out?
This project builds and compares four ML models on 2,000 students, then uses K-means clustering to discover natural schedule profiles and compare burnout rates across them.
Key finding: Two students can have identical schedules and very different burnout outcomes depending on the support around them. Busy but supported students burn out far less than overloaded achievers with the same workload.
Best model: Logistic Regression — ROC-AUC 0.94 on held-out data
Topics: classification, clustering, feature-importance, student-wellbeing, scikit-learn

Composer Fingerprinting in Classical Piano
A symbolic music analysis on GiantMIDI-Piano (1,123 MIDI files, 10 composers) with held-out-era evaluation and conformal uncertainty quantification.
Can structural features of a MIDI score identify its composer? Most published results on this task use random splits that allow the model to see the same composer in both training and test. This project runs three increasingly strict evaluation protocols and quantifies how much of the apparent accuracy depends on that leakage.
Key finding: 69.9% top-1 accuracy and 89% top-3 accuracy under random CV. Era-level accuracy is 84%. 86.5% of errors are within the correct era — confusing Chopin with Schumann is far more common than confusing Chopin with Bach. Composer classification is partly an era-classification problem in disguise.
Methods: pitch-class histograms, interval-class histograms, Forte trichord types, Lerdahl pitch-space steps, calibrated LightGBM, isotonic calibration, APS conformal prediction sets, leave-one-era-out CV, SHAP
Topics: composer-classification, symbolic-music, midi, lightgbm, conformal-prediction, shap
Links:

Kaggle: https://www.kaggle.com/code/danieljhuang/composer-fingering-in-classical-piano


More projects coming soon.
