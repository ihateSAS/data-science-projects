# Can Student Schedules Predict Burnout?

> A machine learning project on the hidden cost of being busy.

This project asks a simple question: **given a student's daily schedule — sleep, homework, screen time, activities, support from family and friends — can we predict whether they feel highly burned out?**

It walks through exploratory data analysis, four machine learning models, feature importance, and a clustering step that discovers natural "schedule profiles" of students.

## Results at a glance

| Model | Cross-val ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | 0.90 | **0.94** |
| Gradient Boosting | 0.89 | 0.92 |
| Random Forest | 0.89 | 0.92 |
| Decision Tree | 0.78 | 0.82 |

**Top drivers of burnout** (after controlling for everything else):
- 🛌 **Less sleep** and **poorer sleep quality** — the strongest protective factors when present
- 📚 **More homework hours** and **more tests per week**
- 👨‍👩‍👧 **Lower family, friend, and teacher support** — *independent* protective effects
- 📱 **Screen time** matters mostly *when sleep is already short* — an interaction effect

**Schedule profiles found via clustering:**

| Profile | Pattern | Burnout rate |
|---|---|---|
| Overloaded Achievers | Heavy homework + many activities + low sleep | Very high |
| Screen-Time Night Owls | High screens, short sleep, low sleep quality | High |
| Under-Supported Students | Low support across all sources | High |
| Busy but Supported | Heavy workload offset by strong support | Moderate |
| Balanced Students | Average everything, good support | Low |

The key insight: **"Busy but Supported"** and **"Overloaded Achievers"** look identical on a schedule alone. What separates them is the *support* around them. That's the lever a school can actually pull.

## Project structure

```
.
├── data/
│   ├── student_burnout.csv          # The dataset (2,000 students)
│   └── generate_dataset.py          # Script that produced it
├── notebooks/
│   └── burnout_analysis.ipynb       # Full analysis with all outputs
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
git clone https://github.com/<your-username>/student-burnout-prediction.git
cd student-burnout-prediction
pip install -r requirements.txt
jupyter notebook notebooks/burnout_analysis.ipynb
```

Or open the notebook directly on Kaggle / Colab — every cell already has its output rendered, so you can read end-to-end without running anything.

## About the data

The dataset is **synthetic but realistic** — 2,000 simulated students with relationships drawn from adolescent health research. The data-generation script (`data/generate_dataset.py`) is included and documented, so you can:

- Inspect exactly what relationships are baked in (no hidden ground truth)
- Tweak the parameters and regenerate to test how the analysis behaves
- Replace it entirely with real survey data — the modeling pipeline is column-name driven and works on any CSV with the same schema

### Schema

| Column | Type | Description |
|---|---|---|
| `student_id` | int | Unique identifier |
| `grade` | int | 9–12 |
| `gender` | str | Female / Male / Nonbinary |
| `sleep_hours` | float | Average hours per night |
| `sleep_quality` | int | Self-rated 1–5 |
| `homework_hours` | float | Average per day |
| `tests_per_week` | int | Quizzes + exams |
| `extracurricular_hours` | float | Per week, all activities |
| `num_activities` | int | Number of distinct extracurriculars |
| `screen_time_hours` | float | Phone / social media per day |
| `commute_minutes` | int | One-way to school |
| `family_support` | int | Self-rated 1–5 |
| `friend_support` | int | Self-rated 1–5 |
| `teacher_support` | int | Self-rated 1–5 |
| `self_rated_stress` | int | 1–5 (not used as feature) |
| `burnout_score` | int | 1–5 (not used as feature; basis of target) |
| `high_burnout` | int | **Target.** 1 if `burnout_score` ≥ 4 |

A few rows have missing values in `screen_time_hours`, `commute_minutes`, and `teacher_support` to reflect realistic survey data — the pipeline imputes them.

## Methodology highlights

- **Train/test split** is stratified on the target (80/20).
- **5-fold cross-validation** for model selection, then a single held-out evaluation.
- **Self-rated stress and burnout score are excluded as features** — including them would be predicting burnout from a near-synonym. Only upstream schedule/environment variables are used.
- **Missing values** are median-imputed inside the pipeline (no leakage).
- **Feature importance** is reported via both random forest impurity importance *and* standardized logistic regression coefficients (which give signs/direction).

## Limitations and honest caveats

- The data is synthetic. Real student survey data has messier relationships, more confounds, and harder-to-predict outliers. Expect real-world ROC-AUC to be substantially lower than 0.94.
- Burnout here is a binary derived from a 1–5 self-report — a coarse measurement. A continuous outcome (e.g. the Maslach Burnout Inventory) would be more sensitive.
- Cross-sectional design means we can describe *associations*, not *causes*. We can't say homework "causes" burnout, only that students with more homework hours report more burnout.
- Self-rated support is itself colored by mood — a burned-out student may rate their support lower than a third party would.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this analysis is useful in your own work or class project:

```
Student Burnout Prediction (2026). Synthetic adolescent schedule dataset and ML analysis.
```
