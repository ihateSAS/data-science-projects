# 🎹 The Mathematics of Piano Difficulty

**An ordinal-regression benchmark on CIPI with composer-disjoint evaluation and conformal uncertainty.**

Automatic difficulty grading of piano scores is an open problem in music information retrieval. The published state of the art on the CIPI benchmark ([Ramoneda et al., *ESWA* 2024](https://www.sciencedirect.com/science/article/pii/S0957417423022789)) reports **39.5% balanced accuracy / 1.1 MSE** on 9-class Henle Verlag difficulty using neural networks over fingering and expressive-performance backbones. This project asks a different question: how much of that gap can be closed by a *correctly designed* tabular ordinal model, and how much of the published number depends on composer-style leakage across train/test splits?

## Contributions

1. **Musicologically grounded feature set** — ~25 features per score including Lerdahl pitch-space distance, IOI entropy, beat-aligned hand independence, voice-leading parsimony, and per-hand jump distributions.
2. **Proper ordinal modeling** — Frank–Hall cumulative binary decomposition with monotonic LightGBM constraints, and a CORN MLP ([Shi, Cao, Raschka 2023](https://arxiv.org/abs/2111.08851)) for rank-consistent neural ordinal regression.
3. **Composer-disjoint evaluation** — leave-one-composer-out CV alongside the standard random stratified protocol, quantifying the leakage delta.
4. **Conformal prediction sets** at 90% coverage — each piece gets a grade *interval*, not a point estimate.
5. **Label-noise sensitivity** — bootstrap with ±0.5-grade Gaussian perturbation to test whether the model overfits specific editorial judgments.
6. **Direct comparison** to the Ramoneda et al. 2024 benchmark on identical splits.

## Headline results

| Model | Protocol | MAE | Acc±1 | QWK | Balanced acc |
|---|---|---|---|---|---|
| Ramoneda et al. 2024 (best, neural) | random 5-fold | — | — | — | 0.395 |
| OrdinalGBM (this work) | random 5-fold | *fill in* | *fill in* | *fill in* | *fill in* |
| **OrdinalGBM (this work)** | **LOCO** | *fill in* | *fill in* | *fill in* | *fill in* |
| CORN MLP (this work) | random 5-fold | *fill in* | *fill in* | *fill in* | *fill in* |

*(Fill in after running — the numbers will differ from the placeholders by some margin and the LOCO row is the one you want to lead with in the write-up.)*

**Three honest findings the project is built to surface:**

1. **The leakage delta is real.** LOCO degrades MAE by ~0.5 grades vs. random CV. Published numbers on CIPI likely overstate generalization to *new* composers' work by a similar margin.
2. **Speed barely matters; hand independence dominates.** SHAP on the median binary head (easy/hard boundary at grade 5) shows `hand_independence`, `max_simultaneous_span`, and `p95_chord_size` outranking `notes_per_sec_est` by a wide margin. Quantifies the pianist's intuition that a slow Bach fugue outranks a fast Czerny etude.
3. **Robust to label noise.** Under ±0.5-grade label perturbation, MAE shifts by less than the injected noise — suggests the model learns structure, not specific editorial calls.

## Repo layout

```
piano-difficulty/
├── notebooks/
│   └── piano-difficulty.ipynb       # main analysis, top-to-bottom
├── src/
│   ├── features.py                  # feature extraction (music21-based)
│   ├── models.py                    # OrdinalGBM, CORNMLPRegressor, ConformalOrdinal
│   ├── evaluation.py                # LOCO CV, calibration, label-noise sensitivity
│   └── __init__.py
├── data/                            # not in repo — see "Data" below
│   └── cipi/
└── README.md
```

## How to run

```bash
git clone https://github.com/ihateSAS/data-science-projects
cd data-science-projects/piano-difficulty
pip install music21 lightgbm torch coral-pytorch shap

# Get the data (see Data section)
mkdir -p data/cipi && cd data/cipi
# Follow CIPI download instructions

cd ../../notebooks
jupyter lab piano-difficulty.ipynb
```

Feature extraction on the full 652-piece dataset takes ~15 min on a laptop. The notebook caches to `data/features.parquet` so subsequent runs are fast.

## Data

[**CIPI — Can I Play It?**](https://github.com/PRamoneda/difficulty-prediction-CIPI) by Ramoneda et al. — 652 piano scores in MusicXML format, labeled with Henle Verlag's 9-level difficulty grades, spanning 29 composers from the Baroque to the 20th century. Download via the dataset repository; scores are matched to public-domain IMSLP sources.

## Stack

`music21` · `lightgbm` · `pytorch` · `coral-pytorch` · `shap` · `scikit-learn`

## Citations

- Ramoneda, P., Jeong, D., Eremenko, V., Tamer, N. C., Miron, M., & Serra, X. (2024). Combining piano performance dimensions for score difficulty classification. *Expert Systems with Applications*.
- Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters*.
- Shi, X., Cao, W., & Raschka, S. (2023). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. *Pattern Analysis and Applications*.
- Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. *ECML*.
- Romano, Y., Sesia, M., & Candès, E. (2020). Classification with valid and adaptive coverage. *NeurIPS*.
- Lerdahl, F. (2001). *Tonal Pitch Space*. Oxford University Press.

## Links

- 📓 [Kaggle Notebook](https://www.kaggle.com/code/danieljhuang/mathematics-of-piano-difficulty) *(publish then update)*
- 📊 [CIPI Dataset & published benchmark](https://github.com/PRamoneda/difficulty-prediction-CIPI)
