# Medical Appointment No-Shows Analysis with Privacy Protection

## 📋 Project Overview

This project analyzes medical appointment no-show patterns while implementing comprehensive **data privacy techniques**. We demonstrate how to extract valuable insights from sensitive healthcare data while protecting patient privacy through k-anonymity, l-diversity, t-closeness, and differential privacy.

## 🎯 Research Question

**Why do patients miss their scheduled medical appointments?**

Using machine learning on privacy-protected data, we identify key factors influencing appointment attendance and evaluate the utility-privacy tradeoff.

## 📊 Dataset

- **Source**: Medical Appointments No-Show Dataset (Vitória, Brazil)
- **Records**: 110,527 appointments (110,526 after cleaning)
- **Features**: 14 variables including patient demographics, medical conditions, and appointment details
- **Target**: Binary classification (Show/No-Show, ~20% no-show rate)

### Key Variables:
- **Direct Identifiers**: PatientId, AppointmentID (removed)
- **Quasi-Identifiers**: Age, Gender, Neighbourhood (protected via generalization)
- **Sensitive Attributes**: Medical conditions (Hipertension, Diabetes, Alcoholism, Handcap), Scholarship (Bolsa Família)
- **Temporal Features**: DaysBetween, LogDaysBetween, AppointmentDayOfWeek, ScheduledDayOfWeek

## 🔒 Privacy Protection Techniques

### 1. **K-Anonymity** (k ≥ 92)
- Generalized age into 5 clinical strata (0-14, 15-24, 25-44, 45-64, 65+)
- Mapped 81 neighbourhoods to 9 administrative regions (based on Wikipedia geographic data)
- Ensures each quasi-identifier combination has at least 92 individuals

### 2. **L-Diversity** (l = 2)
- Implemented via **value perturbation** (not suppression) - 100% data retention
- Ensures diversity in sensitive medical + socioeconomic attributes (Hipertension, Diabetes, Alcoholism, Handcap, Scholarship)
- **Note**: No-show (target variable) is NOT treated as sensitive for L-diversity to preserve target distribution for modeling
- 108/108 groups achieve l=2 diversity

### 3. **T-Closeness** (t ≤ 0.2) - Monitoring Only
- Maintains distribution similarity between groups and global population
- 108/108 groups comply with t-closeness threshold
- **Monitoring only for No-show**: We compute T-closeness on No-show rate to verify target distribution is preserved, but do NOT enforce it (target variable preserved for modeling)

### 4. **Differential Privacy** (ε ≈ 1.73 nats per attribute)
- Randomized response applied to 5 sensitive binary attributes:
  - Hipertension, Diabetes, Alcoholism, Handcap (medical PHI)
  - Scholarship (socioeconomic indicator)
- Privacy parameters: p = 0.3, q = 0.5
- Epsilon formula: ε = ln(max(a₀/b₀, a₁/b₁)) using likelihood ratios

### Privacy Technique Interaction
- **L-diversity and T-closeness are pre-processing checks**; after applying DP, classical k/l/t guarantees may not hold exactly
- **DP becomes our formal privacy guarantee** - provides mathematically rigorous ε-differential privacy bounds regardless of adversary's background knowledge
- **Reproducibility**: Global random seed (42) ensures deterministic perturbation and randomized response

## 🤖 Machine Learning Pipeline

### Key Design Decisions:

1. **Group-Based Train/Test Split**: Split by PatientId to prevent data leakage from repeated appointments
2. **One-Hot Encoding**: For Region (9 categories), Neighbourhood (81 categories), and weekdays - avoids fake ordinal relationships
3. **Calibrated Probabilities**: CalibratedClassifierCV with sigmoid calibration
4. **Threshold Optimization**: Maximize accuracy subject to recall ≥ 0.80

### Model Configurations:

| Model | Configuration |
|-------|---------------|
| **Logistic Regression** | max_iter=500, class_weight='balanced' |
| **Random Forest** | 400 trees, max_depth=16, class_weight='balanced' |
| **XGBoost** | 400 trees, max_depth=8, lr=0.05, scale_pos_weight=4 |

### Model Performance:

| Dataset | Model | Accuracy | Recall | ROC-AUC | Threshold |
|---------|-------|----------|--------|---------|-----------|
| Anonymized | Logistic Regression | 0.580 | 0.805 | 0.707 | 0.18 |
| Anonymized | Random Forest | 0.584 | 0.797 | 0.716 | 0.19 |
| **Anonymized** | **XGBoost** | **0.579** | **0.798** | **0.712** | **0.18** |
| Non-Anonymized | Logistic Regression | 0.583 | 0.810 | 0.719 | 0.18 |
| Non-Anonymized | Random Forest | 0.597 | 0.793 | 0.730 | 0.20 |
| Non-Anonymized | XGBoost | 0.597 | 0.794 | 0.729 | 0.19 |

**Privacy-Utility Tradeoff**: Only ~1-2% accuracy loss while ensuring strong privacy guarantees.

## 🔍 Key Findings

### Top Predictors of No-Shows (XGBoost Feature Importance):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **DaysBetween** | 0.090 |
| 2 | **Age_Group** | 0.036 |
| 3 | Region_Jardim da Penha | 0.036 |
| 4 | ApptDOW_0 (Monday) | 0.035 |
| 5 | Region_Santo Antônio | 0.033 |

### Insights:

1. **Lead Time is Critical**: Same-day appointments have only 4.6% no-show rate vs 28.5% for scheduled ahead
2. **Age Matters**: Younger patients (18-35) show higher no-show rates
3. **Geographic Variation**: Regional factors (distance, transportation, socioeconomic) significantly impact attendance
4. **Chronic Conditions**: Patients with chronic conditions are MORE likely to attend
5. **SMS Paradox**: SMS reminders show selection bias (sent to high-risk patients)

### Actionable Recommendations:

1. Reduce scheduling lead times where possible
2. Implement universal SMS reminders (not just high-risk)
3. Target interventions at young adult demographic
4. Address transportation barriers in underserved regions
5. Use prediction model to identify high-risk appointments proactively

## 📁 Repository Structure

```
.
├── Notebook_20251206_converted.py    # Main analysis script
├── Database.csv                       # Medical appointments dataset
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── output.txt                         # Execution log
├── output_png/                        # Generated visualizations (17 plots)
│   ├── 01_quick_look_eda.png
│   ├── 02_noshow_distribution.png
│   ├── 03_correlation_analysis.png
│   ├── ...
│   └── 15_final_feature_importance.png
└── old/                               # Reference notebooks
    ├── Notebook 20251206.ipynb
    ├── Reference1.ipynb
    ├── Reference2.ipynb
    └── Reference3.ipynb
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yuntongSH/Medical-Appointment-No-Shows.git
cd Medical-Appointment-No-Shows
```

2. Create virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python Notebook_20251206_converted.py
```

Output will be saved to `output.txt` and visualizations to `output_png/`.

### Required Libraries

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## 💡 Technical Highlights

### Privacy Implementation:
- **Age Generalization**: 5 clinical strata (0-14, 15-24, 25-44, 45-64, 65+) based on healthcare age groupings
- **Regional Mapping**: Based on official Vitória administrative regions (Wikipedia source)
- **Randomized Response**: Correct epsilon calculation using likelihood ratios (nats, not bits)
- **L-Diversity via Perturbation**: No data suppression - 100% record retention
- **Reproducibility**: Global random seed (RANDOM_SEED = 42) for deterministic results

### Model Design:
- **Group-based splitting**: Prevents same patient appearing in train and test sets
- **One-hot encoding**: For categorical features without ordinal meaning
- **Calibration plots**: Reliability diagrams with Brier scores
- **Threshold tuning**: Maximize accuracy subject to minimum recall constraint

### Features Engineered:
- `DaysBetween`: Days from scheduling to appointment (date-normalized)
- `LogDaysBetween`: log1p transform to reduce skewness of extreme wait times
- `HealthBurden`: Sum of medical conditions (0-4)

### Features Removed (Redundant):
- ~~`IsSameDay`~~: 100% correlated with DaysBetween==0
- ~~`LeadTime_Category`~~: Redundant categorical binning of DaysBetween
- ~~`IsWeekend`~~: Only 39 weekend appointments in 110k records

## 📈 Visualizations

The analysis generates 17 publication-ready plots:
- EDA and distribution analysis
- Correlation heatmaps
- Regional mapping and characteristics
- L-diversity before/after comparison
- Differential privacy parameter sweep
- Model comparison (Accuracy, Recall, ROC-AUC)
- Calibration curves
- Feature importance comparison
- Privacy-utility tradeoff visualization

## 🎓 Educational Value

This project demonstrates:
1. **Privacy-Preserving Data Analysis**: Real-world application of k-anonymity, l-diversity, t-closeness, and differential privacy
2. **Healthcare Analytics**: Domain-specific insights from medical appointment data
3. **Machine Learning Best Practices**: Group-based splitting, calibration, threshold optimization
4. **Interpretable AI**: Feature importance analysis for explainable predictions
5. **Utility-Privacy Tradeoff**: Quantifying the cost of privacy protection (~1-2% accuracy loss)

## 📝 Privacy Guarantees Summary

| Technique | Parameter | Achieved | Notes |
|-----------|-----------|----------|-------|
| K-Anonymity | k | ≥ 92 | After age/region generalization |
| L-Diversity | l | 2 (108/108 groups) | Medical + socioeconomic attributes only |
| T-Closeness | t | ≤ 0.2 (108/108 groups) | Monitoring only for No-show |
| Differential Privacy | ε | ~1.73 nats per attribute | Formal guarantee after DP |

## 👥 Authors

- **Project Team**: Data Privacy in AI Course
- **Institution**: Université Paris Dauphine-PSL
- **Year**: 2025

## 📚 References

1. Sweeney, L. (2002). k-Anonymity: A Model for Protecting Privacy
2. Machanavajjhala, A., et al. (2007). l-Diversity: Privacy Beyond k-Anonymity
3. Li, N., et al. (2007). t-Closeness: Privacy Beyond k-Anonymity and l-Diversity
4. Dwork, C. (2006). Differential Privacy
5. Warner, S.L. (1965). Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias

## 📞 Contact

- Repository: https://github.com/yuntongSH/Medical-Appointment-No-Shows
- Issues: Use GitHub issue tracker

---

**Note**: This project balances academic research with practical privacy protection. All privacy techniques are implemented from scratch for educational transparency, with correct mathematical formulations (e.g., epsilon in natural log units).
