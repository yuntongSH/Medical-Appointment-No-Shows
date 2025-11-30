# Medical Appointment No-Shows Analysis with Privacy Protection

## 📋 Project Overview

This project analyzes medical appointment no-show patterns while implementing comprehensive **data privacy techniques**. We demonstrate how to extract valuable insights from sensitive healthcare data while protecting patient privacy through k-anonymity, l-diversity, t-closeness, and differential privacy.

## 🎯 Research Question

**Why do patients miss their scheduled medical appointments?**

Using machine learning on privacy-protected data, we identify key factors influencing appointment attendance and evaluate the utility-privacy tradeoff.

## 📊 Dataset

- **Source**: Medical Appointments No-Show Dataset
- **Records**: 110,527 appointments
- **Features**: 14 variables including patient demographics, medical conditions, and appointment details
- **Target**: Binary classification (Show/No-Show)

### Key Variables:
- **Direct Identifiers**: PatientId, AppointmentID (removed)
- **Quasi-Identifiers**: Age, Gender, Neighbourhood (protected)
- **Sensitive Attributes**: Medical conditions (Hipertension, Diabetes, Alcoholism, Handcap)
- **Behavioral Features**: SMS reminders, scholarship status
- **Temporal Features**: Days between scheduling and appointment, day of week

## 🔒 Privacy Protection Techniques

### 1. **K-Anonymity** (k ≥ 31)
- Generalized age into 6 age groups
- Mapped 81 neighbourhoods to 9 administrative regions
- Ensures each quasi-identifier group has at least 31 individuals

### 2. **L-Diversity** (l = 2)
- Ensures diversity in sensitive attributes (No-show status)
- Each k-anonymous group has at least 2 different No-show values

### 3. **T-Closeness** (t ≤ 0.2)
- Maintains distribution similarity between groups and global population
- No-show rate in each group stays within 20% of global rate

### 4. **Differential Privacy** (ε ≈ 2.4)
- Randomized response applied to medical conditions
- Probability parameters: p = 0.3, q = 0.5
- Adds controlled noise while preserving statistical utility

## 🤖 Machine Learning Models

### Reinforced Model Configurations:

#### **Random Forest**
- 500 trees (n_estimators)
- Max depth: 20
- Bootstrap sampling with OOB scoring
- Balanced class weights
- Optimized split parameters

#### **XGBoost**
- 500 trees (n_estimators)
- Max depth: 10
- Learning rate: 0.03
- Advanced regularization (L1=0.1, L2=1.0, gamma=0.1)
- Multi-level sampling (85%)

### Model Performance:

| Dataset | Model | Accuracy | Recall | ROC-AUC |
|---------|-------|----------|--------|---------|
| Anonymized | Random Forest | ~0.64 | ~0.68 | ~0.72 |
| Anonymized | XGBoost | ~0.62 | ~0.74 | ~0.72 |
| Non-Anonymized | Random Forest | ~0.65 | ~0.72 | ~0.74 |
| Non-Anonymized | XGBoost | ~0.64 | ~0.74 | ~0.74 |

**Privacy-Utility Tradeoff**: Minimal accuracy loss (~1-2%) while ensuring strong privacy guarantees.

## 🔍 Key Findings

### Top Factors for Appointment No-Shows:

1. **DaysBetween** (Most Important)
   - Time gap between scheduling and appointment
   - Longer waits → higher no-show probability

2. **Age**
   - Different age groups show varying attendance patterns
   - Younger patients more likely to miss appointments

3. **SMS Reminders**
   - Critical intervention for reducing no-shows
   - Significant difference in attendance with/without SMS

4. **Geographic Location (Region)**
   - Distance and accessibility impact attendance
   - Regional variations in no-show rates

5. **Socioeconomic Factors (Scholarship)**
   - Social welfare participation correlates with attendance patterns

### SHAP Analysis:
- Provides interpretable AI insights
- Accounts for feature interactions
- Validates feature importance rankings

## 📁 Repository Structure

```
.
├── Medical_Appointments_Analysis.ipynb  # Main analysis notebook
├── Database.csv                          # Medical appointments dataset
├── README.md                             # This file
├── Reference1.ipynb                      # Reference implementation 1
├── Reference2.ipynb                      # Reference implementation 2
└── Reference3.ipynb                      # Reference implementation 3
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

### Running the Analysis

1. Clone the repository:
```bash
git clone https://github.com/yuntongSH/Medical-Appointment-No-Shows.git
cd Medical-Appointment-No-Shows
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # or install manually
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Medical_Appointments_Analysis.ipynb
```

4. Execute cells sequentially (full execution takes ~10-15 minutes with reinforced models)

## 💡 Technical Highlights

### Privacy Implementation:
- **Age Generalization**: 6 bins (0-17, 18-29, 30-44, 45-59, 60-74, 75+)
- **Regional Mapping**: 9 administrative regions from 81 neighbourhoods
- **Randomized Response**: Medical conditions protected with ε-differential privacy
- **Temporal Generalization**: Day-level precision (exact timestamps removed)

### Model Interpretability:
- Feature importance analysis (default and SHAP-based)
- Confusion matrices and classification reports
- ROC-AUC curves for performance evaluation
- Comprehensive visualizations

### Utility-Privacy Balance:
- Privacy guarantees: k≥31, l=2, t≤0.2, ε≈2.4
- Minimal accuracy loss: <2% compared to non-anonymized
- Maintains high recall for no-show detection

## 📈 Results Visualization

The notebook includes:
- Age distribution before/after generalization
- Regional no-show rate comparisons
- L-diversity analysis (4-subplot visualization)
- T-closeness distribution
- Feature importance comparisons (4 models)
- SHAP summary plots and beeswarm charts
- Performance comparison visualizations

## 🎓 Educational Value

This project demonstrates:
1. **Privacy-Preserving Data Analysis**: Real-world application of privacy techniques
2. **Healthcare Analytics**: Domain-specific insights from medical data
3. **Machine Learning**: Binary classification with imbalanced data
4. **Interpretable AI**: SHAP values for explainable predictions
5. **Utility-Privacy Tradeoff**: Quantifying the cost of privacy protection

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional privacy techniques (k-map, mondrian partitioning)
- Deep learning models with privacy
- Real-time prediction system
- Extended feature engineering

## 📝 License

This project is for educational purposes. Dataset source and usage comply with data sharing agreements.

## 👥 Authors

- **Project Team**: Data Privacy in AI Course
- **Institution**: Université Paris Dauphine-PSL
- **Year**: 2025

## 📚 References

1. k-Anonymity: Protecting Privacy by Generalizing Data
2. l-Diversity: Privacy Beyond k-Anonymity
3. t-Closeness: Privacy Beyond l-Diversity
4. Differential Privacy: A Survey of Results
5. SHAP: A Unified Approach to Interpreting Model Predictions

## 📞 Contact

For questions or collaboration:
- Repository: https://github.com/yuntongSH/Medical-Appointment-No-Shows
- Issues: Use GitHub issue tracker

---

**Note**: This project balances academic research with practical privacy protection. All techniques are implemented from scratch for educational transparency.
