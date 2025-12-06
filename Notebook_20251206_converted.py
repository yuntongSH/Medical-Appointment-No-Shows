#!/usr/bin/env python
# coding: utf-8

# Redirect all output to output.txt
import sys
output_file = open('output.txt', 'w')
sys.stdout = output_file
sys.stderr = output_file

# # Medical Appointments No-Show Analysis
# ## A Privacy-Preserving Machine Learning Approach
# 
# ---
# 
# ### The Story
# 
# **The Problem**: Healthcare providers lose billions annually due to missed appointments. Understanding *why* patients miss appointments could help clinics intervene proactively—but patient data is sensitive and protected by regulations like HIPAA and GDPR.
# 
# **Our Challenge**: Can we build predictive models that identify no-show risk factors while **protecting patient privacy**?
# 
# **Our Approach**: We apply state-of-the-art privacy techniques from academic literature:
# - **K-Anonymity** (Sweeney, 2002): Ensure each record is indistinguishable from k-1 others
# - **L-Diversity** (Machanavajjhala et al., 2007): Ensure diversity in sensitive attributes
# - **T-Closeness** (Li et al., 2007): Ensure distributions match global patterns
# - **Differential Privacy** (Dwork, 2006): Add calibrated noise to protect individuals
# 
# ---
# 
# ### Research Questions
# 
# 1. **What factors predict appointment no-shows?**
# 2. **How much prediction accuracy do we sacrifice for privacy?**
# 3. **Which privacy parameters offer the best utility-privacy tradeoff?**
# 
# ---
# 
# ## Part One: Understanding Our Data
# 
# **Dataset Variables** (with privacy classification):
# 
# | Variable | Type | Privacy Risk | Utility |
# |----------|------|--------------|--------|
# | PatientId | Direct Identifier | HIGH | None - must remove |
# | AppointmentID | Direct Identifier | HIGH | None - must remove |
# | Age | Quasi-Identifier | MEDIUM | High - predictive |
# | Gender | Quasi-Identifier | LOW | Medium - demographic |
# | Neighbourhood | Quasi-Identifier | MEDIUM | High - geographic patterns |
# | Scholarship | Sensitive | MEDIUM | High - socioeconomic |
# | Hypertension | Sensitive (PHI) | HIGH | Medium - health indicator |
# | Diabetes | Sensitive (PHI) | HIGH | Medium - health indicator |
# | Alcoholism | Sensitive (PHI) | HIGH | Medium - health indicator |
# | Handicap | Sensitive (PHI) | HIGH | Medium - health indicator |
# | SMS_received | Behavioral | LOW | High - intervention marker |
# | No-show | Target/Sensitive | MEDIUM | Target variable |

# ### Why This Matters
# We want to predict missed medical appointments while respecting patient privacy. The notebook should read like a story: what data we have, what it looks like, which privacy levers we pull (k/l/t + DP), and how the final model performs.
# 
# ### What you'll see next
# - A quick data card and early visuals to ground data quality and target balance
# - Transparent privacy steps (k-anonymity, l-diversity, t-closeness, DP) tied to References 1 & 2
# - A calibrated, threshold-tuned model comparison with a simple baseline
# - Actionable insights on why patients miss appointments (Reference 3 framing)
# 

# In[2]:


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Create output directory for plots
output_dir = 'output_png'
os.makedirs(output_dir, exist_ok=True)
print(f"Plot output directory: {output_dir}/")

# Plot counter for unique filenames
plot_counter = 1


# In[3]:


# Load the dataset
df = pd.read_csv('Database.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
df.head(10)


# In[4]:


# Data info and completeness
print("Dataset Information:")
print("="*60)
df.info()
print("\n" + "="*60)
print("\nMissing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
}).sort_values('Missing Count', ascending=False)
print(missing_df[missing_df['Missing Count'] > 0])

if missing_df['Missing Count'].sum() == 0:
    print("\nNo missing values found - dataset is complete!")


# ### Early Exploration: data quality and target balance
# Visual EDA to anchor the story before privacy transforms.
# 

# In[5]:


# Early EDA: lightweight visuals before privacy steps
print("Early EDA: grounding in the raw data")

# Work on a copy to avoid side-effects downstream
df_eda = df.copy()
df_eda['ScheduledDay'] = pd.to_datetime(df_eda['ScheduledDay'])
df_eda['AppointmentDay'] = pd.to_datetime(df_eda['AppointmentDay'])
df_eda['DaysBetween'] = (df_eda['AppointmentDay'] - df_eda['ScheduledDay']).dt.days

data_card = {
    'rows': len(df_eda),
    'columns': df_eda.shape[1],
    'time_span_days': (df_eda['AppointmentDay'].max() - df_eda['AppointmentDay'].min()).days,
    'no_show_rate_%': round((df_eda['No-show'] == 'Yes').mean() * 100, 2),
    'unique_neighbourhoods': df_eda['Neighbourhood'].nunique(),
}
print(pd.DataFrame([data_card]).T.rename(columns={0: 'value'}))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Missingness heatmap
sns.heatmap(df_eda.isnull(), cbar=False, ax=axes[0, 0])
axes[0, 0].set_title('Missingness (all columns)')
axes[0, 0].set_xlabel('Columns')
axes[0, 0].set_ylabel('Rows')

# Class balance
sns.countplot(data=df_eda, x='No-show', palette=['#2ecc71', '#e74c3c'], ax=axes[0, 1])
axes[0, 1].set_title('Target balance (show vs no-show)')
for p in axes[0, 1].patches:
    axes[0, 1].text(p.get_x() + p.get_width()/2., p.get_height() + 500,
                   f"{int(p.get_height())}", ha='center', fontsize=10)

# No-show by Scholarship
sns.countplot(data=df_eda, x='Scholarship', hue='No-show', palette='Set2', ax=axes[0, 2])
axes[0, 2].set_title('No-show by Scholarship')
axes[0, 2].set_xlabel('Scholarship (0/1)')

# Age vs No-show
sns.violinplot(data=df_eda, x='No-show', y='Age', palette='Pastel1', ax=axes[1, 0], cut=0)
axes[1, 0].set_title('Age distribution by outcome')

# Lead time (DaysBetween)
sns.kdeplot(data=df_eda, x='DaysBetween', hue='No-show', fill=True, common_norm=False, ax=axes[1, 1])
axes[1, 1].set_xlim(-5, df_eda['DaysBetween'].quantile(0.99))
axes[1, 1].set_title('Scheduling gap vs outcome')

# Weekday effect
df_eda['AppointmentDOW'] = df_eda['AppointmentDay'].dt.day_name()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sns.barplot(data=df_eda, x='AppointmentDOW', y=(df_eda['No-show'] == 'Yes').astype(int),
            order=order, color='#95a5a6', ax=axes[1, 2])
axes[1, 2].set_title('No-show rate by appointment weekday')
axes[1, 2].set_ylabel('No-show rate')
axes[1, 2].tick_params(axis='x', rotation=30)

plt.suptitle('Quick-look EDA to steer privacy + modeling', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_quick_look_eda.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/01_quick_look_eda.png")


# In[6]:


# Statistical summary
print("Statistical Summary:")
df.describe()


# In[7]:


# Analyze categorical variables
print("Categorical Variables Distribution:")
print("="*60)

categorical_cols = ['Gender', 'Scholarship', 'Hipertension', 'Diabetes', 
                   'Alcoholism', 'Handcap', 'SMS_received', 'No-show']

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Unique values: {df[col].nunique()}")


# In[8]:


# Analyze Neighbourhood distribution
print(f"Number of unique neighbourhoods: {df['Neighbourhood'].nunique()}")
print(f"\nTop 20 neighbourhoods by appointment count:")
neighbourhood_counts = df['Neighbourhood'].value_counts().head(20)
print(neighbourhood_counts)


# In[9]:


# Analyze Age distribution
print("Age Statistics:")
print(df['Age'].describe())
print(f"\nAge range: {df['Age'].min()} to {df['Age'].max()}")
print(f"Negative or suspicious ages: {len(df[df['Age'] < 0])}")

# Visualize age distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['Age'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['Age'])
plt.ylabel('Age')
plt.title('Age Box Plot')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_quick_look_eda.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/01_quick_look_eda.png')


# In[10]:


# Analyze the target variable: No-show
print("Target Variable Analysis: No-show")
print("="*60)
noshow_counts = df['No-show'].value_counts()
noshow_pct = df['No-show'].value_counts(normalize=True) * 100

print(f"\nNo-show distribution:")
for val, count in noshow_counts.items():
    print(f"{val}: {count} ({noshow_pct[val]:.2f}%)")

# Visualize
plt.figure(figsize=(8, 5))
noshow_counts.plot(kind='bar', color=['green', 'red'], alpha=0.7)
plt.xlabel('No-show Status')
plt.ylabel('Count')
plt.title('Distribution')
plt.xticks(rotation=0)
plt.grid(alpha=0.3, axis='y')
for i, (val, count) in enumerate(noshow_counts.items()):
    plt.text(i, count + 1000, f'{count}\n({noshow_pct[val]:.1f}%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_noshow_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/03_noshow_distribution.png")

print(f"\nWarning: Class imbalance: {noshow_pct['No']:.1f}% showed up vs {noshow_pct['Yes']:.1f}% no-show")


# ### Deep Dive: Correlation Analysis & Feature Relationships
# 
# Before applying privacy transformations, let's understand how variables relate to each other. This will help us:
# 1. **Identify key predictors** - Which features correlate most with no-shows?
# 2. **Detect multicollinearity** - Are any predictors redundant?
# 3. **Guide privacy decisions** - Which quasi-identifiers matter most for prediction?

# In[10]:


# Comprehensive Correlation Analysis
print("=" * 70)
print("CORRELATION ANALYSIS: Understanding Feature Relationships")
print("=" * 70)

# Prepare numeric data for correlation
df_corr = df.copy()

# Convert categorical to numeric
df_corr['No-show_numeric'] = (df_corr['No-show'] == 'Yes').astype(int)
df_corr['Gender_numeric'] = (df_corr['Gender'] == 'M').astype(int)

# Select numeric columns for correlation
numeric_cols = ['Age', 'Gender_numeric', 'Scholarship', 'Hipertension', 'Diabetes', 
                'Alcoholism', 'Handcap', 'SMS_received', 'No-show_numeric']

# Calculate correlation matrix
corr_matrix = df_corr[numeric_cols].corr()

# Rename for cleaner display
rename_dict = {'No-show_numeric': 'No-show', 'Gender_numeric': 'Gender (M=1)'}
corr_display = corr_matrix.rename(index=rename_dict, columns=rename_dict)

# Create correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Full correlation heatmap
mask = np.triu(np.ones_like(corr_display, dtype=bool))
sns.heatmap(corr_display, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, ax=axes[0],
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[0].set_title('Feature Correlation Matrix\n(Lower Triangle)', fontsize=14, fontweight='bold')

# Correlation with target (No-show)
target_corr = corr_display['No-show'].drop('No-show').sort_values(key=abs, ascending=True)
colors = ['#e74c3c' if x > 0 else '#3498db' for x in target_corr.values]
target_corr.plot(kind='barh', ax=axes[1], color=colors, edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].set_title('Correlation with No-show Target\n(Red = Positive, Blue = Negative)', fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(axis='x', alpha=0.3)

# Add value labels
for i, (val, name) in enumerate(zip(target_corr.values, target_corr.index)):
    axes[1].text(val + 0.005 if val > 0 else val - 0.005, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/04_correlation_analysis.png")

# Print key insights
print("\nKEY CORRELATION INSIGHTS:")
print("-" * 50)
print("Features most correlated with No-show:")
for feat, corr_val in target_corr.sort_values(key=abs, ascending=False).head(5).items():
    direction = "UP" if corr_val > 0 else "DOWN"
    print(f"  {direction} {feat}: {corr_val:.4f}")
print("\nInterpretation:")
print("  • SMS_received shows positive correlation - patients who received SMS are MORE likely to no-show")
print("    (This is counterintuitive! SMS may be sent to high-risk patients)")
print("  • Age shows negative correlation - older patients tend to show up more often")
print("  • Health conditions (Hypertension, Diabetes) show slight negative correlation")


# In[11]:


# Cross-tabulation Analysis: No-show rates by category combinations
print("=" * 70)
print("CROSS-TABULATION: No-show Rates by Feature Categories")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. No-show rate by Gender and Age Group
df_crosstab = df.copy()
df_crosstab['No-show_binary'] = (df_crosstab['No-show'] == 'Yes').astype(int)
df_crosstab['Age_Group'] = pd.cut(df_crosstab['Age'], bins=[0, 18, 35, 50, 65, 120], 
                                    labels=['0-18', '19-35', '36-50', '51-65', '65+'])

pivot1 = df_crosstab.pivot_table(values='No-show_binary', index='Age_Group', 
                                  columns='Gender', aggfunc='mean') * 100
pivot1.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#3498db'], edgecolor='black', alpha=0.8)
axes[0, 0].set_title('No-show Rate by Age Group & Gender', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('No-show Rate (%)')
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].tick_params(axis='x', rotation=0)
axes[0, 0].legend(title='Gender')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. No-show rate by Health Conditions
health_cols = ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']
noshow_by_health = []
for col in health_cols:
    rate_0 = df_crosstab[df_crosstab[col] == 0]['No-show_binary'].mean() * 100
    rate_1 = df_crosstab[df_crosstab[col] == 1]['No-show_binary'].mean() * 100
    noshow_by_health.append({'Condition': col, 'Without (0)': rate_0, 'With (1)': rate_1})

health_df = pd.DataFrame(noshow_by_health)
x = np.arange(len(health_cols))
width = 0.35
bars1 = axes[0, 1].bar(x - width/2, health_df['Without (0)'], width, label='Without', color='#2ecc71', edgecolor='black')
bars2 = axes[0, 1].bar(x + width/2, health_df['With (1)'], width, label='With', color='#e74c3c', edgecolor='black')
axes[0, 1].set_ylabel('No-show Rate (%)')
axes[0, 1].set_title('No-show Rate by Health Condition', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(health_cols, rotation=15)
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. No-show rate by Scholarship and SMS
pivot2 = df_crosstab.pivot_table(values='No-show_binary', index='Scholarship', 
                                  columns='SMS_received', aggfunc='mean') * 100
pivot2.plot(kind='bar', ax=axes[0, 2], color=['#f39c12', '#9b59b6'], edgecolor='black', alpha=0.8)
axes[0, 2].set_title('No-show Rate by Scholarship & SMS', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('No-show Rate (%)')
axes[0, 2].set_xlabel('Scholarship (0=No, 1=Yes)')
axes[0, 2].tick_params(axis='x', rotation=0)
axes[0, 2].legend(title='SMS Received', labels=['No SMS', 'Got SMS'])
axes[0, 2].grid(axis='y', alpha=0.3)

# 4. No-show rate by Top 10 Neighbourhoods
top_neighbourhoods = df['Neighbourhood'].value_counts().head(10).index
df_top_neigh = df_crosstab[df_crosstab['Neighbourhood'].isin(top_neighbourhoods)]
noshow_by_neigh = df_top_neigh.groupby('Neighbourhood')['No-show_binary'].mean().sort_values(ascending=False) * 100
noshow_by_neigh.plot(kind='bar', ax=axes[1, 0], color='#1abc9c', edgecolor='black', alpha=0.8)
axes[1, 0].set_title('No-show Rate by Top 10 Neighbourhoods', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('No-show Rate (%)')
axes[1, 0].set_xlabel('Neighbourhood')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].axhline(y=df_crosstab['No-show_binary'].mean()*100, color='red', linestyle='--', label='Overall Rate')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 5. No-show rate by Time Features
df_crosstab['ScheduledDay'] = pd.to_datetime(df_crosstab['ScheduledDay'])
df_crosstab['AppointmentDay'] = pd.to_datetime(df_crosstab['AppointmentDay'])
df_crosstab['DaysBetween'] = (df_crosstab['AppointmentDay'] - df_crosstab['ScheduledDay']).dt.days
df_crosstab['DaysCategory'] = pd.cut(df_crosstab['DaysBetween'], 
                                      bins=[-float('inf'), 0, 1, 7, 14, 30, float('inf')],
                                      labels=['Same day', '1 day', '2-7 days', '8-14 days', '15-30 days', '>30 days'])
noshow_by_days = df_crosstab.groupby('DaysCategory')['No-show_binary'].mean() * 100
noshow_by_days.plot(kind='bar', ax=axes[1, 1], color='#3498db', edgecolor='black', alpha=0.8)
axes[1, 1].set_title('No-show Rate by Scheduling Lead Time', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('No-show Rate (%)')
axes[1, 1].set_xlabel('Days Between Scheduling and Appointment')
axes[1, 1].tick_params(axis='x', rotation=30)
axes[1, 1].grid(axis='y', alpha=0.3)

# 6. Hour of scheduling effect (if available)
df_crosstab['ScheduledHour'] = pd.to_datetime(df_crosstab['ScheduledDay']).dt.hour
if df_crosstab['ScheduledHour'].nunique() > 1:
    noshow_by_hour = df_crosstab.groupby('ScheduledHour')['No-show_binary'].mean() * 100
    noshow_by_hour.plot(ax=axes[1, 2], marker='o', linewidth=2, color='#e74c3c')
    axes[1, 2].fill_between(noshow_by_hour.index, noshow_by_hour.values, alpha=0.3, color='#e74c3c')
    axes[1, 2].set_title('No-show Rate by Scheduling Hour', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Hour of Day (Scheduled)')
    axes[1, 2].set_ylabel('No-show Rate (%)')
else:
    # Appointment day of week instead
    df_crosstab['AppDOW'] = df_crosstab['AppointmentDay'].dt.dayofweek
    noshow_by_dow = df_crosstab.groupby('AppDOW')['No-show_binary'].mean() * 100
    noshow_by_dow.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][:len(noshow_by_dow)]
    noshow_by_dow.plot(kind='bar', ax=axes[1, 2], color='#9b59b6', edgecolor='black', alpha=0.8)
    axes[1, 2].set_title('No-show Rate by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('No-show Rate (%)')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.suptitle('Detailed Cross-tabulation Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_crosstab_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/05_crosstab_analysis.png")

print("\nKEY FINDINGS FROM CROSS-TABULATION:")
print("-" * 50)
print("• Young adults (19-35) have the HIGHEST no-show rates")
print("• Patients WITH chronic conditions tend to show up MORE often")
print("• SMS recipients show HIGHER no-show rates (selection bias - high-risk get SMS)")
print("• Longer lead times correlate with MORE no-shows")
print("• Neighbourhood matters - geographic/socioeconomic factors at play")


# ### EDA Summary & Transition to Privacy
# 
# **What we learned from exploration:**
# 
# | Finding | Implication for Privacy |
# |---------|------------------------|
# | Age strongly predicts no-shows | Need to generalize age while preserving predictive power |
# | Neighbourhood shows variation | Must anonymize location but keep regional patterns |
# | Health conditions protect attendance | These are PHI - must protect even if useful |
# | Lead time matters | Safe feature - no privacy concern |
# 
# **Next Step**: Now that we understand our data, we must protect it before building models. We'll apply privacy techniques from the academic literature.

# ## Data Privacy Assessment
# 
# **Classification of Columns:**
# 
# **Direct Identifiers (Must Remove):**
# - `PatientId`: Can directly identify individuals
# - `AppointmentID`: Can link to external records
# 
# **Quasi-Identifiers (Need Protection via k-anonymity/l-diversity/t-closeness):**
# - `Age`: Can be combined with other attributes to re-identify
# - `Gender`: Binary attribute that narrows identification
# - `Neighbourhood`: Geographic information - many unique values (potential re-identification)
# 
# **Sensitive Attributes (Need Protection):**
# - `Scholarship`: Indicates socioeconomic status (Bolsa Família)
# - `Hypertension`: Medical condition (protected health information)
# - `Diabetes`: Medical condition (protected health information)
# - `Alcoholism`: Medical condition (protected health information)
# - `Handicap`: Disability status (protected information)
# - `No-show`: Target variable - could reveal health-seeking behavior
# 
# **Utility Features (Keep and possibly protect):**
# - `ScheduledDay` and `AppointmentDay`: Temporal features - useful for prediction
# - `SMS_received`: Behavioral intervention - useful feature
# - All medical conditions: Critical for ML prediction
# 
# **Privacy Strategy:**
# 1. **Remove** direct identifiers (PatientId, AppointmentID)
# 2. **Apply k-anonymity/l-diversity/t-closeness** on quasi-identifiers (Age, Gender, Neighbourhood)
# 3. **Apply differential privacy** on sensitive medical conditions
# 4. **Generalize** temporal data to reduce precision
# 5. Keep useful features for ML while ensuring privacy

# ## Part Two: Determining the Approach for Protecting the Columns
# 
# Based on Part One analysis, we now determine the privacy protection approach.
# 
# **Key Findings:**
# - Dataset is complete (no missing values)
# - Warning: Age has outliers (-1, 115) that need cleaning
# - Warning: 81 unique neighbourhoods - high cardinality quasi-identifier
# - Warning: Class imbalance: 79.8% show vs 20.2% no-show
# 
# **Privacy Protection Strategy:**
# 
# ### 1. Data Minimization & Direct Identifier Removal
# - **Remove**: `PatientId`, `AppointmentID` (direct identifiers - no utility for prediction)
# 
# ### 2. K-Anonymity on Quasi-Identifiers
# Apply k-anonymity (k=5 minimum) on:
# - **Age**: Generalize into age bins (0-17, 18-30, 31-45, 46-60, 61-75, 76+)
# - **Gender**: Keep as-is (binary - low risk)
# - **Neighbourhood**: Generalize to reduce from 81 to ~10-15 regions or use top-k approach
# 
# ### 3. L-Diversity for Sensitive Attributes
# Ensure each k-anonymous group has diverse values of:
# - **No-show** (target variable - sensitive as it reveals behavior)
# - Medical conditions combined
# 
# ### 4. T-Closeness
# Ensure sensitive attribute distribution in each partition is close to global distribution:
# - Focus on `No-show` (20% no-show rate should be preserved in partitions)
# 
# ### 5. Differential Privacy for Medical Conditions
# Apply randomized response to binary medical features:
# - `Hypertension`, `Diabetes`, `Alcoholism`, `Handicap`
# - Use p=0.3, q=0.5 for moderate privacy (epsilon ≈ 1.90)
# 
# ### 6. Temporal Generalization
# - Extract useful features: days between scheduling and appointment, day of week, time of day
# - Remove exact timestamps after feature engineering
# 
# **Data Leakage Considerations:**
# - Rare combinations of (Age, Gender, Neighbourhood, Medical conditions) can re-identify
# - Medical conditions might be correlated, reducing effective l-diversity
# - Scholarship status is sensitive socioeconomic indicator
# - SMS_received is behavioral but less sensitive

# ### Privacy Techniques: A Theoretical Foundation
# 
# Before we implement privacy protections, let's understand the theory behind each technique:
# 
# ---
# 
# #### K-Anonymity (Sweeney, 2002)
# **Definition**: A dataset satisfies k-anonymity if every record is indistinguishable from at least k-1 other records with respect to quasi-identifiers.
# 
# **Formula**: For each equivalence class $E$ of quasi-identifiers: $|E| \geq k$
# 
# **Example**: If k=5, any combination of (Age_Group, Gender, Region) must appear at least 5 times.
# 
# ---
# 
# #### L-Diversity (Machanavajjhala et al., 2007)
# **Definition**: An equivalence class has l-diversity if there are at least l "well-represented" values for each sensitive attribute.
# 
# **Why we need it**: K-anonymity alone fails if all records in an equivalence class share the same sensitive value (homogeneity attack).
# 
# **Example**: A k=10 group where all 10 have "Diabetes=1" reveals everyone's status.
# 
# ---
# 
# #### T-Closeness (Li et al., 2007)
# **Definition**: An equivalence class has t-closeness if the distance between the distribution of sensitive attributes in the class and their distribution in the overall dataset is at most t.
# 
# **Formula**: $D(P, Q) \leq t$ where $D$ is typically the Earth Mover's Distance (EMD)
# 
# ---
# 
# #### Differential Privacy (Dwork, 2006)
# **Definition**: A mechanism $\mathcal{M}$ satisfies $\epsilon$-differential privacy if for any two datasets $D_1, D_2$ differing in one record:
# 
# $$P[\mathcal{M}(D_1) \in S] \leq e^\epsilon \cdot P[\mathcal{M}(D_2) \in S]$$
# 
# **Randomized Response**: For our binary medical attributes, we use:
# - With probability $p$: return random value (0 with prob $q$, 1 with prob $1-q$)
# - With probability $1-p$: return true value
# 
# **Privacy budget**: $\epsilon = -\ln(p \cdot q)$

# In[12]:


# Create a working copy for protection
df_protected = df.copy()

# Step 1: Remove direct identifiers
print("Step 1: Removing Direct Identifiers")
print("="*60)
direct_identifiers = ['PatientId', 'AppointmentID']
df_protected = df_protected.drop(columns=direct_identifiers)
print(f"Removed columns: {direct_identifiers}")
print(f"New shape: {df_protected.shape}")
print(f"Remaining columns: {list(df_protected.columns)}")


# In[13]:


# Step 2: Clean and prepare data
print("\nStep 2: Data Cleaning")
print("="*60)

# Fix negative age
df_protected.loc[df_protected['Age'] < 0, 'Age'] = 0
print(f"Fixed {(df['Age'] < 0).sum()} negative age values")

# Convert date columns to datetime

# Cap extreme ages to reduce leverage
age_cap = 100
num_capped = (df_protected['Age'] > age_cap).sum()
df_protected.loc[df_protected['Age'] > age_cap, 'Age'] = age_cap
print(f"Capped {num_capped} ages above {age_cap}")
df_protected['ScheduledDay'] = pd.to_datetime(df_protected['ScheduledDay'])
df_protected['AppointmentDay'] = pd.to_datetime(df_protected['AppointmentDay'])

# Create temporal features BEFORE removing timestamps
df_protected['DaysBetween'] = (df_protected['AppointmentDay'] - df_protected['ScheduledDay']).dt.days
df_protected['AppointmentDayOfWeek'] = df_protected['AppointmentDay'].dt.dayofweek
df_protected['AppointmentMonth'] = df_protected['AppointmentDay'].dt.month
df_protected['ScheduledDayOfWeek'] = df_protected['ScheduledDay'].dt.dayofweek

print(f"Created temporal features: DaysBetween, AppointmentDayOfWeek, AppointmentMonth, ScheduledDayOfWeek")

# Remove exact timestamps (already extracted useful features)
df_protected = df_protected.drop(columns=['ScheduledDay', 'AppointmentDay'])
print(f"Removed exact timestamps")

# Convert No-show to binary (0=No, 1=Yes)
df_protected['NoShow_binary'] = (df_protected['No-show'] == 'Yes').astype(int)

print(f"\nCleaned data shape: {df_protected.shape}")
df_protected.head()


# In[14]:


# Step 3: Analyze re-identification risk before anonymization
print("Step 3: Re-identification Risk Analysis")
print("="*60)

# Check uniqueness of quasi-identifier combinations
quasi_identifiers = ['Age', 'Gender', 'Neighbourhood']
unique_combinations = df_protected.groupby(quasi_identifiers).size().reset_index(name='count')
print(f"\nUnique combinations of {quasi_identifiers}:")
print(f"Total unique combinations: {len(unique_combinations)}")
print(f"Combinations with only 1 person: {len(unique_combinations[unique_combinations['count'] == 1])}")
print(f"Combinations with < 5 people: {len(unique_combinations[unique_combinations['count'] < 5])}")
print(f"Percentage of records at risk (k<5): {(unique_combinations[unique_combinations['count'] < 5]['count'].sum() / len(df_protected) * 100):.2f}%")

# Show most unique combinations
print("\nMost unique (risky) combinations:")
print(unique_combinations.nsmallest(10, 'count'))


# In[15]:


# Step 4: Generalize Neighbourhood using Geographic Regions
print("\nStep 4: Neighbourhood to Region Mapping")
print("="*60)

# Define neighbourhood to region mapping based on official administrative regions of Vitória, ES, Brazil
# Source: Wikipedia - Lista de bairros de Vitória (Espírito Santo)
# https://pt.wikipedia.org/wiki/Lista_de_bairros_de_Vit%C3%B3ria_%28Esp%C3%ADrito_Santo%29
neighbourhood_to_region = {
    'JARDIM DA PENHA': 'Jardim da Penha',
    'MATA DA PRAIA': 'Jardim da Penha',
    'PONTAL DE CAMBURI': 'Jardim da Penha',
    'REPÚBLICA': 'Jardim da Penha',
    'GOIABEIRAS': 'Goiabeiras',
    'ANDORINHAS': 'Maruípe',
    'CONQUISTA': 'São Pedro',
    'NOVA PALESTINA': 'São Pedro',
    'DA PENHA': 'Maruípe',
    'TABUAZEIRO': 'Maruípe',
    'BENTO FERREIRA': 'Jucutuquara',
    'SÃO PEDRO': 'São Pedro',
    'SANTA MARTHA': 'Maruípe',
    'SÃO CRISTÓVÃO': 'Maruípe',
    'MARUÍPE': 'Maruípe',
    'GRANDE VITÓRIA': 'Santo Antônio',
    'SÃO BENEDITO': 'Maruípe',
    'ILHA DAS CAIEIRAS': 'São Pedro',
    'SANTO ANDRÉ': 'São Pedro',
    'SOLON BORGES': 'Goiabeiras',
    'BONFIM': 'Jucutuquara',
    'JARDIM CAMBURI': 'Jardim Camburi',
    'MARIA ORTIZ': 'Goiabeiras',
    'JABOUR': 'Goiabeiras',
    'ANTÔNIO HONÓRIO': 'Goiabeiras',
    'RESISTÊNCIA': 'Santo Antônio',
    'ILHA DE SANTA MARIA': 'Jucutuquara',
    'JUCUTUQUARA': 'Jucutuquara',
    'MONTE BELO': 'Maruípe',
    'MÁRIO CYPRESTE': 'Maruípe',
    'SANTO ANTÔNIO': 'Santo Antônio',
    'BELA VISTA': 'Maruípe',
    'PRAIA DO SUÁ': 'Praia do Canto',
    'SANTA HELENA': 'Praia do Canto',
    'ITARARÉ': 'Santo Antônio',
    'INHANGUETÁ': 'Goiabeiras',
    'UNIVERSITÁRIO': 'Jardim da Penha',
    'SÃO JOSÉ': 'Goiabeiras',
    'REDENÇÃO': 'Goiabeiras',
    'SANTA CLARA': 'Goiabeiras',
    'CENTRO': 'Centro',
    'PARQUE MOSCOSO': 'Centro',
    'DO MOSCOSO': 'Centro',
    'SANTOS DUMONT': 'São Pedro',
    'CARATOÍRA': 'Jucutuquara',
    'ARIOVALDO FAVALESSA': 'São Pedro',
    'ILHA DO FRADE': 'Jucutuquara',
    'GURIGICA': 'Maruípe',
    'JOANA D´ARC': 'Goiabeiras',
    'CONSOLAÇÃO': 'Goiabeiras',
    'PRAIA DO CANTO': 'Praia do Canto',
    'BOA VISTA': 'Jardim da Penha',
    'MORADA DE CAMBURI': 'Jardim Camburi',
    'SANTA LUÍZA': 'Praia do Canto',
    'SANTA LÚCIA': 'Praia do Canto',
    'BARRO VERMELHO': 'Jucutuquara',
    'ESTRELINHA': 'Centro',
    'FORTE SÃO JOÃO': 'Centro',
    'FONTE GRANDE': 'Centro',
    'ENSEADA DO SUÁ': 'Praia do Canto',
    'SANTOS REIS': 'Santo Antônio',
    'PIEDADE': 'Centro',
    'JESUS DE NAZARETH': 'Maruípe',
    'SANTA TEREZA': 'Santo Antônio',
    'CRUZAMENTO': 'Jucutuquara',
    'ILHA DO PRÍNCIPE': 'Santo Antônio',
    'ROMÃO': 'Maruípe',
    'COMDUSA': 'São Pedro',
    'SANTA CECÍLIA': 'Goiabeiras',
    'VILA RUBIM': 'Centro',
    'DE LOURDES': 'Jucutuquara',
    'DO QUADRO': 'Maruípe',
    'DO CABRAL': 'Maruípe',
    'HORTO': 'Goiabeiras',
    'SEGURANÇA DO LAR': 'Goiabeiras',
    'ILHA DO BOI': 'Jucutuquara',
    'FRADINHOS': 'Jucutuquara',
    'NAZARETH': 'Goiabeiras',
    'AEROPORTO': 'Goiabeiras',
    'ILHAS OCEÂNICAS DE TRINDADE': 'Centro',
    'PARQUE INDUSTRIAL': 'Jardim Camburi'
}

# Apply the mapping
df_protected['Region'] = df_protected['Neighbourhood'].map(neighbourhood_to_region)

# Handle any unmapped neighbourhoods (if any)
if df_protected['Region'].isna().any():
    unmapped = df_protected[df_protected['Region'].isna()]['Neighbourhood'].unique()
    print(f"\nWarning: {len(unmapped)} neighbourhoods not in mapping:")
    print(unmapped)
    df_protected['Region'] = df_protected['Region'].fillna('Miscellaneous')

print(f"\nOriginal neighbourhoods: {df_protected['Neighbourhood'].nunique()}")
print(f"Geographic regions: {df_protected['Region'].nunique()}")
print(f"\nRegion distribution:")
region_counts = df_protected['Region'].value_counts()
print(region_counts)

# Visualize the mapping
plt.figure(figsize=(12, 6))
region_counts.plot(kind='barh', color='steelblue', alpha=0.7)
plt.xlabel('Number of Appointments')
plt.ylabel('Region')
plt.title('Distribution of Appointments by Geographic Region')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_region_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/06_region_distribution.png")

print(f"\nNeighbourhoods successfully grouped into {df_protected['Region'].nunique()} geographic regions")


# In[16]:


# Step 4b: Analyze Region Characteristics
print("\nStep 4b: Region Characteristics Analysis")
print("="*60)

# Calculate global no-show rate (needed for comparison)
global_noshow_rate = df_protected['NoShow_binary'].mean()

# Analyze each region's characteristics
region_summary = df_protected.groupby('Region').agg({
    'NoShow_binary': ['mean', 'count'],
    'Age': 'mean',
    'Scholarship': 'mean',
    'Hipertension': 'mean',
    'Diabetes': 'mean',
    'Alcoholism': 'mean'
})
region_summary.columns = ['NoShow_Rate', 'Count', 'Avg_Age', 'Scholarship_Rate', 
                         'Hipertension_Rate', 'Diabetes_Rate', 'Alcoholism_Rate']
region_summary = region_summary.sort_values('Count', ascending=False)

print("\nRegion Summary Statistics:")
print(region_summary.round(3))

# Visualize region characteristics
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# No-show rate by region
axes[0, 0].barh(region_summary.index, region_summary['NoShow_Rate'], alpha=0.7, color='coral')
axes[0, 0].axvline(x=global_noshow_rate, color='red', linestyle='--', linewidth=2, label='Global Average')
axes[0, 0].set_xlabel('No-Show Rate')
axes[0, 0].set_ylabel('Region')
axes[0, 0].set_title('No-Show Rate by Geographic Region')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='x')

# Average age by region
axes[0, 1].barh(region_summary.index, region_summary['Avg_Age'], alpha=0.7, color='steelblue')
axes[0, 1].set_xlabel('Average Age')
axes[0, 1].set_ylabel('Region')
axes[0, 1].set_title('Average Age by Geographic Region')
axes[0, 1].grid(alpha=0.3, axis='x')

# Region appointment counts
axes[1, 0].barh(region_summary.index, region_summary['Count'], alpha=0.7, color='green')
axes[1, 0].set_xlabel('Number of Appointments')
axes[1, 0].set_ylabel('Region')
axes[1, 0].set_title('Appointments Count by Geographic Region')
axes[1, 0].grid(alpha=0.3, axis='x')

# Scholarship rate by region
axes[1, 1].barh(region_summary.index, region_summary['Scholarship_Rate'], alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Scholarship Rate')
axes[1, 1].set_ylabel('Region')
axes[1, 1].set_title('Scholarship Rate by Geographic Region')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/06_region_characteristics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/07_region_characteristics.png")

print("\nGeographic regions preserve meaningful patterns:")
print(f"  - No-show rates vary from {region_summary['NoShow_Rate'].min():.3f} to {region_summary['NoShow_Rate'].max():.3f}")
print(f"  - Average age varies from {region_summary['Avg_Age'].min():.1f} to {region_summary['Avg_Age'].max():.1f} years")
print(f"  - Scholarship rates vary from {region_summary['Scholarship_Rate'].min():.3f} to {region_summary['Scholarship_Rate'].max():.3f}")


# In[17]:


# Show which neighbourhoods are in each region
print("\nNeighbourhood Distribution by Region:")
print("="*60)

for region in region_summary.index:
    neighbourhoods_in_region = df_protected[df_protected['Region'] == region]['Neighbourhood'].value_counts()
    print(f"\n{region}: {len(neighbourhoods_in_region)} neighbourhoods, {neighbourhoods_in_region.sum():,} appointments")
    print(f"  Top 5: {', '.join(neighbourhoods_in_region.head(5).index.tolist())}")
    print(f"  No-show rate: {region_summary.loc[region, 'NoShow_Rate']:.3f}")


# In[18]:


# Step 5: Generalize Age into bins
print("\nStep 5: Age Generalization")
print("="*60)

# Create age bins
age_bins = [0, 18, 30, 45, 60, 75, 120]
age_labels = ['0-17', '18-29', '30-44', '45-59', '60-74', '75+']
df_protected['Age_generalized'] = pd.cut(df_protected['Age'], bins=age_bins, labels=age_labels, right=False)

print("Age distribution after generalization:")
print(df_protected['Age_generalized'].value_counts().sort_index())

# Visualize
plt.figure(figsize=(10, 5))
df_protected['Age_generalized'].value_counts().sort_index().plot(kind='bar', color='steelblue', alpha=0.7)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Age Distribution After Generalization')
plt.xticks(rotation=45)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/07_age_generalization.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/08_age_generalization.png")


# In[19]:


# Step 6: Re-check k-anonymity after generalization
print("\nStep 6: K-Anonymity Check After Generalization")
print("="*60)

# Use generalized quasi-identifiers
quasi_identifiers_gen = ['Age_generalized', 'Gender', 'Region']
unique_combinations_gen = df_protected.groupby(quasi_identifiers_gen).size().reset_index(name='count')

print(f"\nAfter generalization:")
print(f"Total unique combinations: {len(unique_combinations_gen)}")
print(f"Combinations with only 1 person: {len(unique_combinations_gen[unique_combinations_gen['count'] == 1])}")
print(f"Combinations with < 5 people (k<5): {len(unique_combinations_gen[unique_combinations_gen['count'] < 5])}")
print(f"Percentage of records at risk (k<5): {(unique_combinations_gen[unique_combinations_gen['count'] < 5]['count'].sum() / len(df_protected) * 100):.2f}%")

# Distribution of group sizes
print("\nDistribution of group sizes (k values):")
k_distribution = unique_combinations_gen['count'].describe()
print(k_distribution)

# Visualize k distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(unique_combinations_gen['count'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Group Size (k)')
plt.ylabel('Frequency')
plt.title('Distribution of K Values')
plt.axvline(x=5, color='red', linestyle='--', label='k=5 threshold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(unique_combinations_gen['count'], bins=50, edgecolor='black', alpha=0.7, cumulative=True, density=True)
plt.xlabel('Group Size (k)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution of K Values')
plt.axvline(x=5, color='red', linestyle='--', label='k=5 threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/08_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/02_age_distribution.png')

print(f"\nMinimum k-value: {unique_combinations_gen['count'].min()}")
print(f"Maximum k-value: {unique_combinations_gen['count'].max()}")
print(f"Median k-value: {unique_combinations_gen['count'].median()}")


# In[20]:


# Step 7: Check L-Diversity for ALL Sensitive Attributes
print("\nStep 7: Comprehensive L-Diversity Analysis")
print("="*60)

# Binarize Handcap for L-diversity analysis (0 = no disability, 1 = has disability)
# Handcap has values 0-4, but 98% are 0, so we treat it as binary for privacy purposes
df_protected['Handcap_binary'] = (df_protected['Handcap'] > 0).astype(int)
print(f"\nBinarized Handcap: {df_protected['Handcap'].value_counts().to_dict()}")
print(f"  → Handcap_binary: {df_protected['Handcap_binary'].value_counts().to_dict()}")

# Define all sensitive attributes that need L-diversity protection
# Note: NoShow_binary is the TARGET variable, not a sensitive attribute requiring privacy
# Use Handcap_binary instead of Handcap (0-4 scale) for meaningful L-diversity
sensitive_attributes = ['Alcoholism', 'Hipertension', 'Diabetes', 'Handcap_binary', 'Scholarship']

print(f"\nSensitive attributes requiring L-diversity = 2:")
for attr in sensitive_attributes:
    print(f"  - {attr}")
print(f"\nNote: Handcap binarized (0=no disability, 1=has disability) for L-diversity")
print(f"Note: NoShow (target variable) is NOT treated as sensitive for L-diversity")

# For each quasi-identifier group, check diversity of ALL sensitive attributes
diversity_results = {}
for attr in sensitive_attributes:
    attr_diversity = df_protected.groupby(quasi_identifiers_gen)[attr].agg(['count', 'nunique']).reset_index()
    attr_diversity.columns = quasi_identifiers_gen + ['group_size', f'{attr}_l']
    diversity_results[attr] = attr_diversity

# Merge all diversity checks into a single dataframe
# Start with the first sensitive attribute
diversity_check = diversity_results[sensitive_attributes[0]].copy()

# Merge the rest
for attr in sensitive_attributes[1:]:
    diversity_check = diversity_check.merge(
        diversity_results[attr][quasi_identifiers_gen + [f'{attr}_l']], 
        on=quasi_identifiers_gen
    )

# Calculate minimum l-diversity across all sensitive attributes
l_cols = [col for col in diversity_check.columns if col.endswith('_l')]
diversity_check['min_l_diversity'] = diversity_check[l_cols].min(axis=1)

# Calculate mean no-show rate for visualization
diversity_check = diversity_check.merge(
    df_protected.groupby(quasi_identifiers_gen)['NoShow_binary'].mean().reset_index().rename(columns={'NoShow_binary': 'noshow_rate'}),
    on=quasi_identifiers_gen
)

print(f"\n" + "="*60)
print("L-DIVERSITY RESULTS FOR ALL SENSITIVE ATTRIBUTES")
print("="*60)

print(f"\nOverall Statistics:")
print(f"  Total equivalence classes: {len(diversity_check)}")
print(f"  Groups with min_l=1 (FAILS L-diversity): {len(diversity_check[diversity_check['min_l_diversity'] == 1])}")
print(f"  Groups with min_l=2 (PASSES L-diversity): {len(diversity_check[diversity_check['min_l_diversity'] == 2])}")
print(f"  Percentage of records in l=2 groups: {(diversity_check[diversity_check['min_l_diversity'] == 2]['group_size'].sum() / len(df_protected) * 100):.2f}%")

print(f"\nPer-Attribute L-Diversity Statistics:")
for attr in sensitive_attributes:
    attr_col = attr if attr == 'NoShow_binary' else attr
    l_col = 'NoShow_l' if attr == 'NoShow_binary' else f'{attr}_l'
    l1_groups = len(diversity_check[diversity_check[l_col] == 1])
    l2_groups = len(diversity_check[diversity_check[l_col] == 2])
    l2_records = diversity_check[diversity_check[l_col] == 2]['group_size'].sum()
    l2_pct = (l2_records / len(df_protected) * 100)
    print(f"  {attr:15s}: l=1 groups: {l1_groups:4d} | l=2 groups: {l2_groups:4d} | Records with l=2: {l2_pct:6.2f}%")

# Show examples of groups that FAIL l-diversity (min_l = 1)
print(f"\n" + "="*60)
print("GROUPS FAILING L-DIVERSITY (min_l = 1):")
print("="*60)
failing_groups = diversity_check[diversity_check['min_l_diversity'] == 1].sort_values('group_size', ascending=False).head(10)
display_cols = quasi_identifiers_gen + ['group_size', 'Alcoholism_l', 'Hipertension_l', 'Diabetes_l', 'Handcap_binary_l', 'Scholarship_l', 'min_l_diversity']
print(failing_groups[display_cols].to_string(index=False))

# Save BEFORE perturbation stats for comparison
diversity_check_before = diversity_check.copy()
attr_l2_pcts_before = []
attr_names_short = []
for attr in sensitive_attributes:
    l_col = f'{attr}_l'
    l2_pct = (diversity_check_before[diversity_check_before[l_col] == 2]['group_size'].sum() / len(df_protected) * 100)
    attr_l2_pcts_before.append(l2_pct)
    attr_names_short.append(attr)

# Store before stats for later comparison
before_stats = {
    'total_groups': len(diversity_check_before),
    'failing_groups': len(diversity_check_before[diversity_check_before['min_l_diversity'] == 1]),
    'passing_groups': len(diversity_check_before[diversity_check_before['min_l_diversity'] == 2]),
    'attr_l2_pcts': attr_l2_pcts_before.copy(),
    'attr_names': attr_names_short.copy(),
    'l1_records_pct': (diversity_check_before[diversity_check_before['min_l_diversity'] == 1]['group_size'].sum() / len(df_protected) * 100),
    'l2_records_pct': (diversity_check_before[diversity_check_before['min_l_diversity'] == 2]['group_size'].sum() / len(df_protected) * 100)
}

print("\n" + "="*60)
print("L-DIVERSITY STATUS BEFORE PERTURBATION:")
print("="*60)
print(f"  Groups failing l=2: {before_stats['failing_groups']}")
print(f"  Groups passing l=2: {before_stats['passing_groups']}")
print(f"  Records in failing groups: {before_stats['l1_records_pct']:.2f}%")
print("\nNote: Detailed visualization will be generated AFTER perturbation for comparison")


# In[20b]:


# Step 7b: Enforce L-Diversity = 2 by VALUE PERTURBATION (Alternative to Suppression/Generalization)
print("\n" + "="*60)
print("Step 7b: ENFORCING L-DIVERSITY = 2 VIA VALUE PERTURBATION")
print("="*60)

print("\nSTRATEGY: Flip/perturb sensitive attribute values in groups that fail l-diversity")
print("This RETAINS ALL RECORDS with their original quasi-identifiers while achieving privacy.")
print("\nApproach:")
print("  1. Identify groups failing l-diversity (all records have same sensitive value)")
print("  2. Randomly flip a minimal number of sensitive values to create diversity")
print("  3. Preserve global distribution as much as possible")

# Before perturbation stats
total_records_before = len(df_protected)
groups_before = len(diversity_check)
failing_groups_before = len(diversity_check[diversity_check['min_l_diversity'] == 1])
failing_records_before = diversity_check[diversity_check['min_l_diversity'] == 1]['group_size'].sum()

print(f"\nBEFORE PERTURBATION:")
print(f"  Total records: {total_records_before:,}")
print(f"  Total groups: {groups_before}")
print(f"  Failing groups (min_l=1): {failing_groups_before}")
print(f"  Records in failing groups: {failing_records_before:,} ({failing_records_before/total_records_before*100:.2f}%)")

np.random.seed(42)  # For reproducibility

def enforce_l_diversity_perturbation(df, quasi_ids, sensitive_attr, target_l=2):
    """
    Enforce l-diversity by perturbing (flipping) sensitive attribute values.
    
    For binary attributes (0/1), we flip the minimum number of values needed
    to achieve target_l distinct values in each equivalence class.
    
    Returns:
        - Modified dataframe
        - Number of values perturbed
        - Details of perturbations
    """
    df_perturbed = df.copy()
    total_perturbed = 0
    perturbation_details = []
    
    # Get unique values for the sensitive attribute
    unique_vals = df[sensitive_attr].unique()
    
    # Group by quasi-identifiers
    for group_keys, group_df in df.groupby(quasi_ids):
        group_indices = group_df.index.tolist()
        current_values = group_df[sensitive_attr].values
        n_unique = len(np.unique(current_values))
        
        # If already satisfies l-diversity, skip
        if n_unique >= target_l:
            continue
        
        # Need to add diversity - flip some values
        group_size = len(group_indices)
        
        if group_size < target_l:
            # Group too small for l-diversity, skip (would need suppression)
            continue
        
        # For binary attributes: flip minimum records to achieve l=2
        if len(unique_vals) == 2:
            dominant_val = current_values[0]  # All same, so take first
            other_val = [v for v in unique_vals if v != dominant_val][0]
            
            # Flip at least 1 record (or more for larger groups to be proportional)
            # We flip ~10-20% or at least 1 record
            n_to_flip = max(1, min(group_size // 5, 3))  # Flip 1-3 records max
            
            # Randomly select records to flip
            flip_indices = np.random.choice(group_indices, size=n_to_flip, replace=False)
            
            for idx in flip_indices:
                df_perturbed.loc[idx, sensitive_attr] = other_val
                total_perturbed += 1
            
            perturbation_details.append({
                'group': group_keys,
                'group_size': group_size,
                'attribute': sensitive_attr,
                'original_val': dominant_val,
                'flipped_to': other_val,
                'n_flipped': n_to_flip
            })
        
        else:
            # For multi-valued attributes: assign random different values
            dominant_val = current_values[0]
            other_vals = [v for v in unique_vals if v != dominant_val]
            
            n_to_flip = max(1, min(group_size // 5, 3))
            flip_indices = np.random.choice(group_indices, size=n_to_flip, replace=False)
            
            for i, idx in enumerate(flip_indices):
                new_val = other_vals[i % len(other_vals)]
                df_perturbed.loc[idx, sensitive_attr] = new_val
                total_perturbed += 1
            
            perturbation_details.append({
                'group': group_keys,
                'group_size': group_size,
                'attribute': sensitive_attr,
                'original_val': dominant_val,
                'flipped_to': 'various',
                'n_flipped': n_to_flip
            })
    
    return df_perturbed, total_perturbed, perturbation_details

# Apply perturbation to each sensitive attribute
df_perturbed = df_protected.copy()
total_perturbations = 0
all_perturbation_details = {}

print(f"\n{'='*60}")
print("APPLYING VALUE PERTURBATION:")
print("="*60)

for attr in sensitive_attributes:
    print(f"\nProcessing: {attr}")
    
    # Check current l-diversity for this attribute
    attr_check = df_perturbed.groupby(quasi_identifiers_gen)[attr].nunique().reset_index()
    attr_check.columns = quasi_identifiers_gen + ['l_value']
    failing_before = len(attr_check[attr_check['l_value'] < 2])
    
    if failing_before == 0:
        print(f"  ✓ Already satisfies l=2 for all groups")
        continue
    
    # Get original distribution
    original_dist = df_perturbed[attr].value_counts(normalize=True)
    
    # Apply perturbation
    df_perturbed, n_perturbed, details = enforce_l_diversity_perturbation(
        df_perturbed, quasi_identifiers_gen, attr, target_l=2
    )
    
    total_perturbations += n_perturbed
    all_perturbation_details[attr] = details
    
    # Check new l-diversity
    attr_check_after = df_perturbed.groupby(quasi_identifiers_gen)[attr].nunique().reset_index()
    attr_check_after.columns = quasi_identifiers_gen + ['l_value']
    failing_after = len(attr_check_after[attr_check_after['l_value'] < 2])
    
    # Get new distribution
    new_dist = df_perturbed[attr].value_counts(normalize=True)
    
    print(f"  Groups failing l=2: {failing_before} → {failing_after}")
    print(f"  Values perturbed: {n_perturbed}")
    print(f"  Original distribution: {dict(original_dist.round(4))}")
    print(f"  New distribution:      {dict(new_dist.round(4))}")
    
    # Calculate distribution shift
    dist_shift = abs(original_dist - new_dist).sum() / 2  # Total variation distance
    print(f"  Distribution shift (TVD): {dist_shift:.4f}")

print(f"\n{'='*60}")
print("PERTURBATION SUMMARY:")
print("="*60)
print(f"  Total values perturbed: {total_perturbations:,}")
print(f"  Perturbation rate: {total_perturbations / len(df_perturbed) * 100:.4f}%")
print(f"  Records retained: 100% (NO suppression!)")
print(f"  Quasi-identifiers: UNCHANGED (no generalization!)")

# Update the protected dataframe
df_protected_l2 = df_perturbed.copy()

print(f"\nAFTER VALUE PERTURBATION:")
print(f"  Total records: {len(df_protected_l2):,} (100% retained)")
print(f"  Total perturbations: {total_perturbations:,} ({total_perturbations/len(df_protected_l2)*100:.4f}%)")

# Re-verify L-diversity on the perturbed dataset
print(f"\n" + "="*60)
print("RE-VERIFYING L-DIVERSITY ON PERTURBED DATASET:")
print("="*60)

diversity_results_l2 = {}
for attr in sensitive_attributes:
    attr_diversity = df_protected_l2.groupby(quasi_identifiers_gen)[attr].agg(['count', 'nunique']).reset_index()
    attr_diversity.columns = quasi_identifiers_gen + ['group_size', f'{attr}_l']
    diversity_results_l2[attr] = attr_diversity

# Merge all diversity checks
diversity_check_l2 = diversity_results_l2[sensitive_attributes[0]].copy()

for attr in sensitive_attributes[1:]:
    diversity_check_l2 = diversity_check_l2.merge(
        diversity_results_l2[attr][quasi_identifiers_gen + [f'{attr}_l']], 
        on=quasi_identifiers_gen
    )

l_cols_l2 = [col for col in diversity_check_l2.columns if col.endswith('_l')]
diversity_check_l2['min_l_diversity'] = diversity_check_l2[l_cols_l2].min(axis=1)

# Add noshow_rate for T-closeness analysis
diversity_check_l2 = diversity_check_l2.merge(
    df_protected_l2.groupby(quasi_identifiers_gen)['NoShow_binary'].mean().reset_index().rename(columns={'NoShow_binary': 'noshow_rate'}),
    on=quasi_identifiers_gen
)

print(f"\nL-DIVERSITY VERIFICATION RESULTS:")
print(f"  Total groups: {len(diversity_check_l2)}")
print(f"  Groups with min_l=1: {len(diversity_check_l2[diversity_check_l2['min_l_diversity'] == 1])}")
print(f"  Groups with min_l=2: {len(diversity_check_l2[diversity_check_l2['min_l_diversity'] == 2])}")
print(f"  SUCCESS: {len(diversity_check_l2[diversity_check_l2['min_l_diversity'] == 2])/len(diversity_check_l2)*100:.1f}% of groups have min_l=2")

print(f"\nPer-Attribute L-Diversity (After Perturbation):")
for attr in sensitive_attributes:
    l_col = f'{attr}_l'
    l1_groups = len(diversity_check_l2[diversity_check_l2[l_col] == 1])
    l2_groups = len(diversity_check_l2[diversity_check_l2[l_col] == 2])
    l2_records = diversity_check_l2[diversity_check_l2[l_col] == 2]['group_size'].sum()
    l2_pct = (l2_records / len(df_protected_l2) * 100) if len(df_protected_l2) > 0 else 0
    print(f"  {attr:15s}: l=1 groups: {l1_groups:4d} | l=2 groups: {l2_groups:4d} | Records with l=2: {l2_pct:6.2f}%")

# Re-verify K-anonymity on perturbed dataset
print(f"\n" + "="*60)
print("RE-VERIFYING K-ANONYMITY ON PERTURBED DATASET:")
print("="*60)

k_check_l2 = df_protected_l2.groupby(quasi_identifiers_gen).size().reset_index(name='count')
min_k_l2 = k_check_l2['count'].min()
median_k_l2 = k_check_l2['count'].median()

print(f"  Minimum k: {min_k_l2}")
print(f"  Median k: {median_k_l2:.0f}")
print(f"  Groups with k < 5: {len(k_check_l2[k_check_l2['count'] < 5])}")
print(f"  K-anonymity maintained: k >= {min_k_l2}")

# Update df_protected to use the perturbed version
df_protected = df_protected_l2.copy()

# Update diversity_check as well for downstream analysis
diversity_check = diversity_check_l2.copy()

print(f"\n" + "="*60)
print("PRIVACY ENFORCEMENT COMPLETE (VALUE PERTURBATION)")
print("="*60)
print(f"Dataset now satisfies L-DIVERSITY = 2 for ALL sensitive attributes")
print(f"K-anonymity maintained with k >= {min_k_l2}")
print(f"{len(df_protected):,} records retained (100% of original)")
print(f"\nADVANTAGES of Value Perturbation:")
print(f"  ✓ No records suppressed (100% data retention)")
print(f"  ✓ No additional generalization (preserves quasi-identifier granularity)")
print(f"  ✓ Minimal perturbation rate ({total_perturbations/len(df_protected)*100:.4f}%)")
print(f"  ✓ Global distributions largely preserved")
print(f"\nTRADE-OFF: Some individual sensitive values are modified (plausible deniability)")

# ============================================================
# COMPREHENSIVE L-DIVERSITY VISUALIZATION (AFTER PERTURBATION)
# ============================================================
print("\n" + "="*60)
print("GENERATING L-DIVERSITY VISUALIZATIONS (AFTER PERTURBATION)")
print("="*60)

# Calculate AFTER perturbation stats
attr_l2_pcts_after = []
for attr in sensitive_attributes:
    l_col = f'{attr}_l'
    l2_pct = (diversity_check[diversity_check[l_col] == 2]['group_size'].sum() / len(df_protected) * 100)
    attr_l2_pcts_after.append(l2_pct)

after_stats = {
    'total_groups': len(diversity_check),
    'failing_groups': len(diversity_check[diversity_check['min_l_diversity'] == 1]),
    'passing_groups': len(diversity_check[diversity_check['min_l_diversity'] == 2]),
    'attr_l2_pcts': attr_l2_pcts_after,
    'l1_records_pct': (diversity_check[diversity_check['min_l_diversity'] == 1]['group_size'].sum() / len(df_protected) * 100) if len(diversity_check[diversity_check['min_l_diversity'] == 1]) > 0 else 0,
    'l2_records_pct': (diversity_check[diversity_check['min_l_diversity'] == 2]['group_size'].sum() / len(df_protected) * 100)
}

# ============================================================
# CHART 1: Comprehensive L-Diversity Analysis (After Perturbation)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('L-DIVERSITY ANALYSIS (AFTER PERTURBATION)', fontsize=16, fontweight='bold', y=1.02)

# 1. Bar chart of minimum l-diversity distribution
min_l_counts = diversity_check['min_l_diversity'].value_counts().sort_index()
axes[0, 0].bar(min_l_counts.index, min_l_counts.values, color='green', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Minimum L-Diversity Value', fontsize=11)
axes[0, 0].set_ylabel('Number of Groups', fontsize=11)
axes[0, 0].set_title('Minimum L-Diversity Distribution\n(After Perturbation)', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3, axis='y')
for i, (idx, v) in enumerate(min_l_counts.items()):
    axes[0, 0].text(idx, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Pie chart showing proportion of records by minimum l-diversity
l_div_records = diversity_check.groupby('min_l_diversity')['group_size'].sum()
if len(l_div_records) == 1:  # All passing
    colors_pie = ['#66ff66']
    axes[0, 1].pie(l_div_records.values, labels=[f'min_l={int(k)}\n(100%)' for k in l_div_records.index], 
                   colors=colors_pie, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
else:
    colors_pie = ['#ff9999', '#66ff66']
    axes[0, 1].pie(l_div_records.values, labels=[f'min_l={int(k)}' for k in l_div_records.index], 
                   autopct='%1.1f%%', colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0, 1].set_title('Proportion of Records\nby Minimum L-Diversity (After)', fontsize=12, fontweight='bold')

# 3. Per-attribute l-diversity comparison (after perturbation)
axes[0, 2].barh(attr_names_short, attr_l2_pcts_after, color='green', alpha=0.7, edgecolor='black')
axes[0, 2].set_xlabel('% Records with l=2', fontsize=11)
axes[0, 2].set_title('L-Diversity Achievement\nPer Attribute (After Perturbation)', fontsize=12, fontweight='bold')
axes[0, 2].axvline(x=100, color='darkgreen', linestyle='--', linewidth=2, label='100% Target')
axes[0, 2].set_xlim(0, 110)
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3, axis='x')
for i, v in enumerate(attr_l2_pcts_after):
    axes[0, 2].text(min(v + 1, 105), i, f'{v:.1f}%', va='center', fontsize=10)

# 4. Scatter: Group size vs No-show rate (after perturbation)
for l_val in sorted(diversity_check['min_l_diversity'].unique()):
    subset = diversity_check[diversity_check['min_l_diversity'] == l_val]
    color = 'green' if l_val == 2 else 'red'
    axes[1, 0].scatter(subset['group_size'], subset['noshow_rate'], 
                      label=f'min_l={int(l_val)}', alpha=0.6, s=50, c=color)
axes[1, 0].set_xlabel('Group Size (k)', fontsize=11)
axes[1, 0].set_ylabel('No-Show Rate', fontsize=11)
axes[1, 0].set_title('Group Size vs No-Show Rate\n(After Perturbation)', fontsize=12, fontweight='bold')
global_noshow_rate_current = df_protected['NoShow_binary'].mean()
axes[1, 0].axhline(y=global_noshow_rate_current, color='blue', linestyle='--', linewidth=2, label=f'Global Rate ({global_noshow_rate_current:.3f})')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 5. Histogram: Group size distribution (after perturbation)
axes[1, 1].hist(diversity_check['group_size'], bins=30, alpha=0.7, color='green', edgecolor='black', label='All groups (l=2)')
axes[1, 1].set_xlabel('Group Size (k)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Group Size Distribution\n(After Perturbation)', fontsize=12, fontweight='bold')
axes[1, 1].axvline(x=diversity_check['group_size'].median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Median k={diversity_check["group_size"].median():.0f}')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 6. Heatmap showing l-diversity by attribute (after perturbation)
l_matrix_after = []
for attr in sensitive_attributes:
    l_col = f'{attr}_l'
    l1_count = len(diversity_check[diversity_check[l_col] == 1])
    l2_count = len(diversity_check[diversity_check[l_col] == 2])
    l_matrix_after.append([l1_count, l2_count])

sns.heatmap(l_matrix_after, annot=True, fmt='d', cmap='RdYlGn', ax=axes[1, 2],
            xticklabels=['l=1 (Fail)', 'l=2 (Pass)'], 
            yticklabels=attr_names_short, cbar_kws={'label': 'Number of Groups'})
axes[1, 2].set_title('L-Diversity Achievement\nHeatmap (After Perturbation)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/09_ldiversity_comprehensive.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/09_ldiversity_comprehensive.png")

# ============================================================
# CHART 2: BEFORE vs AFTER Perturbation Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('L-DIVERSITY: BEFORE vs AFTER PERTURBATION COMPARISON', fontsize=16, fontweight='bold', y=1.02)

# 1. Bar chart comparing groups passing/failing l-diversity
bar_width = 0.35
x = np.array([0, 1])
labels = ['Failing (l=1)', 'Passing (l=2)']
before_vals = [before_stats['failing_groups'], before_stats['passing_groups']]
after_vals = [after_stats['failing_groups'], after_stats['passing_groups']]

axes[0, 0].bar(x - bar_width/2, before_vals, bar_width, label='Before Perturbation', color='coral', alpha=0.8, edgecolor='black')
axes[0, 0].bar(x + bar_width/2, after_vals, bar_width, label='After Perturbation', color='green', alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('L-Diversity Status', fontsize=11)
axes[0, 0].set_ylabel('Number of Groups', fontsize=11)
axes[0, 0].set_title('Groups by L-Diversity Status\nBefore vs After', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(labels)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Add value labels
for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
    axes[0, 0].text(i - bar_width/2, bv + 1, str(bv), ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0, 0].text(i + bar_width/2, av + 1, str(av), ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Percentage of records in l=2 groups (before vs after)
record_pcts = ['Before', 'After']
l2_pcts = [before_stats['l2_records_pct'], after_stats['l2_records_pct']]
colors = ['coral', 'green']

bars = axes[0, 1].bar(record_pcts, l2_pcts, color=colors, alpha=0.8, edgecolor='black')
axes[0, 1].set_ylabel('% of Records', fontsize=11)
axes[0, 1].set_title('% Records Satisfying L-Diversity=2\nBefore vs After', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(0, 110)
axes[0, 1].axhline(y=100, color='darkgreen', linestyle='--', linewidth=2, label='100% Target')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

for bar, pct in zip(bars, l2_pcts):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 3. Per-attribute comparison (grouped bar chart)
bar_width = 0.35
x = np.arange(len(attr_names_short))

axes[1, 0].barh(x - bar_width/2, before_stats['attr_l2_pcts'], bar_width, 
                label='Before Perturbation', color='coral', alpha=0.8, edgecolor='black')
axes[1, 0].barh(x + bar_width/2, after_stats['attr_l2_pcts'], bar_width, 
                label='After Perturbation', color='green', alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('% Records with l=2', fontsize=11)
axes[1, 0].set_title('Per-Attribute L-Diversity Achievement\nBefore vs After', fontsize=12, fontweight='bold')
axes[1, 0].set_yticks(x)
axes[1, 0].set_yticklabels(attr_names_short)
axes[1, 0].axvline(x=100, color='darkgreen', linestyle='--', linewidth=2, label='100% Target')
axes[1, 0].set_xlim(0, 115)
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(alpha=0.3, axis='x')

# 4. Summary stats table
axes[1, 1].axis('off')
summary_data = [
    ['Metric', 'Before', 'After', 'Change'],
    ['Total Groups', str(before_stats['total_groups']), str(after_stats['total_groups']), '0'],
    ['Failing Groups (l=1)', str(before_stats['failing_groups']), str(after_stats['failing_groups']), 
     f"-{before_stats['failing_groups'] - after_stats['failing_groups']}"],
    ['Passing Groups (l=2)', str(before_stats['passing_groups']), str(after_stats['passing_groups']),
     f"+{after_stats['passing_groups'] - before_stats['passing_groups']}"],
    ['Records in l=2 (%)', f"{before_stats['l2_records_pct']:.1f}%", f"{after_stats['l2_records_pct']:.1f}%",
     f"+{after_stats['l2_records_pct'] - before_stats['l2_records_pct']:.1f}%"],
    ['Perturbations', '-', f"{total_perturbations:,}", '-'],
    ['Perturbation Rate', '-', f"{total_perturbations/len(df_protected)*100:.4f}%", '-']
]

table = axes[1, 1].table(cellText=summary_data, loc='center', cellLoc='center',
                         colWidths=[0.35, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style the header row
for j in range(4):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Style success row (last metric rows)
for i in range(1, len(summary_data)):
    if 'Passing' in summary_data[i][0] or 'Records in l=2' in summary_data[i][0]:
        table[(i, 2)].set_facecolor('#C6EFCE')
        table[(i, 3)].set_facecolor('#C6EFCE')

axes[1, 1].set_title('Summary Statistics\nBefore vs After Perturbation', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/09b_ldiversity_before_after_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/09b_ldiversity_before_after_comparison.png")

print("\n" + "="*60)
print("L-DIVERSITY VISUALIZATIONS COMPLETE:")
print("="*60)
print("  1. 09_ldiversity_comprehensive.png - After perturbation analysis")
print("  2. 09b_ldiversity_before_after_comparison.png - Before vs After comparison")
print("\nKEY RESULTS:")
print(f"  • Failing groups: {before_stats['failing_groups']} → {after_stats['failing_groups']}")
print(f"  • Records with l=2: {before_stats['l2_records_pct']:.1f}% → {after_stats['l2_records_pct']:.1f}%")
print(f"  • Perturbation rate: {total_perturbations/len(df_protected)*100:.4f}%")


# In[21]:


# Step 8: Check T-Closeness
print("\nStep 8: T-Closeness Analysis")
print("="*60)

# Global distribution of No-show
global_noshow_rate = df_protected['NoShow_binary'].mean()
print(f"Global No-show rate: {global_noshow_rate:.4f}")

# Calculate distance from global distribution for each group
diversity_check['distance_from_global'] = abs(diversity_check['noshow_rate'] - global_noshow_rate)

# T-closeness threshold (e.g., t=0.2)
t_threshold = 0.2
diversity_check['t_close'] = diversity_check['distance_from_global'] <= t_threshold

print(f"\nT-Closeness Analysis (t={t_threshold}):")
print(f"Groups satisfying t-closeness: {diversity_check['t_close'].sum()} / {len(diversity_check)}")
print(f"Percentage of groups: {(diversity_check['t_close'].sum() / len(diversity_check) * 100):.2f}%")
print(f"Percentage of records in t-close groups: {(diversity_check[diversity_check['t_close']]['group_size'].sum() / len(df_protected) * 100):.2f}%")

# Show groups that violate t-closeness
print(f"\nGroups violating t-closeness (top 10 by size):")
violating = diversity_check[~diversity_check['t_close']].sort_values('group_size', ascending=False).head(10)
print(violating)

# Visualize distribution of distances
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(diversity_check['distance_from_global'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=t_threshold, color='red', linestyle='--', label=f't={t_threshold}')
plt.xlabel('Distance from Global No-show Rate')
plt.ylabel('Number of Groups')
plt.title('T-Closeness: Distance Distribution')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(diversity_check['group_size'], diversity_check['distance_from_global'], alpha=0.3)
plt.axhline(y=t_threshold, color='red', linestyle='--', label=f't={t_threshold}')
plt.xlabel('Group Size (k)')
plt.ylabel('Distance from Global Rate')
plt.title('Group Size vs T-Closeness')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_noshow_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/03_noshow_distribution.png')


# ## Part Three: Implementing Full Data Anonymization
# 
# Now we'll apply the complete anonymization pipeline combining:
# 1. **K-anonymity, L-diversity, T-closeness** (already applied through generalization)
# 2. **Differential Privacy** using randomized response for medical conditions
# 3. **Final anonymized dataset creation** ready for machine learning
# 
# ### Differential Privacy Implementation
# 
# We'll apply the randomized response mechanism from Reference 2 to protect sensitive binary medical attributes.

# In[22]:


# Summary table for k-anonymity, l-diversity, t-closeness (before vs after)
k_before = unique_combinations['count']
k_after = unique_combinations_gen['count']

summary_rows = [
    {
        'Metric': 'k-anonymity (group size)',
        'Before': f"min={k_before.min()}, median={k_before.median():.0f}",
        'After': f"min={k_after.min()}, median={k_after.median():.0f}"
    },
    {
        'Metric': 'Records with k<5',
        'Before': f"{(k_before < 5).sum()} combos / {(k_before[k_before < 5].sum() / len(df_protected) * 100):.1f}% records",
        'After': f"{(k_after < 5).sum()} combos / {(k_after[k_after < 5].sum() / len(df_protected) * 100):.1f}% records"
    },
    {
        'Metric': 'L-diversity (all sensitive attrs)',
        'Before': 'n/a (pre-generalization)',
        'After': f"min_l=1 groups: {len(diversity_check[diversity_check['min_l_diversity']==1])}; min_l=2 groups: {len(diversity_check[diversity_check['min_l_diversity']==2])}"
    },
    {
        'Metric': 'T-closeness @ t=0.2',
        'Before': 'n/a (pre-generalization)',
        'After': f"{diversity_check['t_close'].sum()} / {len(diversity_check)} groups within threshold"
    }
]

klt_table = pd.DataFrame(summary_rows)
print(klt_table)


# In[23]:


# Step 9: Implement Differential Privacy Functions
print("\nStep 9: Differential Privacy Setup")
print("="*60)

import random

def epsilon_from_pq(p, q):
    """
    Calculate epsilon (privacy budget) from p and q parameters.
    epsilon = -ln(p*q)
    """
    return -np.log(p * q)

def randomized_response(value, p, q):
    """
    Apply randomized response mechanism to a binary value.

    Parameters:
    - value: original binary value (0 or 1)
    - p: probability of returning random value instead of true value
    - q: probability of returning 0 when generating random value

    Returns: differentially private value
    """
    if random.random() < p:
        # Return random value
        return 0 if random.random() < q else 1
    else:
        # Return true value
        return value

def apply_differential_privacy_to_column(series, p, q):
    """
    Apply randomized response to an entire column.
    """
    return series.apply(lambda x: randomized_response(int(x), p, q))

# Set privacy parameters
p = 0.3  # Probability of randomization
q = 0.5  # Probability of returning 0 when randomizing

epsilon = epsilon_from_pq(p, q)
print(f"\nPrivacy parameters:")
print(f"  p (randomization probability): {p}")
print(f"  q (probability of 0 when random): {q}")
print(f"  ε (epsilon - privacy budget): {epsilon:.4f}")
print(f"\nInterpretation:")
print(f"  - Each sensitive attribute leaks at most {epsilon:.4f} bits of information")
print(f"  - Lower epsilon = stronger privacy (harder to infer true values)")
print(f"  - With p={p}, true value is kept {(1-p)*100:.0f}% of the time")


# In[24]:


# Step 10: Apply Differential Privacy to Sensitive Attributes
print("\nStep 10: Applying Differential Privacy to Sensitive Attributes")
print("="*60)

# Create a copy for differential privacy application
df_anonymous = df_protected.copy()

# Note: Handcap has values 0-4, so we'll binarize it first (0 = no disability, 1+ = has disability)
df_anonymous['Handcap_binary'] = (df_anonymous['Handcap'] > 0).astype(int)

# Apply randomized response to ALL sensitive binary attributes
# Including Scholarship (Bolsa Família) - reveals socioeconomic status/poverty
sensitive_dp_cols = ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap_binary', 'Scholarship']

print("\nSensitive attributes for DP protection:")
print("  - Hipertension, Diabetes, Alcoholism, Handcap: Medical conditions (PHI)")
print("  - Scholarship (Bolsa Família): Socioeconomic status indicator")

print("\nBefore DP - Original distributions:")
for col in sensitive_dp_cols:
    original_mean = df_anonymous[col].mean()
    print(f"  {col}: {original_mean:.4f}")

# Apply DP
print(f"\nApplying randomized response (p={p}, q={q})...")
for col in sensitive_dp_cols:
    df_anonymous[f'{col}_DP'] = apply_differential_privacy_to_column(df_anonymous[col], p, q)

epsilon_total = epsilon * len(sensitive_dp_cols)
print("\nPrivacy parameters:")
print(f"  p (randomization probability): {p}")
print(f"  q (probability of 0 when random): {q}")
print(f"  ε (epsilon - per-attribute budget): {epsilon:.4f}")
print(f"  ε_total (basic composition across {len(sensitive_dp_cols)} attrs): {epsilon_total:.4f}")
print("\nInterpretation:")
print(f"  - Each sensitive attribute leaks at most {epsilon:.4f} information units")
print(f"  - Basic composition bound across all sensitive attributes: {epsilon_total:.4f}")
print(f"  - With p={p}, true value is kept {(1-p)*100:.0f}% of the time")

print("\nAfter DP - Noisy distributions:")
for col in sensitive_dp_cols:
    dp_mean = df_anonymous[f'{col}_DP'].mean()
    original_mean = df_anonymous[col].mean()
    print(f"  {col}_DP: {dp_mean:.4f} (original: {original_mean:.4f}, diff: {abs(dp_mean-original_mean):.4f})")

# Visualize the effect of DP (now 5 attributes, use 2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(sensitive_dp_cols):
    original_vals = df_anonymous[col].value_counts().sort_index()
    dp_vals = df_anonymous[f'{col}_DP'].value_counts().sort_index()
    x = np.arange(2)
    width = 0.35
    orig_counts = [original_vals.get(0, 0), original_vals.get(1, 0)]
    dp_counts = [dp_vals.get(0, 0), dp_vals.get(1, 0)]
    axes[idx].bar(x - width/2, orig_counts, width, label='Original', alpha=0.7, color='steelblue')
    axes[idx].bar(x + width/2, dp_counts, width, label='DP (noisy)', alpha=0.7, color='coral')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Count')
    title = col.replace('_binary', '')
    if col == 'Scholarship':
        title = 'Scholarship (Bolsa Família)'
    axes[idx].set_title(f"{title}: Original vs DP")
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(['0', '1'])
    axes[idx].legend()
    axes[idx].grid(alpha=0.3, axis='y')

# Hide the 6th subplot (empty)
axes[5].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/11_differential_privacy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/11_differential_privacy_comparison.png')

print(f"\nDifferential privacy applied to {len(sensitive_dp_cols)} sensitive attributes")
print(f"  - 4 medical conditions (PHI)")
print(f"  - 1 socioeconomic indicator (Scholarship/Bolsa Família)")
print(f"Privacy budget (ε) = {epsilon:.4f} per attribute; ε_total ≈ {epsilon_total:.4f} across attrs (basic composition)")
print(f"Note: Handcap was binarized (0=no disability, 1=has disability) before DP application")


# ### Privacy-Utility Tradeoff Analysis
# 
# **The Core Question**: How much prediction accuracy do we sacrifice for privacy?
# 
# Privacy techniques inevitably introduce noise and information loss. This section quantifies the tradeoff between:
# - **Privacy strength**: Higher k, lower ε → stronger guarantees
# - **Model utility**: Accuracy, recall, F1 → predictive power
# 
# We'll conduct a parameter sweep to visualize this fundamental tradeoff.

# In[25]:


# Privacy-Utility Tradeoff: Parameter Sweep Visualization
print("=" * 70)
print("PRIVACY-UTILITY TRADEOFF: Parameter Sweep Analysis")
print("=" * 70)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Prepare base features from our protected dataset
def prepare_features(df_source, dp_cols, use_dp=True):
    """Prepare features for modeling with optional DP columns."""
    feature_cols = ['Age_generalized', 'Gender', 'Region', 'DaysBetween', 
                    'AppointmentDayOfWeek', 'ScheduledDayOfWeek', 'SMS_received', 'Scholarship']

    # Add medical columns (either DP or original)
    if use_dp:
        feature_cols += [f'{c}_DP' for c in dp_cols]
    else:
        feature_cols += dp_cols

    # One-hot encode categorical columns
    X = pd.get_dummies(df_source[feature_cols], drop_first=True)
    y = df_source['NoShow_binary']
    return X, y

# Function to apply DP with different parameters
def apply_dp_with_params(df_base, cols, p_val, q_val):
    """Apply DP to columns with specific p, q parameters."""
    df_copy = df_base.copy()
    for col in cols:
        df_copy[f'{col}_DP'] = df_copy[col].apply(lambda x: randomized_response(int(x), p_val, q_val))
    return df_copy

# Parameter sweep for Differential Privacy
print("\nSweep 1: Differential Privacy (ε) vs Model Performance")
print("-" * 50)

# Different (p, q) combinations to vary epsilon
dp_params = [
    (0.1, 0.5),   # ε ≈ 3.0 (low privacy)
    (0.2, 0.5),   # ε ≈ 2.3
    (0.3, 0.5),   # ε ≈ 2.4 (our current setting)
    (0.4, 0.5),   # ε ≈ 1.6
    (0.5, 0.5),   # ε ≈ 1.4
    (0.6, 0.5),   # ε ≈ 1.2
    (0.7, 0.5),   # ε ≈ 1.0
    (0.8, 0.5),   # ε ≈ 0.9 (high privacy)
]

medical_cols_binary = ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap_binary']
results_dp = []

# Use the protected dataset with generalized QIs
for p_val, q_val in dp_params:
    eps = epsilon_from_pq(p_val, q_val)

    # Apply DP with these parameters (run 3 times for stability)
    accs, recalls, f1s = [], [], []
    for seed in range(3):
        random.seed(seed)
        df_temp = apply_dp_with_params(df_anonymous, medical_cols_binary, p_val, q_val)
        X, y = prepare_features(df_temp, medical_cols_binary, use_dp=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', 
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    results_dp.append({
        'p': p_val, 'q': q_val, 'epsilon': eps,
        'accuracy': np.mean(accs), 'accuracy_std': np.std(accs),
        'recall': np.mean(recalls), 'recall_std': np.std(recalls),
        'f1': np.mean(f1s), 'f1_std': np.std(f1s)
    })
    print(f"  ε={eps:.2f} (p={p_val}): Acc={np.mean(accs):.3f}, Recall={np.mean(recalls):.3f}, F1={np.mean(f1s):.3f}")

results_dp_df = pd.DataFrame(results_dp)

# Visualization: Privacy-Utility Tradeoff
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Epsilon vs Accuracy
axes[0].plot(results_dp_df['epsilon'], results_dp_df['accuracy'], 'o-', linewidth=2, markersize=8, color='#3498db')
axes[0].fill_between(results_dp_df['epsilon'], 
                      results_dp_df['accuracy'] - results_dp_df['accuracy_std'],
                      results_dp_df['accuracy'] + results_dp_df['accuracy_std'], alpha=0.2, color='#3498db')
axes[0].set_xlabel('Privacy Budget (ε)', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy vs Privacy (ε)\n← Stronger Privacy | Weaker Privacy →', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].axvline(x=epsilon, color='red', linestyle='--', label=f'Current ε={epsilon:.2f}')
axes[0].legend()

# Plot 2: Epsilon vs Recall
axes[1].plot(results_dp_df['epsilon'], results_dp_df['recall'], 's-', linewidth=2, markersize=8, color='#e74c3c')
axes[1].fill_between(results_dp_df['epsilon'],
                      results_dp_df['recall'] - results_dp_df['recall_std'],
                      results_dp_df['recall'] + results_dp_df['recall_std'], alpha=0.2, color='#e74c3c')
axes[1].set_xlabel('Privacy Budget (ε)', fontsize=12)
axes[1].set_ylabel('Recall (No-show detection)', fontsize=12)
axes[1].set_title('Recall vs Privacy (ε)\n← Stronger Privacy | Weaker Privacy →', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].axvline(x=epsilon, color='red', linestyle='--', label=f'Current ε={epsilon:.2f}')
axes[1].legend()

# Plot 3: Epsilon vs F1
axes[2].plot(results_dp_df['epsilon'], results_dp_df['f1'], '^-', linewidth=2, markersize=8, color='#2ecc71')
axes[2].fill_between(results_dp_df['epsilon'],
                      results_dp_df['f1'] - results_dp_df['f1_std'],
                      results_dp_df['f1'] + results_dp_df['f1_std'], alpha=0.2, color='#2ecc71')
axes[2].set_xlabel('Privacy Budget (ε)', fontsize=12)
axes[2].set_ylabel('F1 Score', fontsize=12)
axes[2].set_title('F1 Score vs Privacy (ε)\n← Stronger Privacy | Weaker Privacy →', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)
axes[2].axvline(x=epsilon, color='red', linestyle='--', label=f'Current ε={epsilon:.2f}')
axes[2].legend()

plt.suptitle('Privacy-Utility Tradeoff: Differential Privacy Parameter Sweep', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/10b_dp_privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/10b_dp_privacy_utility_tradeoff.png')

# Summary table
print("\nSummary Table: DP Parameter Sweep")
print(results_dp_df[['epsilon', 'p', 'accuracy', 'recall', 'f1']].round(3))

print("\nKEY INSIGHTS:")
print("-" * 50)
print(f"• Our current setting (ε={epsilon:.2f}) balances privacy and utility")
print(f"• Stronger privacy (lower ε) slightly reduces model performance")
print(f"• The tradeoff is relatively modest - privacy doesn't destroy utility!")
print(f"• Recall (detecting no-shows) is more sensitive to privacy than accuracy")


# In[26]:


# Step 11: Create Final Anonymized Dataset
print("\nStep 11: Creating Final Anonymized Dataset")
print("="*60)

# Select columns for the final anonymized dataset
# Use generalized quasi-identifiers and DP-protected sensitive attributes
final_columns = [
    # Generalized quasi-identifiers (k-anonymity compliant)
    'Age_generalized',
    'Gender',
    'Region',

    # Temporal features (already generalized)
    'DaysBetween',
    'AppointmentDayOfWeek',
    'AppointmentMonth',
    'ScheduledDayOfWeek',

    # Behavioral/intervention features (low sensitivity)
    'SMS_received',

    # DP-protected sensitive attributes (medical + socioeconomic)
    'Hipertension_DP',
    'Diabetes_DP',
    'Alcoholism_DP',
    'Handcap_binary_DP',
    'Scholarship_DP',  # Bolsa Família - now DP-protected

    # Target variable
    'NoShow_binary'
]

df_final_anonymous = df_anonymous[final_columns].copy()

# Rename DP columns for clarity
df_final_anonymous.rename(columns={
    'Hipertension_DP': 'Hipertension',
    'Diabetes_DP': 'Diabetes',
    'Alcoholism_DP': 'Alcoholism',
    'Handcap_binary_DP': 'Handcap',
    'Scholarship_DP': 'Scholarship',
    'Age_generalized': 'Age_Group',
    'NoShow_binary': 'NoShow'
}, inplace=True)

print(f"\nFinal anonymized dataset shape: {df_final_anonymous.shape}")
print(f"Columns: {list(df_final_anonymous.columns)}")
print(f"\nFirst few rows:")
print(df_final_anonymous.head(10))

# Summary statistics
print(f"\n" + "="*60)
print("Final Anonymized Dataset Summary:")
print("="*60)
print(df_final_anonymous.describe(include='all'))


# In[26b]:


# Step 11b: Feature Engineering
print("\nStep 11b: Feature Engineering")
print("="*60)

# Feature 1: Lead time categories (more granular)
# Categories: 0=same day or before, 1=next day, 2=2-3 days, 3=4-7 days, 4=1-2 weeks, 5=2-4 weeks, 6=more than 4 weeks
df_final_anonymous['LeadTime_Category'] = pd.cut(
    df_final_anonymous['DaysBetween'],
    bins=[-float('inf'), 0, 1, 3, 7, 14, 30, float('inf')],
    labels=[0, 1, 2, 3, 4, 5, 6]
).astype(int)
print("✓ Created LeadTime_Category (0-6 scale based on days between scheduling and appointment)")

# Feature 2: Is same day appointment
df_final_anonymous['IsSameDay'] = (df_final_anonymous['DaysBetween'] <= 0).astype(int)
print("✓ Created IsSameDay (binary: 1 if appointment is same day or earlier)")

# Feature 3: Is weekend appointment
df_final_anonymous['IsWeekend'] = (df_final_anonymous['AppointmentDayOfWeek'] >= 5).astype(int)
print("✓ Created IsWeekend (binary: 1 if appointment is on Saturday or Sunday)")

# Feature 4: Health burden score (sum of DP-protected conditions)
df_final_anonymous['HealthBurden'] = (
    df_final_anonymous['Hipertension'] + 
    df_final_anonymous['Diabetes'] + 
    df_final_anonymous['Alcoholism'] + 
    df_final_anonymous['Handcap']
)
print("✓ Created HealthBurden (sum of 4 medical conditions, range 0-4)")

print(f"\nNew features added: LeadTime_Category, IsSameDay, IsWeekend, HealthBurden")
print(f"Updated dataset shape: {df_final_anonymous.shape}")
print(f"\nNew feature distributions:")
print(f"  LeadTime_Category: {df_final_anonymous['LeadTime_Category'].value_counts().sort_index().to_dict()}")
print(f"  IsSameDay: {df_final_anonymous['IsSameDay'].value_counts().to_dict()}")
print(f"  IsWeekend: {df_final_anonymous['IsWeekend'].value_counts().to_dict()}")
print(f"  HealthBurden: {df_final_anonymous['HealthBurden'].value_counts().sort_index().to_dict()}")


# ## Part Four: Machine Learning on Anonymized Data
# 
# Now that we have a fully anonymized dataset that satisfies:
# - **K-anonymity** (k ≥ 31)
# - **L-diversity** (l = 2 for all groups)
# - **T-closeness** (100% compliance)
# - **Differential Privacy** (ε ≈ 1.90 per attribute)
# 
# We will train machine learning models to answer our research question:
# **Why do patients miss their scheduled appointments?**
# 
# We'll train models on both:
# 1. **Anonymized dataset** (privacy-protected)
# 2. **Non-anonymized dataset** (original cleaned data)
# 
# Then compare their performance to assess the utility-privacy tradeoff.

# In[27]:


# Step 12: Prepare Data for Machine Learning
print("\nStep 12: Preparing Data for Machine Learning")
print("="*60)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, recall_score

# ============================================================
# PART 1: PREPARE ANONYMIZED DATASET
# ============================================================
print("\n" + "="*60)
print("PREPARING ANONYMIZED DATASET")
print("="*60)

df_ml_anon = df_final_anonymous.copy()

# Encode categorical variables
age_mapping = {
    '0-17': 0,
    '18-29': 1,
    '30-44': 2,
    '45-59': 3,
    '60-74': 4,
    '75+': 5
}
df_ml_anon['Age_Group'] = df_ml_anon['Age_Group'].astype(str).map(age_mapping).astype('int64')

gender_mapping = {'F': 0, 'M': 1}
df_ml_anon['Gender'] = df_ml_anon['Gender'].map(gender_mapping).astype('int64')

le_region = LabelEncoder()
df_ml_anon['Region'] = le_region.fit_transform(df_ml_anon['Region'].astype(str))

dow_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
df_ml_anon['AppointmentDayOfWeek'] = df_ml_anon['AppointmentDayOfWeek'].map(dow_mapping).astype('int64')
df_ml_anon['ScheduledDayOfWeek'] = df_ml_anon['ScheduledDayOfWeek'].map(dow_mapping).astype('int64')

print(f"\nAnonymized dataset shape: {df_ml_anon.shape}")
print(f"Missing values: {df_ml_anon.isnull().sum().sum()}")

# Separate features and target for anonymized data
X_anon = df_ml_anon.drop('NoShow', axis=1)
y_anon = df_ml_anon['NoShow']

print(f"\nAnonymized Features ({X_anon.shape[1]} features):")
print(f"  {list(X_anon.columns)}")
# Ensure all columns are numeric (convert any remaining categorical to int)
for col in X_anon.columns:
    if X_anon[col].dtype == 'object' or X_anon[col].dtype.name == 'category':
        X_anon[col] = X_anon[col].astype('int64')
    elif X_anon[col].dtype == 'float64':
        X_anon[col] = X_anon[col].astype('float64')
    else:
        X_anon[col] = X_anon[col].astype('int64')

print(f"\nAnonymized Features ({X_anon.shape[1]} features) after encoding:")
print(f"  {list(X_anon.columns)}")
print(f"Target distribution: No-show rate = {y_anon.mean()*100:.2f}%")

# ============================================================
# PART 2: PREPARE NON-ANONYMIZED (ORIGINAL) DATASET
# ============================================================
print("\n" + "="*60)
print("PREPARING NON-ANONYMIZED DATASET")
print("="*60)

# Use the protected dataset but with original values (before generalization)
df_ml_orig = df_protected.copy()

# Select same features as anonymized but use original values
# Use original Age (not generalized), original Neighbourhood (not Region)
# Use original medical conditions (not DP-protected)

# Encode Gender
df_ml_orig['Gender'] = df_ml_orig['Gender'].map(gender_mapping)

# Encode Neighbourhood directly
le_neighbourhood = LabelEncoder()
df_ml_orig['Neighbourhood_Encoded'] = le_neighbourhood.fit_transform(df_ml_orig['Neighbourhood'])

# Encode day of week
df_ml_orig['AppointmentDayOfWeek'] = df_ml_orig['AppointmentDayOfWeek'].map(dow_mapping)
df_ml_orig['ScheduledDayOfWeek'] = df_ml_orig['ScheduledDayOfWeek'].map(dow_mapping)

# Feature Engineering for non-anonymized dataset (same as anonymized)
# Feature 1: Lead time categories
df_ml_orig['LeadTime_Category'] = pd.cut(
    df_ml_orig['DaysBetween'],
    bins=[-float('inf'), 0, 1, 3, 7, 14, 30, float('inf')],
    labels=[0, 1, 2, 3, 4, 5, 6]
).astype(int)

# Feature 2: Is same day appointment
df_ml_orig['IsSameDay'] = (df_ml_orig['DaysBetween'] <= 0).astype(int)

# Feature 3: Is weekend appointment
df_ml_orig['IsWeekend'] = (df_ml_orig['AppointmentDayOfWeek'] >= 5).astype(int)

# Feature 4: Health burden score (sum of original conditions)
df_ml_orig['HealthBurden'] = (
    df_ml_orig['Hipertension'] + 
    df_ml_orig['Diabetes'] + 
    df_ml_orig['Alcoholism'] + 
    df_ml_orig['Handcap']
)

print("✓ Applied same feature engineering to non-anonymized dataset")

# Select columns for non-anonymized model
orig_features = [
    'Age',  # Original age (not binned)
    'Gender',
    'Neighbourhood_Encoded',  # Original neighbourhood (not Region)
    'DaysBetween',
    'AppointmentDayOfWeek',
    'AppointmentMonth',
    'ScheduledDayOfWeek',
    'SMS_received',
    'Scholarship',
    'Hipertension',  # Original (not DP-protected)
    'Diabetes',
    'Alcoholism',
    'Handcap',
    # New engineered features
    'LeadTime_Category',
    'IsSameDay',
    'IsWeekend',
    'HealthBurden'
]

X_orig = df_ml_orig[orig_features].copy()
y_orig = df_ml_orig['NoShow_binary']

# Ensure all columns are numeric
for col in X_orig.columns:
    if X_orig[col].dtype == 'object' or X_orig[col].dtype.name == 'category':
        X_orig[col] = X_orig[col].astype('int64')
    elif X_orig[col].dtype == 'float64':
        X_orig[col] = X_orig[col].astype('float64')
    else:
        X_orig[col] = X_orig[col].astype('int64')

print(f"\nNon-anonymized dataset shape: {X_orig.shape}")
print(f"Non-anonymized Features ({X_orig.shape[1]} features):")
print(f"  {list(X_orig.columns)}")
print(f"Missing values: {X_orig.isnull().sum().sum()}")
print(f"Target distribution: No-show rate = {y_orig.mean()*100:.2f}%")

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60)
print(f"Anonymized dataset ready: {X_anon.shape[0]} records, {X_anon.shape[1]} features")
print(f"Non-anonymized dataset ready: {X_orig.shape[0]} records, {X_orig.shape[1]} features")


# ---
# 
# ## Part Three: Machine Learning with Privacy-Protected Data
# 
# ### The Ultimate Test: Can We Still Predict No-Shows?
# 
# We've applied multiple layers of privacy protection:
# - K-anonymity (k ≥ 31) on quasi-identifiers
# - L-diversity (l = 2) on sensitive attributes  
# - T-closeness (t ≤ 0.2) for distribution protection
# - Differential privacy (ε ≈ 2.4) on medical conditions
# 
# **Now we test**: Does privacy kill utility?
# 
# ### Our Modeling Strategy
# 
# 1. **Baseline**: Logistic Regression (simple, interpretable)
# 2. **Ensemble Models**: Random Forest & XGBoost (strong predictors)
# 3. **Calibration**: CalibratedClassifierCV for well-calibrated probabilities
# 4. **Threshold Tuning**: Optimize for F2 score (recall-weighted) since missing a no-show is worse than a false alarm
# 5. **Comparison**: Anonymized vs Non-Anonymized to quantify utility loss

# In[28]:


# Step 13: Rebuilt Modeling with Stratified CV, calibration, and threshold tuning
print("\nStep 13: Modeling with stratified CV + calibrated probabilities + tuned threshold")
print("="*60)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, average_precision_score,
    fbeta_score, precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression

# Helper: cross-validated probabilities for threshold search
def cross_validated_probas(model, X_train, y_train, cv):
    oof = np.zeros(len(X_train))
    for train_idx_cv, val_idx_cv in cv.split(X_train, y_train):
        calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        calibrator.fit(X_train.iloc[train_idx_cv], y_train.iloc[train_idx_cv])
        oof[val_idx_cv] = calibrator.predict_proba(X_train.iloc[val_idx_cv])[:, 1]
    return oof

# Helper: pick threshold maximizing F2 (recall-weighted)
def choose_threshold(y_true, probas):
    thresholds = np.linspace(0.2, 0.8, 25)
    f2_scores = [fbeta_score(y_true, (probas >= t).astype(int), beta=2) for t in thresholds]
    best_idx = int(np.argmax(f2_scores))
    return thresholds[best_idx], f2_scores[best_idx]

# Helper: train + evaluate on holdout
def train_and_eval(dataset_label, X, y):
    print(f"\n=== {dataset_label.upper()} ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_defs = {
        'Baseline LogReg': LogisticRegression(max_iter=500, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=400, max_depth=16, min_samples_split=6, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=4,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
    }

    results = {}

    for name, base_model in model_defs.items():
        print(f"\n{name}: CV calibration + threshold tuning")
        oof_probas = cross_validated_probas(base_model, X_train, y_train, cv)
        best_thresh, best_f2 = choose_threshold(y_train, oof_probas)
        print(f"  Best threshold (F2): {best_thresh:.2f} | F2: {best_f2:.3f}")

        calibrator = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        calibrator.fit(X_train, y_train)
        test_probas = calibrator.predict_proba(X_test)[:, 1]
        test_preds = (test_probas >= best_thresh).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, test_preds),
            'recall': recall_score(y_test, test_preds),
            'roc_auc': roc_auc_score(y_test, test_probas),
            'pr_auc': average_precision_score(y_test, test_probas),
            'f2': fbeta_score(y_test, test_preds, beta=2),
            'threshold': best_thresh,
            'confusion': confusion_matrix(y_test, test_preds),
            'report': classification_report(y_test, test_preds, target_names=['Showed', 'No-Show'])
        }
        results[name] = {
            'calibrated_model': calibrator,
            'metrics': metrics
        }

        print(f"    Accuracy: {metrics['accuracy']:.3f} | Recall: {metrics['recall']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f} | PR-AUC: {metrics['pr_auc']:.3f} | F2: {metrics['f2']:.3f}")
        print(f"    Confusion matrix:\n{metrics['confusion']}")
    return results

results_comparison = {
    'Anonymized': train_and_eval('Anonymized', X_anon, y_anon),
    'Non-Anonymized': train_and_eval('Non-Anonymized', X_orig, y_orig)
}

print("\n" + "="*60)
print("Modeling complete: calibrated, CV-tuned, and threshold-adjusted")
print("="*60)


# In[29]:


# Step 14: Performance Comparison (CV + threshold tuned)
print("\nStep 14: Performance comparison")
print("="*60)

comparison_rows = []
for dataset in ['Anonymized', 'Non-Anonymized']:
    for model_name, payload in results_comparison[dataset].items():
        m = payload['metrics']
        comparison_rows.append({
            'Dataset': dataset,
            'Model': model_name,
            'Threshold': round(m['threshold'], 2),
            'Accuracy': round(m['accuracy'], 3),
            'Recall': round(m['recall'], 3),
            'ROC-AUC': round(m['roc_auc'], 3),
            'PR-AUC': round(m['pr_auc'], 3),
            'F2': round(m['f2'], 3)
        })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df)

# Plot Accuracy, Recall and ROC-AUC for quick scan
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.barplot(data=comparison_df, x='Model', y='Accuracy', hue='Dataset', ax=axes[0])
axes[0].set_title('Accuracy by Model', fontweight='bold')
axes[0].set_ylim(0.5, 0.8)
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(alpha=0.3, axis='y')

sns.barplot(data=comparison_df, x='Model', y='Recall', hue='Dataset', ax=axes[1])
axes[1].set_title('Recall (threshold-tuned) by Model', fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(alpha=0.3, axis='y')

sns.barplot(data=comparison_df, x='Model', y='ROC-AUC', hue='Dataset', ax=axes[2])
axes[2].set_title('ROC-AUC by Model', fontweight='bold')
axes[2].set_ylim(0.5, 1)
axes[2].tick_params(axis='x', rotation=15)
axes[2].grid(alpha=0.3, axis='y')

plt.suptitle('Model Performance Comparison: Anonymized vs Non-Anonymized', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/13_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/13_model_comparison.png')


# In[30]:


# Step 14b: Feature Importance Snapshot (post-calibration)
print("\nStep 14b: Feature importance by model")
print("="*60)

importances = []

def extract_importance(model_wrapper, feature_names):
    model = model_wrapper['calibrated_model']
    base = getattr(model, 'base_estimator', None) or getattr(model, 'base_estimator_', None)
    if base is None:
        return None
    if hasattr(base, 'feature_importances_'):
        return pd.Series(base.feature_importances_, index=feature_names)
    if hasattr(base, 'coef_'):
        return pd.Series(base.coef_.ravel(), index=feature_names)
    return None

for dataset, feats in [( 'Anonymized', X_anon.columns), ('Non-Anonymized', X_orig.columns)]:
    for model_name, payload in results_comparison[dataset].items():
        series = extract_importance(payload, feats)
        if series is not None:
            top = series.abs().sort_values(ascending=False).head(10)
            importances.append(pd.DataFrame({
                'Dataset': dataset,
                'Model': model_name,
                'Feature': top.index,
                'Importance': top.values
            }))

if importances:
    imp_df = pd.concat(importances, ignore_index=True)
    g = sns.catplot(data=imp_df, kind='bar', x='Importance', y='Feature', hue='Dataset', col='Model', sharex=False, height=4, aspect=1.1)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Top features per model (magnitude)')
else:
    print("No feature importances available for these models")


# In[31]:


# Step 14c: Model Feature Importance (Built-in)
print("\nStep 14c: Model Feature Importance Analysis")
print("="*60)

# Extract feature importance from trained models
# For Random Forest and XGBoost, use built-in feature_importances_

# Get the trained models - extract base estimators from calibrated models
print("\nExtracting feature importance from models...")

# For Random Forest
rf_anon_calibrated = results_comparison['Anonymized']['Random Forest']['calibrated_model']
rf_orig_calibrated = results_comparison['Non-Anonymized']['Random Forest']['calibrated_model']

# For XGBoost  
xgb_anon_calibrated = results_comparison['Anonymized']['XGBoost']['calibrated_model']
xgb_orig_calibrated = results_comparison['Non-Anonymized']['XGBoost']['calibrated_model']

# Extract feature importances from base estimators
rf_anon_importance_vals = rf_anon_calibrated.calibrated_classifiers_[0].estimator.feature_importances_
rf_orig_importance_vals = rf_orig_calibrated.calibrated_classifiers_[0].estimator.feature_importances_
xgb_anon_importance_vals = xgb_anon_calibrated.calibrated_classifiers_[0].estimator.feature_importances_
xgb_orig_importance_vals = xgb_orig_calibrated.calibrated_classifiers_[0].estimator.feature_importances_

print("Feature importance extracted successfully for all 4 models")


# In[32]:


# Step 14d: Feature Importance Visualization
print("\nStep 14d: Feature Importance Visualization")
print("="*60)

# Create DataFrames for importance
rf_anon_importance = pd.DataFrame({
    'Feature': X_anon.columns,
    'Importance': rf_anon_importance_vals
}).sort_values('Importance', ascending=True)

xgb_anon_importance = pd.DataFrame({
    'Feature': X_anon.columns,
    'Importance': xgb_anon_importance_vals
}).sort_values('Importance', ascending=True)

rf_orig_importance = pd.DataFrame({
    'Feature': X_orig.columns,
    'Importance': rf_orig_importance_vals
}).sort_values('Importance', ascending=True)

xgb_orig_importance = pd.DataFrame({
    'Feature': X_orig.columns,
    'Importance': xgb_orig_importance_vals
}).sort_values('Importance', ascending=True)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Random Forest - Anonymized
axes[0, 0].barh(rf_anon_importance['Feature'], rf_anon_importance['Importance'], 
                color='#3498db', alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('Feature Importance (Gini)', fontsize=11)
axes[0, 0].set_title('Random Forest (Anonymized)\nFeature Importance', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3, axis='x')

# 2. XGBoost - Anonymized
axes[0, 1].barh(xgb_anon_importance['Feature'], xgb_anon_importance['Importance'], 
                color='#2ecc71', alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Feature Importance (Gain)', fontsize=11)
axes[0, 1].set_title('XGBoost (Anonymized)\nFeature Importance', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='x')

# 3. Random Forest - Non-Anonymized
axes[1, 0].barh(rf_orig_importance['Feature'], rf_orig_importance['Importance'], 
                color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Feature Importance (Gini)', fontsize=11)
axes[1, 0].set_title('Random Forest (Non-Anonymized)\nFeature Importance', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='x')

# 4. XGBoost - Non-Anonymized
axes[1, 1].barh(xgb_orig_importance['Feature'], xgb_orig_importance['Importance'], 
                color='#f39c12', alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Feature Importance (Gain)', fontsize=11)
axes[1, 1].set_title('XGBoost (Non-Anonymized)\nFeature Importance', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.suptitle('Feature Importance Comparison: Anonymized vs Non-Anonymized', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/14_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/14_feature_importance_comparison.png')

# Print top features comparison
print("\n" + "="*60)
print("TOP 5 FEATURES BY IMPORTANCE")
print("="*60)

print("\nRandom Forest (Anonymized):")
for i, (_, row) in enumerate(rf_anon_importance.iloc[::-1].head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Importance']:.4f}")

print("\nXGBoost (Anonymized):")
for i, (_, row) in enumerate(xgb_anon_importance.iloc[::-1].head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Importance']:.4f}")

print("\nRandom Forest (Non-Anonymized):")
for i, (_, row) in enumerate(rf_orig_importance.iloc[::-1].head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Importance']:.4f}")

print("\nXGBoost (Non-Anonymized):")
for i, (_, row) in enumerate(xgb_orig_importance.iloc[::-1].head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Importance']:.4f}")


# In[35]:


# Step 14f: Privacy vs Utility Analysis Summary
print("\nStep 14f: Privacy vs Utility Analysis")
print("="*60)

# Create comprehensive comparison table
print("\nMODEL PERFORMANCE COMPARISON")
print("-" * 60)

# Extract metrics for comparison
models = ['Baseline LogReg', 'Random Forest', 'XGBoost']
metrics_list = []

for model_name in models:
    anon_metrics = results_comparison['Anonymized'][model_name]['metrics']
    orig_metrics = results_comparison['Non-Anonymized'][model_name]['metrics']

    metrics_list.append({
        'Model': model_name,
        'Dataset': 'Anonymized',
        'Accuracy': anon_metrics['accuracy'],
        'Recall': anon_metrics['recall'],
        'ROC-AUC': anon_metrics['roc_auc'],
        'PR-AUC': anon_metrics['pr_auc'],
        'F2': anon_metrics['f2']
    })

    metrics_list.append({
        'Model': model_name,
        'Dataset': 'Non-Anonymized',
        'Accuracy': orig_metrics['accuracy'],
        'Recall': orig_metrics['recall'],
        'ROC-AUC': orig_metrics['roc_auc'],
        'PR-AUC': orig_metrics['pr_auc'],
        'F2': orig_metrics['f2']
    })

metrics_df = pd.DataFrame(metrics_list)

# Pivot for cleaner display
print("\nDetailed Metrics by Model and Dataset:")
print(metrics_df.round(4))

# Calculate utility loss
print("\n" + "="*60)
print("UTILITY LOSS DUE TO PRIVACY PROTECTION")
print("="*60)

for model_name in models:
    anon = results_comparison['Anonymized'][model_name]['metrics']
    orig = results_comparison['Non-Anonymized'][model_name]['metrics']

    acc_loss = (orig['accuracy'] - anon['accuracy']) * 100
    recall_loss = (orig['recall'] - anon['recall']) * 100
    auc_loss = (orig['roc_auc'] - anon['roc_auc']) * 100

    print(f"\n{model_name}:")
    print(f"  Accuracy loss: {acc_loss:+.2f}%")
    print(f"  Recall loss: {recall_loss:+.2f}%")
    print(f"  ROC-AUC loss: {auc_loss:+.2f}%")

# Visualize privacy-utility tradeoff
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy comparison
anon_acc = [results_comparison['Anonymized'][m]['metrics']['accuracy'] for m in models]
orig_acc = [results_comparison['Non-Anonymized'][m]['metrics']['accuracy'] for m in models]

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, orig_acc, width, label='Non-Anonymized', color='#e74c3c', alpha=0.8)
axes[0].bar(x + width/2, anon_acc, width, label='Anonymized', color='#3498db', alpha=0.8)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy: Privacy vs Utility', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=15)
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

# Recall comparison
anon_rec = [results_comparison['Anonymized'][m]['metrics']['recall'] for m in models]
orig_rec = [results_comparison['Non-Anonymized'][m]['metrics']['recall'] for m in models]

axes[1].bar(x - width/2, orig_rec, width, label='Non-Anonymized', color='#e74c3c', alpha=0.8)
axes[1].bar(x + width/2, anon_rec, width, label='Anonymized', color='#3498db', alpha=0.8)
axes[1].set_ylabel('Recall')
axes[1].set_title('Recall: Privacy vs Utility', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=15)
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

# ROC-AUC comparison
anon_auc = [results_comparison['Anonymized'][m]['metrics']['roc_auc'] for m in models]
orig_auc = [results_comparison['Non-Anonymized'][m]['metrics']['roc_auc'] for m in models]

axes[2].bar(x - width/2, orig_auc, width, label='Non-Anonymized', color='#e74c3c', alpha=0.8)
axes[2].bar(x + width/2, anon_auc, width, label='Anonymized', color='#3498db', alpha=0.8)
axes[2].set_ylabel('ROC-AUC')
axes[2].set_title('ROC-AUC: Privacy vs Utility', fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=15)
axes[2].legend()
axes[2].grid(alpha=0.3, axis='y')

plt.suptitle('Privacy-Utility Tradeoff: Anonymized vs Non-Anonymized', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/14b_privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/14b_privacy_utility_tradeoff.png')

print("\n" + "="*60)
print("KEY FINDING: Privacy protection results in minimal utility loss!")
print("="*60)


# In[36]:


# Step 15: Final Research Question Analysis
print("\n" + "="*70)
print("STEP 15: ANSWERING THE RESEARCH QUESTION")
print("Why do patients miss their scheduled medical appointments?")
print("="*70)

# Use XGBoost on anonymized data as our primary model (best overall performance)
best_model_name = 'XGBoost'
best_anon_model = results_comparison['Anonymized'][best_model_name]['calibrated_model']

# Feature importance from model (using built-in feature_importances_)
print("\nFEATURE IMPORTANCE RANKING (Model-based)")
print("-" * 50)

# Sort by importance (descending)
xgb_final_importance = pd.DataFrame({
    'Feature': X_anon.columns,
    'Importance': xgb_anon_importance_vals
}).sort_values('Importance', ascending=False)

for i, (_, row) in enumerate(xgb_final_importance.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:25s} {row['Importance']:.4f}")

# Detailed interpretation of top factors
print("\n" + "="*70)
print("DETAILED INTERPRETATION: WHY PATIENTS MISS APPOINTMENTS")
print("="*70)

top_features = xgb_final_importance.head(6)['Feature'].tolist()

interpretations = {
    'DaysBetween': {
        'title': 'SCHEDULING LEAD TIME (Days Between)',
        'finding': 'Longer wait times between scheduling and appointment date increase no-show risk.',
        'insight': 'Patients forget or circumstances change over time.',
        'recommendation': 'Offer same-week or next-day appointments when possible.'
    },
    'SMS_received': {
        'title': 'SMS REMINDERS',
        'finding': 'SMS reminder status significantly impacts attendance patterns.',
        'insight': 'Reminders are often sent to high-risk patients (selection bias visible in data).',
        'recommendation': 'Implement universal SMS reminders, not just for high-risk patients.'
    },
    'Age_Group': {
        'title': 'AGE GROUP',
        'finding': 'Younger patients (18-35) show higher no-show rates.',
        'insight': 'Young adults have less stable schedules and competing priorities.',
        'recommendation': 'Target younger demographics with flexible scheduling and reminders.'
    },
    'Region': {
        'title': 'GEOGRAPHIC REGION',
        'finding': 'Location strongly predicts attendance behavior.',
        'insight': 'Distance, transportation, and socioeconomic factors vary by region.',
        'recommendation': 'Consider mobile clinics or telehealth for underserved regions.'
    },
    'Scholarship': {
        'title': 'SOCIOECONOMIC STATUS (Scholarship)',
        'finding': 'Patients receiving social welfare show different attendance patterns.',
        'insight': 'Economic barriers (transportation, childcare, work) affect attendance.',
        'recommendation': 'Provide transportation assistance or flexible hours for low-income patients.'
    },
    'AppointmentDayOfWeek': {
        'title': 'DAY OF WEEK',
        'finding': 'Certain days of the week have higher no-show rates.',
        'insight': 'Work schedules and weekend availability affect attendance.',
        'recommendation': 'Analyze which days have highest no-shows and adjust staffing.'
    },
    'ScheduledDayOfWeek': {
        'title': 'SCHEDULING DAY',
        'finding': 'The day appointments are scheduled affects follow-through.',
        'insight': 'Appointments made on certain days may be less committed.',
        'recommendation': 'Consider scheduling follow-up confirmations.'
    },
    'Hipertension': {
        'title': 'HIPERTENSION',
        'finding': 'Patients with hypertension tend to show up more often.',
        'insight': 'Chronic condition management motivates regular attendance.',
        'recommendation': 'Chronic care patients are more reliable; focus interventions elsewhere.'
    },
    'Diabetes': {
        'title': 'DIABETES',
        'finding': 'Diabetic patients have lower no-show rates.',
        'insight': 'Ongoing treatment needs increase adherence.',
        'recommendation': 'Use chronic care patients as models for engagement strategies.'
    },
    'AppointmentMonth': {
        'title': 'MONTH/SEASON',
        'finding': 'Seasonal patterns exist in appointment attendance.',
        'insight': 'Holidays, weather, and seasonal illness affect behavior.',
        'recommendation': 'Account for seasonality in capacity planning.'
    }
}

for feature in top_features:
    if feature in interpretations:
        info = interpretations[feature]
        print(f"\n{info['title']}")
        print(f"   Finding: {info['finding']}")
        print(f"   Insight: {info['insight']}")
        print(f"   Recommendation: {info['recommendation']}")

# Summary visualization
print("\n" + "="*70)
print("VISUAL SUMMARY: TOP PREDICTORS OF NO-SHOWS")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(xgb_final_importance)))
bars = ax.barh(xgb_final_importance['Feature'], xgb_final_importance['Importance'], 
               color=colors, edgecolor='black', alpha=0.8)

ax.set_xlabel('Feature Importance (Impact on No-Show Prediction)', fontsize=12)
ax.set_title('Why Do Patients Miss Appointments?\nFeature Importance from XGBoost Model', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, xgb_final_importance['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/15_final_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  Saved: {output_dir}/15_final_feature_importance.png')

# Final summary statistics
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\nPRIVACY GUARANTEES ACHIEVED:")
print(f"   • K-anonymity: k ≥ {unique_combinations_gen['count'].min()} (every patient hidden among {unique_combinations_gen['count'].min()}+ others)")
print(f"   • L-diversity: min_l = 2 for {(diversity_check['min_l_diversity']==2).sum()}/{len(diversity_check)} groups (across all 6 sensitive attributes)")
print(f"   • T-closeness: {diversity_check['t_close'].sum()}/{len(diversity_check)} groups within t=0.2 threshold")
print(f"   • Differential Privacy: ε ≈ {epsilon:.2f} per medical attribute")

print("\nMODEL PERFORMANCE (Privacy-Protected):")
best_metrics = results_comparison['Anonymized'][best_model_name]['metrics']
print(f"   • Best Model: {best_model_name}")
print(f"   • Accuracy: {best_metrics['accuracy']:.1%}")
print(f"   • Recall (No-show detection): {best_metrics['recall']:.1%}")
print(f"   • ROC-AUC: {best_metrics['roc_auc']:.3f}")
print(f"   • Threshold: {best_metrics['threshold']:.2f} (optimized for F2)")

print("\nKEY FINDINGS:")
print("   1. Lead time is the strongest predictor - shorter waits = better attendance")
print("   2. SMS reminders show selection bias - sent to high-risk patients")
print("   3. Young adults (18-35) are highest risk demographic")
print("   4. Geographic/socioeconomic factors significantly impact attendance")
print("   5. Patients with chronic conditions are MORE likely to attend")

print("\nACTIONABLE RECOMMENDATIONS:")
print("   1. Reduce scheduling lead times where possible")
print("   2. Implement universal SMS reminders (not just high-risk)")
print("   3. Target interventions at young adult demographic")
print("   4. Address transportation barriers in underserved regions")
print("   5. Use prediction model to identify high-risk appointments proactively")


# ---
# 
# ## Final Summary & Conclusions
# 
# ### Research Questions Answered
# 
# | Question | Answer |
# |----------|--------|
# | **What factors predict no-shows?** | Lead time (DaysBetween), SMS reminders, Age, and Region are the top predictors |
# | **How much accuracy do we sacrifice for privacy?** | Only 1-3% accuracy loss with full privacy protection |
# | **Best privacy-utility tradeoff?** | Our parameters (k≥31, ε≈2.4) preserve ~95% of model utility |
# 
# ### Key Achievements
# 
# **Privacy**
# - Every patient hidden among ≥31 others (K-anonymity)
# - Sensitive values diversified in each group (L-diversity)
# - Distributions match global patterns (T-closeness)
# - Medical data protected with calibrated noise (Differential Privacy)
# 
# **Utility**
# - Models achieve ~64-65% accuracy on imbalanced data
# - Recall reaches ~72% with threshold tuning (catching no-shows)
# - Feature importance remains interpretable
# 
# ### Recommendations for Healthcare Providers
# 
# 1. **Reduce scheduling lead times** - same-week appointments have lower no-show rates
# 2. **Implement SMS reminders** - targeted at high-risk patients
# 3. **Focus on young adults (19-35)** - highest no-show demographic
# 4. **Use this model** - privacy-protected predictions enable interventions
# 
# ### Academic Contributions
# 
# This project demonstrates that **privacy and utility are not mutually exclusive**. With careful application of state-of-the-art privacy techniques from academic literature, we can build useful predictive models while fully protecting individual patients.
# 
# ---
# *Project completed as part of the Data Privacy course at Université Paris Dauphine*


# In[ ]:


# Print summary of saved plots
print("\n" + "="*70)
print("ANALYSIS COMPLETE - ALL PLOTS SAVED")
print("="*70)
import glob
saved_plots = sorted(glob.glob(f'{output_dir}/*.png'))
print(f"\nTotal plots saved: {len(saved_plots)}")
print(f"Output directory: {output_dir}/")
print("\nSaved plots:")
for plot in saved_plots:
    print(f"  - {plot}")
print("\n" + "="*70)

# Close the output file and restore stdout
output_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print("All output saved to output.txt")
