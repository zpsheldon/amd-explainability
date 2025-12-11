# Exploratory Data Analysis for B2AI Voice Dataset
# Author: Imran Isa-Dutse
# Goal: Explore acoustic and phenotype data to understand relationships linked to voice disorders.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load data
features = pd.read_csv('static_features.tsv', sep='\t')
labels = pd.read_csv('phenotype.tsv', sep='\t')

# Merge on participant_id
df = pd.merge(features, labels, on='participant_id', how='inner')
print(f"Merged dataset shape: {df.shape}")

# 2. Create voice disorder label
voice_disorder_cols = ['laryng_cancer', 'benign_cord_lesion', 'rrp', 'spas_dys', 'voc_fold_paralysis']
for col in voice_disorder_cols:
    if col not in df.columns:
        print(f"Warning: column {col} not found.")

df['voice_disorder'] = df[voice_disorder_cols].any(axis=1).astype(int)

# 3. Inspect target distribution
sns.countplot(x='voice_disorder', data=df)
plt.title('Voice Disorder Distribution')
plt.show()

# 4. Missing data overview
plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False)
plt.title('Missing Data Heatmap')
plt.show()

# 5. Basic statistics
print(df.describe().T)

# 6. Correlation heatmap for acoustic features
phenotype_cols_to_exclude = list(labels.columns)
other_cols_to_exclude = ['session_id', 'voice_disorder', 'task_name', 'transcription']
all_exclusions = phenotype_cols_to_exclude + other_cols_to_exclude
acoustic_cols = [col for col in df.columns if col not in all_exclusions]
plt.figure(figsize=(12,8))
sns.heatmap(df[acoustic_cols].corr().abs(), cmap='coolwarm', center=0, cbar_kws={'shrink':.8})
plt.title('Feature Correlation Heatmap')
plt.show()

# 7. PCA visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[acoustic_cols].select_dtypes(include=np.number).fillna(0))
y = df['voice_disorder']

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection of Voice Features')
plt.show()

print('Explained variance ratio:', pca.explained_variance_ratio_)

# 8. Key feature comparisons
key_features = ['mean_f0_hertz', 'jitterLocal_sma3nz_amean', 'shimmerLocaldB_sma3nz_amean', 'HNRdBACF_sma3nz_amean']

for feat in key_features:
    if feat in df.columns:
        plt.figure(figsize=(6,4))
        sns.kdeplot(data=df, x=feat, hue='voice_disorder', fill=True)
        plt.title(f'Distribution of {feat} by Voice Disorder')
        plt.show()
    else:
        print(f"Feature {feat} not found.")

# 9. Feature importance (quick baseline)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create a clean dataframe by dropping rows with missing values in acoustic features
clean_df = df.dropna(subset=acoustic_cols)

# Now, define X_clean and y_clean from this clean dataframe
# This ensures X and y are perfectly aligned
X_clean = clean_df[acoustic_cols].select_dtypes(include=np.number)
y_clean = clean_df['voice_disorder']

print(f"Shape of X_clean: {X_clean.shape}")
print(f"Shape of y_clean: {y_clean.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X_clean.columns).sort_values(ascending=False)[:15]
plt.figure(figsize=(8,5))
importances.plot(kind='barh')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.show()

print("EDA complete. Proceed to modeling next (Logistic Regression / XGBoost / LSTM).")
