# =====================================================
# Mobile Game In-App Purchases A/B Testing Analysis
# =====================================================

# ------------------------------
# Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# ------------------------------
# Step 1: Load Dataset (LOCAL PATH)
# ------------------------------
DATA_PATH = r"E:\Kaggle Dataset\mobile_game_in_app_purchases.csv"

# Verify file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("CSV file not found. Check file name and path.")

df = pd.read_csv(DATA_PATH)

# ------------------------------
# Step 2: Initial Inspection
# ------------------------------
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------
# Step 3: Data Cleaning
# ------------------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Gender'] = df['Gender'].fillna('Unknown')
df['Device'] = df['Device'].fillna(df['Device'].mode()[0])
df['InAppPurchaseAmount'] = df['InAppPurchaseAmount'].fillna(0)

categorical_cols = [
    'Country',
    'GameGenre',
    'SpendingSegment',
    'PaymentMethod'
]

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ------------------------------
# Step 4: Exploratory Data Analysis
# ------------------------------
plt.figure()
sns.histplot(df['InAppPurchaseAmount'], bins=50, kde=True)
plt.title("Distribution of In-App Purchase Amounts")
plt.xlabel("Purchase Amount ($)")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.countplot(
    x='SpendingSegment',
    data=df,
    order=['Whale', 'Dolphin', 'Minnow']
)
plt.title("Spending Segment Distribution")
plt.show()

plt.figure()
sns.boxplot(x='Device', y='InAppPurchaseAmount', data=df)
plt.title("Revenue by Device")
plt.show()

# ------------------------------
# Step 5: A/B Testing (iOS vs Android)
# ------------------------------
group_ios = df[df['Device'] == 'iOS']['InAppPurchaseAmount']
group_android = df[df['Device'] == 'Android']['InAppPurchaseAmount']

plt.figure()
sns.boxplot(
    x='Device',
    y='InAppPurchaseAmount',
    data=df[df['Device'].isin(['iOS', 'Android'])]
)
plt.title("A/B Test: iOS vs Android Revenue")
plt.show()

# Normality Test
shapiro_ios = stats.shapiro(group_ios.sample(500, random_state=1)) if len(group_ios) > 500 else stats.shapiro(group_ios)
shapiro_android = stats.shapiro(group_android.sample(500, random_state=1)) if len(group_android) > 500 else stats.shapiro(group_android)

print("\nShapiro Test Results:")
print("iOS:", shapiro_ios)
print("Android:", shapiro_android)

# Statistical Test
if shapiro_ios.pvalue > 0.05 and shapiro_android.pvalue > 0.05:
    test_name = "Independent T-Test"
    stat, p_value = stats.ttest_ind(group_ios, group_android, equal_var=False)
else:
    test_name = "Mann-Whitney U Test"
    stat, p_value = stats.mannwhitneyu(group_ios, group_android, alternative='two-sided')

print(f"\n{test_name} Results")
print("Statistic:", stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("✅ Statistically significant difference between iOS and Android spending.")
else:
    print("❌ No statistically significant difference between iOS and Android spending.")

# ------------------------------
# Step 6: Business Insights
# ------------------------------
print("\nAverage Revenue by Game Genre:")
print(df.groupby('GameGenre')['InAppPurchaseAmount'].mean().sort_values(ascending=False))

print("\nAverage Revenue by Spending Segment:")
print(df.groupby('SpendingSegment')['InAppPurchaseAmount'].mean())

pivot = df.pivot_table(
    values='InAppPurchaseAmount',
    index='Country',
    columns='Device',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".2f")
plt.title("Average Revenue by Country and Device")
plt.show()
