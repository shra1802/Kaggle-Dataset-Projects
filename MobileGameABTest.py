# ------------------------------
# Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import kagglehub

# ------------------------------
# Step 1: Download the dataset
# ------------------------------
path = kagglehub.dataset_download("pratyushpuri/mobile-game-in-app-purchases-dataset-2025")
print("Path to dataset files:", path)

# ------------------------------
# Step 2: Load dataset
# ------------------------------
# Replace with actual CSV file name inside the downloaded folder
df = pd.read_csv(f"{path}/mobile_game_in_app_purchases.csv")

# Quick overview
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# ------------------------------
# Step 3: Data Cleaning
# ------------------------------
# Fill missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Gender with 'Unknown'
df['Gender'] = df['Gender'].fillna('Unknown')

# Fill missing Device with mode
df['Device'] = df['Device'].fillna(df['Device'].mode()[0])

# Fill missing InAppPurchaseAmount with 0
df['InAppPurchaseAmount'] = df['InAppPurchaseAmount'].fillna(0)

# Fill other categorical columns with 'Unknown'
categorical_cols = ['Country', 'GameGenre', 'SpendingSegment', 'PaymentMethod']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# ------------------------------
# Step 4: EDA - Revenue & Spending Segment
# ------------------------------
sns.histplot(df['InAppPurchaseAmount'], bins=50, kde=True)
plt.title("Distribution of In-App Purchase Amounts")
plt.xlabel("Purchase Amount ($)")
plt.ylabel("Frequency")
plt.show()

sns.countplot(x='SpendingSegment', data=df, order=['Whale', 'Dolphin', 'Minnow'])
plt.title("Spending Segment Distribution")
plt.show()

# Revenue by Device
sns.boxplot(x='Device', y='InAppPurchaseAmount', data=df)
plt.title("Revenue by Device")
plt.show()

# ------------------------------
# Step 5: A/B Testing Example
# Compare Revenue between iOS and Android users
# ------------------------------
group_ios = df[df['Device'] == 'iOS']['InAppPurchaseAmount']
group_android = df[df['Device'] == 'Android']['InAppPurchaseAmount']

# Visual comparison
sns.boxplot(x='Device', y='InAppPurchaseAmount', data=df[df['Device'].isin(['iOS','Android'])])
plt.title("In-App Purchases: iOS vs Android")
plt.show()

# Check normality
shapiro_ios = stats.shapiro(group_ios)
shapiro_android = stats.shapiro(group_android)
print("Shapiro Test iOS:", shapiro_ios)
print("Shapiro Test Android:", shapiro_android)

# Perform t-test if normal, else Mann-Whitney U
if shapiro_ios.pvalue > 0.05 and shapiro_android.pvalue > 0.05:
    t_stat, p_val = stats.ttest_ind(group_ios, group_android, equal_var=False)
    print("\nT-test results: t-statistic =", t_stat, "p-value =", p_val)
else:
    u_stat, p_val = stats.mannwhitneyu(group_ios, group_android, alternative='two-sided')
    print("\nMann-Whitney U test: U-statistic =", u_stat, "p-value =", p_val)

if p_val < 0.05:
    print("Significant difference in spending between iOS and Android users.")
else:
    print("No significant difference in spending between iOS and Android users.")

# ------------------------------
# Step 6: Advanced Insights (Optional)
# ------------------------------
# Average spending by GameGenre
genre_revenue = df.groupby('GameGenre')['InAppPurchaseAmount'].mean().sort_values(ascending=False)
print("\nAverage Revenue by Game Genre:")
print(genre_revenue)

# Average spending by Spending Segment
segment_revenue = df.groupby('SpendingSegment')['InAppPurchaseAmount'].mean()
print("\nAverage Revenue by Spending Segment:")
print(segment_revenue)

# Revenue heatmap by Country and Device
pivot = df.pivot_table(values='InAppPurchaseAmount', index='Country', columns='Device', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title("Average Revenue by Country and Device")
plt.show()
