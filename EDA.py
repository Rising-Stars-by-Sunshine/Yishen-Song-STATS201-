#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Adjust the path as necessary
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('iris.data', header=None, names=column_names)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Unique species in the dataset
print("\nUnique species in the dataset:")
print(df['species'].unique())

# Pairplot to visualize feature relationships
print("\nGenerating pairplot...")
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# Correlation heatmap
print("\nGenerating correlation heatmap...")
correlation_matrix = df.iloc[:, :-1].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot for each feature grouped by species
print("\nGenerating boxplots for each feature grouped by species...")
for column in column_names[:-1]:  # Exclude 'species'
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='species', y=column, data=df)
    plt.title(f"Boxplot of {column} by Species")
    plt.show()

# Histograms for feature distributions
print("\nGenerating histograms for feature distributions...")
df.hist(bins=15, figsize=(10, 8), edgecolor='black')
plt.suptitle("Feature Distribution Histograms")
plt.show()


# In[ ]:




