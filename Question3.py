import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine = load_wine()

df=pd.DataFrame(wine.data, columns=wine.feature_names)

corr_matrix=df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Wine Dataset")
plt.show()

corr=corr_matrix.copy()

import numpy as np
np.fill_diagonal(corr.values.np.nan)

max_corr= corr.max().max()

feature_pair = np.where(corr==max_corr)

feature1=corr.index[feature_pair[0][0]]
feature2=corr.columns[feature_pair[1][0]]

print("\n ---Strongest Positive Corrrealtion--- ")
print("Feature 1 : ", feature1 )
print("Feature 2 : ", feature2 )
print("Correlation: ", max_corr)