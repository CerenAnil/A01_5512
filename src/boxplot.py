from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd
import os

housing = fetch_california_housing(as_frame=True)
df = housing.frame

columns = ["HouseAge"]

fig, ax = plt.subplots(figsize=(10, 6))
df[columns].boxplot(ax=ax)
ax.set_title("California Housing Dataset - Feature Distributions")
ax.set_ylabel("Value")
ax.set_xlabel("Feature")

os.makedirs("figs", exist_ok=True)
fig.savefig("figs/boxplot.png", bbox_inches="tight")
print("Saved figs/boxplot.png")
