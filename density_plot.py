import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


df = pd.read_csv('df_joined_clean.csv')
# print(df.shape)
# print(df.head())

# Calculate likes per download

df["likes_per_download"] = df["likes"] / df["downloads"].replace(0, pd.NA)
df["likes_per_download"].describe()

# Set seaborn theme
sns.set_theme(style="ticks", palette="pastel", font_scale=1.1)

# Color for the density plot

blue = "#A1C9F4"

# Density plot with scatter overlay

plt.figure(figsize=(8, 5))
sns.kdeplot(
    x=df["co₂ cost"],
    y=df["likes_per_download"],
    fill=True,
    color=blue,
    alpha=0.6,
    levels=20,
    thresh=0,
    cut=1
)
plt.xlim(0, 80)
sns.scatterplot(
    x=df["co₂ cost"],
    y=df["likes_per_download"],
    color="black",
    s=25,
    alpha=0.6
)

# zone of best in class llms

zone = patches.Rectangle(
    (df["co₂ cost"].min(), df["likes_per_download"].quantile(0.5)),   # lower left corner
    df["co₂ cost"].quantile(0.5) - df["co₂ cost"].min(),             # width
    df["likes_per_download"].max() - df["likes_per_download"].quantile(0.5),  # height
    linewidth=1.8,
    edgecolor="#4CAF50",   # kräftiges Grün
    facecolor="#4CAF50",
    alpha=0.25,
    label="Ballanced best zone"
)

ax = plt.gca()
ax.add_patch(zone)

plt.xlabel("CO2-Cost")
plt.ylabel("Likes / Download")
plt.title("Density distribution 'Likes / Download' vs. 'CO2 costs'")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

plt.savefig("figures/density_plot.png", dpi=300, bbox_inches="tight")
