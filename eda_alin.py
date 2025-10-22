import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


df_all = pd.read_csv("huggingface_llm_metadata.csv")
#df_100 = pd.read_csv("huggingface_100_llm_metadata.csv")
df_lb = pd.read_csv("hf_leaderboard.csv")

df_all.duplicated().sum()
df_lb.duplicated().sum()


# df_lb has duplicate rows
duplicates=df_lb[df_lb.duplicated(keep=False)]
duplicates.sort_values("Rank", ascending=True, inplace=True)
print(duplicates)

# drop'em

df_lb = df_lb.sort_values("Rank", ascending=True)
df_lb = df_lb.drop_duplicates(subset=["Model"], keep="first").reset_index(drop=True)
df_lb.duplicated().sum() 

# all cleaned 


# this is not needed as in postitron you can see the data similar to R
display(df_all.head())  
print(df_all.columns.tolist())
print(df_all.isna().sum().sort_values(ascending=False).head(10))
print(df_lb.isna().sum().sort_values(ascending=False).head(10))


#intersections

print(set(df_all.columns).intersection(df_lb.columns))


# creting a function to have same names for models

def clean_model_name(series):
    return (
        series.astype(str)
        .str.strip()                     
        .str.lower()                     
        .str.replace(r"\s+", " ", regex=True)  
        .str.replace(r"[^a-z0-9_\-/\.]", "", regex=True)  
    )

df_all["model"] = clean_model_name(df_all["modelId"])
df_all.drop(columns=["modelId"], inplace=True)
print(df_all.columns.tolist())


df_lb["model"] = clean_model_name(df_lb["Model"])
df_lb.drop(columns=["Model", "Unnamed: 11"], inplace=True, errors="ignore")
print(df_lb.columns.tolist())


df_all.columns = df_all.columns.str.lower()
df_lb.columns = df_lb.columns.str.lower()




# inner join 

df_joined = pd.merge(
    df_all,
    df_lb,
    on="model",
    how="inner"
)

df_joined = df_joined[["model", "model_type", "rank", "type"] 
    + [c for c in df_joined.columns if c not in ["model", "model_type", "rank", "type"]]]



# missing data 

missing_summary = df_joined.isna().sum().sort_values(ascending=False)
print(missing_summary)

#easier to see column names
df_joined.head(3).T

#what models have the NA's since only a 10 we can list them

na_rows = df_joined[df_joined["model_size"].isna() | df_joined["language"].isna()]
na_rows_display = na_rows[["model", "model_size", "language"]]
display(na_rows_display)


import requests
from bs4 import BeautifulSoup
import re



def get_model_size(model_id):
    """
    Extracts model size info (e.g., '88.2M params', '1.5B parameters') from a Hugging Face model card.
    """
    url = f"https://huggingface.co/{model_id}"
    headers = {"User-Agent": "Mozilla/5.0"}  
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Joining all visible text (lowercase for easier searching)
    text = " ".join(soup.stripped_strings).lower()

    # Using regex to find things like '124m params', '1.5b parameters'
    match = re.search(r"(\d+(\.\d+)?\s*[mb]\s*(params|parameters))", text)
    if match:
        return match.group(1)
    else:
        return None




missing = df_joined[df_joined["model_size"].isna()]["model"]
print(missing)

for model_name in missing:
    size = get_model_size(model_name)
    print(f"{model_name}: {size}")
    df_joined.loc[df_joined["model"] == model_name, "model_size"] = size



# cannot get DialoGpt-medium
# url = "https://huggingface.co/microsoft/DialoGPT-medium"
# html = requests.get(url).text
# match = re.search(r"(\d+\.?\d*\s*[mb]\s*parameters?)", html, re.IGNORECASE)
# print(match.group(1) if match else "Not found")



# cleaning model_size variable and transforming to billions for consistency

def clean_model_size(value):
    if pd.isna(value):
        return None
    value = str(value).lower().strip()
    # Extract number and unit (e.g., "124m", "1.5b")
    match = re.search(r"(\d+\.?\d*)\s*([mb])", value)
    if match:
        num = float(match.group(1))
        unit = match.group(2)
        # Convert everything to billions for consistency
        if unit == "m":
            num = num / 1000  # in billions
        return num
    else:
        return None



df_joined["model_size"] = df_joined["model_size"].apply(clean_model_size)

#cannot find it so will add it manualy from here:
#https://huggingface.co/microsoft/DialoGPT-medium
df_joined.loc[df_joined["model"] == "microsoft/dialogpt-medium", "model_size"] = 0.147


missing = df_joined[df_joined["model_size"].isna()]["model"]
print(missing)


# Convert strings like '45.3 %' to float 45.3 
percent_cols = ["average", "ifeval", "bbh", "math", "gpqa", "musr", "mmlu-pro"]

def clean_percent(series):
    return (
        pd.to_numeric(series.astype(str)
                      .str.replace("%", "", regex=False)
                      .str.strip(),
                      errors="coerce")
            )

for col in percent_cols:
    df_joined[col] = clean_percent(df_joined[col])


# Convert strings like '0.89 kg' to float 0.89
# remove kg in co2

kg_cols = ["co₂ cost"]

def clean_kg(series):
    return (
        pd.to_numeric(series.astype(str)
                      .str.replace("kg", "", regex=False)
                      .str.strip(),
                      errors="coerce")
    )


for col in kg_cols:
    df_joined[col] = clean_kg(df_joined[col])


# import os
# print(os.getcwd())

symbol_map = {
    "🟢": "open-source",
    "💬": "chat",
    "🔶": "proprietary"
}

df_joined["type"] = df_joined["type"].map(symbol_map).fillna("unknown")

df_joined.to_csv("df_joined_clean.csv", index=False)




sns.set_theme(style="ticks", palette="pastel", font_scale=1.1)
#plt.rcParams["figure.figsize"] = (8, 5)

#distribution of model_size

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_joined, x="model_size", color="skyblue")
plt.title("Distribution of Model Sizes")
plt.xlabel("Model Size (Billions)")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df_joined["model_size"], bins=30, kde=True)

plt.title("Distribution of Model Sizes (in billions)")
plt.xlabel("Model Size (B)")
plt.ylabel("Count")
plt.show()

#compare open-source vs proprietary models,


plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_joined,
    x="model_size",
    y="average",
    hue="type",              # your real grouping column
    split=True,
    inner="quart",
    palette="pastel",
    linewidth=1
)
plt.title("Distribution of Average Score by Model Size and Type")
plt.xlabel("Model Size (B)")
plt.ylabel("Average Score")
plt.show()

# Averge vs Model-Size

y = "average"       
x = "model_size"

plt.figure(figsize=(8,5))
sns.scatterplot(data=df_joined, x=x, y=y, alpha=0.8)
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.title("Average vs Model Size")
plt.show()

plt.figure(figsize=(7,5))
sns.regplot(data=df_joined, x=x, y=y, scatter_kws=dict(alpha=0.6), line_kws=dict(alpha=0.9))
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.title("Average vs Model Size (Linear Fit)")
plt.show()

from plotnine import ggplot, aes, geom_point, geom_smooth, labs, theme_bw, scale_x_log10, facet_wrap
# for LOESS pip install scikit-misc


plt.figure(figsize=(7,5))
(
    ggplot(df_joined, aes(x="model_size", y="average"))
    + geom_point(alpha=0.5, color="steelblue")
    + geom_smooth(method="loess", color="black", se=False)   # ggplot’s LOESS smoother
    + labs(
        title="Average Score vs Model Size (LOESS smoother)",
        x="Model Size (B)",
        y="Average Score (%)"
    )
    + theme_bw()
)



# used scale_x_log10 so we can plot better the small models
plt.figure(figsize=(7,5))
(
    ggplot(df_joined, aes(x="model_size", y="average"))
    + geom_point(alpha=0.5, color="darkorange")
    + geom_smooth(method="loess", color="black", se=False)
    + scale_x_log10()
    + labs(
        title="Average Score vs Model Size (log scale)",
        x="Model Size (log10 Billions)",
        y="Average Score (%)"
    )
    + theme_bw()
)

df_joined[["model_size", "average"]].corr()

import statsmodels.api as sm

X = sm.add_constant(df_joined["model_size"])
y = df_joined["average"]
model = sm.OLS(y, X).fit()
print(model.summary())



from scipy.stats import ttest_ind

open_src = df_joined.loc[df_joined["type"].str.contains("open", case=False, na=False), "average"]
prop = df_joined.loc[df_joined["type"].str.contains("prop", case=False, na=False), "average"]

t_stat, p_val = ttest_ind(open_src, prop, equal_var=False)
print(f"T-statistic: {t_stat:.3f},  p-value: {p_val:.4f}")




# we can clearly see thatthe bigger the model the higher the average score

# faceting by type


plt.figure(figsize=(7,5))
(
    ggplot(df_joined, aes(x="model_size", y="average"))
    + geom_point(alpha=0.6, color="coral")
    #+ geom_smooth(method="lm", color="black", se=False)
    # CANNOT DO LOESS BECAUSE NOT ENOUGH POINTS.
    #+ geom_smooth(method="loess", color="black", se=False)
    + facet_wrap("~ type")        # or "~ model_type"
    + scale_x_log10()
    + labs(
        title="Average Score vs Model Size by Type (log scale)",
        x="Model Size (log10 Billions)",
        y="Average Score (%)"
    )
    + theme_bw()
)
 # CANNOT DO FACET WRAP WITH SMOOTHER OR lm BECAUSE NOT ENOUGH POINTS.


import plotly.express as px



fig = px.scatter(
    df_joined,
    x="model_size",
    y="average",
    color="type",
    color_discrete_map={
        "open-source": "green",
        "chat/dialogue model": "orange",
        "restricted/proprietary": "red"
    },
    hover_name="model",
    hover_data={
        "model_size": True,
        "average": True,
        "type": True
        },
    size="downloads" if "downloads" in df_joined.columns else None,
    log_x=True,
    title="Avg. Score vs Model Size (scale by nr. of downloads) - interactive"
)

# Move legend inside top-left
fig.update_layout(
    legend=dict(
        x=0.02,            # distance from left edge (0 = left, 1 = right)
        y=0.98,            # distance from bottom (1 = top)
        bgcolor="rgba(255,255,255,0.6)",  # translucent white background
        bordercolor="black",
        borderwidth=1
    ),
    xaxis_title="Model Size (Billions, log scale)",
    yaxis_title="Average Score (%)",
    template="plotly_white"
)

fig.show()


# buble size based on Model_size - does not make sense because you cannot see the smaller bubles
# fig = px.scatter(
#     df_joined,
#     x="model_size",
#     y="average",
#     size="model_size",              
#     color="type",
#     hover_name="model",
#     hover_data={
#         "model_size": True,
#         "average": True,
#         "type": True
#     },
#     log_x=True,
#     title="Average Score vs Model Size (bubbles scaled by model size)"
# )

# fig.update_layout(
#     legend=dict(
#         x=0.02,
#         y=0.98,
#         bgcolor="rgba(255,255,255,0.6)",
#         bordercolor="black",
#         borderwidth=1
#     ),
#     xaxis_title="Model Size (Billions, log scale)",
#     yaxis_title="Average Score (%)",
#     template="plotly_white"
# )

# # Optional: scale bubble size range for better readability
# fig.update_traces(marker=dict(sizeref=2.*max(df_joined["model_size"])/(40**2), sizemode='area'))

# fig.show()



# Efficiency: score per billion parameters and per kg CO₂ (emmission during testing)
df_joined["score_per_b"] = df_joined["average"] / df_joined["model_size"]
df_joined["score_per_kg"] = df_joined["average"] / df_joined["co₂ cost"]

# Show top models by efficiency
eff_top = (
    df_joined[["model", "type", "model_size", "co₂ cost", "average", "score_per_b", "score_per_kg"]]
    .sort_values("score_per_b", ascending=False)
    .head(12)
)
display(eff_top)


plt.figure(figsize=(8,5))
sns.boxplot(data=df_joined, x="type", y="score_per_b", palette="pastel")
plt.title("Score per Billion Parameters by Model Type")
plt.xlabel("Model Type")
plt.ylabel("Score per Billion Parameters")
plt.show()

#this makes sense because the proprietary are bigger models more CO2 consumed not a very usefull metric anyway


# 1) Prepare data: log-transform size
tmp = df_joined[["model_size", "average"]].copy()
tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
tmp = tmp[tmp["model_size"] > 0]  # guard
tmp["log_size"] = np.log10(tmp["model_size"])

# 2) Fit linear model: average = a * log10(size) + b  (no external libs)
x = tmp["log_size"].values
y = tmp["average"].values
a, b = np.polyfit(x, y, 1)  # slope a, intercept b
y_hat = a*x + b

# R^2 (coefficient of determination)
ss_res = np.sum((y - y_hat)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res/ss_tot

print(f"Slope (a): {a:.3f}")
print(f"Intercept (b): {b:.3f}")
print(f"R²: {r2:.3f}")
# average score vs size explains 62% of the variation.

# 3) Plot: scatter + fitted line in log-size space
plt.figure(figsize=(8,5))
sns.scatterplot(data=tmp, x="log_size", y="average", alpha=0.7)
# line across the x-range
xr = np.linspace(tmp["log_size"].min(), tmp["log_size"].max(), 200)
plt.plot(xr, a*xr + b, color="black", linewidth=2, label=f"fit: avg = {a:.2f}·log10(size)+{b:.2f}\nR²={r2:.2f}")
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.xlabel("log10(Model Size in Billions)")
plt.ylabel("Average Score (%)")
plt.title("Scaling relation: Average vs log10(Model Size)")
plt.legend()
plt.tight_layout()
plt.show()


# 4) Same points with original x on log-scale and fitted curve transformed back
plt.figure(figsize=(8,5))
sns.scatterplot(data=df_joined, x="model_size", y="average", alpha=0.7)
plt.xscale("log")
# transform the line back: x_original in B → log10(x_original)
x_original = np.logspace(np.log10(df_joined["model_size"].min()),
                         np.log10(df_joined["model_size"].max()), 200)
plt.plot(x_original, a*np.log10(x_original) + b, color="black", linewidth=2, label="log-fit")
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.xlabel("Model Size (Billions, log scale)")
plt.ylabel("Average Score (%)")
plt.title("Average vs Model Size (log x) with log-fit line")
plt.legend()
plt.tight_layout()
plt.show()

#Two-group t-test (open-source vs proprietary)

from scipy.stats import ttest_ind, levene

g1 = df_joined.loc[df_joined["type"]=="open-source", "average"].dropna()
g2 = df_joined.loc[df_joined["type"]=="chat", "average"].dropna()

print(f"n_open_source={len(g1)}, n_chat={len(g2)}")

# Variance check (Levene)
W, p_lev = levene(g1, g2, center="median")
print(f"Levene W={W:.3f}, p={p_lev:.4f} (p>0.05 → ok to assume equal variances)")


# Welch t-test ----------------
t, p = ttest_ind(g1, g2, equal_var=False)
print(f"Welch t={t:.3f}, p={p:.4g}")

def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    s_pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / s_pooled

print(f\"Cohen's d={cohens_d(g1,g2):.2f}\")


# ANOVA + TUCKEY -----------------

import statsmodels.api as sm
import statsmodels.formula.api as smf


anova_model = smf.ols("average ~ C(model_type)", data=df_joined).fit()
print(sm.stats.anova_lm(anova_model, typ=2))

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tuk = pairwise_tukeyhsd(
    endog=df_joined["average"].values,
    groups=df_joined["model_type"].astype(str).values,
    alpha=0.05
)
print(tuk.summary())

# All rows show reject = False
# none of the pairwise differences between these model families is statistically significant at the 5% level

# stats based on the model type
(df_joined.groupby("model_type")["average"]
   .agg(["count","mean","std","min","max"])
   .sort_values("mean", ascending=False))

# omnibus ANOVA. Tukey is most meaningful if the overall ANOVA finds a group effect
anova = smf.ols("average ~ C(model_type)", data=df_joined).fit()
print(sm.stats.anova_lm(anova, typ=2))

# p = 0.316
# no statistically significant difference in mean average across the 7 model families
# if we are expecting difference we need more data. or more groups

# ANCOVA
ancova = smf.ols("average ~ np.log10(model_size) + C(model_type)", data=df_joined).fit()
print(ancova.summary())

#Controlling for family, each 10× increase in model size is associated with ~15.6** percentage-point higher average score** (p < 0.001);
#family effects are not significant.

residuals = ancova.resid
fitted = ancova.fittedvalues

plt.figure(figsize=(6,5))
sns.scatterplot(x=fitted, y=residuals, alpha=0.7)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Fitted Values (Predicted Average %)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.tight_layout()
plt.show()

# Normal Q–Q plot
sm.qqplot(residuals, line="45", fit=True)
plt.title("Normal Q–Q Plot of Residuals")
plt.tight_layout()
plt.show()
