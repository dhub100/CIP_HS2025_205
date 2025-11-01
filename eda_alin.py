import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import requests
from bs4 import BeautifulSoup
import re

df_all = pd.read_csv("huggingface_llm_metadata.csv")
#df_100 = pd.read_csv("huggingface_100_llm_metadata.csv")
df_lb = pd.read_csv("hf_scraped.csv")



# -------------------------- Check for gaps / missing data ----------------------

# data inspect 
display(df_all.head())  
display(df_lb.head())

print(df_all.columns.tolist())
print(df_lb.columns.tolist())

# ----------- 1a Check for duplicate row

print(df_all.duplicated().sum())
print(df_lb.duplicated().sum())


# df_lb had initally duplicate rows since this code is not needed
# code is commented it out since now is redudant

# duplicates=df_lb[df_lb.duplicated(keep=False)]
# duplicates.sort_values("Rank", ascending=True, inplace=True)
# print(duplicates)

#---------- 1b we droped duplicate rows

# df_lb = df_lb.sort_values("Rank", ascending=True)
# df_lb = df_lb.drop_duplicates(subset=["Model"], keep="first").reset_index(drop=True)
# df_lb.duplicated().sum() 


# ----------- 1c Checking for na


print(df_all.isna().sum().sort_values(ascending=False).head(10))
print(df_lb.isna().sum().sort_values(ascending=False).head(10))

# explanation:
# df all does not have model size for 22 of the models, 
# after join we will have to do scrape on the LLM dedicated pages for the model size

#intersections - looking for columns with the same name

print(set(df_all.columns).intersection(df_lb.columns))
# no intersection - we will create variables with the same name so we can join

# -------------1c creting a function to clean the names for models (make name of the model the same)

def clean_model_name(series):
    return (
        series.astype(str)
        .str.strip()                     
        .str.lower()                     
        .str.replace(r"\s+", " ", regex=True)  
        .str.replace(r"[^a-z0-9_\-/\.]", "", regex=True)  
    )

# --------------1d creating the model variables in both dataset and droping the unnecessary column

#df_all dataset
df_all["model"] = clean_model_name(df_all["modelId"])
df_all.drop(columns=["modelId", "language", "pipeline_tag"], inplace=True)
print(df_all.columns.tolist())


# language column will not be using so we droped language as well
# pipeline tag variable is the same accross all models "text-generation" - drop
# we extracted the date from createdAt and lastModifed columns

# df_lb dataset
df_lb["model"] = clean_model_name(df_lb["Model"])
df_lb.drop(columns=["Model", "Unnamed: 11", "Unnamed: 0"], inplace=True, errors="ignore")
print(df_lb.columns.tolist())


# ------------- 1e Make all column names lower string

df_all.columns = df_all.columns.str.lower()
df_lb.columns = df_lb.columns.str.lower()




# ----------------- 1f inner join (4 Format your dataset)

df_joined = pd.merge(
    df_all,
    df_lb,
    on="model",
    how="inner"
)


# moving columns to the front
df_joined = df_joined[
    ["model", "model_type", "rank", "type", "gated"]
    + [c for c in df_joined.columns if c not in ["model", "model_type", "rank", "type", "gated"]]
]



# ------------- 1g missing data 

missing_summary = df_joined.isna().sum().sort_values(ascending=False)
print(missing_summary)



# what models have the NA's since only a 10 we can list them
na_rows = df_joined[df_joined["model_size"].isna()]
na_rows_display = na_rows[["model", "model_size"]]
display(na_rows_display)


import requests
from bs4 import BeautifulSoup
import re

# ----------- 1h webscraper

def get_model_size(model_id):
    """
    Extracts model size info from a Hugging Face model card.
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

# do another check 
missing = df_joined[df_joined["model_size"].isna()]["model"]
print(missing)

#cannot find it so will add it manualy from here:
#https://huggingface.co/microsoft/DialoGPT-medium
df_joined.loc[df_joined["model"] == "microsoft/dialogpt-medium", "model_size"] = 0.147


missing = df_joined[df_joined["model_size"].isna()]["model"]
print(missing)



# -----------------2. columns datatypes and Changes

df_joined.info()
df_joined.dtypes.value_counts()
#easier to see column names
df_joined.head(3).T

# we will have to change severl columns, date, percentages, remove kg

#--------2a transform dates

df_joined["created_at"] = pd.to_datetime(df_joined["createdat"]).dt.date
df_joined["last_modified"] = pd.to_datetime(df_joined["lastmodified"]).dt.date
df_joined.drop(columns=["createdat", "lastmodified"], inplace=True)


df_joined.head(3).T

#--------2b cleaning model_size variable and transforming to billions for consistency
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



# ----------------2c Convert strings like '45.3 %' to float 45.3 

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


# -------------2d Convert strings like '0.89 kg' to float 0.89
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

# -------------2e  map symbols

symbol_map = {
    "🟢": "open-source",
    "💬": "chat",
    "🔶": "proprietary"
}

df_joined["type"] = df_joined["type"].map(symbol_map).fillna("unknown")

#df_joined.to_csv("df_joined_clean.csv", index=False)


#----------------3 check if values lie in the expected range 

# Rank must be positive
bad_rank = df_joined.query("rank <= 0 or rank.isna()", engine="python")
print(len(bad_rank))

#Model size should be > 0 and < 70
bad_size = df_joined.query("model_size <= 0 or model_size >= 80 or model_size.isna()", engine="python")
print(len(bad_size))
#display(bad_size[["model", "model_size"]])

# scores bewteen 0-100
score_cols = ["average", "ifeval", "bbh", "math", "gpqa", "musr", "mmlu-pro"]

for col in score_cols:
    if col in df_joined.columns:
        invalid = df_joined[(df_joined[col] < 0) | (df_joined[col] > 100)]
        if not invalid.empty:
            print(f"{col}: {len(invalid)}")

# co2 should be negative
if "co₂ cost" in df_joined.columns:
    bad_co2 = df_joined[df_joined["co₂ cost"] < 0]
    print(len(bad_co2))

# date check > 2022

if "created_at" in df_joined.columns:
    bad_dates = df_joined[
        (df_joined["created_at"] < np.datetime64("2015-01-01")) |
        (df_joined["created_at"] > np.datetime64("today"))
        ]
    print(len(bad_dates))


#-------------------------- Identify outliers 

# define the IQR bounds 
def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

benchmark_cols = ["average", "ifeval", "bbh", "math", "gpqa", "musr", "mmlu-pro"]
numeric_cols = ["rank", "model_size", "co₂ cost"] + [c for c in benchmark_cols if c in df_joined.columns]
numeric_cols = [c for c in numeric_cols if c in df_joined.columns]
print(numeric_cols)

# display the bounds
for col in numeric_cols:
    s = df_joined[col].dropna()
    if s.empty:
        continue
    lb, ub = iqr_bounds(s)
    print(f"{col}: lower={lb:.2f}, upper={ub:.2f}")


# here in Theory we can apply the IQR treatement using IQR clipping 
# values outside the IQR will be capped
# we only have 40 rows so we will not apply the clipping since the large models will be clipped
# for col in numeric_cols:
#     s = df_joined[col]
#     if s.dropna().nunique() <= 1:
#         continue
#     lb, ub = iqr_bounds(s)
#     df_joined[col] = s.clip(lower=lb, upper=ub)



# -----------5 Enrich your dataset with at least one column

#avg score per bilion parameters - can show performance per billion parm
df_joined["score_per_billion"] = (df_joined["average"] / df_joined["model_size"]).round(2)


# enviroment efficient models- expected that theythe bigger the model the more c02 consumtion
df_joined["score_per_co2"] = (df_joined["average"] / df_joined["co₂ cost"]).round(2)


df_joined.to_pickle("df_joined_clean.pkl")



# ---------------- data exploration

import os
os.makedirs("figures", exist_ok=True)

# def to save figures

def savefig(name):
    plt.savefig(f"figures/{name}", dpi=300, bbox_inches="tight")
    plt.close()


# Define colors
sns.set_theme(style="ticks", palette="pastel", font_scale=1.1)

# ---- Define consistent pastel colors for reuse
COLORS = {
    "main": "#A1C9F4",       # blue (for numeric distributions)
    "accent": "#FFB482",     # pastel orange (for scatter / highlights)
    "cat1": "#A1C9F4",       # category 1
    "cat2": "#FFB482",       # category 2
    "cat3": "#8DE5A1",       # category 3
}

#-----------------------------------------------------------------
#distribution of model_size

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_joined, x="model_size", color=COLORS["main"])
plt.title("Distribution of Model Sizes")
plt.xlabel("Model Size (Billions)")
savefig("Dist_by_model_size_boxplot.png")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df_joined["model_size"], bins=30, kde=True)
plt.title("Distribution of Model Sizes (in billions)")
plt.xlabel("Model Size (B)")
plt.ylabel("Count")
plt.savefig("figures/Dist_by_model_size_hist.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
#compare open-source vs proprietary models,

# violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df_joined,
    x="type",
    y="average",
    hue="type", 
    inner="quart",
    palette="pastel",
     linewidth=1 )
plt.title("Distribution of Average Score by Type")
plt.xlabel("Model Size (B)")
plt.ylabel("Average Score")
plt.savefig("figures/avg_vs_size_violin.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df_joined,
    x="model_size",
    y="average",
    color=COLORS["accent"],
    alpha=0.7,
    edgecolor="white",
    s=80
)
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.title("Average Score vs Model Size")
plt.xlabel("Model Size (Billions)")
plt.ylabel("Average Score (%)")
sns.despine()
savefig("avg_vs_model_size_scatter.png")
plt.show()

# regression line MOdel vs model size

plt.figure(figsize=(8, 5))
sns.regplot(
    data=df_joined,
    x="model_size",
    y="average",
    scatter_kws=dict(color=COLORS["accent"], alpha=0.6, s=70),
    line_kws=dict(color=COLORS["cat2"], lw=2, alpha=0.8)
)
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.title("Average Score vs Model Size (Linear Fit)")
plt.xlabel("Model Size (Billions)")
plt.ylabel("Average Score (%)")
sns.despine()
savefig("avg_vs_model_size_regplot.png")
plt.show()


#log scale x-axis  - patterns should be more visible


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df_joined,
    x="model_size",
    y="average",
    color=COLORS["accent"],
    alpha=0.7,
    edgecolor="white",
    s=80
)
plt.xscale("log")
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.title("Average Score vs Model Size (log scale)")
plt.xlabel("Model Size (Billions, log10 scale)")
plt.ylabel("Average Score (%)")
sns.despine()
savefig("avg_vs_model_size_log_scatter.png")
plt.show()





from plotnine import ggplot, aes, geom_point, geom_smooth, labs, theme_bw, scale_x_log10, facet_wrap
# for LOESS pip install scikit-misc
PASTEL_COLORS = ["#A1C9F4", "#FFB482", "#8DE5A1"]

plt.figure(figsize=(8,5))
#p1 =
(
    ggplot(df_joined, aes(x="model_size", y="average"))
    + geom_point(alpha=0.5, color="#A1C9F4")
    + geom_smooth(method="loess", color="#FFB482", se=False)   # ggplot’s LOESS smoother
    + labs(
        title="Average Score vs Model Size (LOESS smoother)",
        x="Model Size (B)",
        y="Average Score (%)"
    )
    + theme_bw()
)


#p1.save("figures/avg_vs_model_size_loess.png", dpi=300, bbox_inches="tight")


# used scale_x_log10 so we can plot better the small models
#LOESS
plt.figure(figsize=(8,5))
#p_log = 
(
    ggplot(df_joined, aes(x="model_size", y="average"))
    + geom_point(alpha=0.5, color="#A1C9F4")
    + geom_smooth(method="loess", color="#FFB482", se=False)
    + scale_x_log10()
    + labs(
        title="Average Score vs Model Size (log scale)",
        x="Model Size (log10 Billions)",
        y="Average Score (%)"
    )
    + theme_bw()
)
#p_log.save("figures/avg_vs_model_size_loess_log.png", dpi=300, bbox_inches="tight")

# you can clearly see how performance changes across small and large models.

# -------------correlation
#we calculate the Pearson correlation between model size and average score

df_joined[["model_size", "average"]].corr()

#correlation is 0.75 so the bigger the model, the higher the averge score

import statsmodels.api as sm

X = sm.add_constant(df_joined["model_size"])
y = df_joined["average"]
model = sm.OLS(y, X).fit()
print(model.summary())

# The intercept (13.54) is the expected average score (in %) when the model size is 0 B (not meaningful)
# The slope is 0.45 meaning that for every 1 bilion parameter increase
#                             the average score increase with approx 0.45 points
# p <0.001 so it is statistically significant
# Larger models tend to achieve higher average benchmark scores
# model size alone explains roughly half of the performance variation. Rsq = 0.56

# FITTED REGRESSION LINE

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df_joined,
    x="model_size",
    y="average",
    color="#A1C9F4",
    alpha=0.7,
    s=80,
    edgecolor="white"
)
# Regression line
X = sm.add_constant(df_joined["model_size"])
y = df_joined["average"]
model = sm.OLS(y, X).fit()
pred = model.predict(X)
plt.plot(df_joined["model_size"], pred, color="#FFB482", lw=2, label="OLS Fit")
plt.title("Average Score vs Model Size (OLS Fit)")
plt.xlabel("Model Size (Billions)")
plt.ylabel("Average Score (%)")
plt.legend()
sns.despine()
plt.savefig("figures/avg_vs_model_size_OLSfit.png", dpi=300, bbox_inches="tight")
plt.show()







from scipy.stats import ttest_ind

open_src = df_joined.loc[df_joined["type"].str.contains("open", case=False, na=False), "average"]
prop = df_joined.loc[df_joined["type"].str.contains("prop", case=False, na=False), "average"]

t_stat, p_val = ttest_ind(open_src, prop, equal_var=False)
print(f"T-statistic: {t_stat:.3f},  p-value: {p_val:.4f}")

# There is no significant difference in average benchmark scores between open-source and proprietary models



import plotly.express as px

# Define pastel color map consistent with all plots
PASTEL_BLUE = COLORS["main"] if "COLORS" in globals() else "#A1C9F4"

g = sns.relplot(
    data=df_joined,
    x="model_size",
    y="average",
    col="type",
    kind="scatter",
    color=PASTEL_BLUE,
    alpha=0.7,
    height=4,
    aspect=1.1,
    col_wrap=3
)

# Log scale + percent formatter + tidy titles/labels
for ax in g.axes.flatten():
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Model Size (Billions, log10)")
    ax.set_ylabel("Average Score (%)")

g.set_titles(col_template="{col_name}")
g.figure.suptitle("Average Score vs Model Size by Type (log scale)", y=1.03)
sns.despine()

# Save the whole faceted figure
g.figure.savefig("figures/avg_vs_size_by_type_facets.png", dpi=300, bbox_inches="tight")
plt.show()


# buble size based on Model_size - does not make sense because you cannot see the smaller bubles



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


plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df_joined,
    x="type",
    y="score_per_b",
    palette="pastel"
)
plt.title("Score per Billion Parameters by Model Type")
plt.xlabel("Model Type")
plt.ylabel("Score per Billion Parameters")
sns.despine()
savefig("score_per_billion_by_type_boxplot.png")
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
