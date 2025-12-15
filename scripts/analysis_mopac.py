import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from snakemake.script import snakemake
# import itertools
import pathlib

plt.rcParams["font.size"] = 25
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.title_fontsize"] = 15
plt.rcParams["axes.labelsize"] = 20

inpaths = [pathlib.Path(x) for x in snakemake.input["inp"]]
outpaths_mopac = [pathlib.Path(x) for x in snakemake.input["mopac"]]


solvents = [x.stem for x in inpaths]
input_data = {solv: pd.read_csv(path) for solv, path in zip(solvents, inpaths)}
output_data = {solv: pd.read_csv(path, names=["ID", "homo_1", "homo", "lumo"]) for solv, path in zip(solvents, outpaths_mopac)}

for k, v in input_data.items():    
    # Calculate Q1, Q3, and IQR
    Q1 = v["N Params"].quantile(0.25)
    Q3 = v["N Params"].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter outliers
    input_data[k] = v[(v["N Params"] >= lower_bound) & (v["N Params"] <= upper_bound)]

dat = {k: pd.merge(v, output_data[k], on="ID") for k, v in input_data.items()}

data = pd.concat(dat.values())


data["mu"] = 0.5*(data["lumo"] + data["homo"])
data["eta"] = data["lumo"] - data["homo"]
data["gamma"] = data["lumo"] - 2* data["homo"] + data["homo_1"]
data["omega"] = data["mu"]**2 / (2*data["eta"])
data["NG"] = -1* data["omega"] * (1 + data["mu"]*data["gamma"]/(3*data["eta"]**2) - (4*data["eta"]**2) / (data["mu"]*data["gamma"]) + (4*data["eta"]**4)/(3*data["mu"]**2 * data["gamma"]**2) )
data["ND"] = data["homo"] - (-9.061)


# N_G plot
linr = linregress(data["NG"], data["N Params"])
minmax = [data["NG"].min(),data["NG"].max()]
fig, ax = plt.subplots(figsize=(15,8))
sns.scatterplot(data, x="NG", y="N Params", hue="Solvent", ax=ax)
r2 = r2_score(data["N Params"], data["NG"])
rmse = np.sqrt(mean_squared_error(data["N Params"], data["NG"]))
ax.text(0.5,0.85, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes)
ax.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
ax.set(xlabel="$N_G$", ylabel="$N_M$", title="MOPAC")
fig.tight_layout()
fig.savefig("imgs/ng_mopac.pdf")


# N_D plot
linr = linregress(data["ND"], data["N Params"])
minmax = [data["ND"].min(),data["ND"].max()]
fig, ax = plt.subplots(figsize=(15,8))
sns.scatterplot(data, x="ND", y="N Params", hue="Solvent", ax=ax)
r2 = r2_score(data["N Params"], data["ND"])
rmse = np.sqrt(mean_squared_error(data["N Params"], data["ND"]))
ax.text(0.75,0.1, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes)
ax.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
ax.set(xlabel="$N_D$", ylabel="$N_M$", title="MOPAC")
fig.tight_layout()
fig.savefig("imgs/nd_mopac.pdf")

