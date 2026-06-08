import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from snakemake.script import snakemake
import itertools
import pathlib

plt.rcParams["font.size"] = 25
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.title_fontsize"] = 15
plt.rcParams["axes.labelsize"] = 20

inpaths = [pathlib.Path(x) for x in snakemake.input["inp"]]
outpaths_xtb = [pathlib.Path(x) for x in snakemake.input["xtb"]]


solvents = [x.stem for x in inpaths]
input_data = {solv: pd.read_csv(path) for solv, path in zip(solvents, inpaths)}
output_data = {solv: pd.read_csv(path, index_col=False,
                              names=["ID", "homo_1", "homo", "lumo", "fuk_minus", "fuk_plus", "chrg",
                                    "nei_1_fuk", "nei_2_fuk", "nei_3_fuk",
                                    "nei_1_chrg", "nei_2_chrg", "nei_3_chrg",
                                    "sasa"]) for solv, path in zip(solvents, outpaths_xtb)}

for k, v in input_data.items():    
    v = v.sort_values('N Params', ascending=False).drop_duplicates(subset='Smiles', keep='first') 
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


# add variables - cdft descriptors and other
data["mu"] = 0.5*(data["lumo"] + data["homo"])
data["eta"] = data["lumo"] - data["homo"]
data["gam"] = data["lumo"] - 2* data["homo"] + data["homo_1"]
data["omega"] = data["mu"]**2 / (2*data["eta"])
data["NG"] = -1* data["omega"] * (1 + data["mu"]*data["gam"]/(3*data["eta"]**2) - (4*data["eta"]**2) / (data["mu"]*data["gam"]) + (4*data["eta"]**4)/(3*data["mu"]**2 * data["gam"]**2) )
data["ND"] = data["homo"] - (-9.061)

solvs = pd.read_csv(f"output/dbs/solvents.csv", index_col=False, names=["Solvent", "homo_1_solv", "homo_solv", "lumo_solv", "dipole_solv"])
solvs["mu_solv"] = 0.5*(solvs["lumo_solv"] + solvs["homo_solv"])
solvs["eta_solv"] = solvs["lumo_solv"] - solvs["homo_solv"]
solvs["gam_solv"] = solvs["lumo_solv"] - 2* solvs["homo_solv"] + solvs["homo_1_solv"]
solvs["omega_solv"] = solvs["mu_solv"]**2 / (2*solvs["eta_solv"])
data = pd.merge(data, solvs, on="Solvent")

data["solv"] = data["Solvent"].replace({"water": 78.4,
                                        "ch2cl2": 8.9,
                                        "DMF": 36.7,
                                        "THF": 7.6,
                                        "acetonitrile": 35.7,
                                        "DMSO": 46.7,
                                        })

data["solvent"] = data["Solvent"].replace({"water": "water",
                                        "ch2cl2": "DCM",
                                        "DMF": "DMF",
                                        "THF": "THF",
                                        "acetonitrile": "ACN",
                                        "DMSO": "DMSO",
                                        })

# N_G plot
linr = linregress(data["NG"], data["N Params"])
minmax = [data["NG"].min(),data["NG"].max()]
fig, ax = plt.subplots(figsize=(15,8))
sns.scatterplot(data, x="NG", y="N Params", hue="solvent", ax=ax)
r2 = r2_score(data["N Params"], data["NG"]*linr.slope + linr.intercept)
rmse = np.sqrt(mean_squared_error(data["N Params"], data["NG"]*linr.slope + linr.intercept))
ax.text(0.2,0.1, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes)
ax.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
ax.set(xlabel="$N_G$", ylabel="$N_M$", xlim=(-82, -15), title="xTB")
fig.tight_layout()
fig.savefig("imgs/ng_xtb.pdf")


# N_D plot
linr = linregress(data["ND"], data["N Params"])
minmax = [data["ND"].min(),data["ND"].max()]
fig, ax = plt.subplots(figsize=(15,8))
sns.scatterplot(data, x="ND", y="N Params", hue="solvent", ax=ax)
r2 = r2_score(data["N Params"], data["ND"]*linr.slope + linr.intercept)
rmse = np.sqrt(mean_squared_error(data["N Params"], data["ND"]*linr.slope + linr.intercept))
ax.text(0.75,0.1, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes)
ax.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
ax.set(xlabel="$N_D$", ylabel="$N_M$", title="xTB")
fig.tight_layout()
fig.savefig("imgs/nd_xtb.pdf")


# calculate N_{emp} of all data
# old version
# data["pysr"] = data["dipole_solv"] + (data["dipole_solv"]*data["fuk_minus"] - data["chrg"]-2.69)**2 * (data["homo"]+data['homo_solv']+data["mu_solv"]+30.5) + 13.9

# after revision
data["pysr"] = data["dipole_solv"] + (data["chrg"] - data["dipole_solv"] * data["fuk_minus"] + 2.89)**2 * (data["homo"] + data["homo_1_solv"] + data["mu_solv"] + 31.0) + 14.5


# N_{emp} plot
linr = linregress(data["pysr"], data["N Params"])
r2 = r2_score(data["N Params"], data["pysr"]*linr.slope + linr.intercept)
minmax = [data["pysr"].min(),data["pysr"].max()]
fig, ax = plt.subplots(figsize=(15,8))
sns.scatterplot(data, x="pysr", y="N Params", hue="solvent", ax=ax)
rmse = np.sqrt(mean_squared_error(data["N Params"], data["pysr"]*linr.slope + linr.intercept))
ax.text(0.2,0.85, f"$R^2={r2:.3f}$\nRMSE={rmse:.3f}", transform=ax.transAxes)
ax.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
ax.set(xlabel="$N_{emp}$", ylabel="$N_M$")
fig.tight_layout()
fig.savefig("imgs/xtb_all.pdf")


# N_{emp} of all data per solvent
fig, ax = plt.subplots(3,2, figsize=(25,25))
for axx, solv in zip(itertools.chain.from_iterable(ax), solvents):
    plotdata = data[data["Solvent"]==solv]
    minmax = [plotdata["pysr"].min(),plotdata["pysr"].max()]
    sns.scatterplot(plotdata, x="pysr", y="N Params", ax=axx)
    r2 = r2_score(plotdata["N Params"], plotdata["pysr"]*linr.slope + linr.intercept)
    rmse = np.sqrt(mean_squared_error(plotdata["N Params"], plotdata["pysr"]*linr.slope + linr.intercept))
    if solv == "ch2cl2":
        solv = "DCM"
    axx.set(title=f"{solv}", xlabel="$N_{PySR}$", ylabel="$N_M$")
    axx.text(0.1,0.8, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=axx.transAxes)
    axx.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
fig.tight_layout()
fig.savefig("imgs/xtb_all_per_solvents.pdf")


# Per solvent N_{emp}
# old version

# data["pysr_water"] = (0.968 - data["nei_1_chrg"])/(data["nei_1_fuk"]/(1.57-data["gam"])-8.55*data["nei_2_chrg"]+0.604) + 13.3
# data["pysr_ACN"] = -data["gam"]+data["homo"]+np.sqrt((np.sqrt(data["fuk_minus"])*(data["gam"]/(data["nei_1_chrg"]*(data["fuk_minus"]-0.013))+1.55*10**3)+data["sasa"]))
# data["pysr_DCM"] = data["homo"]*(data["nei_3_chrg"]+(data["chrg"]-data["fuk_minus"])*(data["homo"]+12)-np.sqrt((data["chrg"]+data["nei_1_chrg"]+0.265)**2))
# data["pysr_DMSO"] = data["eta"] - 17.8 - 359/(-0.584*data["gam"]+data["homo"])+1/(data["sasa"]*(data['gam']-0.693))
# data["pysr_DMF"] = 51.4 - 0.375/(data["fuk_minus"]*data["sasa"]+data["homo_1"]) + (7.97*data["nei_1_chrg"]-8.29)/data["fuk_minus"]
# data["pysr_THF"] = (data["nei_1_fuk"]*(data["sasa"]-data["gam"]**2) - (data["nei_1_chrg"]+0.0127*data["sasa"]+0.262)/data["nei_1_chrg"])**2 + 13.1 

# after revision
data["pysr_water"] = -33.3 * data["nei_2_fuk"] + 15.5 - 1 / (data["nei_1_chrg"] + (data["chrg"]**2 - 0.248)**3 / (data["nei_1_chrg"] + data["nei_1_fuk"])**3)
data["pysr_THF"] = 15.4 + 1 / (data["nei_1_chrg"] - np.sqrt(data["nei_2_chrg"]) + 0.213)
data["pysr_ACN"] = (data["mu"]**2 - 31.9) * (data["fuk_minus"] - data["nei_2_fuk"] - 0.306) + np.log((data["nei_2_chrg"] + 1 / (data["gam"] + 0.0853))**2) + 21.0
data["pysr_DMSO"] = data["eta"] + data["homo"] - 278. / (data["gam"] * (-np.sqrt(data["fuk_minus"]) - 0.365) + data["homo"]) + 1 / (data["sasa"] * (data["gam"] - 0.692))
data["pysr_DMF"] = np.sqrt(data["sasa"]) + 26.6 - 2.39 / data["fuk_minus"]
data["pysr_DCM"] = ((data["homo"] + 10.3)**2 - 3.97)**2 * (-data["chrg"] + data["fuk_minus"] - data["nei_3_chrg"] + np.sqrt((data["chrg"] + data["nei_1_chrg"] + 0.260)**2))

# per solvent N_{emp} plot
fig, ax = plt.subplots(3,2, figsize=(25,25))
for axx, solv in zip(itertools.chain.from_iterable(ax), data["solvent"].unique()):
    plotdata = data[data["solvent"]==solv]
    linr = linregress(plotdata[f"pysr_{solv}"], plotdata["N Params"])
    minmax = [plotdata[f"pysr_{solv}"].min(),plotdata[f"pysr_{solv}"].max()]
    sns.scatterplot(plotdata, x=f"pysr_{solv}", y="N Params", ax=axx)
    r2 = r2_score(plotdata["N Params"], plotdata[f"pysr_{solv}"]*linr.slope + linr.intercept)
    rmse = np.sqrt(mean_squared_error(plotdata["N Params"], plotdata[f"pysr_{solv}"]*linr.slope + linr.intercept))
    if solv == "ch2cl2":
        solv = "DCM"
    axx.set(title=f"{solv}", xlabel=f"$N_{{emp}}[{solv}] (n={len(plotdata)})$", ylabel="$N_M$")
    axx.text(0.1,0.8, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=axx.transAxes)
    axx.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
fig.tight_layout()
fig.savefig("imgs/xtb_per_solvents.pdf")

data.to_csv(snakemake.output["csv"])