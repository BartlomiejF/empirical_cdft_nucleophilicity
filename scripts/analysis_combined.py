import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from snakemake.script import snakemake
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams["font.size"] = 25
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.title_fontsize"] = 15
plt.rcParams["axes.labelsize"] = 20

xtb_scaf = pd.read_csv(snakemake.input["xtb_scaf"])


linr = linregress(xtb_scaf["pysr_prediction"], xtb_scaf["N Params"])
r2 = r2_score(xtb_scaf["N Params"], xtb_scaf["pysr_prediction"]*linr.slope + linr.intercept)
rmse = np.sqrt(mean_squared_error(xtb_scaf["N Params"], xtb_scaf["pysr_prediction"]))
minmax = [xtb_scaf["pysr_prediction"].min(),xtb_scaf["pysr_prediction"].max()]
fig, ax = plt.subplots(figsize=(15,8))
sns.scatterplot(xtb_scaf, x="pysr_prediction", y="N Params", ax=ax)
ax.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
ax.text(0.1,0.8, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes)
ax.set(xlabel="$N_{emp}$", ylabel="$N_M$")

fig.tight_layout()
fig.savefig(snakemake.output["img"])