from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import Draw
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from snakemake.script import snakemake
import itertools
import pathlib


plt.rcParams["font.size"] = 25
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.title_fontsize"] = 15
plt.rcParams["axes.labelsize"] = 20


n_smirks_dict = {'Ether': '[OX2:1]([#6;!$(C([OX2])[#7,#8,#15,#16,F,Cl,Br,I]);!$([#6]=[#8]):2])[#6;!$(C([OX2])[#7,#8,#15,#16]);!$([#6]=[#8]):3]>>[CH3][OX3+:1]([*:2])[*:3]',
                 'Ketone': '[OX1H0:1]=[#6X3:2]([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]>>[CH3][OX2H0:1][#6X3+:2]([*:3])[*:4]',
                 'Amide': '[OX1:1]=[CX3;$([CX3][#6]),$([CX3H]):2][#7X3;!R:3]>>[CH3][OX2:1][CX3:2]=[#7X3+:3]',
                 'Enolate': '[#6;$([#6]=,:[#6]-[#8-]),$([#6-]-[#6]=,:[#8]):1]~[#6:2]~[#8;$([#8-]-[#6]=,:[#6]),$([#8]=,:[#6]-[#6-]):3]>>[CH3][#6+0:1][*:2]=[#8+0:3]',
                 'Aldehyde': '[OX1:1]=[$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):2]>>[CH3][OX2:1][#6+:2]',
                 'Imine': '[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):1]=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):2]>>[CH3][NX3+:1]=[*:2]',
                 'Nitranion': '[#7X2-:1]>>[CH3][#7X3+0:1]',
                 'Carbanion': '[#6-;!$([#6X1-]#[#7,#8,#15,#16]):1]>>[CH3][#6+0:1]',
                 'Nitronate': '[#6:1]=[#7+:2](-[#8-:3])-[#8-:4]>>[CH3][#6:1][#7+:2](=[#8+0:3])-[*:4]',
                 'Ester': '[OX1:1]=[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]):2][#8X2H0:3][#6;!$(C=[O,N,S]):4]>>[CH3][OX2:1][#6X3+:2][*:3][*:4]',
                 'Carboxylic acid': '[OX1:1]=[CX3;$([R0][#6]),$([H1R0]):2][$([OX2H]),$([OX1-]):3]>>[CH3][OX2:1][CX3+:2][*:3]',
                 'Amine': '[#7+0;$([N;R;!$([#7X2]);$(N-[#6]);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]),$([NX3+0;!$([#7X3][CX3;$([CX3][#6]),$([CX3H])]=[OX1])]),$([NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]):1]>>[CH3][#7+:1]',
                 'Cyanoalkyl/nitrile anion': '[C:1]=[C:2]=[#7X1-:3]>>[CH3][C:1][C:2]#[#7X1+0:3]',
                 'Nitrile': '[NX1:1]#[CX2;!$(CC=C=[#7X1-]);!$(CC=C):2]>>[CH3][NX2+:1]#[*:2]',
                 'Isonitrile': '[CX1-:1]#[NX2+:2]>>[CH3][CX2+0:1]#[NX2+:2]',
                 'Phenol': '[OX2H:1][$(c(c)c),$([#6X3;R](=[#6X3;R])[#6X3;R]):2]>>[CH3][OX3+1:1][*:2]', # added due to rxn100
                 'Silyl_ether': '[#8X2H0:1][#14X4:2]([!#1:3])([!#1:4])[!#1:5]>>[CH3][#8X3H0+:1][*:2]([*:3])([*:4])[*:5]', # added due to rxn100
                 'Pyridine_like_nitrogen': '[#7X2;$([nX2](:*):*),$([#7X2;R](=[*;R])[*;R]):1]>>[CH3][#7X3+:1]'
}
smarts_dict = {k: v.split(">>")[0] for k,v in n_smirks_dict.items()}

def group_by_scaffold(smiles_series, return_scaffold_smiles=False, scaffold_type='murcko'):
    """
    Group molecules by scaffolds.
    
    scaffold_type options:
    - 'murcko': Bemis-Murcko scaffolds (rings only)
    - 'murcko_generic': Generic Murcko (all atoms as carbon)
    - 'functional': Group by functional groups for non-ring molecules
    """
    scaffold_dict = {}
    scaffold_labels = []
    next_id = 0
    
    for smiles in smiles_series:
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            if scaffold_type == 'murcko_generic':
                scaffold = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol)
                )
                scaffold_smiles = Chem.MolToSmiles(scaffold)
            elif scaffold_type == 'functional':
                # For molecules without rings, use functional group pattern
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                
                if not scaffold_smiles:  # No rings, use functional groups
                    scaffold_smiles = get_functional_group_pattern(mol)
            else:  # murcko
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
            
            # Fallback for empty scaffolds
            if not scaffold_smiles or scaffold_smiles == '':
                scaffold_smiles = 'acyclic'
            
            if return_scaffold_smiles:
                scaffold_labels.append(scaffold_smiles)
            else:
                if scaffold_smiles not in scaffold_dict:
                    scaffold_dict[scaffold_smiles] = next_id
                    next_id += 1
                scaffold_labels.append(scaffold_dict[scaffold_smiles])
        except:
            scaffold_labels.append(None)
    
    return pd.Series(scaffold_labels, index=smiles_series.index)

def get_functional_group_pattern(mol, smiles=None):
    """Create a simple functional group signature for acyclic molecules."""
    # Count key functional groups
    patterns = smarts_dict
    
    groups = []
    for name, pattern in patterns.items():
        if mol.HasSubstructMatch(pattern):
            groups.append(f"{name}")
    
    return '_'.join(groups) if groups else 'aliphatic'

# def get_functional_group(smiles):
#     """Create a simple functional group signature."""
#     # Count key functional groups
#     patterns = smarts_dict
#     mol = Chem.MolFromSmiles(smiles)
    
#     groups = []
#     for name, smarts_string in patterns.items():
#         pattern = Chem.MolFromSmarts(smarts_string)  # Convert SMARTS string to mol object
#         if pattern and mol.HasSubstructMatch(pattern):
#             groups.append(name)
    
#     return '_'.join(groups) if groups else 'aliphatic'



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


data["group_murcko"] = group_by_scaffold(data["Smiles"], return_scaffold_smiles=True, scaffold_type="murcko")


data["mu"] = 0.5*(data["lumo"] + data["homo"])
data["eta"] = data["lumo"] - data["homo"]
data["gam"] = data["lumo"] - 2* data["homo"] + data["homo_1"]
data["omega"] = data["mu"]**2 / (2*data["eta"])
data["NG"] = -1* data["omega"] * (1 + data["mu"]*data["gam"]/(3*data["eta"]**2) - (4*data["eta"]**2) / (data["mu"]*data["gam"]) + (4*data["eta"]**4)/(3*data["mu"]**2 * data["gam"]**2) )
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

data["sol"] = data["Solvent"].replace({"water": "water",
                                        "ch2cl2": "DCM",
                                        "DMF": "DMF",
                                        "THF": "THF",
                                        "acetonitrile": "ACN",
                                        "DMSO": "DMSO",
                                        })


# old equations
# data["pysr_acyclic"] = np.log((-data["omega"]+data["solv"]+(-1.62*data["dipole_solv"]**2+1.62*data["omega_solv"])**2*(data["homo"]-data["homo_1_solv"]-0.889)**2)**2)
# data["pysr_c1ccccc1"] = data["dipole_solv"] - data["gam"] + (data["dipole_solv"]**2 * data["nei_1_fuk"]**2 +0.248*(-2.25*data["chrg"]+data["homo"]))**3 + 22.7
# data["pysr_c1ccncc1"] = -np.log((-4.18*10**3*data["nei_1_fuk"]**2+np.exp(8.38*data["fuk_plus"]))**2) - 28.5 - 473/data["homo"]
# data["pysr_c1ccc2[nH]ccc2c1"] = -data["fuk_plus"]*(data["sasa"]+2.93/(data["mu"]+data["sasa"]+1.04))-0.252*data["homo_1"]*data["mu"]+31.1
# data["pysr_C1CCNC1"] = np.log(np.sqrt((0.0152*data["gam"]*(6.26-data["sasa"])+data["nei_3_chrg"])**2))+18.2-1/(data["mu"]+data["sasa"])
# data["pysr_c1ccc(P(c2ccccc2)c2ccccc2)cc1"] = -9.97*data["gam"]*data["nei_1_chrg"]+1/(data["chrg"]-0.893+3.04/data["eta"])+293/data["omega"]

# after revision
data["pysr_acyclic"] = data["homo"] + np.sqrt(data["dipole_solv"] * (data["mu"] * data["nei_3_fuk"] * data["omega_solv"]**2 + data["solv"] + 111.) - 1 / (0.138 * data["omega"] - 0.807))
data["pysr_c1ccccc1"] = -data["gam"] + (-data["homo"] - 2.53 - 1.61 / (data["dipole_solv"] + data["nei_1_chrg"] - 1.71)) * (2 * data["fuk_minus"] + data["homo"] + 12.4)
data["pysr_c1ccncc1"] = np.sqrt((7.49 * data["homo_1"] + 87.8)**2) + 10.0
data["pysr_c1ccc2[nH]ccc2c1"] = 33.2 + (data["homo_1"] + data["mu"])**2 * (-0.0684)
data["pysr_C1CCNC1"] = -np.sqrt(data["nei_3_chrg"]) + 18.7 - 1 / (data["homo"] + 11.6)
data["pysr_c1ccc(P(c2ccccc2)c2ccccc2)cc1"] = 3.98 * data["gam"] + 3.98 * np.sqrt((data["omega"] - 12.8)**2)


fig, ax = plt.subplots(3,2, figsize=(25,25))
for axx, gr in zip(itertools.chain.from_iterable(ax), data["group_murcko"].value_counts().head(6).index):
    plotdata = data[data["group_murcko"]==gr]
    minmax = [plotdata[f"pysr_{gr}"].min(),plotdata[f"pysr_{gr}"].max()]

    mol = Chem.MolFromSmiles(gr)
    if mol:
        img = Draw.MolToImage(mol, kekulize=True)
        # make white background transparent
        img = img.convert("RGBA")
        data_img = img.getdata()
        new_data = [(255, 255, 255, 0) if item[:3] == (255, 255, 255) else item for item in data_img]
        img.putdata(new_data)
        ins = inset_axes(axx, width="20%", height="20%", loc="lower right")
        ins.imshow(img)
        ins.axis("off")
    else:
        axx.text(0.8, 0.1, f"{gr}", transform=axx.transAxes)

    sns.scatterplot(plotdata, x=f"pysr_{gr}", y="N Params", ax=axx, hue="sol")
    axx.set(xlabel=f"$N_{{emp}}<{gr}> (n={len(plotdata)})$", ylabel="$N_M$")
    linr = linregress(plotdata[f"pysr_{gr}"], plotdata["N Params"])
    r2 = r2_score(plotdata["N Params"], plotdata[f"pysr_{gr}"]*linr.slope + linr.intercept)
    rmse = np.sqrt(mean_squared_error(plotdata["N Params"], plotdata[f"pysr_{gr}"]*linr.slope + linr.intercept))
    axx.text(0.3,0.87, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=axx.transAxes)
    axx.plot(minmax, [x*linr.slope + linr.intercept for x  in minmax], ls=":", linewidth=3, alpha=0.7, c="k")
    axx.legend(loc="upper left")

# pldat = data[(data["group_murcko"] == "C1CCNC1") & ~(data["Solvent"] == "ch2cl2")]
# minmax = [pldat[f"pysr_C1CCNC1"].min(),pldat[f"pysr_C1CCNC1"].max()]
# r2 = r2_score(pldat["N Params"], pldat[f"pysr_C1CCNC1"]*linr.slope + linr.intercept)
# rmse = np.sqrt(mean_squared_error(pldat["N Params"], pldat[f"pysr_C1CCNC1"]*linr.slope + linr.intercept))
# ax[2][0].text(0.7,0.4, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", color="r", transform=ax[2][0].transAxes)
fig.tight_layout()
fig.savefig("imgs/xtb_per_scaffold.pdf")

def get_pysr_prediction(row):
    scaffold = row["group_murcko"]
    
    # old
    # if scaffold == "acyclic":
    #     return np.log((-row["omega"] + row["solv"] + (-1.62*row["dipole_solv"]**2 + 1.62*row["omega_solv"])**2 * (row["homo"] - row["homo_1_solv"] - 0.889)**2)**2)
    # elif scaffold == "c1ccccc1":
    #     return row["dipole_solv"] - row["gam"] + (row["dipole_solv"]**2 * row["nei_1_fuk"]**2 + 0.248*(-2.25*row["chrg"] + row["homo"]))**3 + 22.7
    # elif scaffold == "c1ccncc1":
    #     return -np.log((-4.18e3*row["nei_1_fuk"]**2 + np.exp(8.38*row["fuk_plus"]))**2) - 28.5 - 473/row["homo"]
    # elif scaffold == "c1ccc2[nH]ccc2c1":
    #     return -row["fuk_plus"]*(row["sasa"] + 2.93/(row["mu"] + row["sasa"] + 1.04)) - 0.252*row["homo_1"]*row["mu"] + 31.1
    # elif scaffold == "C1CCNC1":
    #     return np.log(np.sqrt((0.0152*row["gam"]*(6.26 - row["sasa"]) + row["nei_3_chrg"])**2)) + 18.2 - 1/(row["mu"] + row["sasa"])
    # elif scaffold == "c1ccc(P(c2ccccc2)c2ccccc2)cc1":
    #     return -9.97*row["gam"]*row["nei_1_chrg"] + 1/(row["chrg"] - 0.893 + 3.04/row["eta"]) + 293/row["omega"]
    # else:
    #     return np.nan

    # revised
    if scaffold == "acyclic":
        return row["homo"] + np.sqrt(row["dipole_solv"] * (row["mu"] * row["nei_3_fuk"] * row["omega_solv"]**2 + row["solv"] + 111.) - 1 / (0.138 * row["omega"] - 0.807))
    elif scaffold == "c1ccccc1":
        return -row["gam"] + (-row["homo"] - 2.53 - 1.61 / (row["dipole_solv"] + row["nei_1_chrg"] - 1.71)) * (2 * row["fuk_minus"] + row["homo"] + 12.4)
    elif scaffold == "c1ccncc1":
        return np.sqrt((7.49 * row["homo_1"] + 87.8)**2) + 10.0
    elif scaffold == "c1ccc2[nH]ccc2c1":
        return 33.2 + (row["homo_1"] + row["mu"])**2 * (-0.0684)
    elif scaffold == "C1CCNC1":
        return -np.sqrt(row["nei_3_chrg"]) + 18.7 - 1 / (row["homo"] + 11.6)
    elif scaffold == "c1ccc(P(c2ccccc2)c2ccccc2)cc1":
        return 3.98 * row["gam"] + 3.98 * np.sqrt((row["omega"] - 12.8)**2)
    else:
        return row["dipole_solv"] + (row["chrg"] - row["dipole_solv"] * row["fuk_minus"] + 2.89)**2 * (row["homo"] + row["homo_1_solv"] + row["mu_solv"] + 31.0) + 14.5
    
data["pysr_prediction"] = data.apply(get_pysr_prediction, axis=1)

data.to_csv(snakemake.output["csv"])