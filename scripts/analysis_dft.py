import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
import morfeus
import check_nucleo_sites
from scipy.stats import linregress
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
import json

plt.rcParams["font.size"] = 25
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["legend.title_fontsize"] = 15
plt.rcParams["axes.labelsize"] = 20

# ── Load JSON data ─────────────────────────────────────────────────────────────
data = []
for fil in snakemake.input["jsons"]:
    with open(fil, "r") as f:
        jsn = json.load(f)
        cid = int(jsn["molecule"].split("/")[-1][:-4])
        data.append({
            "ID"        : cid,
            "NG"         : jsn["reactivity_indices"]["NG"],
            "ND"         : jsn["reactivity_indices"]["ND"],
            "homo"       : jsn["orbital_energies"]["homo_eV"],
            "lumo"       : jsn["orbital_energies"]["lumo_eV"],
            "homo_1"     : jsn["orbital_energies"]["homo_1_eV"],
            "mol"        : Chem.MolFromMolFile(f"output/pyscf/{cid}_opt.mol",
                                               sanitize=False, removeHs=False),
            "atoms"      : jsn["atoms"],
            "fukui_minus": jsn["fukui_minus_hirshfeld"],
            "fukui_plus" : jsn["fukui_plus_hirshfeld"],
            "cm5"        : jsn["cm5_N"],
        })

mayr = pd.read_csv(snakemake.input["mayr"], names=["cid", "smiles", "N", "solvent"])

# ── Feature extraction ─────────────────────────────────────────────────────────
rows = []
for dat in data:
    cid       = dat["ID"]
    mol       = dat["mol"]
    fuks      = dat["fukui_minus"]
    fuks_plus = dat["fukui_plus"]
    chrgs     = dat["cm5"]

    rank  = [fuk * chrg for fuk, chrg in zip(fuks, chrgs)]
    atoms = list(mol.GetAtoms())
    sites, *_ = check_nucleo_sites.find_nucleophilic_sites(mol)

    if len(sites) == 0:
        sites = list(range(len(rank)))

    rank_sites = [rank[i] for i in sites]
    rank_ind   = rank.index(min(rank_sites))

    neighbours = atoms[rank_ind].GetNeighbors()
    neighbours = [
        [nei.GetIdx(), nei.GetAtomicNum(), nei.GetNeighbors(), chrgs[nei.GetIdx()]]
        for nei in neighbours
    ]
    for n in neighbours:
        n[2] = sum([a.GetAtomicNum() for a in n[2]]) - atoms[rank_ind].GetAtomicNum()

    fuk_neighbours_fuk  = [0] * 3
    fuk_neighbours_chrg = [0] * 3
    neighbours.sort(reverse=True, key=lambda x: (x[1], x[2], -x[3]))

    for i, ind in enumerate(neighbours[:3]):
        fuk_neighbours_fuk[i]  += fuks[ind[0]]
        fuk_neighbours_chrg[i] += chrgs[ind[0]]

    elems, coords = morfeus.read_geometry(f"output/pyscf/{cid}_opt.xyz")
    sasa          = morfeus.SASA(elems, coords)
    sasa_fuk_max  = sasa.atom_areas[rank_ind + 1]

    rows.append([
        cid,
        dat["NG"],
        dat["ND"],
        dat["homo"],
        dat["lumo"],
        dat["homo_1"],
        fuks[rank_ind],
        fuks_plus[rank_ind],
        chrgs[rank_ind],
        *fuk_neighbours_fuk,
        *fuk_neighbours_chrg,
        sasa_fuk_max,
    ])

rows = pd.DataFrame(rows, columns=[
    "cid", "NG", "ND", "homo", "lumo", "homo_1",
    "fuk_minus", "fuk_plus", "chrg",
    "nei_1_fuk", "nei_2_fuk", "nei_3_fuk",
    "nei_1_chrg", "nei_2_chrg", "nei_3_chrg",
    "sasa"
])

data = pd.merge(mayr, rows, on="cid")

data["solv"] = data["solvent"].replace({
    "water"       : 78.4,
    "ch2cl2"      : 8.9,
    "DMF"         : 36.7,
    "THF"         : 7.6,
    "acetonitrile": 35.7,
    "DMSO"        : 46.7,
})

data["dipole_solv"] = data["solvent"].replace({"water": 2.6089})
data["omega_solv"]  = 0.7003
data["homo_1_solv"] = -11.2164
data["homo_solv"] = -9.2129
data["mu_solv"] = -3.8691

data["mu"]    = 0.5 * (data["lumo"] + data["homo"])
data["eta"]   = data["lumo"] - data["homo"]
data["gam"]   = data["lumo"] - 2 * data["homo"] + data["homo_1"]
data["omega"] = data["mu"]**2 / (2 * data["eta"])

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_regression(data, x_col, y_col, xlabel, ylabel, outpath, text_pos=(0.1, 0.8), **kwargs):
    fig, ax = plt.subplots(figsize=(15, 8))
    linr    = linregress(data[x_col], data[y_col])
    minmax  = [data[x_col].min(), data[x_col].max()]
    r2      = r2_score(data[y_col], data[x_col] * linr.slope + linr.intercept)
    rmse    = np.sqrt(mean_squared_error(data[y_col], data[x_col] * linr.slope + linr.intercept))
    sns.scatterplot(data, x=x_col, y=y_col, ax=ax, **kwargs)
    ax.plot(minmax, [x * linr.slope + linr.intercept for x in minmax],
            ls=":", linewidth=3, alpha=0.7, c="k")
    ax.text(*text_pos, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

plot_regression(data, "NG",  "N", "$N_G$", "$N_M$", "imgs/ng_dft.pdf")
plot_regression(data, "ND",  "N", "$N_D$", "$N_M$", "imgs/nd_dft.pdf")


# old
# data["pysr"] = (2.6089 + (2.6089 * data["fuk_minus"] - data["chrg"] - 2.69)**2
#                 * (data["homo"] + (-9.2128) + -3.8691 + 30.5) + 13.9)

# revision
data["pysr"] = data["dipole_solv"] + (data["chrg"] - data["dipole_solv"] * data["fuk_minus"] + 2.89)**2 * (data["homo"] + data["homo_1_solv"] + data["mu_solv"] + 31.0) + 14.5

plot_regression(data, "pysr", "N", "$N_{emp}$", "$N_M$", "imgs/pysr_dft.pdf",
                text_pos=(0.1, 0.3))

# old
# data["pysr_water"] = ((0.968 - data["nei_1_chrg"])
#                       / (data["nei_1_fuk"] / (1.57 - data["gam"])
#                          - 8.55 * data["nei_2_chrg"] + 0.604) + 13.3)

# revision
data["pysr_water"] = -33.3 * data["nei_2_fuk"] + 15.5 - 1 / (data["nei_1_chrg"] + (data["chrg"]**2 - 0.248)**3 / (data["nei_1_chrg"] + data["nei_1_fuk"])**3)

plot_regression(data, "pysr_water", "N", "$N_{emp}[water]$", "$N_M$",
                "imgs/pysr_water_dft.pdf", text_pos=(0.8, 0.8))

# ── Scaffold grouping ──────────────────────────────────────────────────────────
def group_by_scaffold(smiles_series, return_scaffold_smiles=False):
    scaffold_dict   = {}
    scaffold_labels = []
    next_id = 0

    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f'  Invalid SMILES: {smiles}')
            scaffold_labels.append(None)
            continue

        scaffold        = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() > 0 else 'acyclic'

        if return_scaffold_smiles:
            scaffold_labels.append(scaffold_smiles)
        else:
            if scaffold_smiles not in scaffold_dict:
                scaffold_dict[scaffold_smiles] = next_id
                next_id += 1
            scaffold_labels.append(scaffold_dict[scaffold_smiles])

    return pd.Series(scaffold_labels, index=smiles_series.index)

data["group_murcko"] = group_by_scaffold(data["smiles"], return_scaffold_smiles=True)

# ── Scaffold-based predictions ─────────────────────────────────────────────────
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
d = data.dropna(subset=["pysr_prediction"])

plot_regression(d, "pysr_prediction", "N", "$N_{emp}<scaffold>$", "$N_M$",
                "imgs/pysr_scaffold_dft.pdf", text_pos=(0.5, 0.3))


data.rename(columns={"cid": "ID"}, inplace=True)

data.to_csv(snakemake.output["csv"])

