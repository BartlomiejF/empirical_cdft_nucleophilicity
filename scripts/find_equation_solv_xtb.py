from pysr import PySRRegressor
import pandas as pd
from sklearn.metrics import r2_score
from snakemake.script import snakemake
import pathlib


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

data["mu"] = 0.5*(data["lumo"] + data["homo"])
data["eta"] = data["lumo"] - data["homo"]
data["gam"] = data["lumo"] - 2* data["homo"] + data["homo_1"]
data["omega"] = data["mu"]**2 / (2*data["eta"])

data["solv"] = data["Solvent"].replace({"water": 78.4,
                                        "ch2cl2": 8.9,
                                        "DMF": 36.7,
                                        "THF": 7.6,
                                        "acetonitrile": 35.7,
                                        "DMSO": 46.7,
                                        })

solvs = pd.read_csv(snakemake.input["solv"], index_col=False, names=["Solvent", "homo_1_solv", "homo_solv", "lumo_solv", "dipole_solv"])
solvs["mu_solv"] = 0.5*(solvs["lumo_solv"] + solvs["homo_solv"])
solvs["eta_solv"] = solvs["lumo_solv"] - solvs["homo_solv"]
solvs["gam_solv"] = solvs["lumo_solv"] - 2* solvs["homo_solv"] + solvs["homo_1_solv"]
solvs["omega_solv"] = solvs["mu_solv"]**2 / (2*solvs["eta_solv"])

data = pd.merge(data, solvs, on="Solvent")


for solv in data["Solvent"].unique():

    d = data[data["Solvent"]==solv]
    n_points = len(d)

    # Scale model complexity based on number of points
    if n_points < 30:
        maxsize = 12
        parsimony = 0.01
        select_k_features = 4
    elif n_points < 60:
        maxsize = 18
        parsimony = 0.005
        select_k_features = 6
    else:
        maxsize = 30
        parsimony = 0.0032
        select_k_features = 8

    model = PySRRegressor(
        maxsize=maxsize,
        niterations=1000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "exp", "log", "sqrt", "square",
            "inv(x) = 1/x",
            "cube(x) = x^3",
        ],
        extra_sympy_mappings={
            "inv":  lambda x: 1/x,
            "cube": lambda x: x**3,
        },
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="best",
        populations=20,
        population_size=100,
        parsimony=parsimony,
        complexity_of_operators={
            "/":    2,
            "log":  2,
            "exp":  2,
            "cube": 2,
        },
        ncycles_per_iteration=550,
        turbo=True,
        batching=True,
        batch_size=50,
        adaptive_parsimony_scaling=20.0,
        warmup_maxsize_by=0.0,
        weight_add_node=0.79,
        weight_insert_node=5.0,
        weight_delete_node=1.6,
        weight_simplify=0.01,
        weight_mutate_constant=0.048,
        weight_mutate_operator=0.47,
        weight_swap_operands=0.1,
        optimizer_algorithm="BFGS",
        optimizer_nrestarts=3,
        should_optimize_constants=True,
        deterministic=True,
        random_state=42,
        parallelism="serial",
        progress=True,
        select_k_features=select_k_features,
    )

    X = d.loc[:, ['mu', 'eta', 'gam', 'omega',
                 'fuk_minus', 'fuk_plus', "chrg",
                 "homo_1", "homo", "lumo",
                 "nei_1_fuk", "nei_2_fuk", "nei_3_fuk",
                 "nei_1_chrg", "nei_2_chrg", "nei_3_chrg",
                 "sasa", 'mu_solv', 'eta_solv', 'gam_solv', 'omega_solv',
                 "homo_1_solv", "homo_solv", "lumo_solv", "dipole_solv", "solv"
                 ]]
    y = d["N Params"]
    model.fit(X, y)

    eqs = model.equations_.copy()

    eqs["r2"] = None

    for i in eqs.index:
        y_pred = model.predict(X, index=i)
        eqs.loc[i, "r2"] = r2_score(y, y_pred)

    eqs_filtered = eqs[eqs["complexity"] <= maxsize * 0.75]

    eqs_sorted = eqs_filtered.sort_values(by="r2", ascending=False)

    top3 = eqs_sorted.head(3)

    latex_equations = []

    for eq_id in top3.index:
        latex_equations.append(model.latex(eq_id))

    with open(snakemake.output["eqs"], "a") as f:
        f.write(f"{solv} (n={n_points})\n")
        f.write("\n".join(latex_equations))
        f.write("\n\n")