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



x = data.loc[:, ['mu', 'eta', 'gam', 'omega',
                 'fuk_minus', 'fuk_plus', "chrg",
                 "homo_1", "homo", "lumo",
                 "nei_1_fuk", "nei_2_fuk", "nei_3_fuk",
                 "nei_1_chrg", "nei_2_chrg", "nei_3_chrg",
                 "sasa", 'mu_solv', 'eta_solv', 'gam_solv', 'omega_solv',
                 "homo_1_solv", "homo_solv", "lumo_solv", "dipole_solv", "solv"
                 ]]
y = data["N Params"]


model = PySRRegressor(
    maxsize=30,
    niterations=1000,
    
    # Operators - Enhanced for chemistry
    binary_operators=["+", "-", "*", "/", "pow"],  # Added pow for power laws
    unary_operators=[
        "exp", "log", "sqrt", "square",  # Added square - common in chemistry
        "inv(x) = 1/x",
        "cube(x) = x^3",  # Useful for volume/distance relationships
    ],
    extra_sympy_mappings={
        "inv": lambda x: 1/x,
        "cube": lambda x: x**3,
    },
    
    # Loss and selection
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    model_selection="best",  # Consider "accuracy" to see Pareto frontier
    
    # Population parameters - Optimized
    populations=20,  # Reduced from 30 - diminishing returns beyond 20
    population_size=100,  # Reduced from 200 - faster with similar quality
    
    # Complexity control - CRITICAL for chemistry
    parsimony=0.0032,  # Slightly increased for simpler models
    complexity_of_operators={
        "/": 2,      # Division can cause instability
        "pow": 3,    # Power operations are complex
        "log": 2,    # Logarithms need careful handling
        "exp": 2,    # Exponentials grow rapidly
        "cube": 2,   # Moderate complexity
    },
    
    # Performance upgrades
    ncycles_per_iteration=550,  # Up from 10 - MUCH better evolution
    turbo=True,  # Faster evaluation on CPU
    batching=True,  # Enable if you have >5000 data points
    batch_size=50,  # Adjust based on your dataset size
    
    # Search improvements
    adaptive_parsimony_scaling=20.0,  # Dynamic complexity penalty
    warmup_maxsize_by=0.0,  # Start with full complexity immediately
    
    # Mutation weights - Fine-tuned for chemistry
    weight_add_node=0.79,
    weight_insert_node=5.0,  # Favor insertions - builds structure
    weight_delete_node=1.6,
    weight_simplify=0.01,  # Algebraic simplification
    weight_mutate_constant=0.048,
    weight_mutate_operator=0.47,
    weight_swap_operands=0.1,
    
    # Optimization
    optimizer_algorithm="BFGS",  # Better than default NelderMead for smooth landscapes
    optimizer_nrestarts=3,  # Multiple restarts for constant optimization
    
    # Constraints for physical meaning
    constraints={
        "pow": (-1, 5),  # Limit power range to physically reasonable values
        "/": (9, 9),     # Allow division everywhere (adjust if needed)
    },
    
    # Useful features
    should_optimize_constants=True,
    deterministic=True,  # Set False for stochastic search if exploring
    random_state=99,  # Reproducibility
    parallelism="serial",
    
    # Output control
    # verbosity=1,  # Set to 0 for less output, 2 for debugging
    progress=True,
    select_k_features=8,
)

model.fit(x,y)

eqs = model.equations_.copy()

eqs["r2"] = None

for i in eqs.index:
    y_pred = model.predict(x, index=i)
    eqs.loc[i, "r2"] = r2_score(y, y_pred)


eqs_filtered = eqs[eqs["complexity"] <= 20]

eqs_sorted = eqs_filtered.sort_values(by="r2", ascending=False)

top3 = eqs_sorted.head(3)

latex_equations = []

for eq_id in top3.index:
    latex_equations.append(model.latex(eq_id))

with open(snakemake.output["xtb"], "w") as f:
    f.write("\n".join(latex_equations))
