import pandas as pd
import pathlib
from snakemake.script import snakemake

data = pd.read_csv(snakemake.input["csvs"], header=None)

for path_str in snakemake.output["smi"]:
    path = pathlib.Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)

    cid = path.stem  # remove '.smi' to match CSV
    sol = str(path.parent).split("/")[-1]
    row = data[(data.iloc[:,0] == int(cid)) & (data.iloc[:,3] == sol)]

    with open(path, "w") as f:
        f.write(str(row[1].iloc[0]))
