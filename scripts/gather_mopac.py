import pathlib
from snakemake.script import snakemake

outpaths = [pathlib.Path(pth) for pth in snakemake.output["paths"]]
outpaths = {str(k.stem).split("_")[1]: k for k in outpaths}
for pth in outpaths.values():
    pth.parent.mkdir(parents=True, exist_ok=True)

outprops = {k: [] for k in outpaths.keys()}

for path_props in snakemake.input["props"]:
    path_props = pathlib.Path(path_props)

    cid = path_props.stem
    with open(path_props, "r") as f:
        fil = f.readlines()

    uhf = "UHF" in fil[0]
    no_levels = fil[0].split("=")[1].strip()
    fill = [x for x in fil[1:] if len(x) >5]
    rootnos = fill[0::3]
    energies = fill[2::3]
    energies_sp = [x.strip().split() for x in energies]
    rootnos_sp = [x.strip().split()[2:] for x in rootnos]
    indices = [(i, x.index(no_levels)) for i, x in enumerate(rootnos_sp) if no_levels in x][0]

    if uhf:
        lumo_ind = indices[1]
        homo_ind = lumo_ind - 1
        homo_1_ind = lumo_ind - 2
        lumo = energies_sp[indices[0]][lumo_ind]
        if homo_ind < 0:
            homo = energies_sp[indices[0]-1][homo_ind]
        else:
            homo = energies_sp[indices[0]][homo_ind]
        if homo_1_ind < 0:
            homo_1 = energies_sp[indices[0]-1][homo_1_ind]
        else:
            homo_1 = energies_sp[indices[0]][homo_1_ind]
    else:
        homo_ind = indices[1]
        lumo_ind = homo_ind + 1
        homo_1_ind = homo_ind - 1
        homo = energies_sp[indices[0]][homo_ind]
        if lumo_ind > len(energies_sp[0])-1:
            lumo = energies_sp[indices[0]+1][0]
        else:
            lumo = energies_sp[indices[0]][lumo_ind]
        if homo_1_ind < 0:
            homo_1 = energies_sp[indices[0]-1][homo_1_ind]
        else:
            homo_1 = energies_sp[indices[0]][homo_1_ind]

    row = [cid]
    row.extend([homo_1, homo, lumo])

    row = ",".join(row)

    print(row)

    for k, v in outprops.items():
        if k in str(path_props):
            v.append(row)

for k, v in outprops.items():
    with open(outpaths[k], "w") as f:
        f.write("\n".join(v))
