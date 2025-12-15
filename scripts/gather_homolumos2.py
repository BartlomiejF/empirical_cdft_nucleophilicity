from rdkit import Chem
import morfeus
import pathlib
from snakemake.script import snakemake
import check_nucleo_sites

outpaths = [pathlib.Path(pth) for pth in snakemake.output["paths"]]
outpaths = {str(k.stem): k for k in outpaths}
for pth in outpaths.values():
    pth.parent.mkdir(parents=True, exist_ok=True)

outprops = {k: [] for k in outpaths.keys()}

for path_props, path_mol, xyz_path in zip(snakemake.input["propsfiles"],
                                              snakemake.input["molfiles"],
                                              snakemake.input["xyz_paths"]):
    path_props = pathlib.Path(path_props)

    mol = Chem.MolFromMolFile(str(path_mol), removeHs=False)
    if mol == None:
        mol = Chem.MolFromMolFile(str(path_mol), sanitize=False, removeHs=False)

    cid = path_props.stem
    with open(path_props, "r") as f:
        props = f.readlines()
    
    props_hl = props[1:4]
    props_hl = [lin.strip().split() for lin in props_hl]
    props_hl = [props_hl[0][-1], props_hl[1][-2], props_hl[2][-2]]

    charges = []
    props_fuk = []
    char = True
    for line in props[5:]:
        if "f(+)" in line:
            char = False
            continue
        if char:
            charges.append(line)
        else:
            props_fuk.append(line)
        

    props_fuk_neg = [float(lin.strip().split()[2]) for lin in props_fuk]
    props_fuk_pos = [float(lin.strip().split()[1]) for lin in props_fuk]

    charges = [float(lin.strip().split()[-1]) for lin in charges]

    rank = [fuk*chrg for fuk, chrg in zip(props_fuk_neg, charges)]

    atoms = mol.GetAtoms()

    sites, *_ = check_nucleo_sites.find_nucleophilic_sites(mol)
    # if rank_ind in sites:
    #     print(True)
    # else:
    #     print(False, cid)
    #     with open("not_in_sites", "a") as f:
    #         f.write(f"{cid}\n")

    if len(sites) == 0:
        sites = list(range(len(rank)))

    rank_sites = [rank[i] for i in sites]
    rank_ind = rank.index(min(rank_sites))

    neighbours = atoms[rank_ind].GetNeighbors()
    neighbours = [[nei.GetIdx(), nei.GetAtomicNum(), nei.GetNeighbors(), charges[nei.GetIdx()]] for nei in neighbours]
    for n in neighbours:
        n[2] = sum([a.GetAtomicNum() for a in n[2]]) - atoms[rank_ind].GetAtomicNum()
    # fuk_neighbours_fuk = [[0,0,0]]*3
    # fuk_neighbours_chrg = [[0,0,0]]*3
    fuk_neighbours_fuk = [0]*3
    fuk_neighbours_chrg = [0]*3
    neighbours.sort(reverse=True, key=lambda x: (x[1], x[2], -x[3]))

    for i, ind in enumerate(neighbours[:3]):
        fuk_neighbours_fuk[i] += props_fuk_neg[ind[0]]
        fuk_neighbours_chrg[i] += charges[ind[0]]

    # fuk_neighbours_fuk.sort(reverse=True, key=lambda x: (x[1], x[2], x[0]))
    # fuk_neighbours_chrg.sort(reverse=True, key=lambda x: (x[1], x[2], x[0]))

    elems, coords = morfeus.read_geometry(str(xyz_path))
    sasa = morfeus.SASA(elems, coords)
    sasa_fuk_max = sasa.atom_areas[rank_ind+1]

    row = [cid]
    row.extend(props_hl)
    row.append(str(props_fuk_neg[rank_ind]))
    row.append(str(props_fuk_pos[rank_ind]))
    row.append(str(charges[rank_ind]))
    row.extend([str(x) for x in fuk_neighbours_fuk])
    row.extend([str(x) for x in fuk_neighbours_chrg])
    row.append(str(sasa_fuk_max))
    # print(row)
    row = ",".join(row)

    for k, v in outprops.items():
        if k in str(path_props):
            v.append(row)

for k, v in outprops.items():
    with open(outpaths[k], "w") as f:
        f.write("\n".join(v))
