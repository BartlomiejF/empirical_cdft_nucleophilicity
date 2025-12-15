import pathlib
from snakemake.script import snakemake


out = []
for propsfile in snakemake.input["propsfiles"]:
    pth = pathlib.Path(propsfile)

    solv = pth.stem

    with open(pth, "r") as f:
        props = f.readlines()
    
    props_hl = props[1:4]
    props_hl = [lin.strip().split() for lin in props_hl]
    props_hl = [props_hl[0][-1], props_hl[1][-2], props_hl[2][-2]]

    dipole_moment = props[4].split()[-1]

    # out.append(solv)
    # out.extend(props_hl)
    # out.append(dipole_moment)
    out.append(",".join([solv, *props_hl, dipole_moment]))

with open(snakemake.output["gathered"], "w") as f:
    f.write("\n".join(out))