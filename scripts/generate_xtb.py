from rdkit import Chem
from rdkit.Chem import Descriptors
from snakemake.script import snakemake
# from rdkit.Chem  import AllChem
import pathlib

def xtb_generator(molecule) -> tuple:
    mol = Chem.AddHs(molecule)

    # AllChem.EmbedMolecule(mol)

    charge = Chem.GetFormalCharge(mol)

    spin_multiplicity = (Descriptors.NumValenceElectrons(mol))%2 *2*0.5 + 1 

    return charge, spin_multiplicity

path = pathlib.Path(snakemake.input["mol"])
path.parents[0].mkdir(parents=True, exist_ok=True)

mol = Chem.MolFromMolFile(str(path))
if mol == None:
    mol = Chem.MolFromMolFile(str(path), sanitize=False)
    
chrg, multiplicity = xtb_generator(mol)

with open(path.parents[0]/".CHRG", "w") as f:
    f.write(str(int(chrg)))

with open(path.parents[0]/".UHF", "w") as f:
    f.write(str(int(multiplicity)))