import pathlib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from snakemake.script import snakemake
from rdkit import Chem
from rdkit.Chem import SaltRemover


def generate_3d_from_smiles(smiles, outfile, num_confs=20):
    """Generate a single 3D conformer from SMILES and save as MOL file."""
    print(f"reading {smiles}\n cid: {outfile.split('raw_')[1]}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"❌ Invalid SMILES: {smiles}")

    desaler = SaltRemover.SaltRemover()
    mol = desaler.StripMol(mol)

    mol = Chem.AddHs(mol)

    # ETKDGv3 embedding (robust and chemistry-aware)
    params = rdDistGeom.ETKDGv3()
    # params.numThreads = 10

    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)
        if not conf_ids:
            print(f"❌ Failed to embed conformer for {smiles}")
            raise RuntimeError(f"Failed to generate 3D geometry for {smiles} after {max_attempts} attempts.")
        else:
            # Just pick the first conformer
            best_conf_id = conf_ids[0]

            # Save only that conformer
            Chem.MolToMolFile(mol, outfile, confId=best_conf_id)
            print(f"✅ Saved conformer {best_conf_id} to {outfile}")
    except:
        success = False
        for attempt in range(num_confs):
            try:
                status = rdDistGeom.EmbedMolecule(mol, params)
                if status != 0:
                    print(f"⚠️ Embedding failed on attempt {attempt+1}")
                    continue
                else:
                    success = True
                    break
            
            except Exception as e:
                print(f"⚠️ Error on attempt {attempt+1}: {e}")
            
        if success:
            Chem.MolToMolFile(mol, outfile)
            print(f"✅ Saved to {outfile}")

with open(snakemake.input["smi"], "r") as f:
    smiles = f.read()

smiles = smiles.strip()

path = pathlib.Path(snakemake.output["mol"])
path.parent.mkdir(parents=True, exist_ok=True)

generate_3d_from_smiles(smiles, str(path), snakemake.params["num_confs"])