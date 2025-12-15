from rdkit import Chem
import numpy as np

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
                 'Pyridine_like_nitrogen': '[#7X2;$([nX2](:*):*),$([#7X2;R](=[*;R])[*;R]):1]>>[CH3][#7X3+:1]', # added due to rxn100
                 'anion_with_charge_minus1': '[*-:1]>>[CH3][*+0:1]', # added to capture additional sites 
                 'double_bond': '[*;!$([!X4;!#1;!#6:1])+0:1]=[*+0:2]>>[CH3][*:1]-[*+1:2]', # added to capture additional sites
                 'double_bond_neighbouratom_with_charge_plus1': '[*;!$([!X4;!#1;!#6:1])+0:1]=[*+1:2]>>[CH3][*:1]-[*+2:2]', # added to capture additional sites
                 'triple_bond': '[*;!$([!X4;!#1;!#6:1])+0:1]#[*+0:2]>>[CH3][*:1]=[*+1:2]', # added to capture additional sites 
                 'triple_bond_neighbouratom_with_charge_plus1': '[*;!$([!X4;!#1;!#6:1])+0:1]#[*+1:2]>>[CH3][*:1]=[*+2:2]', # added to capture additional sites
                 'atom_with_lone_pair': '[!X4;!#1;!#6:1]>>[CH3][*+1:1]', # added to capture additional sites
                }


def find_nucleophilic_sites_file(molfile):
    copy_rdkit_mol = Chem.MolFromMolFile(molfile, removeHs=False)
    if copy_rdkit_mol == None:
        copy_rdkit_mol = Chem.MolFromMolFile(str(molfile), sanitize=False, removeHs=False)
    Chem.Kekulize(copy_rdkit_mol)

    nuc_sites = []
    nuc_names = []
    nuc_smirks = []
    for name, smirks in n_smirks_dict.items():
        smarts = smirks.split('>>')[0]
        if copy_rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            sites = [x[0] for x in copy_rdkit_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=False)]
            # sites = remove_identical_atoms(copy_rdkit_mol, sites)
            for site in sites:
                if site not in nuc_sites:
                    nuc_sites.append(site)
                    nuc_names.append(name)
                    nuc_smirks.append(smirks)
    
    # Remove sites with same canonical rank
    idx_list = []
    rank_kept = []
    atom_rank = list(Chem.CanonicalRankAtoms(copy_rdkit_mol, breakTies=False))
    for idx, atom in enumerate(nuc_sites):
        if atom_rank[atom] not in rank_kept:
            rank_kept.append(atom_rank[atom])
            idx_list.append(idx)

    nuc_sites = np.array(nuc_sites)[idx_list].tolist()
    nuc_names = np.array(nuc_names)[idx_list].tolist()
    nuc_smirks = np.array(nuc_smirks)[idx_list].tolist()

    return nuc_sites, nuc_names, nuc_smirks


def find_nucleophilic_sites(mol):
    copy_rdkit_mol = Chem.Mol(mol, True)
    Chem.Kekulize(copy_rdkit_mol)

    nuc_sites = []
    nuc_names = []
    nuc_smirks = []
    for name, smirks in n_smirks_dict.items():
        smarts = smirks.split('>>')[0]
        if copy_rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            sites = [x[0] for x in copy_rdkit_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=False)]
            # sites = remove_identical_atoms(copy_rdkit_mol, sites)
            for site in sites:
                if site not in nuc_sites:
                    nuc_sites.append(site)
                    nuc_names.append(name)
                    nuc_smirks.append(smirks)
    
    # Remove sites with same canonical rank
    idx_list = []
    rank_kept = []
    atom_rank = list(Chem.CanonicalRankAtoms(copy_rdkit_mol, breakTies=False))
    for idx, atom in enumerate(nuc_sites):
        if atom_rank[atom] not in rank_kept:
            rank_kept.append(atom_rank[atom])
            idx_list.append(idx)

    nuc_sites = np.array(nuc_sites)[idx_list].tolist()
    nuc_names = np.array(nuc_names)[idx_list].tolist()
    nuc_smirks = np.array(nuc_smirks)[idx_list].tolist()

    return nuc_sites, nuc_names, nuc_smirks