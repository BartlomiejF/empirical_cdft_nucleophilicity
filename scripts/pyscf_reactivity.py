import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'

import sys
log_file = open(snakemake.log[0], 'w')
sys.stdout = log_file
sys.stderr = log_file

from pyscf import gto, dft, solvent, lib
from pyscf.hirshfeld import HirshfeldAnalysis
from pyscf.geomopt.geometric_solver import optimize
from cm5calculator import calc_cm5
from ase import Atoms
import numpy as np
import json

lib.num_threads(8)
lib.param.MAX_MEMORY = 16000
hartree2ev = 27.2114
bohr2ang   = 0.529177

# ── Snakemake inputs ──────────────────────────────────────────────────────────
xyz_file  = snakemake.input["xyz"]
chrg_file = snakemake.input["chrg"]
uhf_file  = snakemake.input["uhf"]

# ── Read inputs ───────────────────────────────────────────────────────────────
def read_xyz(xyz_file):
    with open(xyz_file) as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    coords = ''.join(lines[2:2+natoms])
    return coords

def read_charge(chrg_file):
    with open(chrg_file) as f:
        charge = int(f.read().strip())
    print(f'  Charge read from {chrg_file}: {charge}')
    return charge

def read_spin(uhf_file):
    with open(uhf_file) as f:
        spin = int(f.read().strip()) - 1
    print(f'  Spin read from {uhf_file}: {spin}')
    return spin

def get_spin_neighbour(n_electrons):
    """Determine spin for N±1 state based on electron count."""
    if (n_electrons - 1) % 2 == 0:
        return 0   # even electrons -> singlet
    else:
        return 1   # odd electrons -> doublet

coords     = read_xyz(xyz_file)
mol_charge = read_charge(chrg_file)
spin_N     = read_spin(uhf_file)

# ── Detect N±1 spins ──────────────────────────────────────────────────────────
_mol_tmp = gto.Mole()
_mol_tmp.atom    = coords
_mol_tmp.charge  = mol_charge
_mol_tmp.basis   = 'sto-3g'
_mol_tmp.verbose = 0
_mol_tmp.build()
n_electrons = _mol_tmp.nelectron
spin_cat    = get_spin_neighbour(n_electrons)      # N-1: remove one electron
spin_an     = get_spin_neighbour(n_electrons + 2)  # N+1: add one electron (check parity of n+1)

# fix: parity check should be on actual n±1 electron count
spin_cat = 0 if (n_electrons - 1) % 2 == 0 else 1
spin_an  = 0 if (n_electrons + 1) % 2 == 0 else 1

print(f'\n  Total electrons : {n_electrons}')
print(f'  Charge          : {mol_charge}')
print(f'  Spin N          : {spin_N} (multiplicity {spin_N+1})')
print(f'  Spin N-1        : {spin_cat} (multiplicity {spin_cat+1})')
print(f'  Spin N+1        : {spin_an} (multiplicity {spin_an+1})')

# ── Helper functions ──────────────────────────────────────────────────────────
def build_mf(coords, charge, spin, xc='PBE0', basis='ma-def2-svp', eps=78.3553, unit='Angstrom'):
    mol = gto.Mole()
    mol.atom    = coords
    mol.charge  = charge
    mol.spin    = spin
    mol.basis   = basis
    mol.unit    = unit
    mol.verbose = 4
    mol.build()

    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf.xc          = xc
    mf.grids.level = 5
    mf.max_memory  = 16000
    mf = solvent.ddCOSMO(mf)
    mf.with_solvent.eps = eps
    return mf

def run_sp(coords, charge, spin, **kwargs):
    mf = build_mf(coords, charge, spin, **kwargs)
    mf.kernel()
    return mf.mol, mf

def get_orbital_energies(mf, mol, spin):
    if spin == 0:
        nocc   = mol.nelectron // 2
        mo_e   = mf.mo_energy
        homo_1 = mo_e[nocc - 2] * hartree2ev
        homo   = mo_e[nocc - 1] * hartree2ev
        lumo   = mo_e[nocc]     * hartree2ev
    else:
        mo_e_a = mf.mo_energy[0]
        mo_e_b = mf.mo_energy[1]
        nocc_a = mol.nelec[0]
        nocc_b = mol.nelec[1]

        homo_a = mo_e_a[nocc_a - 1] * hartree2ev
        homo_b = mo_e_b[nocc_b - 1] * hartree2ev
        homo   = max(homo_a, homo_b)
        homo_1 = sorted([mo_e_a[nocc_a - 2], mo_e_b[nocc_b - 1]],
                         reverse=True)[0] * hartree2ev
        lumo   = mo_e_b[nocc_b] * hartree2ev

        print(f'  HOMO (alpha): {homo_a:.4f} eV')
        print(f'  HOMO (beta) : {homo_b:.4f} eV')
        print(f'  LUMO (alpha): {mo_e_a[nocc_a] * hartree2ev:.4f} eV')
        print(f'  LUMO (beta) : {lumo:.4f} eV')

    return homo_1, homo, lumo

def get_hirsh_cm5(mf, mol, molecule_ase):
    H     = HirshfeldAnalysis(mf).run()
    hirsh = H.result['charge_eff']
    cm5   = calc_cm5(molecule_ase, hirsh)
    return hirsh, cm5

# ── Step 1: Geometry optimization ────────────────────────────────────────────
print(f'\n>>> Step 1: Geometry optimization (charge={mol_charge}, spin={spin_N}, ma-def2-svp)')
mf_opt = build_mf(coords, charge=mol_charge, spin=spin_N, basis='ma-def2-svp', unit='Angstrom')
mf_opt.kernel()
mol_opt = optimize(mf_opt)
mol_opt.tofile(snakemake.output["xyz_opt"])
print(f'  Optimized geometry saved to {snakemake.output["xyz_opt"]}')

opt_coords = mol_opt.atom
atoms_list = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
coords_ang = mol_opt.atom_coords() * bohr2ang
molecule   = Atoms(symbols=atoms_list, positions=coords_ang)

# ── Step 2: Ground state SP (N) ───────────────────────────────────────────────
print(f'\n>>> Step 2: Ground state SP (charge={mol_charge}, spin={spin_N}, ma-def2-svp)')
mol_N, mf_N = run_sp(opt_coords, charge=mol_charge, spin=spin_N, unit='Bohr')

homo_1, homo, lumo = get_orbital_energies(mf_N, mol_N, spin_N)
gap   = lumo - homo
ip    = -homo
ea    = -lumo
eta   = lumo - homo
mu    = 0.5 * (lumo + homo)
omega = mu**2 / (2 * eta)
gamma = lumo - 2*homo + homo_1
NG    = -1*omega*(1 + mu*gamma/(3*eta**2) - 4*eta**2/(mu*gamma) + 4*eta**4/(3*mu**2*gamma**2))
ND    = homo - (-9.061)

dip       = mf_N.dip_moment(unit='Debye', verbose=0)
dip_total = float(np.linalg.norm(dip))

hirsh_N, cm5_N = get_hirsh_cm5(mf_N, mol_N, molecule)

# ── Step 3: Oxidized state SP (N-1) ───────────────────────────────────────────
print(f'\n>>> Step 3: Oxidized state SP (charge={mol_charge+1}, spin={spin_cat}, ma-def2-svp)')
mol_cat, mf_cat = run_sp(opt_coords, charge=mol_charge+1, spin=spin_cat, unit='Bohr')
hirsh_cat, cm5_cat = get_hirsh_cm5(mf_cat, mol_cat, molecule)

# ── Step 4: Reduced state SP (N+1) ────────────────────────────────────────────
print(f'\n>>> Step 4: Reduced state SP (charge={mol_charge-1}, spin={spin_an}, ma-def2-svp)')
mol_an, mf_an = run_sp(opt_coords, charge=mol_charge-1, spin=spin_an, unit='Bohr')
hirsh_an, cm5_an = get_hirsh_cm5(mf_an, mol_an, molecule)

# ── Fukui indices ─────────────────────────────────────────────────────────────
# sign convention: atomic charges (Z - electrons)
f_minus_hirsh = hirsh_cat - hirsh_N    # nucleophilic attack
f_minus_cm5   = cm5_cat   - cm5_N
f_plus_hirsh  = hirsh_N   - hirsh_an   # electrophilic attack
f_plus_cm5    = cm5_N     - cm5_an
f_zero_hirsh  = (f_plus_hirsh + f_minus_hirsh) / 2   # radical attack
f_zero_cm5    = (f_plus_cm5   + f_minus_cm5)   / 2

# ── Write output JSON ─────────────────────────────────────────────────────────
results = {
    'molecule'  : xyz_file,
    'charge'    : mol_charge,
    'spin_N'    : spin_N,
    'spin_cat'  : spin_cat,
    'spin_an'   : spin_an,
    'atoms'     : atoms_list,
    'orbital_energies': {
        'homo_1_eV' : homo_1,
        'homo_eV'   : homo,
        'lumo_eV'   : lumo,
        'gap_eV'    : gap,
    },
    'reactivity_indices': {
        'ip_eV'  : ip,
        'ea_eV'  : ea,
        'eta_eV' : eta,
        'mu_eV'  : mu,
        'omega_eV': omega,
        'gamma'  : gamma,
        'NG'     : NG,
        'ND'     : ND,
    },
    'dipole_moment_Debye'   : dip_total,
    'hirshfeld_N'           : hirsh_N.tolist(),
    'cm5_N'                 : cm5_N.tolist(),
    'hirshfeld_cat'         : hirsh_cat.tolist(),
    'cm5_cat'               : cm5_cat.tolist(),
    'hirshfeld_an'          : hirsh_an.tolist(),
    'cm5_an'                : cm5_an.tolist(),
    'fukui_minus_hirshfeld' : f_minus_hirsh.tolist(),
    'fukui_minus_cm5'       : f_minus_cm5.tolist(),
    'fukui_plus_hirshfeld'  : f_plus_hirsh.tolist(),
    'fukui_plus_cm5'        : f_plus_cm5.tolist(),
    'fukui_zero_hirshfeld'  : f_zero_hirsh.tolist(),
    'fukui_zero_cm5'        : f_zero_cm5.tolist(),
}

with open(snakemake.output["json"], 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n  Results saved to {snakemake.output["json"]}')

log_file.flush()
log_file.close()