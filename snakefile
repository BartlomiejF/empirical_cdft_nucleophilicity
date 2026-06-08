def get_cids(file_path):
    import pandas as pd
    return pd.read_csv(file_path)["ID"].astype(str).tolist()


solvents = ["water", "THF", "acetonitrile", "DMSO", "DMF", "ch2cl2"]

# Precompute SMI targets
smi_targets = []
for solv in solvents:
    for cid in get_cids(f"input/dbs/{solv}.csv"):
        smi_targets.append(f"input/smiles/{solv}/{cid}.smi")


rule all:
    input:
        "output/equation/eq_solvents_xtb.tex",
        "output/equation/eq_all_xtb.tex",
        "output/equation/eq_scaffolds_xtb.tex",
        "output/dbs/solvents.csv",
        expand("input/solvents/{solv}/{solv}.props", solv=solvents),
        "imgs/ng_xtb.pdf",
        "imgs/nd_xtb.pdf",
        "imgs/xtb_all.pdf",
        "imgs/xtb_all_per_solvents.pdf",
        "imgs/xtb_per_solvents.pdf",
        "imgs/xtb_per_scaffold.pdf",
        "imgs/ng_mopac.pdf",
        "imgs/nd_mopac.pdf",
        "imgs/xtb_combined.pdf",
        expand('output/pyscf/{mol}_reactivity.json', mol=get_cids(f"input/dbs/water.csv")[:-2]),
        "output/pyscf_water/water_reactivity.json",
        "imgs/ng_dft.pdf",
        "imgs/nd_dft.pdf",
        "imgs/pysr_dft.pdf",
        "imgs/pysr_water_dft.pdf",
        "imgs/pysr_scaffold_dft.pdf",




# TODO: cleaning rule

rule gather_sol_csvs:
    input:
        csvs=expand("input/dbs/{solv}.csv", solv=solvents)
    output:
        csv="input/csv_all.csv"
    shell:
        """
        cat {input.csvs} > input/tempout.csv
        grep -v ID input/tempout.csv > {output.csv}
        rm input/tempout.csv
        """


rule generate_smi_files:
    input:
        csvs="input/csv_all.csv"
    output:
        smi=smi_targets
    script:
        "scripts/generate_smiles.py"


rule generate_mol_files:
    input:
        smi="input/smiles/{solv}/{cid}.smi"
    output:
        mol="input/mols/{solv}/{cid}/raw_{cid}.mol"
    params:
        num_confs=20
    script:
        "scripts/gen_conformer.py"


rule gen_solv_3d:
    input:
        smi = "input/solvents/smiles/{solv}.smi",
    
    output:
        "input/solvents/{solv}/{solv}.xyz"

    shell:
        """
        mkdir -p input/solvents/{wildcards.solv}
        obabel {input.smi} -oxyz -O {output} --gen3D
        """


rule calculate_solvents:
    input:
        xyz = "input/solvents/{solv}/{solv}.xyz"

    output:
        "input/solvents/{solv}/{solv}.props"

    shell:
        """
        cd input/solvents/{wildcards.solv}
        /home/bartek/xtb-dist/bin/xtb {wildcards.solv}.xyz --opt extreme --alpb {wildcards.solv} --acc 0.01 --gfn 1 -P {threads} > gfn1.out
        sed -n '/Property Printout/,/HOMO-LUMO/p' gfn1.out > gfn1props.props
        echo '         #    Occupation            Energy/Eh            Energy/eV' >> {wildcards.solv}.props
        grep -a \(HOMO\) -A1 -B1 gfn1props.props | tail -n3 >> {wildcards.solv}.props
        grep Debye gfn1props.props >> {wildcards.solv}.props
        sed -n '/CM5 charges/,/Wiberg/p' gfn1props.props | head -n -2 | cut -c 1-28 >> {wildcards.solv}.props
        """


rule gather_solvent_props:
    input:
        propsfiles = expand("input/solvents/{solv}/{solv}.props", solv=solvents)

    output:
        gathered = "output/dbs/solvents.csv"

    script:
        "scripts/gather_homolumos_solv.py"


rule generate_xtb_files:
    input:
        mol = "input/mols/{solv}/{cid}/raw_{cid}.mol"
    output:
        chrg = "input/mols/{solv}/{cid}/.CHRG",
        uhf = "input/mols/{solv}/{cid}/.UHF"
    script:
        "scripts/generate_xtb.py"


rule xtb_calculate:
    input:
        mol = "input/mols/{solv}/{cid}/raw_{cid}.mol",
        chrg = "input/mols/{solv}/{cid}/.CHRG",
        uhf = "input/mols/{solv}/{cid}/.UHF"
    output:
        mol = "input/mols/{solv}/{cid}/{cid}.mol",
        out = "input/mols/{solv}/{cid}/{cid}.out",
        props = "input/mols/{solv}/{cid}/{cid}.props"
    threads: 3
    shell:
        """
        cd input/mols/{wildcards.solv}/{wildcards.cid}
        /home/bartek/xtb-dist/bin/xtb raw_{wildcards.cid}.mol --opt extreme --alpb {wildcards.solv} --acc 0.01 --gfn 1 -P {threads} > gfn1.out
        sed -n '/Property Printout/,/HOMO-LUMO/p' gfn1.out > gfn1props.props
        echo '         #    Occupation            Energy/Eh            Energy/eV' >> {wildcards.cid}.props
        grep -a \(HOMO\) -A1 -B1 gfn1props.props | tail -n3 >> {wildcards.cid}.props
        sed -n '/CM5 charges/,/Wiberg/p' gfn1props.props | head -n -2 | cut -c 1-28 >> {wildcards.cid}.props
        /home/bartek/xtb-dist/bin/xtb xtbopt.mol --vfukui --alpb {wildcards.solv} --acc 0.01 --gfn 1 -P {threads} > fukui.out
        sed -n '/Fukui functions:/,/---------/p' fukui.out | head -n -1 | tail -n +2 >> {wildcards.cid}.props
        cp xtbopt.mol {wildcards.cid}.mol
        cat gfn1.out fukui.out >> {wildcards.cid}.out
        """


homolumo_input = []
molfiles_input = []
xyzs_input = []
for solv in solvents:
    for cid in get_cids(f"input/dbs/{solv}.csv"):
        homolumo_input.append(f"input/mols/{solv}/{cid}/{cid}.props")
        molfiles_input.append(f"input/mols/{solv}/{cid}/{cid}.mol")
        xyzs_input.append(f"input/mols/{solv}/{cid}/{cid}.xyz")

rule generate_xyzs:
    input:
        "input/mols/{solv}/{cid}/{cid}.mol"
    
    output:
        "input/mols/{solv}/{cid}/{cid}.xyz"
    
    shell:
        """
        obabel {input} -oxyz -O {output}
        """


rule gather_homolumos:
    input:
        propsfiles = homolumo_input,
        molfiles = molfiles_input,
        xyz_paths = xyzs_input

    output:
        paths = expand("output/dbs/{solv}.csv", solv=solvents)
    
    script:
        "scripts/gather_homolumos2.py"


rule generate_mopac:
    input:
        mol = "input/mols/{solvent}/{cid}/{cid}.mol",
        chrg = "input/mols/{solvent}/{cid}/.CHRG",
        uhf = "input/mols/{solvent}/{cid}/.UHF"

    output:
        mop = "input/mopac/{solvent}/{cid}/{cid}.mop"
    
    shell:
        """
        mkdir -p input/mopac/{wildcards.solvent}/{wildcards.cid}
        obabel -imol {input.mol} -omop -O {output.mop}
        CHRG=$(<{input.chrg})
        UHF=$(<{input.uhf})
        if [ "$UHF" -eq 1 ]; then
            MULTI="SINGLET"
        elif [ "$UHF" -eq 2 ]; then
            MULTI="DOUBLET"
        fi
        if [ "{wildcards.solvent}" = "water" ]; then
            EPS=78.4
        elif [ "{wildcards.solvent}" = "DMSO" ]; then
            EPS=46.7
        elif [ "{wildcards.solvent}" = "DMF" ]; then
            EPS=36.7
        elif [ "{wildcards.solvent}" = "ch2cl2" ]; then
            EPS=8.9
        elif [ "{wildcards.solvent}" = "THF" ]; then
            EPS=7.6
        elif [ "{wildcards.solvent}" = "acetonitrile" ]; then
            EPS=35.7
        fi
        sed -i "s/PUT KEYWORDS HERE/PM7 PRECISE XYZ ENPART STATIC SUPER BONDS OPT LARGE PRTXYZ DISP ALLVEC EPS=$EPS $MULTI CHARGE=$CHRG GRAPHF/g" {output.mop}
        echo "" >> {output.mop}
        echo "OLDGEO FORCE PM7" >> {output.mop}
        """


rule mopac_calculate:
    input:
        mop = "input/mopac/{solvent}/{cid}/{cid}.mop"
    
    output:
        out = "input/mopac/{solvent}/{cid}/{cid}.out",
    
    shell:
        """
        mopac {input.mop}
        """


rule extract_mopac_homolumo:
    input:
        "input/mopac/{solvent}/{cid}/{cid}.out"

    output:
        "input/mopac/{solvent}/{cid}/{cid}.props"

    shell:
        """
        grep "NO. OF FILLED LEVELS\|NO. OF ALPHA ELECTRONS" {input} | head -n1  > {output}
        grep "Root No." -A5 {input} >> {output}
        """


mopac_props = []
for solv in solvents:
    for cid in get_cids(f"input/dbs/{solv}.csv"):
        mopac_props.append(f"input/mopac/{solv}/{cid}/{cid}.props")

rule read_mopac_props:
    input:
        props = mopac_props

    output:
        paths = expand("output/dbs/mopac_{solv}.csv", solv=solvents)

    script:
        "scripts/gather_mopac.py"


rule find_equation_xtb:
    input:
        inp = expand("input/dbs/{solv}.csv", solv=solvents),
        xtb = expand("output/dbs/{solv}.csv", solv=solvents),
        solv = "output/dbs/solvents.csv"

    output:
        xtb = "output/equation/eq_all_xtb.tex",

    script:
        "scripts/find_equation_xtb.py"


rule find_equation_solv_xtb:
    input:
        inp = expand("input/dbs/{solv}.csv", solv=solvents),
        xtb = expand("output/dbs/{solv}.csv", solv=solvents),
        solv = "output/dbs/solvents.csv"

    output:
        eqs = "output/equation/eq_solvents_xtb.tex"

    script:
        "scripts/find_equation_solv_xtb.py"


rule find_equation_scaffold_xtb:
    input:
        inp = expand("input/dbs/{solv}.csv", solv=solvents),
        xtb = expand("output/dbs/{solv}.csv", solv=solvents),
        solv = "output/dbs/solvents.csv"

    output:
        eqs = "output/equation/eq_scaffolds_xtb.tex"

    script:
        "scripts/find_equation_scaffold_xtb.py"

rule visualize_xtb:
    input:
        inp = expand("input/dbs/{solv}.csv", solv=solvents),
        xtb = expand("output/dbs/{solv}.csv", solv=solvents)

    output:
        "imgs/ng_xtb.pdf",
        "imgs/nd_xtb.pdf",
        "imgs/xtb_all.pdf",
        "imgs/xtb_all_per_solvents.pdf",
        "imgs/xtb_per_solvents.pdf",
        csv = "output/xtb_indices.csv"

    script:
        "scripts/analysis.py"


rule visualize_mopac:
    input:
        inp = expand("input/dbs/{solv}.csv", solv=solvents),
        mopac = expand("output/dbs/mopac_{solv}.csv", solv=solvents)

    output:
        "imgs/ng_mopac.pdf",
        "imgs/nd_mopac.pdf",
        csv = "output/mopac_indices.csv"

    script:
        "scripts/analysis_mopac.py"


rule visualize_xtb_scaffold:
    input:
        inp = expand("input/dbs/{solv}.csv", solv=solvents),
        xtb = expand("output/dbs/{solv}.csv", solv=solvents)

    output:
        "imgs/xtb_per_scaffold.pdf",
        csv = "output/xtb_indices_scaffold.csv"

    script:
        "scripts/analysis_scaffolds.py"


rule visualize_xtb_combined:
    input:
        xtb_scaf = "output/xtb_indices_scaffold.csv"

    output:
        img = "imgs/xtb_combined.pdf",

    script:
        "scripts/analysis_combined.py"


# dft_cids = get_cids(f"input/dbs/water.csv")[:-2:2]
dft_cids = get_cids(f"input/dbs/water.csv")[:-2]

rule prepare_pyscf_files:
    input:
        xyzs = expand('input/mols/water/{cid}/{cid}.xyz', cid=dft_cids),
        chrgs = expand('input/mols/water/{cid}/.CHRG', cid=dft_cids),
        uhfs = expand('input/mols/water/{cid}/.UHF', cid=dft_cids),

    output:
        xyzs = expand('input/pyscf/{cid}.xyz', cid=dft_cids),
        chrgs = expand('input/pyscf/.{cid}.CHRG', cid=dft_cids),
        uhfs = expand('input/pyscf/.{cid}.UHF', cid=dft_cids)

    run:
        import os
        import shutil

        # Ensure directory exists
        os.makedirs("input/pyscf", exist_ok=True)
        os.makedirs("output/pyscf/logs", exist_ok=True)

        # We iterate through the IDs to map input paths to output paths
        for cid in dft_cids:
            # Copy XYZ: input/mols/water/1/1.xyz -> input/pyscf/1.xyz
            shutil.copy(f"input/mols/water/{cid}/{cid}.xyz", f"input/pyscf/{cid}.xyz")
            
            # Copy CHRG: input/mols/water/1/.CHRG -> input/pyscf/.1.CHRG
            shutil.copy(f"input/mols/water/{cid}/.CHRG", f"input/pyscf/.{cid}.CHRG")
            
            # Copy UHF: input/mols/water/1/.UHF -> input/pyscf/.1.UHF
            shutil.copy(f"input/mols/water/{cid}/.UHF", f"input/pyscf/.{cid}.UHF")


rule pyscf_reactivity:
    input:
        xyz  = 'input/pyscf/{mol}.xyz',
        chrg = 'input/pyscf/.{mol}.CHRG',
        uhf = 'input/pyscf/.{mol}.UHF'
    output:
        xyz_opt = 'output/pyscf/{mol}_opt.xyz',
        json    = 'output/pyscf/{mol}_reactivity.json',
    log:
        'output/pyscf/logs/{mol}.log'
    threads: 8
    script:
        'scripts/pyscf_reactivity.py'


rule pyscf_xyz_2_mol:
    input:
        "output/pyscf/{mol}_opt.xyz"

    output:
        "output/pyscf/{mol}_opt.mol"

    shell:
        """
        obabel {input} -omol -O {output}
        """


rule prepare_water_pyscf_files:
    output:
        chrg = 'input/solvents/water/.CHRG',
        uhf = 'input/solvents/water/.UHF'

    shell:
        """
        mkdir -p input/solvents/water output/pyscf_water
        echo 0 > {output.chrg}
        echo 1 > {output.uhf}
        """


rule pyscf_reactivity_water:
    input:
        xyz  = 'input/solvents/water/xtbopt.xyz',
        chrg = 'input/solvents/water/.CHRG',
        uhf = 'input/solvents/water/.UHF'
    output:
        xyz_opt = 'output/pyscf_water/water_opt.xyz',
        json    = 'output/pyscf_water/water_reactivity.json',
    log:
        'output/pyscf_water/water.log'
    threads: 8
    script:
        'scripts/pyscf_reactivity.py'


rule visualize_dft_results:
    input:
        jsons = expand('output/pyscf/{mol}_reactivity.json', mol=dft_cids),
        mayr = "input/csv_all.csv",
        mols = expand("output/pyscf/{mol}_opt.mol", mol=dft_cids),

    output:
        "imgs/ng_dft.pdf",
        "imgs/nd_dft.pdf",
        "imgs/pysr_dft.pdf",
        "imgs/pysr_water_dft.pdf",
        "imgs/pysr_scaffold_dft.pdf",
        csv = "output/dft_indices.csv"
    
    script:
        "scripts/analysis_dft.py"


