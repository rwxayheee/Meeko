#!/usr/bin/env python

""""
micromamba create -c conda-forge -n meeko_tutorial_py39 python=3.9 -y
micromamba activate meeko_tutorial_py39

micromamba install -c conda-forge numpy scipy rdkit gemmi -y

git clone --single-branch --branch rna_flex https://github.com/rwxayheee/Meeko.git
cd Meeko; pip install --use-pep517 -e .; cd ..

git clone --single-branch --branch develop https://github.com/forlilab/scrubber.git
cd scrubber; pip install --use-pep517 -e .; cd ..

pip install prody
"""

from scrubber import Scrub
from rdkit import Chem

from meeko import CovalentBuilder
import prody

from meeko import MoleculePreparation
preparator = MoleculePreparation()
from meeko import PDBQTWriterLegacy
from meeko import Polymer
from meeko import ResidueChemTemplates

def get_fragments_by_atom_indices(mol: Chem.Mol, 
                                  atom_idx1: int, atom_idx2: int) -> tuple[tuple[int], tuple[int]]:

    """
    Splits mol into two frags by bond between atoms w/ atom_idx1 and atom_idx2
    """

    bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
    if bond is None:
        raise ValueError(f"No bond exists between the specified atom indices {atom_idx1, atom_idx2}.")

    emol = Chem.EditableMol(mol)
    emol.RemoveBond(atom_idx1, atom_idx2)
    mol = emol.GetMol()

    index_fragments = Chem.GetMolFrags(mol, asMols=False)
    if len(index_fragments) != 2:
        raise ValueError(f"Expected 2 fragments, but got {len(index_fragments)}.")

    frag1_indices, frag2_indices = index_fragments
    if atom_idx1 in frag1_indices:
        return (frag1_indices, frag2_indices)
    else:
        return (frag2_indices, frag1_indices)

"""
Ligand Specifications
"""

# Basename for the ligand
lig_name = "PBI_URA"

# Ligand Smiles, preferably isomeric Smiles w/ a specified protonation state
mysmi = "CN(C(N1)=O)C=C(/N=C/CCCC2=CC=CC=C2)C1=O"

# Smarts pattern and indices in Smarts pattern 
# to define covalent atoms within the ligand
tether_smarts, tether_smarts_indices = ("[CH3][NX3H0]", (0,1))
# The closer atom to the receptor core needs to be put first, 
# followed by the other atom that is closer to the ligand. 

"""
Receptor Specifications
"""

# PDB ID or PDB filename of the starting receptor structure
pdb_token = "3zd5"

# ProDy selection language to define receptor atoms
rec_atoms_selection = "not water"

# A string to define two "attractor" atoms within the receptor
# Format: 
# "Chain_ID:Residue_Name:Residue_Number:Atom_Name1,Atom_Name2"
cov_atoms_selection = "B:5BU:11:C1',N1"
# The closer atom to the receptor core needs to be put first, 
# followed by the other atom that is closer to the ligand. 

"""
Prepare Ligand
"""

mymol = Chem.MolFromSmiles(mysmi)
Chem.AddHs(mymol)
Chem.SanitizeMol(mymol)
scrub = Scrub(skip_acidbase=True, skip_tautomers=True)
scrubbed_mols = list(scrub(mymol))

print(f"Number of scrubbed mols: {len(scrubbed_mols)}")
mol_0 = scrubbed_mols[0]

pdb_prody_mol = prody.parsePDB(pdb_token)
rec_prody_mol = pdb_prody_mol.select(rec_atoms_selection)
builder = CovalentBuilder(rec_prody_mol, cov_atoms_selection)

chain, res, num, cov_atom_names = cov_atoms_selection.split(":")
cov_name1, cov_name2 = (x.strip() for x in cov_atom_names.split(","))
lig_counter = 0
for cov_lig in builder.process(mol_0, tether_smarts, tether_smarts_indices):
    root_atom_index = cov_lig.indices[0]
    molsetups = preparator.prepare(
                    cov_lig.mol,
                    root_atom_index=root_atom_index,
                    not_terminal_atoms=[root_atom_index],
                )
    for molsetup in molsetups:
        pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetup)
        if success:
            pdbqt_string = (
                PDBQTWriterLegacy.adapt_pdbqt_for_autodock4_flexres(
                    pdbqt_string, res, chain, num, skip_rename_ca_cb=True
                    )
                )
            lig_counter += 1
            lig_fn = f"{lig_name}_{lig_counter}.pdbqt" 
            with open(lig_fn, "w") as f:
                f.write(pdbqt_string)
            print(f"File written: {lig_fn}")
        else:
            print(error_msg)

print(f"Number of prepared ligs: {lig_counter}")

"""
Prepare Receptor
"""

templates = ResidueChemTemplates.create_from_defaults()
polymer = Polymer.from_prody(
    rec_prody_mol,
    templates,
    preparator,
    default_altloc="A",
)

res_id = f"{chain}:{num}"
res_atom_names = polymer.monomers[res_id].atom_names
atom_idx1, atom_idx2 = (res_atom_names.index(cov_name1), res_atom_names.index(cov_name2))

keep_indices, delete_indices = get_fragments_by_atom_indices(polymer.monomers[res_id].rdkit_mol, 
                                             atom_idx1, atom_idx2)
                                             
for atom_idx in delete_indices: 
    molsetup_mapidx_inverse = {v:k for k,v in polymer.monomers[res_id].molsetup_mapidx.items()}
    polymer.monomers[res_id].molsetup.delete_atom(molsetup_mapidx_inverse[atom_idx])

rec_fn = f"{pdb_token}_rec.pdbqt"
with open(rec_fn, "w") as f:
    f.write(PDBQTWriterLegacy.write_from_polymer(polymer)[0])
print(f"File written: {rec_fn}")

