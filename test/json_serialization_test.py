import collections
import json
import meeko
import numpy as np
import pathlib
import pytest

from meeko import (
    Monomer,
    Polymer,
    MoleculePreparation,
    MoleculeSetup,
    RDKitMoleculeSetup,
    ResiduePadder,
    ResidueTemplate,
    ResidueChemTemplates,
    PDBQTWriterLegacy,
)

from meeko import polymer
from meeko.molsetup import Atom, Bond, Ring, RingClosureInfo, Restraint

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from meeko.utils.pdbutils import PDBAtomInfo

try:
    import openforcefields
    _got_openff = True
except ImportError as err:
    _got_openff = False

# from ..meeko.utils.pdbutils import PDBAtomInfo

pkgdir = pathlib.Path(meeko.__file__).parents[1]

# Test Data
ahhy_example = pkgdir / "test/polymer_data/AHHY.pdb"
just_one_ALA_missing = (
    pkgdir / "test/polymer_data/just-one-ALA-missing-CB.pdb"
)

# Polymer creation data
chem_templates = ResidueChemTemplates.create_from_defaults()
mk_prep = MoleculePreparation()


# region Fixtures
@pytest.fixture
def populated_polymer():
    file = open(ahhy_example)
    pdb_str = file.read()
    polymer = Polymer.from_pdb_string(
        pdb_str, chem_templates, mk_prep, blunt_ends=[("A:1", 0)]
    )
    return polymer

@pytest.fixture
def polymer_with_missing_residue():
    file = open(just_one_ALA_missing)
    pdb_str = file.read()
    polymer = Polymer.from_pdb_string(
        pdb_str, chem_templates, mk_prep, blunt_ends=[("A:1", 0)]
    )
    return polymer

@pytest.fixture
def object_factory(populated_polymer):
    polymer = populated_polymer
    res_chem_templates = polymer.residue_chem_templates

    def _make(cls):
        if cls is Polymer:
            return populated_polymer
        if cls is Monomer:
            return polymer.monomers["A:1"]
        if cls is RDKitMoleculeSetup:
            return polymer.monomers["A:1"].molsetup
        if cls is ResidueChemTemplates:
            return polymer.residue_chem_templates
        if cls is ResidueTemplate:
            return res_chem_templates.residue_templates["G"]
        if cls is ResiduePadder:
            return res_chem_templates.padders["5-prime"]
        raise ValueError(f"Unsupported class: {cls}")
    return _make

# Registry of attributes to skip per class
EQUALITY_SKIP_FIELDS = {
    RDKitMoleculeSetup: {"atom_true_count" },
    Monomer: {"template", "link_labels"},
}

# region Test Cases

@pytest.mark.parametrize("cls", [
    RDKitMoleculeSetup,
    Monomer,
    Polymer,
    ResidueTemplate,
    ResiduePadder,
    ResidueChemTemplates,
])

def deep_assert_equal(decoded, original, path="root"):
    """Recursively compares two objects with support for type-aware handling and skip lists."""
    if type(decoded) != type(original):
        raise AssertionError(f"[{path}] Type mismatch: {type(decoded)} != {type(original)}")

    # Basic types
    if isinstance(decoded, (int, float, bool, str)):
        assert decoded == original, f"[{path}] Value mismatch: {decoded} != {original}"
        return

    # Dicts
    if isinstance(decoded, dict):
        assert decoded.keys() == original.keys(), f"[{path}] Dict keys mismatch"
        for key in decoded:
            deep_assert_equal(decoded[key], original[key], path=f"{path}.{key}")
        return

    # Lists or Tuples
    if isinstance(decoded, (list, tuple)):
        assert len(decoded) == len(original), f"[{path}] Length mismatch"
        for i, (d_item, o_item) in enumerate(zip(decoded, original)):
            deep_assert_equal(d_item, o_item, path=f"{path}[{i}]")
        return

    # Numpy arrays
    if isinstance(decoded, np.ndarray):
        assert np.allclose(decoded, original), f"[{path}] Numpy arrays not equal"
        return

    # RDKit Molecules
    if isinstance(decoded, Chem.Mol):
        decoded_smiles = Chem.MolToSmiles(decoded)
        original_smiles = Chem.MolToSmiles(original)
        if decoded_smiles != original_smiles:
            print(f"[DEBUG] Mol mismatch at {path}")
            print(f"Original: {original_smiles}")
            print(f"Decoded:  {decoded_smiles}")
            #import pdb; pdb.set_trace()  # Optional: step through interactively
        assert decoded_smiles == original_smiles, f"[{path}] Mol SMILES mismatch"

    # RDKit Reactions
    if isinstance(decoded, rdChemReactions.ChemicalReaction):
        assert rdChemReactions.ReactionToSmarts(decoded) == rdChemReactions.ReactionToSmarts(original), f"[{path}] Reaction SMARTS mismatch"
        return

    # Custom objects with attributes
    if hasattr(decoded, "__dict__"):
        cls = type(decoded)
        skip_attrs = EQUALITY_SKIP_FIELDS.get(cls, set())

        for attr in vars(original):
            if attr in skip_attrs:
                continue
            if not hasattr(decoded, attr):
                raise AssertionError(f"[{path}] Missing attribute: {attr}")
            decoded_val = getattr(decoded, attr)
            original_val = getattr(original, attr)
            deep_assert_equal(decoded_val, original_val, path=f"{path}.{attr}")
        return

    # Fallback
    assert decoded == original, f"[{path}] Fallback mismatch: {decoded} != {original}"

# region Standard Test Cases
@pytest.mark.parametrize("cls", [
    RDKitMoleculeSetup,
    Monomer,
    Polymer,
    ResidueTemplate,
    ResiduePadder,
    ResidueChemTemplates,
])
def test_json_roundtrip(cls, object_factory):
    obj = object_factory(cls)
    json_str = obj.to_json()
    decoded = cls.from_json(json_str)
    assert isinstance(decoded, cls)
    deep_assert_equal(decoded, obj)

# endregion

# region Other Test Cases

def test_pdbqt_writing_from_decoded_polymer(populated_polymer):
    """
    Takes a fully populated Polymer, writes a PDBQT string from it, encodes and decodes it, writes
    another PDBQT string from the decoded polymer, and then checks that the PDBQT strings are identical.

    Parameters
    ----------
    populated_polymer: Polymer
        Takes as input a populated Polymer object.

    Returns
    -------
    None
    """

    starting_polymer = populated_polymer
    starting_pdbqt = PDBQTWriterLegacy.write_from_polymer(starting_polymer)
    json_str = starting_polymer.to_json()
    decoded_polymer = Polymer.from_json(json_str)
    decoded_pdbqt = PDBQTWriterLegacy.write_from_polymer(decoded_polymer) 
    assert decoded_pdbqt == starting_pdbqt
    return


def test_load_reference_json():
    fn = str(pkgdir/"test"/"polymer_data"/"AHHY_reference_fewer_templates.json")
    with open(fn) as f:
        json_string = f.read()
    polymer = Polymer.from_json(json_string)
    assert len(polymer.get_valid_monomers()) == 4
    return


@pytest.mark.skipif(not _got_openff, reason="requires openff-forcefields")
def test_dihedral_equality():
    mk_prep = MoleculePreparation(
        merge_these_atom_types=(),
        dihedral_model="openff",
    )
    fn = str(pkgdir/"test"/"flexibility_data"/"non_sequential_atom_ordering_01.mol")
    mol = Chem.MolFromMolFile(fn, removeHs=False)
    starting_molsetup = mk_prep(mol)[0]
    json_str = starting_molsetup.to_json()
    decoded_molsetup = RDKitMoleculeSetup.from_json(json_str)
    deep_assert_equal(starting_molsetup, decoded_molsetup)
    return


def test_broken_bond(): 
    fn = str(pkgdir / "test" / "macrocycle_data" / "lorlatinib.mol")
    mol = Chem.MolFromMolFile(fn, removeHs=False)
    mk_prep_untyped = MoleculePreparation(untyped_macrocycles=True)
    starting_molsetup = mk_prep_untyped(mol)[0]
    decoded_molsetup = RDKitMoleculeSetup.from_json(starting_molsetup.to_json())
    count_rotatable = 0
    count_breakable = 0
    for bond_id, bond_info in decoded_molsetup.bond_info.items():
        count_rotatable += bond_info.rotatable
        count_breakable += bond_info.breakable
    assert count_rotatable == 10
    assert count_breakable == 1

# endregion

