from rdkit import Chem


def canonicalize_smi(smi, remove_Hs=False, sanitize=True, remove_atom_mapping=False):
    """
    Create canonicalized SMILES.
    By default, RDKit creates canonical smiles.
    https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles
    """
    params = Chem.SmilesParserParams()
    params.removeHs = remove_Hs
    params.sanitize = sanitize

    mol = Chem.MolFromSmiles(smi, params)

    if not remove_Hs:
        mol = Chem.AddHs(mol)

    # Remove atom map numbers, otherwise the smiles string is long and non-readable
    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    
    return Chem.rdmolfiles.MolToSmiles(mol)


def process_reaction(rxn):
    """
    Process and canonicalize reaction SMILES
    """
    reactants, reagents, products = rxn.split(">")

    reactants_c = ".".join(sorted([canonicalize_smi(r, remove_atom_mapping=True) for r in reactants.split(".")]))

    if len(reagents) > 0:
        reagents_c = ".".join(sorted([canonicalize_smi(r, remove_atom_mapping=True) for r in reagents.split(".")]))
    else:
        reagents_c = ''

    products_c = ".".join(sorted([canonicalize_smi(p, remove_atom_mapping=True) for p in products.split(".")]))

    return f"{reactants_c}>{reagents_c}>{products_c}"

