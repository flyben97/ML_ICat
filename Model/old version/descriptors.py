import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors, Descriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.EState import Fingerprinter
from rdkit.ML.Descriptors import MoleculeDescriptors

def rdkit_descriptor(smi):
    """
    Parameters
    ----------
    smi: str
        SMILES of molecules
        
    Returns: 
    -------
    type: list
        If SMILES is valid, a list will be returned.
        If SMILES is not valid, a list containing zero or Flase will be returned.
    
    Source:
    -------  
    RDKit: https://www.rdkit.org/
    """
    mol = Chem.MolFromSmiles(smi)
    if mol:
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(
            [x[0] for x in Descriptors._descList])
        ds = calc.CalcDescriptors(mol)
    else:
        ds = ExplicitBitVect(len(list(Descriptors._descList)))
        ds = fp2string(ds, output='vect')
    return list(ds)

def fp2string(fp, output, fp_type="Others"):

    if fp_type in ["Estate", "EstateIndices"]:
        fp = fp
    elif output == "bit":
        fp = list(fp.GetOnBits())

    elif output == "vect":
        fp = list(fp.ToBitString())
        fp = [1 if val in ["1", 1] else 0 for val in fp]

    elif output == "bool":
        fp = list(fp.ToBitString())
        fp = [1 if val == "1" else -1 for val in fp]

    return fp

def rdkit_fingerprint(smi, fp_type="rdkit", radius=2, max_path=2, fp_length=1024, output="bit"):
    """ Molecular fingerprint generation by rdkit package.
    
    Parameters:
    ------------
    smi: str
        SMILES string.
    fp_type: str
        • Avalon -- Avalon Fingerprint
        • AtomPaires -- Atom-Pairs Fingerprint
        • TopologicalTorsions -- Topological-Torsions Fingerprint
        • MACCSKeys Fingerprint 167
        • RDKit -- RDKit Fingerprint 
        • RDKitLinear -- RDKit linear Fingerprint
        • LayeredFingerprint -- RDKit layered Fingerprint
        • Morgan -- Morgan-Circular Fingerprint
        • FeaturedMorgan -- Morgan-Circular Fingerprint with feature definitions
    radius: int
    max_path: int
    fp_length: int
    output: str
        "bit" -- the index of fp exist
        "vect" -- represeant by 0,1
        "bool" -- represeant by 1,-1
    
    Returns: 
    -------
    type: list
        If SMILES is valid, a list will be returned.
        If SMILES is not valid, a list containing zero or Flase will be returned.
    
    Source:
    -------
    RDKit: https://www.rdkit.org/
    """

    mol = Chem.MolFromSmiles(smi)

    if mol:
        if fp_type == "RDKit":
            fp = Chem.RDKFingerprint(
                mol=mol, maxPath=max_path, fpSize=fp_length)

        elif fp_type == "RDKitLinear":
            fp = Chem.RDKFingerprint(
                mol=mol, maxPath=max_path, branchedPaths=False, fpSize=fp_length)

        elif fp_type == "AtomPaires":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=fp_length)

        elif fp_type == "TopologicalTorsions":
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=fp_length)

        elif fp_type == "MACCSKeys":
            fp = MACCSkeys.GenMACCSKeys(mol)

        elif fp_type == "Morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=fp_length)

        elif fp_type == "FeaturedMorgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, useFeatures=True, nBits=fp_length)

        elif fp_type == "Avalon":
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=fp_length)

        elif fp_type == "LayeredFingerprint":
            fp = Chem.LayeredFingerprint(
                mol, maxPath=max_path, fpSize=fp_length)

        elif fp_type == "Estate":
            fp = list(Fingerprinter.FingerprintMol(mol)[0])

        elif fp_type == "EstateIndices":
            fp = list(Fingerprinter.FingerprintMol(mol)[1])

        else:
            print("Invalid fingerprint type!")

        #fp = fp2string(fp, output, fp_type)

    else:
        if fp_type == "MACCSKeys":
            fp_length = 167
        if fp_type == "Estate":
            fp_length = 79
        if fp_type == "EstateIndices":
            fp_length = 79
        fp = ExplicitBitVect(fp_length)
        fp = fp2string(fp, output='vect')

    return fp