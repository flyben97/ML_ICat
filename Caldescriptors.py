import numpy as np
import pandas as pd
import datetime
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors, Descriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.EState import Fingerprinter
from rdkit.ML.Descriptors import MoleculeDescriptors
import ICatKit

# rdkit_descriptor, rdkit_fingerprint and fp2string reference the code from https://github.com/WhitestoneYang/spoc

def rdkit_descriptor(smi):

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

#------------------------------------------------------------------------------

def smi_fp(smi, df):
    fp_temp = np.zeros((df.shape[0],167)) 
    for i in range(0,df.shape[0]):
        cfp_temp = rdkit_fingerprint(smi[i], fp_type="MACCSKeys")
        for j in range(0,167):
            fp_temp[i][j] = cfp_temp[j]
    fp_temp = pd.DataFrame(fp_temp)
    return fp_temp

def smi_pp(smi, df):

    rd_temp = []  

    for i in range(0,df.shape[0]):
        crd_ds = rdkit_descriptor(smi[i])
        rd_temp.append(crd_ds)  

    pp_temp = pd.DataFrame(rd_temp) 
    pp_temp = pp_temp.fillna(0)  

    return pp_temp


def caldescriptors(df):

    sub_smi = df.iloc[:,1]
    precat_smi = df.iloc[:,2]

    add_smi1 = df.iloc[:,5]
    add_smi2 = df.iloc[:,7]

    sol_smi1 = df.iloc[:,9]
    sol_smi2 = df.iloc[:,10]

    sub_fp_temp = smi_fp(sub_smi,df)
    precat_fp_temp = smi_fp(precat_smi,df)
    add1_fp_temp = smi_fp(add_smi1,df)
    add2_fp_temp = smi_fp(add_smi2,df)
    sol1_fp_temp = smi_fp(sol_smi1,df)
    sol2_fp_temp = smi_fp(sol_smi2,df)

    sub_pp_temp = smi_pp(sub_smi,df)
    precat_pp_temp = smi_pp(precat_smi,df)
    add1_pp_temp = smi_pp(add_smi1,df)
    add2_pp_temp = smi_pp(add_smi2,df)
    sol1_pp_temp = smi_pp(sol_smi1,df)
    sol2_pp_temp = smi_pp(sol_smi2,df)

    eq1 = df.iloc[:,3]
    temp = df.iloc[:,4]
    eq2 = df.iloc[:,6]
    eq3 = df.iloc[:,8]
    vv = df.iloc[:,11:15]

    ICatKit_features = ICatKit.cal_Icatfp(precat_smi, df.shape[0])

    fp_data = {
    'eq1':eq1,
    'temp':temp,
    'eq2':eq2,
    'eq3':eq3,
    'vv':vv,
    'sub': sub_fp_temp,
    'precat': precat_fp_temp,
    'add1': add1_fp_temp,
    'add2': add2_fp_temp,
    'sol1': sol1_fp_temp,
    'sol2': sol2_fp_temp,
    }

    pp_data = {
        'eq1':eq1,
        'temp':temp,
        'eq2':eq2,
        'eq3':eq3,
        'vv':vv,
        'sub': sub_pp_temp,
        'precat': precat_pp_temp,
        'add1': add1_pp_temp,
        'add2': add2_pp_temp,
        'sol1': sol1_pp_temp,
        'sol2': sol2_pp_temp,
    }

    fp_pp_data = {
        'eq1':eq1,
        'temp':temp,
        'eq2':eq2,
        'eq3':eq3,
        'vv':vv,
        'sub': sub_fp_temp,
        'precat': precat_fp_temp,
        'add1': add1_fp_temp,
        'add2': add2_fp_temp,
        'sol1': sol1_fp_temp,
        'sol2': sol2_fp_temp,
        'sub2': sub_pp_temp,
        'precat2': precat_pp_temp,
        'add12': add1_pp_temp,
        'add22': add2_pp_temp,
        'sol12': sol1_pp_temp,
        'sol22': sol2_pp_temp,
        #'ICatKit':ICatKit_features
    }

    #yield_data = df.iloc[:,15]

    targets_temp = df.iloc[:,16]

    features_temp = pd.DataFrame(pd.concat(fp_pp_data, axis=1))

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print('The descriptor data calculation has been completed        ',current_time)

    return targets_temp, features_temp









