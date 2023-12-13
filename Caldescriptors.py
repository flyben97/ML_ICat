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

def smi_add_pp(add_smi1, add_smi2, eq2, eq3, df):

    rd_temp = []  

    for i in range(0,df.shape[0]):


        crd_ds1 = np.array(rdkit_descriptor(add_smi1[i])) 
        crd_ds2 = np.array(rdkit_descriptor(add_smi2[i]))

        add1 = float(eq2[i])
        add2 = float(eq3[i])

        if add1 == 0 and add2 == 0:
            C1 = 0
            C2 = 0
        else:
            C1 = (add1)/(add1+add2)
            C2 = (add2)/(add1+add2)

        crd_ds = crd_ds1 * C1 + crd_ds2 * C2

        rd_temp.append(crd_ds)  

    pp_temp = pd.DataFrame(rd_temp) 
    pp_temp = pp_temp.fillna(0)  

    return pp_temp

def smi_sol_pp(sol_smi1, sol_smi2, vv, df):

    rd_temp = []  

    for i in range(0,df.shape[0]):


        crd_ds1 = np.array(rdkit_descriptor(sol_smi1[i])) 
        crd_ds2 = np.array(rdkit_descriptor(sol_smi2[i]))

        radio_sol = float(vv[i])

        crd_ds = crd_ds1 * (radio_sol)/(radio_sol+1) + crd_ds2 * (radio_sol)/(radio_sol+1)

        rd_temp.append(crd_ds)  

    pp_temp = pd.DataFrame(rd_temp) 
    pp_temp = pp_temp.fillna(0)  

    return pp_temp


def caldescriptors(df,df_solvent):

    sub_smi = df.iloc[:,1]
    product_smi = df.iloc[:,2]
    precat_smi = df.iloc[:,4]

    eq1 = df.iloc[:,5]

    temp = df.iloc[:,7]

    add1_smi = df.iloc[:,8]
    eq2 = df.iloc[:,9]
    add2_smi = df.iloc[:,10]
    eq3 = df.iloc[:,11]

    sol1_smi = df.iloc[:,12]
    sol2_smi = df.iloc[:,13]

    vv = df.iloc[:,14]

    condition = df.iloc[:,15:22]

    DFT_descriptors = df.iloc[:,22:29]

    ee = df.iloc[:,30]


    sol1_result = np.zeros([df.shape[0],5])

    for i in range(0,df.shape[0]):
        for j in range(0,df_solvent.shape[0]):
            if sol1_smi[i] == df_solvent.iloc[j,0]:
                sol1_result[i,0] = df_solvent.iloc[j,1]
                sol1_result[i,1] = df_solvent.iloc[j,2]
                sol1_result[i,2] = df_solvent.iloc[j,3]
                sol1_result[i,3] = df_solvent.iloc[j,4]
                sol1_result[i,4] = df_solvent.iloc[j,5]

    sol1_result = pd.DataFrame(sol1_result)

    sol2_result = np.zeros([df.shape[0],5])

    for i in range(0,df.shape[0]):
        for j in range(0,df_solvent.shape[0]):
            if sol2_smi[i] == df_solvent.iloc[j,0]:
                sol2_result[i,0] = df_solvent.iloc[j,1]
                sol2_result[i,1] = df_solvent.iloc[j,2]
                sol2_result[i,2] = df_solvent.iloc[j,3]
                sol2_result[i,3] = df_solvent.iloc[j,4]
                sol2_result[i,4] = df_solvent.iloc[j,5]

    sol2_result = pd.DataFrame(sol2_result)

    sub_fp_temp = smi_fp(sub_smi,df)
    product_fp_temp = smi_fp(product_smi,df)
    precat_fp_temp = smi_fp(precat_smi,df)

    sub_pp_temp = smi_pp(sub_smi,df)
    product_pp_temp = smi_pp(product_smi,df)
    precat_pp_temp = smi_pp(precat_smi,df)
    add_pp_temp = smi_add_pp(add1_smi, add2_smi, eq2, eq3, df)
    #sol_pp_temp = smi_sol_pp(sol1_smi, sol2_smi, vv, df)
 


    pp_data = {
        'eq1':eq1,
        'temp':temp,
        'remain_condition':condition,
        'sub': sub_pp_temp,
        'sub2': sub_fp_temp,
        'prod':product_pp_temp,
        'prod2':product_fp_temp,
        'precat': precat_pp_temp,
        'precat2': precat_fp_temp,
        'precat_qm': DFT_descriptors,
        'add': add_pp_temp,
        'vv':vv,
        'eq2':eq2,
        'eq3':eq3,
        'sol1': sol1_result,
        'sol2': sol2_result,
    }


    #yield_data = df.iloc[:,15]

    targets_temp = df.iloc[:,30]

    features_temp = pd.DataFrame(pd.concat(pp_data, axis=1)) 

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print('The descriptor data calculation has been completed        ',current_time)

    return targets_temp, features_temp









