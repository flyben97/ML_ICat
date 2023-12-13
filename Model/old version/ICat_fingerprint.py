from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs  # Import DataStructs module

def is_molecule_symmetric(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        # 生成分子的二维坐标
        AllChem.Compute2DCoords(mol)
        
        # 获取分子的非手性SMILES表示法
        non_chiral_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        
        # 判断非手性SMILES是否等于其反向表示
        is_symmetric = non_chiral_smiles == Chem.MolToSmiles(mol, isomericSmiles=False)
        return int(is_symmetric)  # 返回0或1
    else:
        return 0  # 无法处理的情况返回0


def contains_fluorine_element(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        # 遍历分子中的原子
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'F':
                return 1  # 包含氟元素，返回1
        return 0  # 没有氟元素，返回0
    else:
        return 0  # 无法处理的情况返回0


def molecule_contains_NH_group(smiles):
    # 将分子 SMILES 转化为 RDKit 的分子对象
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        # 定义NH基团的SMARTS表示法
        nh_group_smarts = "[N&H2,$([N&H1])]"
        
        # 创建NH基团的分子对象
        nh_group = Chem.MolFromSmarts(nh_group_smarts)
        
        if nh_group is not None:
            # 使用GetSubstructMatches函数来查找NH基团在分子中的匹配
            matches = mol.GetSubstructMatches(nh_group)
            
            # 如果找到匹配，说明分子包含NH基团
            if len(matches) > 0:
                return 1
    return 0


def molecule_contains_fragment(molecule_smiles, fragment_smiles):
    # 将分子 SMILES 转化为 RDKit 的分子对象
    molecule = Chem.MolFromSmiles(molecule_smiles)
    fragment = Chem.MolFromSmiles(fragment_smiles)
    
    if molecule is not None and fragment is not None:
        # 使用SubstructMatch函数来查找片段
        matches = molecule.GetSubstructMatches(fragment)
        
        # 如果找到匹配，说明分子包含片段
        if len(matches) > 0:
            return 1
    return 0


def molecule_contains_trifluoromethyl(smiles):
    # 将分子 SMILES 转化为 RDKit 的分子对象
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None:
        # 定义三氟甲基基团的SMARTS表示法
        trifluoromethyl_smarts = "FC(F)(F)"
        
        # 创建三氟甲基基团的分子对象
        trifluoromethyl = Chem.MolFromSmarts(trifluoromethyl_smarts)
        
        if trifluoromethyl is not None:
            # 使用GetSubstructMatches函数来查找三氟甲基基团在分子中的匹配
            matches = mol.GetSubstructMatches(trifluoromethyl)
            
            # 如果找到匹配，说明分子包含三氟甲基基团
            if len(matches) > 0:
                return 1
    return 0

def calculate_similarity(smiles1, smiles2):
    # Convert the SMILES representations of the molecules to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is not None and mol2 is not None:
        # Generate ECFP fingerprints for the molecules
        fingerprint1 = AllChem.GetMorganFingerprint(mol1, 2)
        fingerprint2 = AllChem.GetMorganFingerprint(mol2, 2)
        
        # Calculate Tanimoto similarity score
        similarity = DataStructs.TanimotoSimilarity(fingerprint1, fingerprint2)
        return similarity
    
    return None