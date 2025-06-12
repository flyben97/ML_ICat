# ⚛️ ICat: Predicting Enantioselectivity in Hypervalent Iodine(III)-Catalyzed Dearomatizations

**📄 Publication:**  
 [https://doi.org/10.31635/ccschem.024.202303774]

## 🌟 Project Summary

This repository provides the code, pretrained models, and datasets for **ICat**, a machine learning framework developed to predict **enantioselectivity (ΔΔG‡)** in **hypervalent iodine(III)-catalyzed asymmetric phenolic dearomatization** reactions.

Our approach combines molecular fingerprints and solvent descriptors with ensemble machine learning algorithms to deliver accurate and generalizable predictions.

## ⚙️ Installation

Make sure your environment uses **Python 3.10.9**. Then install the required packages:

```bash
pip install numpy pandas scikit-learn==1.2.1 torch==2.0.1+cpu optuna==3.3.0
````

## 🚀 Quick Start

1. **Prepare Your Data**
   Place your dataset (e.g. SMILES strings, experimental values) into the `Dataset/` directory.

2. **Generate Descriptors**
   Run `Caldescriptors.py` to calculate molecular fingerprints and solvent descriptors. These are concatenated into feature vectors for model training.

3. **Partition the Dataset (Optional)**
   Use `EDC.py` to split the dataset based on Euclidean distance clustering, useful for model evaluation and analysis.

4. **Run the Main Program**
   Execute the entire prediction pipeline using:

   ```bash
   python ICat_main.py
   ```

5. **Model Ensemble (Optional)**
   Run `Model_ensemble.py` to apply and evaluate a set of ML models defined in the `Model/` folder.

## Notes

* The pretrained model is saved in the `ICat_xgb1205_ee/detaG` directory.
* **Typo Note:** The term "`detaG`" is a consistent misspelling of "`deltaG`" (ΔG) throughout the codebase. It has been preserved for compatibility.

---

## Citation

If you use this repository in your research, please cite:

> **Ben Gao**, **Liu Cai**, **Yuchen Zhang**, **Huaihai Huang**, **Yao Li**, **Xiao-Song Xue**.
> *A machine learning model for predicting enantioselectivity in hypervalent iodine (iii) catalyzed asymmetric phenolic dearomatizations*
> **CCS Chemistry**, 6 (10), 2515–2528, 2024












