The datasets used for building models in this job, as well as the experimental data, are all stored in the 'Dataset' folder.

Before running the program, you need to install packages such as numpy, pandas, rdkit, sklearn, torch, optuna, etc.

1.  Python 3.10.9
2. Scikit-learn 1.2.1
3. RDKit 2022.09.5
4. PyTorch 2.0.1+cpu
5. Optuna 3.3.0

The main function of the program is in ICat_main.py, where the overall computational logic of the entire code is written. The 'Model' folder contains all the machine learning algorithms used in this work. ICat_xgb1205_ee/detaG is the trained model. (I misspelled 'delta' here, and due to not noticing it initially, all instances in the code are written as 'deta.' The correct spelling, based on the correct pronunciation, should be 'delta.)

Caldescriptors.py is a program for calculating descriptors. It concatenates molecular fingerprints calculated from the SMILES of compounds with other descriptors, such as solvent descriptors, to form a structured dataset used to train machine learning models in our work.

EDC.py is a program that partitions the dataset based on the Euclidean distance.

ICat_main.py is the main program.

Model_ensemble.py is a code that calls a collection of machine learning algorithms under the Model folder.









