# Towards contrast- and pathology-agnostic clinical fetal brain MRI segmentation using SynthSeg
This is the code repo to our [paper](https://arxiv.org/abs/2504.10244). 

Our experiments can be run through the scripts located in [scripts/fetal_scripts](https://github.com/ZiyaoShang/synthseg-for-clinical-fetal-brains/tree/main/scripts/fetal_scripts). Derailed instructions and documentations can be found in the respective code files:

feature_extractor.py: Generate the sampling weights when using FeTA_train for training. 
feature_extractor_dhcp.py: Generate the sampling weights when using both FeTA_train and dHCP_fetal_train for training.

fetal_helpers.py: Preprocessing and postprocessing helpers for the training and testing datasets.

train_and_validate_ours.py: Train network and select the checkpoint to be applied to the testing data. 
predict_and_merge_ours.py: Predict using multiple models and merge the results into a final segmentation map.  

## Citation
This pipeline is based on the [Synthseg framework](https://github.com/BBillot/SynthSeg). More instructions can also be found in the original repo.

**Robust machine learning segmentation for large-scale analysisof heterogeneous clinical brain MRI datasets** \
B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
PNAS (2023) \
[ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120#bibliography) | [arxiv](https://arxiv.org/abs/2203.01969) | [bibtex](bibtex.bib) ]

**SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining** \
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias \
Medical Image Analysis (2023) \
[ [article](https://www.sciencedirect.com/science/article/pii/S1361841523000506) | [arxiv](https://arxiv.org/abs/2107.09559) | [bibtex](bibtex.bib) ]

