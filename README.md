# Towards contrast- and pathology-agnostic clinical fetal brain MRI segmentation using SynthSeg
Magnetic resonance imaging (MRI) has played a crucial role in fetal neurodevelopmental research. Structural annotations of MR images are an important step for quantitative analysis of the developing human brain, with Deep learning providing an automated alternative for this otherwise tedious manual process. However, segmentation performances of Convolutional Neural Networks often suffer from domain shift, where the network fails when applied to subjects that deviate from the distribution with which it is trained on. In this work, we aim to train networks capable of automatically segmenting fetal brain MRIs with a wide range of domain shifts pertaining to differences in subject physiology and acquisition environments, in particular shape-based differences commonly observed in pathological cases. We introduce a novel data-driven train-time sampling strategy that seeks to fully exploit the diversity of a given training dataset to enhance the domain generalizability of the trained networks. We adapted our sampler, together with other existing data augmentation techniques, to the SynthSeg framework, a generator that utilizes domain randomization to generate diverse training data, and ran thorough experimentations and ablation studies on a wide range of training/testing data to test the validity of the approaches. Our networks achieved notable improvements in the segmentation quality on testing subjects with intense anatomical abnormalities (p < 1e-4), though at the cost of a slighter decrease in performance in cases with fewer abnormalities. Our work also lays the foundation for future works on creating and adapting data-driven sampling strategies for other training pipelines.  

This is the code repo to our [paper](https://arxiv.org/abs/2504.10244). 

Our experiments can be run through the scripts located in [scripts/fetal_scripts](https://github.com/ZiyaoShang/synthseg-for-clinical-fetal-brains/tree/main/scripts/fetal_scripts). Detailed instructions and documentations can be found in the respective code files:

`feature_extractor.py`: Generate the sampling weights when using FeTA_train for training.

`feature_extractor_dhcp.py`: Generate the sampling weights when using both FeTA_train and dHCP_fetal_train for training.

`fetal_helpers.py`: Preprocessing and postprocessing helpers for the training and testing datasets.

`train_and_validate_ours.py`: Train network and select the checkpoint to be applied to the testing data. 

`predict_and_merge_ours.py`: Predict using multiple models and merge the results into a final segmentation map.  

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

