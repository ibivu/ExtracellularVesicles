# Proteome encoded determinants of protein sorting into extracellular vesicles
### Abstract
Extracellular vesicles (EVs) are membranous structures released by cells into the extracellular space and are thought to be involved in cell-to-cell communication. While EVs and their cargo are promising biomarker candidates, protein sorting mechanisms of proteins to EVs remain unclear. In this study, we ask if it is possible to determine EV association based on the protein sequence. Additionally, we ask what the most important determinants are for EV association. We answer these questions with explainable AI models, using human proteome data from EV databases to train and validate the model. It is essential to correct the datasets for contaminants introduced by coarse EV isolation workflows and for experimental bias caused by mass spectrometry. In this study, we show that it is indeed possible to predict EV association from the protein sequence: a simple sequence-based model for predicting EV proteins achieved an area under the curve of 0.77±0.01, which increased further to 0.84±0.00 when incorporating curated post-translational modification (PTM) annotations. Feature analysis shows that EV associated proteins are stable, polar, and structured with low isoelectric point compared to non-EV proteins. PTM annotations emerged as the most important features for correct classification; specifically palmitoylation is one of the most prevalent EV sorting mechanisms for unique proteins. Palmitoylation and nitrosylation sites are especially prevalent in EV proteins that are determined by very strict isolation protocols, indicating they could potentially serve as quality control criteria for future studies. This computational study offers an effective sequence-based predictor of EV associated proteins with extensive characterisation of the human EV proteome that can explain for individual proteins which factors contribute to their EV association.

This GitHub repository contains all the datasets used in this study with detailed Python notebooks for everyone to reproduce. 

The most relevant files: 

1. The curated EV and non-EV protein dataset containing all the sequence-based features and curated annotations (discovery set): [here](https://github.com/ibivu/ExtracellularVesicles/blob/master/Data/Training/training_data_MS_iso_filter.csv)

2. A text file containing the UniProt IDs of the 5,965 EV proteins of the discovery dataset. This list of proteins is also provided as a supplementary file in the revised version of the manuscript: [here](https://github.com/ibivu/ExtracellularVesicles/blob/master/Data/Curated/EV_proteins.txt)

### If you use our dataset or code, please cite the paper:

Proteome encoded determinants of protein sorting into extracellular vesicles. Katharina Waury, Dea Gogishvili, Rienk Nieuwland, Madhurima Chatterjee, Charlotte E. Teunissen, Sanne Abeln. bioRxiv 2023.02.01.526570; doi: https://doi.org/10.1101/2023.02.01.526570
