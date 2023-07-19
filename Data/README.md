## This folder contains all the datasets used in this study.

## Features
- Folder UniProt contains the list of proteins annotated to have certain post-translational modifications (PTMs)
- Folder PhosphositePlus contains PhosphositePlus annotations of PTMs
- iPTMnet_annotations.txt (iPTMnet annotations)
- NSP2_complete.tab (global secondary structure predictions for all the proteins in the human proteome by NetSurfP2.0)
- SwissPalm_proteins.txt (palmitoylation database annotations)
- TMHMM_tmp.csv (predicted transmembrane regions by TMHMM)
- PTM_MusiteDeep_75 (MusiteDeep predictions of PTMs)
- NSP2_exposed.csv (calculated fractions of exposed amino acid per protein, predictions from NetSurfP2.0)

## ProteomicsDB
ProteomicsDB contains information about the evidence of a protein being able to be detected by mass spectrometry (MS)

## Raw

- exocarta.csv (EV proteome from Exocarta)
- evpedia.csv (EV proteome from EVpedia)
- exocarta_experiments.txt (experimental details from Exocarta)
- vesiclepedia.txt (EV proteome from Vesiclepedia)
- vesiclepedia_experiments.txt (experimental details from Vesiclepedia)

## Curated
- EV_mapped_UP.csv (mapped EV proteins to UniProt)
- EV_proteins.txt (a text file containing the UniProt IDs of the 5,965 EV proteins of the discovery dataset) 
- VPEC_all.csv (Vesiclepedia and Exocarta proteins combined)
- VPEC_mapped.csv (Vesiclepedia and Exocarta proteins mapped to UniProt IDs)
- VP_Isolation_filter.csv (EV proteins selected after isolation method filtering)
- VP_pre2014_mapped.csv (low-confidence EV protein dataset)
- features_human_proteome_MS_filter.csv (the main dataset with all the filters containing true EV proteins with sequence-based predictions and curated annotations)
- features_human_proteome.csv (larger dataset without MS filter)
- High_confidence_EV_proteins.txt (list of EV proteins detected in three recent high-quality EV studies)
- Low_confidence_EV_proteins.txt (list of EV proteins detected prior to 2014 EV studies)
- Isolation_methods_filtered.txt (the list of all isolation methods included in our filtering steps)

## Mapping 
Contains datasets used to map Entrez gene IDs to UniProt

## PROSITE 
Contains the list of matched proteins containing motifs and domains of interest

# Training
Contains 3 different datasets for training our models

- training_data_MS_filter.csv (MS filter)
- training_data_MS_iso_filter.csv (MS filter + isolation method filtering)
- training_data_no_filter.csv (unfiltered)

# Validation 
Contains raw data files from 3 different studies we selected to validate our models 
Using Python notebooks we combine the datasets curate, annotate with necessary features, and try to predict the EV association

