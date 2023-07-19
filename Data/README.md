## This folder contains all the datasets used in this study.

## Curated
- EV_proteins.txt (UniProt IDs of the 5,965 EV proteins of the discovery dataset)
- EV_proteins_Entrez_to_Uniprot.tsv (Entrez Gene ID to UniProt ID mapped EV proteins)
- features_human_proteome_MS_filter.csv (human proteome feature dataset with MS filter)
- features_human_proteome.csv (larger human proteome feature dataset without MS filter)
- High_confidence_EV_proteins.txt (list of EV proteins detected in three recent high-quality EV studies)
- Isolation_methods_filtered.txt (all isolation methods included in our filtering steps)
- Low_confidence_EV_proteins.txt (list of EV proteins in Vesiclepedia database that were detected in EV studies prior to 2014)
- VPEC_all.csv (Vesiclepedia and Exocarta proteins combined)
- VP_Isolation_filter.csv (Vesiclepedia EV proteins selected after isolation method filtering)
- VP_pre2014_mapped.csv (Vesiclepedia low-confidence mapped EV proteins)

## Features
- Folder PhosphositePlus (contains PhosphositePlus files for each included PTM)
- Folder PROSITE contains the list of matched proteins containing motifs and domains of interest
- Folder UniProt (list of proteins annotated according to uniProt keywords, e.g. PTMs 
- iPTMnet_annotations.txt (iPTMnet annotations)
- NSP2_complete.tab (global secondary structure predictions for all the proteins in the human proteome by NetSurfP-2.0)
- NSP2_exposed.csv (calculated fractions of exposed amino acid per protein, predictions from NetSurfP-2.0)
- PTM_MusiteDeep_75 (MusiteDeep predictions of PTMs using .75 cutoff)
- SwissPalm_proteins.txt (SwissPalm palmitoylation annotations)
- TMHMM_tmp.csv (predicted proteins with transmembrane regions by TMHMM)

## ProteomicsDB
Folder contains information about the evidence of a protein being detectable by mass spectrometry according to ProteomicsDB

## Raw
- evpedia.csv (EV proteome from EVpedia)
- exocarta.csv (EV proteome from Exocarta)
- exocarta_experiments.txt (experimental details from Exocarta)
- vesiclepedia.txt (EV proteome from Vesiclepedia)
- vesiclepedia_experiments.txt (experimental details from Vesiclepedia)

# Training
Contains 3 different EV-annotated feature datasets for training the random forest models
- training_data_no_filter.csv (unfiltered)
- training_data_MS_filter.csv (MS filter)
- training_data_MS_iso_filter.csv (MS filter + isolation method filtering)

# Validation 
Contains raw data files from 3 different studies selected to validate our models 

