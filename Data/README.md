## This folder contains all the necessary datasets used to reproduce this study.

## Raw
Folder UniProt contains the list of proteins annotated to have certain post-translational modifications (PTMs)
Folder PhosphositePlus contains PhosphositePlus annotations of PTMs
Folder ProteomicsDB contains the information about the evidence of a protein being able to be detected by mass spectrometry (MS)

- NSP2_complete.tab (global secondary structure predictions for all the proteins in the human proteome by NetSurfP2.0)
!needs to be removed - NSP_exposed_aa.csv 
- SwissPalm_proteins.txt (palmytoilation database annotations)
- TMHMM_tmp.csv (predicted transmembrane regions by TMHMM)
- exocarta.csv (EV proteome from exocarta)
- exocarta_experiments.txt (experimental details from exocarta)
- iPTMnet_annotations.txt (iPTMnet annotations)
- vesiclepedia.txt (EV proteome from vesiclepedia)
- vesiclepedia_experiments.txt (experimental details from vesiclepedia)

## Curated
- EV_mapped_UP.csv (mapped EV proteins to UniProt)
- PTM_MusiteDeep_75 (MusiteDeep predictions of PTMs)
- VPEC_all.csv (Vesiclepedia and exocarta proteins combined)
- VPEC_mapped.csv (Vesiclepedia and exocarta proteins mapped to UniProt IDs)
- VP_Isolation_filter.csv (EV proteins selected after isolation method filtering)
! removed? - VP_old_mapped.csv
- aa_exposed_NSP.csv (calculated fractions of exposed amino acid per protein, predictions from NetSurfP2.0)
- features_human_proteome_MS_filter.csv (the main dataset with all the filters containing true EV proteins with sequence-based predictions and curated annotations)
- features_human_proteome.csv (larger dataset without MS filter)

## Mapping 
Contains datasets used to map Entrez gene IDs to UniProt

## PROSITE 
Containes the list of matched proteins containing motifs and domains of interest

# Training
Contains 3 different datasets for training our models

- training_data_MS_filter.csv (MS filter)
- training_data_MS_iso_filter.csv (MS filter + isolation method filtering)
- training_data_no_filter.csv (unfiltered)

# Validation 
Contains raw data files from 3 different studies we selected to validate our models 
Using python notebooks we combine the datasets curate, annotate with necessary features and try to predict the EV association

