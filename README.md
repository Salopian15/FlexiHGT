# FlexiHGT
A HGT detection suite derived from the HGTPhyloDetect software suite. Allows flexible selection of taxonomic levels of interest and speedy processing and analysis on the proteome scale.

See: https://github.com/SysBioChalmers/HGTphyloDetect Yuan, Le, et al. HGTphyloDetect: facilitating the identification and phylogenetic analysis of horizontal gene transfer. Briefings in Bioinformatics (2023). https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbad035/7031155.


*Installation Instructions*

It is advised to make an anaconda environment with python version 3.12. Diamond and MMseqs can be installed through the bioconda channel for convenience and required python packages can be installed with no interference to the rest of your system. 

*Usage instructions*

At its basic level all that is needed to use FlexiHGT is a proteome fasta file, the species NCBI taxid corresponding to your proteome fasta file (organism the proteins come from) and the path to the database of your search option of choice. An example of this is shown below:

'''
flexihgt input.fasta -q 12344 -db path/to/db/file
'''

The following will use default parameters for the taxonomic level of interest - 'family' , hgt index - 0.5, bitscore parameter - 100 and out_pct - 0.8. Thus, hits that do not share the same family taxonomy as the query proteins will be defined as being in the 'outgroup'. There are several difference

FlexiHGT allows for the selection of the taxonomic levels of interest by the user, with this being used to define what is an in group (within the taxonomic level) and outgroup (outside of the taxonomic level), without any knowledge of the 