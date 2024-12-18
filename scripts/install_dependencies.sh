#!/bin/bash

# Possible for users to install with the bioconda channel on anaconda, miniconda or mamba with the following command:
# conda create env -n flexihgt -c bioconda diamond mmseqs2 python=3.12
# conda install -c bioconda diamond mmseqs2

# Install diamond
echo "Installing diamond"
wget http://github.com/bbuchfink/diamond/releases/download/v2.0.9/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
sudo mv diamond /usr/local/bin/

# Install mmseqs2
echo "Installing mmseqs2"
wget https://mmseqs.com/latest/mmseqs-linux.tar.gz
tar xzf mmseqs-linux.tar.gz
sudo mv mmseqs/bin/mmseqs /usr/local/bin/

echo "Installation of diamond and mmseqs2 completed."