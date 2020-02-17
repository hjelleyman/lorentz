# Special Relativity and other things

_How to use this code_

This is a collection of scripts and notebooks produced as part of a summer project 2019-2020.

## Azure notebook

This repository can be found on Azure notebooks and run there: https://notebooks.azure.com/HJelleyman/projects/lorentz

## Prerequisites

To run this code you need python3 installed and Anaconda for installing python packages.

## How to set up the repository

First download the repository into a folder. This can be done with Github Desktop or with the command
```
mkdir /path/to/where/you/want/this/lorentz
cd /directory/path/lorentz
git clone git@github.com:hjelleyman/lorentz.git
```
Then run the following commands to set up the conda environment with all the required packages.

```
conda create -n lorentz
conda activate lorentz
conda install --file requirements.txt
```
If you wish to deactivate the environment you can use
```
conda deactivate
```
## Running the code

Run the following commands, This will open the code as a jupyter notebook.
```
conda activate lorentz
cd /directory/path/lorentz
jupyter notebook
```