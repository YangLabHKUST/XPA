# XPA
This GitHub project provides the Mixture Model Net for analysis large scale genetic data

## Overview 

The Mixture Model Net software package currently includes two main algorithms, the MMNet linear mixed model for estimating heritability and testing association, and the MMNet cross population phenotype prediction. This software aims to analysis large scale genetic data, for example UKBiobank.

## Download and compile
### Download
You can directly download the latest version of the MMNet software from the Github project. Now we only release the initial version and we keep updating it.
### Compile 
If you want to compile the MMNet software by your own, you can reference to my CMakeList.txt file. Please note that you should modify the path of Intel Math Kernel Library according to your own pc.  
-Intel Math Kernel Library. In order to improve the speed of basic linear algebra involved in the MMNet software, we use the Intel Math Kernel Library.  
-Boost C++ libraries. MMNet uses the Boost program options libraries to deal with the input arguments.
### Running MMNet examples
To run the MMNet executable, you only need to invoke ./MMNET on the Linux command line with the required parameters in the format - -option optionValue. There are two running examples in the followings to show how to run MMNet linear mixed model and cross population analysis.
  A toy example for MMNet linear mixed model analysis. 
  ```
  ./MMNET --bfile genotype --phenoFile phenotype --phenoCol diabetes --numThreads 8 
  --covarFile cov.txt --associationTest true --outputFile ./result.txt
  ```
  A toy example for MMNet cross population analysis. 
  ```
  ./MMNET --bfile genotype --phenoFile phenotype --phenoCol diabetes --covarFile cov.txt 
  --auxbfile auxgenotype --auxphenoFile auxphenotype --auxphenoCol diabetes 
  --auxcovarFile auxcov.txt--predbfile predgenotype --precovarFile precov.txt --numThreads 8 
  --geneticCorr true --outputFile ./result
  ```
### Contact Info
If you have comments or questions about the MMNet software, please contact Shunkang Zhang, szhangcj@connect.ust.hk. Welcome more suggestions to further improve the software.

### License
MMNet is free software under the GNU General Public License v3.0 (GPLv3).

### Acknowledgement
This software partially refers to BOLT-LMM software which provides the solid baseline for our implementation. This research is supported by Professor Can Yang at The Hong Kong University of Science and Technology and Professor Xiang Wan at Shenzhen Research Institute of Big Data.
