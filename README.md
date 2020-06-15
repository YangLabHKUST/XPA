# XPA
This GitHub project provides the implementation of XPA

## Overview 

The XPA aims to analysis large scale genetic cross-population analysis.

## Download and compile
### Download
You can directly download the latest version of the XPA software from the Github project. Now we only release the initial version and we keep updating it.
### Compile 
If you want to compile the XPA software by your own, you can reference to my CMakeList.txt file. Please note that you should modify the path of Intel Math Kernel Library according to your own pc.  
-Intel Math Kernel Library. In order to improve the speed of basic linear algebra involved in the MMNet software, we use the Intel Math Kernel Library.  
-Boost C++ libraries. MMNet uses the Boost program options libraries to deal with the input arguments.
### Running XPA examples
To run the MMNet executable, you only need to invoke ./XPA on the Linux command line with the required parameters in the format - -option optionValue. The following is an example to show how to run XPA.
  ```
  A toy example for XPA. 
  ```
  ./XPA --bfile genotype --phenoFile phenotype --phenoCol diabetes --covarFile cov.txt 
  --auxbfile auxgenotype --auxphenoFile auxphenotype --auxphenoCol diabetes 
  --auxcovarFile auxcov.txt--predbfile predgenotype --precovarFile precov.txt --numThreads 8 
  --geneticCorr true --outputFile ./result
  ```
### Contact Info
If you have comments or questions about the XPA software, please contact Shunkang Zhang, szhangcj@connect.ust.hk. Welcome more suggestions to further improve the software.

### License
XPA is free software under the GNU General Public License v3.0 (GPLv3).

### Acknowledgement
This software partially refers to BOLT-LMM software which provides the solid baseline for our implementation. This research is supported by Professor Can Yang at The Hong Kong University of Science and Technology and Professor Xiang Wan at Shenzhen Research Institute of Big Data.
