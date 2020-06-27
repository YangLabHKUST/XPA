# XPA
This GitHub project provides the implementation of XPA

## Overview 

The XPA aims to perform large-scale cross-population analysis.

## Download and compile
### Download
You can directly download the latest version of the XPA software from the Github project. Now we only release the initial version and we keep updating it.
### Compile 
If you want to compile the XPA software by your own, you can reference to my CMakeList.txt file. Please note that you should modify the path of Intel Math Kernel Library according to your own pc.  
-Intel Math Kernel Library. In order to improve the speed of basic linear algebra involved in the XPA software, we use the Intel Math Kernel Library.  
-Boost C++ libraries. XPA uses the Boost program options libraries to deal with the input arguments.

After you change the library path, you can compile it with the following command
```
mkdir -p build
cd build
cmake ..
make -j 4
```
### Running XPA examples
To run the XPA executable, you only need to invoke ./XPA on the Linux command line with the required parameters in the format - -option optionValue. The following is an example to show how to run XPA.
  ```
  ./XPA --bfile genotype --phenoFile phenotype --phenoCol diabetes --covarFile cov.txt 
  --auxbfile auxgenotype --auxphenoFile auxphenotype --auxphenoCol diabetes 
  --auxcovarFile auxcov.txt--predbfile predgenotype --precovarFile precov.txt --numThreads 8 
  --geneticCorr true --outputFile ./result
  ```
### Contact Info
If you have comments or questions about the XPA software, please feel free to contact Prof. Can Yang (macyang@ust.hk) and Shunkang Zhang (szhangcj@connect.ust.hk). Welcome more suggestions to further improve the software.

### License
XPA is free software under the GNU General Public License v3.0 (GPLv3).

### Acknowledgement
This software partially refers to the BOLT-LMM software which provides the solid baseline for our implementation. 
