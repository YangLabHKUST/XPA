// Mixture Model Net
// Copyright (C) 2019  Shunkang Zhang
// Copyright (C) 2014-2017  Harvard University
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cctype>

#include "omp.h"
#include <mkl.h>

#include "Parameters.h"
#include "IOUtils.h"
#include "GenoData.h"
#include "AuxGenoData.h"
#include "InfoStructure.h"
#include "CovarBasis.h"
#include "PhenoBasis.h"
#include "MemoryUtils.h"
#include "Timer.h"
#include "NumericUtils.h"
#include "GeneticCorr.h"
#include "LMMCPU.h"

using namespace LMMNET;
using namespace std;

int main(int argc, char **argv) {

  cout << "                      +-----------------------------+" << endl;
  cout << "                      |                         ___ |" << endl;
  cout << "                      |   XPA, version 1.0     /_ / |" << endl;
  cout << "                      |   March 1, 2020        /_/  |" << endl;
  cout << "                      |   Shunkang Zhang       //   |" << endl;
  cout << "                      |                        /    |" << endl;
  cout << "                      +-----------------------------+" << endl;
  cout << endl;

  // output arguments
  cout << "The input arguments are :" << "\n";
  for (int num = 1; num < argc; num++) {
    cout << argv[num++] << " " << argv[num] << "\n";
  }
  cout << endl;

  Timer timer;
  double start_time = timer.get_time();
#ifdef DEBUG
  cout << "***** Debug version of program *****" << endl;
#endif
  Params params;
  if (!params.processCommandLineArgs(argc, argv)) {
    cerr << "Aborting due to error processing command line arguments" << endl;
    cerr << "For list of arguments, run with -h (--help) option" << endl;
    exit(1);
  }

  cout << endl << "Setting number of threads to " << params.numThreads << endl;
  omp_set_num_threads(params.numThreads);
  mkl_set_num_threads(params.numThreads);

  cout << "fam: " << params.famFile << endl;
  cout << "bim(s): ";
  for (uint i = 0; i < params.bimFiles.size(); i++) cout << params.bimFiles[i] << endl;
  cout << "bed(s): ";
  for (uint i = 0; i < params.bedFiles.size(); i++) cout << params.bedFiles[i] << endl;

  /********** SET UP DATA **********/

  cout << endl << "***** Reading genotype data *****" << endl << endl;

  GenoData genoData(params.famFile, params.bimFiles, params.bedFiles, params.removeSnpsFiles,
                    params.removeIndivsFiles, params.modelSnpsFiles, params.maxMissingPerSnp, params.maxMissingPerIndiv);
  const vector<SnpInfo> &snps = genoData.getSnpInfo();

  cout << "Time for setting up dataset and read data " << timer.update_time() << " sec" << endl;

  if ((int) snps.size() > params.maxModelSnps) {
    cerr << "ERROR: Number of SNPs exceeds maxModelSnps = " << params.maxModelSnps << endl;
    exit(1);
  }

  cout << endl << "***** Reading phenotype and covariate data *****" << endl << endl;

  vector<double> maskIndivs(genoData.getNpad());
  genoData.writeMaskIndivs(maskIndivs.data());

  // read phenotype data from files if provided
  PhenoBasis<GenoData> phenobasis(params.phenoFile, genoData, params.phenoCols, maskIndivs, params.phenoUseFam);
  phenobasis.padPhenodbl(); // pad the phenotype data to have the same individuals as genotype data

  vector<vector<double> > phenodbl = phenobasis.getPhenodbl();

  // read covariate data from files if provided
  CovarBasis<GenoData> covarBasis(params.covarFile, genoData, params.covarCols, maskIndivs);
  covarBasis.writeMask(maskIndivs);
  // update the phenotype value with new mask
  // because here we change the mask according to the provided covariate 
  for (uint64 n = 0; n < phenodbl[0].size(); n++) {
    phenodbl[0][n] *= maskIndivs[n];
  }
  covarBasis.projectCovarsVec(phenodbl[0].data());


  cout << "Time for reading and processing covariate and phenotype data " << timer.update_time() << " esc" << endl;

  if (params.geneticCorr) {

    // get the snp reference list from main dataset
    const std::map<std::string, uint64> &snpRef = genoData.getsnpRef();
    cout << endl << "***** Read the auxiliary data *****" << endl;
    AuxGenoData auxData(params.auxfamFile, params.auxbimFiles, params.auxbedFiles,
                        params.auxremoveIndivsFiles, params.maxMissingPerSnp, params.maxMissingPerIndiv, snpRef);

    if (genoData.getM() != auxData.getM()) {
      cerr << "Error: Number of SNPs in two datasets should be equal" << endl;
      exit(1);
    }
    const vector<SnpInfo> &auxSnps = auxData.getSnpInfo();

    // the number of prediction SNPs cannot be too large
    if (auxSnps.size() > params.maxModelSnps) {
      cerr << "ERROR: Number of auxiliary SNPs exceeds maxModelSnps = " << params.maxModelSnps << endl;
      exit(1);
    }

    // the number of prediction snps should be equal to training snps
    if (auxSnps.size() != snps.size()) {
      cerr << "Error: Number of auxiliary SNPs should be equal the number of training SNPs " << endl;
      exit(1);
    }

    vector<double> auxmaskIndivs(auxData.getNpad());
    auxData.writeMaskIndivs(auxmaskIndivs.data());

    // read phenotype data from files if provided
    PhenoBasis<AuxGenoData>
        auxphenobasis(params.auxphenoFile, auxData, params.auxphenoCols, auxmaskIndivs, params.phenoUseFam);
    auxphenobasis.padPhenodbl();

    vector<vector<double> > auxphenodbl = auxphenobasis.getPhenodbl();

    CovarBasis<AuxGenoData> auxCovBasis(params.auxcovarFile, auxData, params.auxcovarCols, auxmaskIndivs);
    auxCovBasis.projectCovarsVec(auxphenodbl[0].data());

    cout << "Time for setting up dataset and read auxiliary data " << timer.update_time() << " sec" << endl;

    GeneticCorr
        geneticCorr(genoData, covarBasis, &maskIndivs[0], auxData, auxCovBasis, &auxmaskIndivs[0], params.snpsPerBlock,
                    params.estIterationTrace, params.estIterationTraceAux, params.estIterationDelta, params.maxIterationConj,
                    params.useExactTrace, params.imputeMethod, params.outputFile);

    cout << "Time for initializing genetic object and normalizing snps is " << timer.update_time() << " sec" << endl;

    geneticCorr.compVCM(phenodbl[0].data(), auxphenodbl[0].data()); // note the order of input arguments

    cout << "Total elapsed time for analysis = " << (timer.get_time() - start_time) << " sec"
         << endl;

    cout << endl << "***** Estimate the fixed effect *****" << endl;

    vector<vector<double> > mainPheno = phenobasis.getPhenodbl();
    vector<vector<double> > auxPheno = auxphenobasis.getPhenodbl();
    geneticCorr.estFixEff(mainPheno[0].data(), auxPheno[0].data(), params.useApproFixEffect); // estimate fix effect with original phenotype data

    cout << "Time for estimating fix effect is " << timer.update_time() << " sec" << endl;

    cout << endl << "***** Estimate the posterior Mean *****" << endl;

    // whether use RAM efficient model
    if (params.RAMeff) {
      cout << endl << "Using RAM efficient model" << endl;
      geneticCorr.estPosteriorMean(params.bimFiles, params.bedFiles, params.auxbimFiles, params.auxbedFiles);
    } else {
      geneticCorr.estPosteriorMean();
    }

    cout << "Time for estimating posterior Mean is " << timer.update_time() << " sec" << endl;

    cout << endl << "***** Read the prediction data *****" << endl << endl;
    GenoData predictData(params.predfamFile, params.predbimFiles, params.predbedFiles, params.predremoveSnpsFiles,
                         params.predremoveIndivsFiles, params.modelSnpsFiles, params.maxMissingPerSnp, params.maxMissingPerIndiv);

    const vector<SnpInfo> &predictSnps = predictData.getSnpInfo();

    // the number of prediction SNPs cannot be too large
    if (predictSnps.size() > params.maxModelSnps) {
      cerr << "ERROR: Number of prediction SNPs exceeds maxModelSnps = " << params.maxModelSnps << endl;
      exit(1);
    }

    // the number of prediction snps should be equal to training snps
    if (predictSnps.size() != snps.size()) {
      cerr << "Error: Number of prediction SNPs should be equal the number of training SNPs " << endl;
      exit(1);
    }

    // read predict covariate data
    vector<double> predmaskIndivs(predictData.getNpad());
    predictData.writeMaskIndivs(predmaskIndivs.data());
    CovarBasis<GenoData> predictCov(params.predcovarFile, predictData, params.precovarCols, predmaskIndivs);

    cout << "Time for setting up dataset and read prediction data " << timer.update_time() << " sec" << endl;

    // predict the new phenotype based on the posterior mean

    double *predictOutput = ALIGN_ALLOCATE_DOUBLES(predictData.getNpad());
    geneticCorr.predict(predictOutput, predictData, predictCov);

    return 0;
  }

  cout << endl << "***** Initializing lmmnet object and normlizing snps *****" << endl << endl;

  LMMCPU lmmcpu(genoData, covarBasis, &maskIndivs[0], params.snpsPerBlock, params.estIterationTrace, params.numChrom,
                params.numCalibSnps, params.maxIterationConj, params.useExactTrace, params.imputeMethod, params.outputFile);

  cout << "Time for initializing lmmnet object and normalizing snps is " << timer.update_time() << " sec" << endl;

  cout << endl << "***** Computing heritability *****" << endl << endl;

  lmmcpu.calHeritability(phenodbl[0].data());

  cout << "Time for computing heritability is " << timer.update_time() << " sec" << endl;

  if (params.prediction) {
    cout << endl << "***** Compute the estimate for fix effect w *****" << endl << endl;

    vector<vector<double> > phenodblorigin = phenobasis.getPhenodbl(); // here we use the original pheno data
    lmmcpu.estimateFixEff(phenodblorigin[0].data(), params.useApproFixEffect);

    cout << "Timer for estimating fix effect " << timer.update_time() << " esc" << endl;

    cout << endl << "***** Compute the posterior mean *****" << endl << endl;
    if (params.RAMeff) {
      cout << endl << "Using RAM efficient model" << endl;
      lmmcpu.computePosteriorMean(params.bimFiles, params.bedFiles, phenodblorigin[0].data(), params.useApproFixEffect);
    } else {
      lmmcpu.computePosteriorMean(phenodblorigin[0].data(), params.useApproFixEffect);
    }

    cout << "Timer for computing posterior mean " << timer.update_time() << " esc" << endl;

    cout << endl << "***** Read the prediction data *****" << endl << endl;
    GenoData singlepredictData(params.predfamFile, params.predbimFiles, params.predbedFiles, params.predremoveSnpsFiles,
                               params.predremoveIndivsFiles, params.modelSnpsFiles, params.maxMissingPerSnp, params.maxMissingPerIndiv);

    const vector<SnpInfo> &predictSnps = singlepredictData.getSnpInfo();

    // the number of prediction SNPs cannot be too large
    if (predictSnps.size() > params.maxModelSnps) {
      cerr << "ERROR: Number of prediction SNPs exceeds maxModelSnps = " << params.maxModelSnps << endl;
      exit(1);
    }

    // the number of prediction snps should be equal to training snps
    if (predictSnps.size() != snps.size()) {
      cerr << "Error: Number of prediction SNPs should be equal the number of training SNPs " << endl;
      exit(1);
    }

    // read predict covariate data
    vector<double> singPredmaskIndivs(singlepredictData.getNpad());
    singlepredictData.writeMaskIndivs(singPredmaskIndivs.data());
    CovarBasis<GenoData> predictCov(params.predcovarFile, singlepredictData, params.precovarCols, singPredmaskIndivs);

    cout << "Time for setting up dataset and read prediction data " << timer.update_time() << " sec" << endl;

    // predict the new phenotype based on the posterior mean

    double *predictOutput = ALIGN_ALLOCATE_DOUBLES(singlepredictData.getNpad());
    lmmcpu.predict(predictOutput, singlepredictData, predictCov);

    cout << "Timer for predict new data " << timer.update_time() << " esc" << endl;

  }

  cout << "Total elapsed time for analysis = " << (timer.get_time() - start_time) << " sec"
       << endl;

  return 0;
}