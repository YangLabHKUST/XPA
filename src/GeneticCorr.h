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

#ifndef LMMNET_GENETICCORR_H
#define LMMNET_GENETICCORR_H

#include "InfoStructure.h"
#include "GenoData.h"
#include "CovarBasis.h"
#include "AuxGenoData.h"

namespace LMMNET {
class GeneticCorr {

 private:
  // dataset for the first genotype and covariate data
  const GenoData &genoData;
  const CovarBasis<GenoData> &covarBasis;

  // dataset for the auxiliary genotype and covariate data
  const AuxGenoData &auxgenoData;
  const CovarBasis<AuxGenoData> &auxcovarBasis;

  // two masks for each dataset
  const double *maskgenoIndivs;
  const double *maskauxIndivs;

  int snpsPerBlock;
  uint64 estIteration;
  uint64 estIterationMain;
  uint64 estIterationAux;
  uint64 estIterationDelta;

  uchar *projMaskSnps; // two datasets must analysis the same snps

  // dataset information for two genotype data
  uint64 Mmain, Maux, M;
  uint64 NpadGeno, NusedGeno;
  uint64 Npadaux, Nusedaux;

  // parameters need to be estimated for the two datasets
  double sigma2g, sigma2e;
  double auxsigma2g, auxsigma2e;
  double delta;
  double corr;

  double (*genosnpLookupTable)[4];
  double (*auxsnpLookupTable)[4];

  double *conjugateResultFixEff;

  std::vector<double> fixEffect;
  std::vector<double> posteriorMean;

  std::string imputeMethod;
  // outputfile name
  const std::string outputFile;

  bool useExactTrace; // whether compute the exact trace

  std::vector<std::pair<double, double> > genoMeanstd; // genotype mean and invstd
  std::vector<std::pair<double, double> > auxMeanstd; // aux genotype mean and invstd

  // hash table to be used in flip value
  std::map<std::string, std::string> flipMap;

  // snp reference position index
  std::vector<uint64> snpIndex;

  // decode snps from two genotype datasets
  inline void buildMaskedGenoSnp(double snpCovCompVec[], uint64 m, double (*work)[4]) const {
    genoData.decodeSnpsVector(snpCovCompVec, maskgenoIndivs, m, genosnpLookupTable[m], work);
  }

  // here we add additional mapping to match the snp position (snpIndex)
  inline void buildMaskedAuxSnp(double snpCovCompVec[], uint64 m, double (*work)[4]) const {
    auxgenoData.decodeSnpsVector(snpCovCompVec, maskauxIndivs, m, auxsnpLookupTable[snpIndex[m]], work);
  }

  // preprocess data
  void initialize(); // initialize the class
  bool normalizeSnps(uint64 m, uint64 numSamples, uint64 numUsed, const double *maskIndivs, double *snpVector,
                     double(*snpLookupTablfe)[4], std::vector<std::pair<double, double> > &meanstd);
  bool checkFlip(const SnpInfo &mainInfo, const SnpInfo &auxInfo);
  void flipLookupTable(const std::vector<SnpInfo> &mainInfo,
                       const std::vector<SnpInfo> &auxInfo,
                       double(*snpLookupTable)[4]);

  // compute the seperate variance component
  double calyVKVy(const double *projectPheno, uint64 numSamples, const char whichDataset) const;
  void multXXT(double *out, const double *vec, const char whichGenotype) const;
  void multXXTTrace(double *out, const double *vec, const char whichData) const; // moment estimate for the trace
  void calTraceMoM(double &kv, double &kvkv, const char whichData);
  double calStandError(const double *projectPheno, double kv, double kvkv, const char whichData) const;

  // estimate correlation
  void multXXTAux(double *out, const double *vec) const;
  void multXXTAuxTrace(double *out, const double *vec) const;
  double estimateDelta(const double *genoProjectPheno, const double *auxProjectPheno);

  // solve the blocks of conjugate gradient
  void multXTmatrix(double *out, const double *matrix, int cols, const char whichDataset) const;
  void multXmatrix(double *out, const double *matrix, int cols, const char whichDataset) const;
  void multXXTConjugate(double *out, const double *matrix, int cols, const char whichDataset) const;
  void multdeltaXXT(double *out, const double *matrix, int cols, const char whichDataset) const;
  void multOmega(double *out, const double *inputMatrix, int cols) const;
  void solveConjugateBatch(double *out, const double *inputMatrix1, const double *inputMatrix2, int col);
  void solveInverse(double *out, double *inputMatrix, double *vec, uint64 cols) const;
  void compuInverse(double *out, const unsigned int row) const;

  // predict on new data
  void buildPredictSnpsLookupTable(uint64 m,
                                   uint64 numPredict,
                                   const double *snpVector,
                                   double (*predictionSnpLookupTable)[4]) const;
  void predictRandomEff(uint64 numPredict, double *predictMaskIndivs, double *randomEff,
                        double (*predictionSnpLookupTable)[4], const GenoData &predictData) const;
  void predictFixEff(uint64 numPredict, double *fixEff, const double *predictCovarMatrix) const;

  // interface for RAM econ model
  void normalizeSingleSnp(uchar* genoLine, double* normalizedSnp, uint64 numSamples, uint64 numUsed); // normalize single SNP and store in normalizedSnp
  void computeSinglePosteriorMean(const vector <string> &bimFiles, const vector <string> &bedFiles, const double* phenoData, char whichDataset);

 public:
  GeneticCorr(const GenoData &_genoData,
              const CovarBasis<GenoData> &_covarBasis,
              const double *_maskgenoIndivs,
              const AuxGenoData &_auxgenoData,
              const CovarBasis<AuxGenoData> &_auxcovarBasis,
              const double *_maskauxIndivs,
              int _snpsPerBlock,
              uint64 _estIteration,
              uint64 _estIterationAux,
              uint64 _estIterationDelta,
              bool _useExactTrace,
              const std::string _inputeMethod,
              const std::string _outputFile);
  ~GeneticCorr();

  void compVCM(const double *genoProjecPheno, const double *auxProjectPheno);
  void estFixEff(const double *mainGenoPheno, const double *auxGenoPheno, bool useApproximate);
  // this function will use the SNPs which have already been loaded
  void estPosteriorMean();
  // this function will read data from file and compute the posterior mean
  // so you need to specify the bimFiles and bedFiles
  void estPosteriorMean(const vector <string> &bimFilesG, const vector <string> &bedFilesG,
                        const vector <string> &bimFilesA, const vector <string> &bedFilesA);
  void predict(double *output, const GenoData &predictData, const CovarBasis<GenoData> &predictCov) const;
};
}

#endif //LMMNET_GENETICCORR_H
