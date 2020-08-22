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

#ifndef LMMNET_LMMCPU_H
#define LMMNET_LMMCPU_H

#include "GenoData.h"
#include "CovarBasis.h"

namespace LMMNET {
class LMMCPU {

 private:
  const GenoData &genoData; // class for genotype data
  const CovarBasis<GenoData> &covarBasis; // class for covariate data

  const double *maskIndivs; // final individual mask

  uint64 M, Npad;
  uint64 Nused;

  int numChrom; // the number of chrom used in model for leave out strategy
  int numCalibSnps;
  int snpsPerBlock; // split the matrix into block to do computation
  uint64 estIteration; // B in the initial algorithm
  uint64 maxIterationConj; // number of max iteration in conjugate gradient

  uchar *projMaskSnps;

  double sigma2g; // variance of signal
  double sigma2e; // variance of noise

  double *L2normXcols; //
  double (*snpLookupTable)[4]; // table for looking up raw genotype data

  bool useExactTrace; // whether compute the exact trace

  double *conjugateResultFixEff;

  std::vector<double> projGenoStd; // project genotype standard deviation
  std::vector<double> VinvyNorm; // l2 norm of Conjugate gradient result
  std::vector<double> fixEffect;
  std::vector<double> posteriorMean;

  double subIntercept;

  std::vector<std::pair<double, double> > Meanstd; // genotype mean and std

  std::vector<int> chromID;
  std::map<int, uint64> numSnpsPerChrom;
  std::vector<uint64> chromStartPoint;

  std::string imputeMethod;

  const std::string outputFile;

  // statistics used to calculate association test
  double calibrationFactor;
  std::vector<double> z;
  std::vector<double> zsquare;

  // initialize the lmmcpu object and normalize snps
  void initialize();
  uchar normalizeSnps(uint64 m, double *snpVector);
  void invnormalizeSnps();

  // decode snps
  inline void buildMaskedSnpCovCompVec(double snpCovCompVec[], uint64 m, double (*work)[4]) const {
    genoData.decodeSnpsVector(snpCovCompVec, maskIndivs, m, snpLookupTable[m], work);
  }

  // compute the heritability
  double calyVKVy(const double *projectPheno) const;
  void multXXT(double *out, const double *vec) const;
  void computeXXT(double *out) const;
  void multXXTTrace(double *out, const double *vec) const; // moment estimate for the trace
  void calTraceMoM(double &kv, double &kvkv) const;
  void calTraceExact(double &kv, double &kvkv) const;
  double calStandError(const double *projectPheno, double kv, double kvkv) const;

  // compute the association test statistics
  void makeConjugateMask(uchar *mask); // make masks for the leave out strategy
  void multXTmatrix(double *out, const double *matrix, int col) const;
  void multXmatrix(double *out, const double *matrix, int col) const;
  void multXXTConjugate(double *out, const double *matrix, const uchar *mask);
  void calConjugateBatch(double *VinvChromy, const double *inputMatrix); // compute batch conjguate gradient
  void calBeta(const double *VinvChromy); // compute beta with leave out strategy
  std::vector<uint64> selectProSnps(int numCalibSnps);

  // compute the estimate of fixed effect based on training data
  void multXXTConjugateWithoutMask(double *out, const double *matrix, unsigned int batchsize);
  void calConjugateWithoutMask(double *Viny, const double *inputVec, int batchsize);

  // overload blas interface
  double dotVec(const double *vec1, const double *vec2, uint64 N) const;
  void scalVec(double *vec, double alpha, uint64 elem) const;
  void compInverse(double *matrix,
                   const unsigned int row) const; // Note that this function will overwrite original matrix directly
  
  void buildPredictSnpsLookupTable(uint64 m,
                                   uint64 numPredict,
                                   double *snpVector,
                                   double (*predictionSnpLookupTable)[4]) const;
  
  /**
   * Compute the random effect based on the predict dataset (\beta^TX)
   * 
   * @param numPredict the number of predict samples
   * @param predictMaskIndivs the mask of removed individuals in predict dataset
   * @param randomEff the output of random effect
   * @param predictionSnpLookupTable the work table of predict dataset
   * @param predictData class dealing with the predict dataset
  */
  void predictRandomEff(uint64 numPredict,
                        double *predictMaskIndivs,
                        double *randomEff,
                        double (*predictionSnpLookupTable)[4],
                        const GenoData &predictData) const;
  
  /**
   * Compute the fix effect based on the predict dataset (W\mu)
   * 
   * @param numPredict the nnumber of predict samples
   * @param fixEff the output of fix effect
   * @param predictCovarMatrix the covariate matrix for predict dataset
  */
  void predictFixEff(uint64 numPredict, double *fixEff, const double *predictCovarMatrix) const;

  // interface for RAM econ model
  void normalizeSingleSnp(uchar* genoLine, double* normalizedSnp, uint64 numSamples, uint64 numUsed); // normalize single SNP and store in normalizedSnp
  void computeSinglePosteriorMean(const vector <string> &bimFiles, const vector <string> &bedFiles, const double* phenoData);


 public:
  LMMCPU(const GenoData &_genoData,
         const CovarBasis<GenoData> &_covarBasis,
         const double _maskIndivs[],
         int _snpsPerBlock,
         uint64 _estIteration,
         int _numChrom,
         int _numCalibSnps,
         uint64 _maxIterationConj,
         bool _useExactTrace,
         const std::string _imputeMethod,
         const std::string _outputFile);
  ~LMMCPU();

  /**
   * Compute the heritability of give phenotype
   * 
   * @param projectPheno project phenotype
  */
  void calHeritability(const double *projectPheno);
  void calCalibrationFactor(const double *projectPheno, bool sampleCalibrationFactor);
  void computeStatistics(std::string &outputFile) const;

  /**
   * Compute the fix effect from training dataset
   * 
   * @param projectPheno project phenotype 
   * @param useApproximate whether use approximation algorithm
  */
  void estimateFixEff(const double *projectPheno, bool useApproximate);

  /**
   * Compute the posterior mean of random effct and the standard way is to use
   * conjugate gradient to solve it. You can also use the approximation algorithm
   * by set useApproximate as true
   * 
   * @param projectPheno project phenotype
   * @param useApproximate whether use approximation algorithm
  */
  void computePosteriorMean(const double* projectPheno, bool useApproximate);

  /**
   * Mak prediction based on the predict dataset and predict covariate
   * output the final result to disk
   * 
   * @param output final prediction output
   * @param predictData class dealing with the predict dataset
   * @param predictCov class dealing with predict covariate 
  */
  void predict(double *output, const GenoData &predictData, const CovarBasis<GenoData> &predictCov) const;

  void computePosteriorMean(const vector <string> &bimFiles, const vector <string> &bedFiles,
                        const double* pheno, bool useApproximate);
};
}
#endif //LMMNET_LMMCPU_H
