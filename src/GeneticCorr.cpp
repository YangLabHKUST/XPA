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

#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <math.h>

#include <mkl.h>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include "omp.h"

#include <Eigen/Dense>

#include "MemoryUtils.h"
#include "NumericUtils.h"
#include "IOUtils.h"
#include "InfoStructure.h"
#include "Timer.h"
#include "GeneticCorr.h"

namespace LMMNET {
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

GeneticCorr::GeneticCorr(const LMMNET::GenoData &_genoData,
                         const LMMNET::CovarBasis<GenoData> &_covarBasis,
                         const double *_maskgenoIndivs,
                         const LMMNET::AuxGenoData &_auxgenoData,
                         const LMMNET::CovarBasis<AuxGenoData> &_auxcovarBasis,
                         const double *_maskauxIndivs,
                         int _snpsPerBlock,
                         uint64 _estIteration,
                         uint64 _estIterationAux,
                         uint64 _estIterationDelta,
                         bool _useExactTrace,
                         const std::string _imputeMethod,
                         const std::string _outputFIle) : genoData(_genoData),
                                                    covarBasis(_covarBasis),
                                                    maskgenoIndivs(_maskgenoIndivs),
                                                    auxgenoData(_auxgenoData),
                                                    auxcovarBasis(_auxcovarBasis),
                                                    maskauxIndivs(_maskauxIndivs),
                                                    snpsPerBlock(_snpsPerBlock),
                                                    estIterationMain(_estIteration),
                                                    estIterationAux(_estIterationAux),
                                                    estIterationDelta(_estIterationDelta),
                                                    useExactTrace(_useExactTrace),
                                                    imputeMethod(_imputeMethod),
                                                    outputFile(_outputFIle) {
  initialize();
}

GeneticCorr::~GeneticCorr() {
  ALIGN_FREE(genosnpLookupTable);
  ALIGN_FREE(auxsnpLookupTable);
  ALIGN_FREE(projMaskSnps);
}

void GeneticCorr::initialize() {
  // get dataset information
  Mmain = genoData.getM(); // the number of snps should be equal
  Maux = auxgenoData.getM(); // total number of snps in auxiliary data
  NpadGeno = genoData.getNpad();
  Npadaux = auxgenoData.getNpad();
  NusedGeno = genoData.getNused();
  Nusedaux = auxgenoData.getNused();

  M = Mmain;

  assert(Mmain == Maux);

  genosnpLookupTable = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(Mmain * sizeof(*genosnpLookupTable));
  memset(genosnpLookupTable, 0, Mmain * sizeof(*genosnpLookupTable));

  auxsnpLookupTable = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(Maux * sizeof(*auxsnpLookupTable));
  memset(auxsnpLookupTable, 0, Maux * sizeof(*auxsnpLookupTable));

  projMaskSnps = ALIGN_ALLOCATE_UCHARS(M);
  genoData.writeMaskSnps(projMaskSnps);

  // get reference snp position
  snpIndex = auxgenoData.getSnpIndex();

  cout << "Snp index size is " << snpIndex.size() << endl;
  cout << "Normalizing snps ......" << endl;

  // normalize the snps parallel and store the lookup table seperately
  double *genosnpVector = ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * NpadGeno);
  double *auxsnpVector = ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * Npadaux);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(omp_get_max_threads() * 256 * sizeof(*workTable));
  double map0129[4] = {0, 1, 2, 9};

  auxMeanstd.reserve(Maux);
  genoMeanstd.reserve(Mmain);

  // normalize snps in main genotype dataset
#pragma omp parallel for
  for (uint64 m = 0; m < Mmain; m++) {
    if (projMaskSnps[m]) {
      genoData.decodeSnpsVector(genosnpVector + (omp_get_thread_num() * NpadGeno),
                                maskgenoIndivs,
                                m,
                                map0129,
                                workTable + (omp_get_thread_num() << 8));
      bool validGeno = normalizeSnps(m,
                                     NpadGeno,
                                     NusedGeno,
                                     maskgenoIndivs,
                                     genosnpVector + (omp_get_thread_num() * NpadGeno),
                                     genosnpLookupTable,
                                     genoMeanstd);
      auxgenoData.decodeSnpsVector(auxsnpVector + (omp_get_thread_num() * Npadaux),
                                   maskauxIndivs,
                                   m,
                                   map0129,
                                   workTable + (omp_get_thread_num() << 8));
      bool validaux = normalizeSnps(snpIndex[m],
                                    Npadaux,
                                    Nusedaux,
                                    maskauxIndivs,
                                    auxsnpVector + (omp_get_thread_num() * Npadaux),
                                    auxsnpLookupTable,
                                    auxMeanstd);
      if (validGeno && validaux) {
        projMaskSnps[m] = 1;
      }
    }
  }

  cout << "Impute missing value by " << imputeMethod << endl;

  // flip the lookup table according to the allel reference
  const vector<SnpInfo> &mainInfo = genoData.getSnpInfo();
  const vector<SnpInfo> &auxInfo = auxgenoData.getSnpInfo();
  flipLookupTable(mainInfo, auxInfo, auxsnpLookupTable);

  ALIGN_FREE(genosnpVector);
  ALIGN_FREE(auxsnpVector);
  ALIGN_FREE(workTable);
}

bool GeneticCorr::normalizeSnps(uint64 m,
                                uint64 numSamples,
                                uint64 numUsed,
                                const double *maskIndivs,
                                double *snpVector,
                                double(*snpLookupTable)[4],
                                std::vector<std::pair<double, double> > &meanstd) {
  double sumGenoNonMissing = 0;
  int numGenoNonMissing = 0;
  for (uint64 n = 0; n < numSamples; n++) {
    if (maskIndivs[n] // important! don't use masked-out values
        && snpVector[n] != 9) {
      sumGenoNonMissing += snpVector[n];
      numGenoNonMissing++;
    }
  }

  if (numGenoNonMissing == 0) return 0;

  // mean-center and replace missing values with mean (centered to 0)
//        double mean = sumGenoNonMissing / numGenoNonMissing;
  double mean = sumGenoNonMissing / (numSamples - 1);
  for (uint64 n = 0; n < numSamples; n++) {
    if (maskIndivs[n]) {
      if (snpVector[n] == 9)
        snpVector[n] = 0;
      else
        snpVector[n] -= mean;
    } else
      assert(snpVector[n] == 0); // buildMaskedSnpVector should've already zeroed
  }

  double meanCenterNorm2 = NumericUtils::norm2(snpVector, numSamples);
  // normalize to Nused-1 (dimensionality of subspace with all-1s vector projected out)
  double invMeanCenterNorm = sqrt((numUsed - 1) / meanCenterNorm2);

  // save lookup of 0129 values: 0, 1/meanCenterNorm, 2/meanCenterNorm, mean/meanCenterNorm
  snpLookupTable[m][0] = -mean * invMeanCenterNorm;
  snpLookupTable[m][1] = (1 - mean) * invMeanCenterNorm;
  snpLookupTable[m][2] = (2 - mean) * invMeanCenterNorm;

  // transform the input to lower case in case of sensitive
  transform(imputeMethod.begin(), imputeMethod.end(), imputeMethod.begin(), ::tolower);
  if (imputeMethod == "zero") {
    snpLookupTable[m][3] = -mean * invMeanCenterNorm;
  } else if (imputeMethod == "mean") {
    snpLookupTable[m][3] = 0;
  } else {
    cout << "Warning: There is no such imputing method. Use the mean value to impute." << endl;
    imputeMethod = "mean";
    snpLookupTable[m][3] = 0;
  }

  meanstd[m] = std::make_pair(mean, invMeanCenterNorm);

  return true;
}

bool GeneticCorr::checkFlip(const SnpInfo &mainInfo, const SnpInfo &auxInfo) {
  std::string allelref = flipMap[mainInfo.allele1] + flipMap[mainInfo.allele2];
  std::string allel = flipMap[auxInfo.allele2] + flipMap[auxInfo.allele1];

  if (allelref[0] == allelref[1])
    return false;
  return (allelref == allel);
}

void GeneticCorr::flipLookupTable(const std::vector<SnpInfo> &mainInfo,
                                  const std::vector<SnpInfo> &auxInfo,
                                  double(*snpLookupTable)[4]) {
  // initialize the flip map in order to convert value quickly
  flipMap["A"] = "A";
  flipMap["T"] = "A";
  flipMap["G"] = "C";
  flipMap["C"] = "C";

  // in this case, we assume that the number of snps is equal between two datasets.
  uint64 totalFlip = 0;
  for (uint64 m = 0; m < Maux; m++) {
    bool flip = checkFlip(mainInfo[m], auxInfo[snpIndex[m]]);

    if (flip) { // if flip, we should convert lookup table value to negative
      snpLookupTable[snpIndex[m]][0] *= -1;
      snpLookupTable[snpIndex[m]][1] *= -1;
      snpLookupTable[snpIndex[m]][2] *= -1;
      snpLookupTable[snpIndex[m]][3] *= -1;
      totalFlip++;
    }
  }

  cout << "The total number of flipped snp is " << totalFlip << endl;
}

double GeneticCorr::calyVKVy(const double *projectPheno, uint64 numSamples, const char whichDataset) const {
  double *temp = ALIGN_ALLOCATE_DOUBLES(numSamples);

  // compute A=XX^TVy
  multXXT(temp, projectPheno, whichDataset);

  // compute yVA
  MKL_INT n = numSamples;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  double yvkvy = cblas_ddot(n, projectPheno, incx, temp, incy) / M;

#ifdef DEBUG
  cout << "yvkvy result is " << yvkvy << endl;
#endif

  ALIGN_FREE(temp);
  return yvkvy;
}

void GeneticCorr::multXXT(double *out, const double *vec, const char whichGenotype) const {
  // set Npad according to different datasets
  uint64 Npad;
  if (whichGenotype == 'G') {
    Npad = NpadGeno;
  } else {
    Npad = Npadaux;
  }

  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(Npad * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * 256 * sizeof(*workTable));

  // store the temp result
  double *temp1 = ALIGN_ALLOCATE_DOUBLES(snpsPerBlock);
  memset(out, 0, Npad * sizeof(double));

  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m])
        if (whichGenotype == 'G')
          buildMaskedGenoSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
        else
          buildMaskedAuxSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
      else
        memset(snpBlock + mPlus * Npad, 0, Npad * sizeof(snpBlock[0]));
    }
    // compute A=X^TV
    MKL_INT row = Npad;
    MKL_INT col = snpsPerBLockCrop;
    double alpha = 1.0;
    MKL_INT lda = Npad;
    MKL_INT incx = 1;
    double beta = 0.0;
    MKL_INT incy = 1;
    cblas_dgemv(CblasColMajor, CblasTrans, row, col, alpha, snpBlock, lda, vec, incx, beta, temp1, incy);

    // compute XA
    double beta1 = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, row, col, alpha, snpBlock, lda, temp1, incx, beta1, out, incy);

  }
  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
  ALIGN_FREE(temp1);
}

void GeneticCorr::multXXTTrace(double *out, const double *vec, const char whichData) const {
  // set Npad for different datasets
  uint64 Npad;
  if (whichData == 'G') {
    Npad = NpadGeno;
  } else {
    Npad = Npadaux;
  }

  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(Npad * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * 256 * sizeof(*workTable));

  // store the temp result
  double *temp1 = ALIGN_ALLOCATE_DOUBLES(snpsPerBlock);
  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m])
        if (whichData == 'G')
          buildMaskedGenoSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
        else
          buildMaskedAuxSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
      else
        memset(snpBlock + mPlus * Npad, 0, Npad * sizeof(snpBlock[0]));
    }

    for (uint64 iter = 0; iter < estIteration; iter++) {
      // compute A=X^TV
      MKL_INT row = Npad;
      MKL_INT col = snpsPerBLockCrop;
      double alpha = 1.0;
      MKL_INT lda = Npad;
      MKL_INT incx = 1;
      double beta = 0.0;
      MKL_INT incy = 1;
      cblas_dgemv(CblasColMajor,
                  CblasTrans,
                  row,
                  col,
                  alpha,
                  snpBlock,
                  lda,
                  vec + iter * Npad,
                  incx,
                  beta,
                  temp1,
                  incy);

      // compute XA
      double beta1 = 1.0;
      cblas_dgemv(CblasColMajor, CblasNoTrans, row, col, alpha, snpBlock, lda, temp1, incx, beta1, out + iter * Npad,
                  incy);

    }

  }
  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
  ALIGN_FREE(temp1);
}

void GeneticCorr::compVCM(const double *genoProjecPheno, const double *auxProjectPheno) {
  Timer timer;
  cout << endl << "***** Computing variance component for main dataset *****" << endl;
  // compute variance component for one dataset
  double yvy = NumericUtils::dot(genoProjecPheno, genoProjecPheno, NpadGeno);
  double yvkvy = calyVKVy(genoProjecPheno, NpadGeno, 'G'); // the third argument G means the main genotype data

#if DEBUG
  cout << "yvy " << yvy << endl;
  cout << "yvkvy " << yvkvy << endl;
#endif

  double kv, kvkv;

  calTraceMoM(kv, kvkv, 'G');

#if DEBUG
  cout << "kv is " << kv << endl;
  cout << "kvkv is " << kvkv << endl;
#endif

  auto temp1 = static_cast<double>(NusedGeno - covarBasis.getC());
  sigma2g = (yvkvy * temp1 - yvy * kv) / (temp1 * kvkv - kv * kv);
  sigma2e = (yvy - kv * sigma2g) / temp1;

  double standardErrorGeno = calStandError(genoProjecPheno, kv, kvkv, 'G');

  cout << "sigma2g " << sigma2g << endl;
  cout << "sigma2e " << sigma2e << endl;
  cout << "heritability is " << sigma2g / (sigma2e + sigma2g) << endl;
  cout << "standard error is " << standardErrorGeno << endl;

  cout << "Timer for computing variance component of main dataset is " << timer.update_time() << " sec" << endl;

  cout << endl << "***** Computing variance component for auxiliary dataset *****" << endl;

  yvy = NumericUtils::dot(auxProjectPheno, auxProjectPheno, Npadaux);
  yvkvy = calyVKVy(auxProjectPheno, Npadaux, 'A');

#if DEBUG
  cout << "yvy " << yvy << endl;
  cout << "yvkvy " << yvkvy << endl;
#endif

  calTraceMoM(kv, kvkv, 'A');
#if DEBUG
  cout << "kv is " << kv << endl;
  cout << "kvkv is " << kvkv << endl;
#endif

  auto temp2 = static_cast<double>(Nusedaux - auxcovarBasis.getC());
  auxsigma2g = (yvkvy * temp2 - yvy * kv) / (temp2 * kvkv - kv * kv);
  auxsigma2e = (yvy - kv * auxsigma2g) / temp2;
  double standardErrorAux = calStandError(auxProjectPheno, kv, kvkv, 'A');

  cout << "sigma2g " << auxsigma2g << endl;
  cout << "sigma2e " << auxsigma2e << endl;
  cout << "heritability is " << auxsigma2g / (auxsigma2e + auxsigma2g) << endl;
  cout << "standard error is " << standardErrorAux << endl;

  cout << "Timer for computing variance component of auxiliary dataset is " << timer.update_time() << " sec" << endl;

  cout << endl << "***** Computing the estimate of delta *****" << endl;
  delta = estimateDelta(genoProjecPheno, auxProjectPheno);

  cout << "delta " << delta << endl;

  corr = delta / sqrt(sigma2g * auxsigma2g);
  cout << "Correlation is " << corr << endl;


}

void GeneticCorr::multXXTAux(double *out, const double *vec) const {

  double *genosnpBlock = ALIGN_ALLOCATE_DOUBLES(NpadGeno * snpsPerBlock);
  double *auxsnpBlock = ALIGN_ALLOCATE_DOUBLES(Npadaux * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * 256 * sizeof(*workTable));

  // store the temp result
  double *temp1 = ALIGN_ALLOCATE_DOUBLES(snpsPerBlock);
  memset(out, 0, NpadGeno * sizeof(double));

  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m]) {
        buildMaskedGenoSnp(genosnpBlock + mPlus * NpadGeno, m, workTable + (omp_get_thread_num() << 8));
        buildMaskedAuxSnp(auxsnpBlock + mPlus * Npadaux, m, workTable + (omp_get_thread_num() << 8));
      } else {
        memset(genosnpBlock + mPlus * NpadGeno, 0, NpadGeno * sizeof(genosnpBlock[0]));
        memset(auxsnpBlock + mPlus * Npadaux, 0, Npadaux * sizeof(auxsnpBlock[0]));
      }
    }
    // compute A=X^TV
    MKL_INT row = Npadaux;
    MKL_INT col = snpsPerBLockCrop;
    double alpha = 1.0;
    MKL_INT lda = row;
    MKL_INT incx = 1;
    double beta = 0.0;
    MKL_INT incy = 1;
    cblas_dgemv(CblasColMajor, CblasTrans, row, col, alpha, auxsnpBlock, lda, vec, incx, beta, temp1, incy);

    // compute XA
    MKL_INT row1 = NpadGeno;
    MKL_INT lda1 = row1;
    double beta1 = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, row1, col, alpha, genosnpBlock, lda1, temp1, incx, beta1, out, incy);

  }
  ALIGN_FREE(genosnpBlock);
  ALIGN_FREE(auxsnpBlock);
  ALIGN_FREE(workTable);
  ALIGN_FREE(temp1);
}

void GeneticCorr::multXXTAuxTrace(double *out, const double *vec) const {
  double *genosnpBlock = ALIGN_ALLOCATE_DOUBLES(NpadGeno * snpsPerBlock);
  double *auxsnpBlock = ALIGN_ALLOCATE_DOUBLES(Npadaux * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * 256 * sizeof(*workTable));

  // store the temp result
  double *temp1 = ALIGN_ALLOCATE_DOUBLES(snpsPerBlock);
  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m]) {
        buildMaskedGenoSnp(genosnpBlock + mPlus * NpadGeno, m, workTable + (omp_get_thread_num() << 8));
        buildMaskedAuxSnp(auxsnpBlock + mPlus * Npadaux, m, workTable + (omp_get_thread_num() << 8));
      } else {
        memset(genosnpBlock + mPlus * NpadGeno, 0, NpadGeno * sizeof(genosnpBlock[0]));
        memset(auxsnpBlock + mPlus * Npadaux, 0, Npadaux * sizeof(auxsnpBlock[0]));
      }
    }

    for (uint64 iter = 0; iter < estIteration; iter++) {
      // compute A=X^TV
      MKL_INT row = Npadaux;
      MKL_INT col = snpsPerBLockCrop;
      double alpha = 1.0;
      MKL_INT lda = row;
      MKL_INT incx = 1;
      double beta = 0.0;
      MKL_INT incy = 1;
      cblas_dgemv(CblasColMajor,
                  CblasTrans,
                  row,
                  col,
                  alpha,
                  auxsnpBlock,
                  lda,
                  vec + iter * Npadaux,
                  incx,
                  beta,
                  temp1,
                  incy);

      // compute XA
      double beta1 = 1.0;
      MKL_INT row1 = NpadGeno;
      MKL_INT lda1 = row1;
      cblas_dgemv(CblasColMajor,
                  CblasNoTrans,
                  row1,
                  col,
                  alpha,
                  genosnpBlock,
                  lda1,
                  temp1,
                  incx,
                  beta1,
                  out + iter * NpadGeno,
                  incy);
    }

  }
  ALIGN_FREE(genosnpBlock);
  ALIGN_FREE(auxsnpBlock);
  ALIGN_FREE(workTable);
  ALIGN_FREE(temp1);
}

double GeneticCorr::estimateDelta(const double *genoProjectPheno, const double *auxProjectPheno) {
  double *XXTy = ALIGN_ALLOCATE_DOUBLES(NpadGeno);
  memset(XXTy, 0, NpadGeno * sizeof(double));

  // compute XXTY
  multXXTAux(XXTy, auxProjectPheno);

  // compute YXXTY
  double yvkvy = NumericUtils::dot(genoProjectPheno, XXTy, NpadGeno) / M;

  cout << "yvkvy in estimating Delta is " << yvkvy << endl;

  // estimate the trace by sampling or exact trace
  boost::mt19937 rng; // I don't seed it on purpouse
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937 &, boost::normal_distribution<> > var_nor(rng, nd);

  uint64 numGaussianAux = estIterationDelta * Npadaux;
  double *gaussianVecAux = ALIGN_ALLOCATE_DOUBLES(numGaussianAux);
  double *result = ALIGN_ALLOCATE_DOUBLES(estIterationDelta * NpadGeno);
  memset(result, 0, estIterationDelta * NpadGeno * sizeof(double));

  for (uint n = 0; n < numGaussianAux; n++) {
    gaussianVecAux[n] = var_nor();
  }

  // project covariates from X1
  for (uint iter = 0; iter < estIterationDelta; iter++) {
    auxcovarBasis.projectCovarsVec(gaussianVecAux + iter * Npadaux);
  }

  multXXTAuxTrace(result, gaussianVecAux);

  // project covariates from X2
  for (uint iter = 0; iter < estIterationDelta; iter++) {
    covarBasis.projectCovarsVec(result + iter * NpadGeno);
  }

  // compute l2 norm
  uint64 norm1 = estIterationDelta * M * M;

  MKL_INT n = estIteration * NpadGeno;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  double kvkv = cblas_ddot(n, result, incx, result, incy) / norm1;

  ALIGN_FREE(XXTy);
  ALIGN_FREE(gaussianVecAux);
  ALIGN_FREE(result);

  return yvkvy / kvkv;
}

void GeneticCorr::calTraceMoM(double &kv, double &kvkv, const char whichData) {
  uint64 Npad;
  if (whichData == 'G') {
    Npad = NpadGeno;
    estIteration = estIterationMain;
  }
  else {
    Npad = Npadaux;
    estIteration = estIterationAux;
  }

  Timer timer;
  boost::mt19937 rng; // I don't seed it on purpouse
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937 &, boost::normal_distribution<> > var_nor(rng, nd);

  uint64 numGaussian = estIteration * Npad;
  double *gaussianVec = ALIGN_ALLOCATE_DOUBLES(numGaussian);
  double *result = ALIGN_ALLOCATE_DOUBLES(estIteration * Npad);
  memset(result, 0, estIteration * Npad * sizeof(double));

  for (uint n = 0; n < numGaussian; n++) {
    gaussianVec[n] = var_nor();
  }

  cout << "Time for setting up compute trace is " << timer.update_time() << " sec" << endl;
  // compute A=XX^TZ for all blocks
  multXXTTrace(result, gaussianVec, whichData);

  // compute VA for all blocks
  if (whichData == 'G') {
    for (uint iter = 0; iter < estIteration; iter++) {
      covarBasis.projectCovarsVec(result + iter * Npad);
    }
  } else {
    for (uint iter = 0; iter < estIteration; iter++) {
      auxcovarBasis.projectCovarsVec(result + iter * Npad);
    }
  }

  // compute l2 norm
  uint64 norm1 = estIteration * M * M;
  uint64 norm2 = estIteration * M;

  MKL_INT n = estIteration * Npad;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  kvkv = cblas_ddot(n, result, incx, result, incy) / norm1;
  kv = cblas_ddot(n, result, incx, gaussianVec, incy) / norm2;

  ALIGN_FREE(gaussianVec);
  ALIGN_FREE(result);
#ifdef DEBUG
  cout << "Calculate MoM trace kvkv " << kvkv << endl;
  cout << "Calculate MoM trace kv " << kv << endl;
#endif
}

double GeneticCorr::calStandError(const double *projectPheno, double kv, double kvkv, const char whichData) const {
  // set the parameters for different datasets
  uint64 Npad, Nused, C;
  double _sigma2g, _sigma2e;
  if (whichData == 'G') {
    Npad = NpadGeno;
    Nused = NusedGeno;
    _sigma2g = sigma2g;
    _sigma2e = sigma2e;
    C = covarBasis.getC();
  } else {
    Npad = Npadaux;
    Nused = Nusedaux;
    _sigma2g = auxsigma2g;
    _sigma2e = auxsigma2e;
    C = auxcovarBasis.getC();
  }

  // compute inverse S matrix column major
  double S[4] = {kvkv, kv, kv, static_cast<double>(Nused - C)};

  // LU decomposition
  lapack_int n = 2;
  lapack_int m = 2;
  lapack_int lda = 2;
  lapack_int ipiv[2];
  lapack_int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, S, lda, ipiv);

  if (info != 0) {
    cerr << "Failed to compute LU decomposition " << endl;
    cerr << "Cannot compute the inverse matrix " << endl;
    exit(1);
  }

  // get inverse matrix
  lapack_int info1 = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, S, lda, ipiv);

  if (info1 != 0) {
    cerr << "Failed to compute LU decomposition " << endl;
    cerr << "Cannot compute the inverse matrix " << endl;
    exit(1);
  }

  // compute cov of theta
  auto* omegaY = ALIGN_ALLOCATE_DOUBLES(Npad);

  multXXT(omegaY, projectPheno, whichData);
  for (uint64 n = 0; n < Npad; n++) {
    omegaY[n] = omegaY[n] * _sigma2g / M + _sigma2e * projectPheno[n];
  }

  double yTomegay = 2 * NumericUtils::dot(projectPheno, omegaY, Npad);

  // compute cov diagnoal
  auto* KVomegay = ALIGN_ALLOCATE_DOUBLES(Npad);
  if (whichData == 'G') {
    covarBasis.projectCovarsVec(omegaY);
  } else {
    auxcovarBasis.projectCovarsVec(omegaY);
  }
  multXXT(KVomegay, omegaY, whichData);
  NumericUtils::scaleElem(KVomegay, static_cast<double>(1) / M, Npad);

  double dig = 2 * NumericUtils::dot(projectPheno, KVomegay, Npad);

  // compute first elem
  auto* VKy = ALIGN_ALLOCATE_DOUBLES(Npad);
  auto* omegaVKy = ALIGN_ALLOCATE_DOUBLES(Npad);

  multXXT(VKy, projectPheno, whichData);
  NumericUtils::scaleElem(VKy, static_cast<double>(1) / M, Npad);
  if (whichData == 'G') {
    covarBasis.projectCovarsVec(VKy);
  } else {
    auxcovarBasis.projectCovarsVec(VKy);
  }
  multXXT(omegaVKy, VKy, whichData);
  for (uint64 n = 0; n < Npad; n++) {
    omegaVKy[n] = omegaVKy[n] * _sigma2g / M + _sigma2e * VKy[n];
  }

  double elem = 2 * NumericUtils::dot(VKy, omegaVKy, Npad);
  elem += _sigma2g * _sigma2g * kv / estIteration;

  double cov[4] = {elem, dig, dig, yTomegay};

  // using eigen to compute the small scale matrix multiplication
  // S^{-1}covS^{-1}
  Eigen::MatrixXd Sinv(2,2);
  Sinv << S[0], S[1],
      S[2], S[3];
  Eigen::MatrixXd temp(2,2);
  temp << cov[0], cov[1],
      cov[2], cov[3];
  Eigen::MatrixXd covTheta = Sinv * temp * Sinv;

  // deltah
  double denominator = _sigma2g * kv + _sigma2e * static_cast<double>(Nused - C);
  double coeff = kv * static_cast<double>(Nused - C) / (denominator * denominator);
  Eigen::Vector2d deltah;
  deltah(0, 0) = coeff * sigma2g; deltah(1, 0) = coeff * (-sigma2e);

  double stdHreg = sqrt(deltah.transpose() * covTheta * deltah);

  return stdHreg;
}

void GeneticCorr::estFixEff(const double *mainGenoPheno, const double *auxGenoPheno, bool useApproximate) {
  // get covarMatrix from two datasets
  const double *covarMatrix = covarBasis.getCovarMatrix();
  const double *auxcovarMatrix = auxcovarBasis.getCovarMatrix();

  // create batch input vectors
  int batchSize1 = 1 + covarBasis.getC();
  double *inputMatrix1 = ALIGN_ALLOCATE_DOUBLES(NpadGeno * batchSize1);
  memcpy(inputMatrix1, mainGenoPheno, NpadGeno * sizeof(double));
  memcpy(inputMatrix1 + NpadGeno, covarMatrix, NpadGeno * covarBasis.getC() * sizeof(double));

  int batchSzie2 = 1 + auxcovarBasis.getC();
  double *inputMatrix2 = ALIGN_ALLOCATE_DOUBLES(Npadaux * batchSzie2);
  memcpy(inputMatrix2, auxGenoPheno, Npadaux * sizeof(double));
  memcpy(inputMatrix2 + Npadaux, auxcovarMatrix, Npadaux * auxcovarBasis.getC() * sizeof(double));

  // solve conjugate gradient, the phenotype result in the first column
  cout << endl << "Solving conjugate gradient in estimaing fix effect" << endl;
  Timer timer;
  conjugateResultFixEff = ALIGN_ALLOCATE_DOUBLES((NpadGeno + Npadaux) * (batchSize1 + batchSzie2 - 1));

  if (!useApproximate) {
    // use the conjugate gradient to compute the exact fix effect
    solveConjugateBatch(conjugateResultFixEff, inputMatrix1, inputMatrix2, (batchSize1 + batchSzie2 - 1));

    cout << "Time for solving conjugate gradient is " << timer.update_time() << " sec" << endl;
  } else {
    // use the approximate fix effect to boost the program performance
    memset(conjugateResultFixEff, 0, (NpadGeno + Npadaux) * (batchSize1 + batchSzie2 - 1) * sizeof(double));
    // copy the first input matrix to the specific section in the whole matrix
    for (int col = 0; col < batchSize1; col++) {
      memcpy(conjugateResultFixEff + col * (NpadGeno + Npadaux), inputMatrix1 + NpadGeno * col, NpadGeno * sizeof(double));
    }

    // copy the second input matrix to the specific section in the whole matrix
    double *temp = conjugateResultFixEff + (NpadGeno + Npadaux) * batchSize1 + NpadGeno;
    for (int col = 0; col < batchSzie2; col++) {
      memcpy(temp + col * (NpadGeno + Npadaux), inputMatrix2 + (col + 1) * Npadaux, Npadaux * sizeof(double));
    }

    // copy the phenotype data into one column
    memcpy(conjugateResultFixEff + NpadGeno, inputMatrix2, Npadaux * sizeof(double));
  }

  uint64 totalSamples = NpadGeno + Npadaux;
  // create whole matrix z by merge two covariate matrix
  double *Z = ALIGN_ALLOCATE_DOUBLES(totalSamples * (batchSize1 + batchSzie2 - 2));
  memset(Z, 0, totalSamples * (batchSize1 + batchSzie2 - 2) * sizeof(double));

  // copy the upper left part of covariate matrix
  for (uint64 col = 0; col < (batchSize1 - 1); col++) {
    memcpy(Z + col * totalSamples, covarMatrix + col * NpadGeno, NpadGeno * sizeof(double));
  }

  double *z_middle = Z + (batchSize1 - 1) * totalSamples + NpadGeno;

  for (uint64 col = 0; col < (batchSzie2 - 1); col++) {
    memcpy(z_middle + col * totalSamples, auxcovarMatrix + col * Npadaux, Npadaux * sizeof(double));
  }


  // ZTomega-1y
  uint64 totalComp = covarBasis.getC() + auxcovarBasis.getC();
  double *ZTOinvZ = ALIGN_ALLOCATE_DOUBLES(totalComp * totalComp);

  // parameters for clbas_dgemm
  MKL_INT m = totalComp;
  MKL_INT n = totalComp;
  MKL_INT k = totalSamples;
  const double alpha = 1.0;
  MKL_INT lda = k;
  MKL_INT ldb = k;
  const double beta = 0.0;
  const MKL_INT ldc = m;
  double *inputVec = conjugateResultFixEff + totalSamples;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, Z, lda, inputVec, ldb,
              beta, ZTOinvZ, ldc);

  // compute inverse instead of solving linear equation
  compuInverse(ZTOinvZ, totalComp);

  // compute ZTOinvy
  double *ZTOinvy = ALIGN_ALLOCATE_DOUBLES(totalComp);
  memset(ZTOinvy, 0, totalComp * sizeof(double));

  // parameters for clbas_dgemv (reuse the old parameters delcaration)
  const MKL_INT m1 = totalSamples;
  const MKL_INT n1 = totalComp;
  const double alpha1 = 1.0;
  const double beta1 = 0.0;
  const MKL_INT lda1 = m1;
  const MKL_INT incx = 1;
  const MKL_INT incy = 1;
  double *vec = conjugateResultFixEff; // the conjugate gradient of phenotype result is the first column

  cblas_dgemv(CblasColMajor, CblasTrans, m1, n1, alpha1, Z, lda1, vec, incx, beta1, ZTOinvy, incy);

  // compute the final fixed effect
  fixEffect.resize(totalComp);

  // save the fix effect
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, ZTOinvZ, m, ZTOinvy, incx, beta, fixEffect.data(), incy);

  ALIGN_FREE(inputMatrix1);
  ALIGN_FREE(inputMatrix2);
  ALIGN_FREE(Z);
  ALIGN_FREE(ZTOinvy);
  ALIGN_FREE(ZTOinvZ);
}

void GeneticCorr::multXTmatrix(double *out, const double *matrix, int cols, const char whichDataset) const {
  // set matrix rows according to the different datasets
  uint64 Npad;
  if (whichDataset == 'G')
    Npad = NpadGeno;
  else
    Npad = Npadaux;

  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(Npad * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(omp_get_max_threads() * 256 * sizeof(*workTable));

  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m])
        if (whichDataset == 'G') // decoder different snps according the different datasets
          buildMaskedGenoSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
        else
          buildMaskedAuxSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
      else
        memset(snpBlock + mPlus * Npad, 0, Npad * sizeof(snpBlock[0]));
    }

    // compute A=X^TV
    MKL_INT m = cols;
    MKL_INT n = snpsPerBLockCrop;
    MKL_INT k = Npad;
    double alpha = 1.0;
    MKL_INT lda = Npad;
    MKL_INT ldb = Npad;
    double beta = 0.0;
    MKL_INT ldc = cols;
    double *temp_out = out + m0 * cols;

//            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, matrix, lda, snpBlock, ldb, beta, temp_out, ldc);
    const char transA = 'T';
    const char transB = 'N';
    dgemm(&transA, &transB, &m, &n, &k, &alpha, matrix, &lda, snpBlock, &ldb, &beta, temp_out, &ldc);
  }

  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
}

void GeneticCorr::multXmatrix(double *out, const double *matrix, int cols, const char whichDataset) const {
  // set matrix rows according to the different datasets
  uint64 Npad;
  if (whichDataset == 'G')
    Npad = NpadGeno;
  else
    Npad = Npadaux;

  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(Npad * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(omp_get_max_threads() * 256 * sizeof(*workTable));

  memset(out, 0, Npad * cols * sizeof(double));

  uint64 ucol = cols;
  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m])
        if (whichDataset == 'G') // decoder different snps according the different datasets
          buildMaskedGenoSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
        else
          buildMaskedAuxSnp(snpBlock + mPlus * Npad, m, workTable + (omp_get_thread_num() << 8));
      else
        memset(snpBlock + mPlus * Npad, 0, Npad * sizeof(snpBlock[0]));
    }

    MKL_INT m = Npad;
    MKL_INT n = cols;
    MKL_INT k = snpsPerBLockCrop;
    double alpha = 1.0;
    MKL_INT lda = m;
    MKL_INT ldb = n;
    MKL_INT ldc = m;
    double beta = 1.0;
    const double *temp = matrix + m0 * ucol;

//            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, snpBlock, lda, temp, ldb, beta, out, ldc);
    const char transA = 'N';
    const char transB = 'T';
    dgemm(&transA, &transB, &m, &n, &k, &alpha, snpBlock, &lda, temp, &ldb, &beta, out, &ldc);
  }

  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
}

void GeneticCorr::multXXTConjugate(double *out, const double *matrix, int cols, const char whichDataset) const {
  // set matrix rows according to the different datasets
  uint64 Npad;
  double vcmbeta;
  double vcmeps;
  if (whichDataset == 'G') {
    Npad = NpadGeno;
    vcmbeta = sigma2g;
    vcmeps = sigma2e;
  } else {
    Npad = Npadaux;
    vcmbeta = auxsigma2g;
    vcmeps = auxsigma2e;
  }

  double *XtransMatrix = ALIGN_ALLOCATE_DOUBLES(M * cols);

  multXTmatrix(XtransMatrix, matrix, cols, whichDataset); // compute X^TY

  multXmatrix(out, XtransMatrix, cols, whichDataset); // compute XX^TY

  for (uint64 col = 0, m = 0; col < cols; col++) {
    double invM = vcmbeta / (double) (M);
    for (uint64 n = 0; n < Npad; n++, m++)
      out[m] = invM * out[m] + vcmeps * matrix[m];
  }

  ALIGN_FREE(XtransMatrix);
}

void GeneticCorr::multdeltaXXT(double *out, const double *matrix, int cols, const char whichDataset) const {
  // if whichDataset == 'G' we first compute X1T matrix, else we first compute X2T matrix
  char order1;
  char order2;
  uint64 Npad;
  if (whichDataset == 'G') {
    order1 = 'G';
    order2 = 'A';
    Npad = Npadaux;
  } else {
    order1 = 'A';
    order2 = 'G';
    Npad = NpadGeno;
  }

  double *XtransMatrix = ALIGN_ALLOCATE_DOUBLES(M * cols);

  multXTmatrix(XtransMatrix, matrix, cols, order1); // compute X^TY

  multXmatrix(out, XtransMatrix, cols, order2); // compute XX^TY
  for (uint64 col = 0, m = 0; col < cols; col++) {
    double invM = delta / (double) (M);
    for (uint64 n = 0; n < Npad; n++, m++)
      out[m] = invM * out[m];
  }

  ALIGN_FREE(XtransMatrix);
}

void GeneticCorr::multOmega(double *out, const double *inputMatrix, int cols) const {
  uint64 totalSamples = NpadGeno + Npadaux;
  // split inputMatrix into two parts to utilize the matrix structure information
  double *upper = ALIGN_ALLOCATE_DOUBLES(NpadGeno * cols);
  double *lowwer = ALIGN_ALLOCATE_DOUBLES(Npadaux * cols);

  for (uint64 col = 0; col < cols; col++) {
    memcpy(upper + col * NpadGeno, inputMatrix + col * totalSamples, NpadGeno * sizeof(double));
    memcpy(lowwer + col * Npadaux, inputMatrix + NpadGeno + col * totalSamples, Npadaux * sizeof(double));
  }

  // compute omega1 matrix
  double *omega1 = ALIGN_ALLOCATE_DOUBLES(NpadGeno * cols);
  multXXTConjugate(omega1, upper, cols, 'G');

  // compute omega2 matrix
  double *omega2 = ALIGN_ALLOCATE_DOUBLES(Npadaux * cols);
  multXXTConjugate(omega2, lowwer, cols, 'A');

  // compute delta X2X1T matrix
  double *deltaX2X1 = ALIGN_ALLOCATE_DOUBLES(Npadaux * cols);
  multdeltaXXT(deltaX2X1, upper, cols, 'G');

  // compute delta X1X2T matrix
  double *deltaX1X2 = ALIGN_ALLOCATE_DOUBLES(NpadGeno * cols);
  multdeltaXXT(deltaX1X2, lowwer, cols, 'A');

  // merge four parts into one whole matrix
  // sum the result first
  NumericUtils::sumElem(omega1, deltaX1X2, NpadGeno * cols);
  NumericUtils::sumElem(omega2, deltaX2X1, Npadaux * cols);

  // merge the two parts into one whole matrix
  for (uint64 col = 0; col < cols; col++) {
    memcpy(out + col * totalSamples, omega1 + col * NpadGeno, NpadGeno * sizeof(double));
    memcpy(out + NpadGeno + col * totalSamples, omega2 + col * Npadaux, Npadaux * sizeof(double));
  }

  ALIGN_FREE(upper);
  ALIGN_FREE(lowwer);
  ALIGN_FREE(omega1);
  ALIGN_FREE(omega2);
  ALIGN_FREE(deltaX1X2);
  ALIGN_FREE(deltaX2X1);
}

void GeneticCorr::solveConjugateBatch(double *out, const double *inputMatrix1, const double *inputMatrix2, int cols) {
  uint64 totalSamples = Npadaux + NpadGeno;

  // allocate memory for the temporary variables
  double *p = ALIGN_ALLOCATE_DOUBLES(totalSamples * cols); // store the batch result
  double *r = ALIGN_ALLOCATE_DOUBLES(totalSamples * cols);

  double *VmultCovCompVecs = ALIGN_ALLOCATE_DOUBLES(totalSamples * cols);

  memset(VmultCovCompVecs, 0, totalSamples * cols * sizeof(double));
  memset(p, 0, totalSamples * cols * sizeof(double));
  memset(r, 0, totalSamples * cols * sizeof(double));

  // assume x = 0, so p = r, create the diagnoal block matrix
  for (uint64 col = 0; col < (1 + covarBasis.getC()); col++) {
    memcpy(p + col * totalSamples, inputMatrix1 + col * NpadGeno, NpadGeno * sizeof(double));
  }

  double *p_middle = p + totalSamples * (1 + covarBasis.getC()) + NpadGeno;
  for (uint64 col = 0; col < auxcovarBasis.getC(); col++) {
    memcpy(p_middle + col * totalSamples, inputMatrix2 + (col + 1) * Npadaux, Npadaux * sizeof(double));
  }

  memcpy(p + NpadGeno, inputMatrix2, Npadaux * sizeof(double)); // two datasets phenotype in one column

  memcpy(r, p, totalSamples * cols * sizeof(double));

  vector<double> rsold(cols), rsnew(cols);
  for (int col = 0; col < cols; col++) {
    double *temp_p = p + col * totalSamples;
    rsold[col] = NumericUtils::dot(temp_p, temp_p, totalSamples);
  }

  vector<double> rsoldOrigin = rsold;

  Timer timer;

  int maxIteration = 100;
  cout << "start iteration" << endl;
  for (int iter = 0; iter < maxIteration; iter++) {

    timer.update_time();
    multOmega(VmultCovCompVecs, p, cols);

    for (uint64 numbatch = 0, m = 0; numbatch < cols; numbatch++) {
      double *p_temp = p + numbatch * totalSamples;
      double *Vp_temp = VmultCovCompVecs + numbatch * totalSamples;

      double alpha = rsold[numbatch] / NumericUtils::dot(p_temp, Vp_temp, totalSamples);
      for (uint64 n = 0; n < totalSamples; n++, m++) {
        out[m] += alpha * p[m];
        r[m] -= alpha * VmultCovCompVecs[m];
      }
    }

    // compute rsnew for each batch
    for (uint64 numbatch = 0; numbatch < cols; numbatch++) {
      double *r_temp = r + numbatch * totalSamples;
      rsnew[numbatch] = NumericUtils::norm2(r_temp, totalSamples);
    }

    // check convergence condition
    bool converged = true;
    for (int numbatch = 0; numbatch < cols; numbatch++) {
      if (sqrt(rsnew[numbatch] / rsoldOrigin[numbatch]) > 5e-4) {
        converged = false;
      }
    }

    // output intermediate result
    double maxRatio = 0;
    double minRatio = 1e9;
    for (int numbatch = 0; numbatch < cols; numbatch++) {
      double currRatio = sqrt(rsnew[numbatch] / rsoldOrigin[numbatch]);
      maxRatio = std::max(maxRatio, currRatio);
      minRatio = std::min(minRatio, currRatio);
    }

    printf(" Iter: %d, time = %.2f, maxRatio = %.4f, minRatio = %.4f, convergeRatio = %.4f \n",
           iter + 1, timer.update_time(), maxRatio, minRatio, 5e-4);

    if (converged) {
      cout << "Conjugate gradient reaches convergence at " << iter + 1 << " iteration" << endl;
      ALIGN_FREE(p);
      ALIGN_FREE(r);
      ALIGN_FREE(VmultCovCompVecs);
      break;
    }

    for (uint64 numbatch = 0, m = 0; numbatch < cols; numbatch++) {
      double r2ratio = rsnew[numbatch] / rsold[numbatch];
      for (uint64 n = 0; n < totalSamples; n++, m++)
        p[m] = r[m] + r2ratio * p[m];
    }

    rsold = rsnew;
  }

}

void GeneticCorr::solveInverse(double *out, double *inputMatrix, double *vec, uint64 cols) const {
  MKL_INT n = cols;
  MKL_INT lda = n;
  MKL_INT ldb = n;
  lapack_int nrhs = 1;
  lapack_int ldx = n;
  lapack_int ipiv[n];
  lapack_int iter = -1;
  lapack_int info1 = LAPACKE_dsgesv(LAPACK_COL_MAJOR, n, nrhs, inputMatrix, lda, ipiv, vec, ldb, out, ldx, &iter);

  if (info1 != 0) {
    cout << "Error: failed to solve linear equations" << endl;
    exit(1);
  }
}

void GeneticCorr::estPosteriorMean() {
  uint64 totalSamples = NpadGeno + Npadaux;
  uint64 totalCom = covarBasis.getC() + auxcovarBasis.getC();
  // compute yhat = omega-1ZW
  double *yhat = ALIGN_ALLOCATE_DOUBLES(totalSamples);
  MKL_INT m = totalSamples;
  MKL_INT n = totalCom;
  double alpha = 1.0;
  MKL_INT lda = m;
  double beta = 0.0;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  double *inputMatrix = conjugateResultFixEff + totalSamples; // the first column is the result of oinvy
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, inputMatrix, lda, fixEffect.data(), incx, beta, yhat, incy);

  // compute omega-1y - yhat
  double *omegaInvy = conjugateResultFixEff;
  double *phenoData = ALIGN_ALLOCATE_DOUBLES(totalSamples);

  for (uint64 n = 0; n < totalSamples; n++) {
    phenoData[n] = omegaInvy[n] - yhat[n];
  }

  // scale the result of conjugate gradient (refer to the document)
  NumericUtils::scaleElem(phenoData, sigma2g, NpadGeno);
  NumericUtils::scaleElem(phenoData + NpadGeno, delta, Npadaux);

  // compute mu = X^TA, where A is a vector
  double *temp_result = ALIGN_ALLOCATE_DOUBLES(M);
  posteriorMean.reserve(M);
  multXTmatrix(posteriorMean.data(), phenoData, 1, 'G'); // X1Tyhat
  multXTmatrix(temp_result, phenoData + NpadGeno, 1, 'A'); // X2Tyhat

  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] += temp_result[m];
    posteriorMean[m] /= sqrt(M);
  }

  double subIntercept = 0;
  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] /= sqrt(M) / genoMeanstd[m].second;
    subIntercept += posteriorMean[m] * genoMeanstd[m].first;
  }

  fixEffect[0] -= subIntercept;

  FileUtils::SafeOfstream fout;
  std::string outputMean = outputFile + "_posteriorMean.txt";
  fout.open(outputMean);
  for (uint64 m = 0; m < M; m++) {
    fout << posteriorMean[m] << "\n";
  }
  fout.close();

  std::string outputFix = outputFile + "_fixeff.txt";
  fout.open(outputFix);
  for (int i = 0; i < totalCom; i++) {
    fout << fixEffect[i] << endl;
  }
  fout.close();

  ALIGN_FREE(yhat);
  ALIGN_FREE(phenoData);
  ALIGN_FREE(temp_result);
}

void GeneticCorr::normalizeSingleSnp(uchar *genoLine, double *normalizedSnp, uint64 numSamples, uint64 numUsed) {
  // compute mean and var for specific SNP
  uchar genoBase = (uchar) 0;
  double sumGenoNonMissing = 0;
  int numGenoNonMissing = 0;
  for (uint64 n = 0; n < numSamples; n++) {
    if (maskIndivs[n] // important! don't use masked-out values
        && (genoLine[n] - genoBase) != 9) {
      sumGenoNonMissing += (genoLine[n] - genoBase);
      numGenoNonMissing++;
    }
  }

  double mean = sumGenoNonMissing / static_cast<double>(numSamples - 1);
  for (uint64 n = 0; n < numSamples; n++) {
    if (maskIndivs[n]) {
      if ((genoLine[n] - genoBase) == 9)
        normalizedSnp[n] = 0;
      else
        normalizedSnp[n] = static_cast<double>(genoLine[n] - genoBase) - mean; // here we transform the char to double
    } else
      assert((genoLine[n] - genoBase) == 0);
  }

  // compute the variance and normalize snp
  double meanCenterNorm2 = NumericUtils::norm2(normalizedSnp, numSamples);
  double invMeanCenterNorm = sqrt(static_cast<double>(numUsed - 1) / meanCenterNorm2);

  for (uint64 n = 0; n < numSamples; n++)
    normalizedSnp[n] *= invMeanCenterNorm;
}

void GeneticCorr::computeSinglePosteriorMean(const vector<string> &bimFiles, const vector<string> &bedFiles,
                                            const double* phenoData, char whichDataset) {
  // set variables according to different datasets
  uint64 numSamples, numUsed, numPad;
  const double* conjugateResult = nullptr;
  if (whichDataset == 'G') {
    numPad = genoData.getNpad();
    numUsed = genoData.getNused();
    numSamples = genoData.getN();
    conjugateResult = phenoData;
  } else {
    numPad = auxgenoData.getNpad();
    numUsed = auxgenoData.getNused();
    numSamples = auxgenoData.getN();
    conjugateResult = phenoData + NpadGeno; // make shift according to different dataset
  }

  // compute mu = X^TA, where A is a vector
  // we need to read data from file and compute the result line by line
  FileUtils::SafeIfstream finBim, finBed;
  uint64 mbed = 0;
  uchar *genoLine = ALIGN_ALLOCATE_UCHARS(numSamples);
  uchar *bedLineIn = ALIGN_ALLOCATE_UCHARS(numPad>>2);
  double* normalizedSnp = ALIGN_ALLOCATE_DOUBLES(numPad);

  // read the main dataset from file and compute the first part result
  for (uint i = 0; i < bedFiles.size(); i++) {
    finBim.open(bimFiles[i]);
    finBed.open(bedFiles[i], std::ios::in | std::ios::binary);
    uchar header[3];
    finBed.read((char *) header, 3);
    if (!finBed || header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01) {
      cerr << "ERROR: Incorrect first three bytes of bed file: " << bedFiles[i] << endl;
      exit(1);
    }

    string line;
    while (getline(finBim, line)) {
      // read bed genotype and normalize
      genoData.readBedLine(genoLine, bedLineIn, finBed);
      normalizeSingleSnp(genoLine, normalizedSnp, numPad, numUsed);
      // store the dot result
      posteriorMean[mbed] += NumericUtils::dot(normalizedSnp, conjugateResult, numPad);
      mbed++;
    }
  }

  ALIGN_FREE(genoLine);
  ALIGN_FREE(bedLineIn);
  ALIGN_FREE(normalizedSnp);
}

void GeneticCorr::estPosteriorMean(const vector <string> &bimFilesG, const vector <string> &bedFilesG,
                                   const vector <string> &bimFilesA, const vector <string> &bedFilesA) {
  uint64 totalSamples = NpadGeno + Npadaux;
  uint64 totalCom = covarBasis.getC() + auxcovarBasis.getC();
  // compute yhat = omega-1ZW
  double *yhat = ALIGN_ALLOCATE_DOUBLES(totalSamples);
  MKL_INT m = totalSamples;
  MKL_INT n = totalCom;
  double alpha = 1.0;
  MKL_INT lda = m;
  double beta = 0.0;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  double *inputMatrix = conjugateResultFixEff + totalSamples; // the first column is the result of oinvy
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, inputMatrix, lda, fixEffect.data(), incx, beta, yhat, incy);

  // compute omega-1y - yhat
  double *omegaInvy = conjugateResultFixEff;
  double *phenoData = ALIGN_ALLOCATE_DOUBLES(totalSamples);

  for (uint64 n = 0; n < totalSamples; n++) {
    phenoData[n] = omegaInvy[n] - yhat[n];
  }

  // scale the result of conjugate gradient (refer to the document)
  NumericUtils::scaleElem(phenoData, sigma2g, NpadGeno);
  NumericUtils::scaleElem(phenoData + NpadGeno, delta, Npadaux);

  // resize posteriorMean and store the result
  posteriorMean.resize(M);

  // compute the main geno dataset result
  computeSinglePosteriorMean(bimFilesG, bedFilesG, phenoData, 'G');

  // compute the aux geno dataset result
  computeSinglePosteriorMean(bimFilesA, bedFilesA, phenoData, 'A');

  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] /= sqrt(M);
  }

  double subIntercept = 0;
  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] /= sqrt(M) / genoMeanstd[m].second;
    subIntercept += posteriorMean[m] * genoMeanstd[m].first;
  }

  fixEffect[0] -= subIntercept;

  FileUtils::SafeOfstream fout;
  std::string outputMean = outputFile + "_posteriorMean.txt";
  fout.open(outputMean);
  for (uint64 m = 0; m < M; m++) {
    fout << posteriorMean[m] << "\n";
  }
  fout.close();

  std::string outputFix = outputFile + "_fixeff.txt";
  fout.open(outputFix);
  for (int i = 0; i < totalCom; i++) {
    fout << fixEffect[i] << endl;
  }
  fout.close();

  ALIGN_FREE(yhat);
  ALIGN_FREE(phenoData);
}

void GeneticCorr::predict(double *output, const GenoData &predictData, const CovarBasis<GenoData> &predictCov) const {
  uint64 numPredict = predictData.getNpad(); // get number of prediction samples

  // build SNPs lookup table for prediction data (impute missing data by 0)
  vector<double> predictMaskIndivs(numPredict);
  predictData.writeMaskIndivs(&predictMaskIndivs[0]);

  double (*predictionSnpLookupTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(M * sizeof(*predictionSnpLookupTable));
  memset(predictionSnpLookupTable, 0, M * sizeof(predictionSnpLookupTable[0]));

  uchar *predictProjMaskSnps = ALIGN_ALLOCATE_UCHARS(M);
  predictData.writeMaskSnps(predictProjMaskSnps); // now we do not exclude snps, so no snp is masked

  double *snpVector = ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * numPredict);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(omp_get_max_threads() * 256 * sizeof(*workTable));
  double map0129[4] = {0, 1, 2, 9};

#pragma omp parallel for
  for (uint64 m = 0; m < M; m++) {
    if (predictProjMaskSnps[m]) {
      predictData.decodeSnpsVector(snpVector + (omp_get_thread_num() * numPredict), predictMaskIndivs.data(),
                                   m, map0129, workTable + (omp_get_thread_num() << 8));
      buildPredictSnpsLookupTable(m,
                                  numPredict,
                                  snpVector + (omp_get_thread_num() * numPredict),
                                  predictionSnpLookupTable);
    }
  }

  // compute random effect based on new lookup table
  double *predictedRandomEff = ALIGN_ALLOCATE_DOUBLES(numPredict);
  predictRandomEff(numPredict, predictMaskIndivs.data(), predictedRandomEff, predictionSnpLookupTable, predictData);

  // compute fix effect based on new pc data
  double *predictedFixEff = ALIGN_ALLOCATE_DOUBLES(numPredict);

  // if the prediction covariate file is empty
  if (predictCov.getC() == 1) {
    memcpy(output, predictedRandomEff, numPredict * sizeof(double));
  } else {
    const double *predictCovMatrix = predictCov.getCovarMatrix(); // get predictCovMatrix
    predictFixEff(numPredict, predictedFixEff, predictCovMatrix);

    for (uint64 n = 0; n < numPredict; n++) {
      output[n] = predictedFixEff[n] + predictedRandomEff[n];
    }
  }

  FileUtils::SafeOfstream fout;
  std::string predictHeight = outputFile + "_predict.txt";
  fout.open(predictHeight);
  fout << "Prediction (rand effect)" << "\t" << "Prediction (rand + fix effect)" << "\n";
  for (uint64 n = 0; n < numPredict; n++) {
    fout << predictedRandomEff[n] << "\t" << output[n] << endl;
  }
  fout.close();

  ALIGN_FREE(predictedRandomEff);
  ALIGN_FREE(predictProjMaskSnps);
  ALIGN_FREE(snpVector);
  ALIGN_FREE(workTable);
  ALIGN_FREE(predictionSnpLookupTable);
}

void GeneticCorr::buildPredictSnpsLookupTable(uint64 m,
                                              uint64 numPredict,
                                              const double *snpVector,
                                              double (*predictionSnpLookupTable)[4]) const {
//      // get the mode of each snp
//      vector <uint64> snpsCounts;
//      snpsCounts.reserve(3);
//      for (uint64 n = 0; n < numPredict; n++) {
//        if (snpVector[n] != 9)
//          snpsCounts[(int)snpVector[n]]++;
//      }
//      // find the mode index in this case, which is one of 0, 1, 2
//      int mode = std::distance(snpsCounts.begin(), std::max_element(snpsCounts.begin(), snpsCounts.end()));

  // get the mean of each snp
  uint64 sum = 0;
  uint64 numIndivs = 0;
  for (uint64 n = 0; n < numPredict; n++) {
    if (snpVector[n] != 9) {
      sum += snpVector[n];
      numIndivs++;
    }
  }

  double mean = static_cast<double>(sum) / numIndivs;

  predictionSnpLookupTable[m][0] = 0;
  predictionSnpLookupTable[m][1] = 1;
  predictionSnpLookupTable[m][2] = 2;

  if (imputeMethod == "zero") {
    predictionSnpLookupTable[m][3] = mean;
  } else if (imputeMethod == "mean") {
    predictionSnpLookupTable[m][3] = 0;
  } else {
    cout << "Warning: There is no such imputing method. Use the mean value to impute." << endl;
    predictionSnpLookupTable[m][3] = mean;
  }
}

void GeneticCorr::predictRandomEff(uint64 numPredict, double *predictMaskIndivs, double *randomEff,
                                   double (*predictionSnpLookupTable)[4], const GenoData &predictData) const {
  // we want to compute Xmu, however we have to decode the X from column major
  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(numPredict * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * 256 * sizeof(*workTable));

  // store the temp result
  memset(randomEff, 0, numPredict * sizeof(double));

  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m]) {
        predictData.decodeSnpsVector(snpBlock + mPlus * numPredict, predictMaskIndivs, m, predictionSnpLookupTable[m],
                                     workTable + (omp_get_thread_num() << 8));
        NumericUtils::scaleElem(snpBlock + mPlus * numPredict, posteriorMean[m], numPredict);
      } else
        memset(snpBlock + mPlus * numPredict, 0, numPredict * sizeof(snpBlock[0]));
    }

    // Todo: optimize the memory read pattarn (also we should avoid race condition)
#pragma omp parallel for
    for (uint64 n = 0; n < numPredict; n++) {
      for (uint64 col = 0; col < snpsPerBLockCrop; col++) {
        randomEff[n] += snpBlock[col * numPredict + n];
      }
    }

  }
  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
}

void GeneticCorr::predictFixEff(uint64 numPredict, double *fixEff, const double *predictCovarMatrix) const {
  const MKL_INT m = numPredict;
  const MKL_INT n = covarBasis.getC(); // predict data should have the same number of components
  const double alpha = 1.0;
  const MKL_INT lda = m;
  const double beta = 0.0;
  const MKL_INT incx = 1;
  const MKL_INT incy = 1;
  cblas_dgemv(CblasColMajor,
              CblasNoTrans,
              m,
              n,
              alpha,
              predictCovarMatrix,
              lda,
              fixEffect.data(),
              incx,
              beta,
              fixEff,
              incy);
}

void GeneticCorr::compuInverse(double *matrix, const unsigned int row) const {
  lapack_int m1 = row;
  lapack_int n1 = row;
  lapack_int lda1 = row;
  lapack_int ipiv[row];
  lapack_int info1 = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m1, n1, matrix, lda1, ipiv);

  if (info1 != 0) {
    cerr << "Failed to compute LU decomposition " << endl;
    exit(1);
  }

  info1 = LAPACKE_dgetri(LAPACK_COL_MAJOR, m1, matrix, lda1, ipiv);

  if (info1 != 0) {
    cerr << "Failed to compute inverse " << endl;
    exit(1);
  }
}
}