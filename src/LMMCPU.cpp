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

#include <stdlib.h>
#include <cstring>
#include <assert.h>
#include <math.h>
#include <cmath>
#include<bits/stdc++.h>

#include <mkl.h>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>

#include <Eigen/Dense>

#include "omp.h"

#include "MemoryUtils.h"
#include "NumericUtils.h"
#include "IOUtils.h"
#include "InfoStructure.h"
#include "Timer.h"
#include "LMMCPU.h"

namespace LMMNET {

using std::vector;
using std::string;
using std::pair;
using std::cout;
using std::endl;
using std::cerr;

LMMCPU::LMMCPU(const LMMNET::GenoData &_genoData,
               const LMMNET::CovarBasis<GenoData> &_covarBasis,
               const double *_maskIndivs,
               int _snpsPerBlock,
               uint64 _estIteration,
               int _numChrom,
               int _numCalibSnps,
               uint64 _maxIterationConj,
               bool _useExactTrace,
               const std::string _imputeMethod,
               const std::string _outputFile) : genoData(_genoData),
                                          covarBasis(_covarBasis),
                                          maskIndivs(_maskIndivs),
                                          snpsPerBlock(_snpsPerBlock),
                                          estIteration(_estIteration),
                                          numChrom(_numChrom),
                                          numCalibSnps(_numCalibSnps),
                                          useExactTrace(_useExactTrace),
                                          maxIterationConj(_maxIterationConj),
                                          imputeMethod(_imputeMethod),
                                          outputFile(_outputFile) {
  initialize();
}

void LMMCPU::initialize() {

  M = genoData.getM();
  Npad = genoData.getNpad();

  // get the value of Nused based on the new maskIndivs
  Nused = 0;
  for (uint64 n = 0; n < Npad; n++) {
    Nused += maskIndivs[n];
  }

  L2normXcols = (double *) ALIGN_ALLOCATE_DOUBLES(M);
  snpLookupTable = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(M * sizeof(*snpLookupTable));

  memset(L2normXcols, 0, M * sizeof(L2normXcols[0]));
  memset(snpLookupTable, 0, M * sizeof(snpLookupTable[0]));

  projMaskSnps = ALIGN_ALLOCATE_UCHARS(M);
  genoData.writeMaskSnps(projMaskSnps); // now we do not exclude snps, so no snp is masked

  cout << "Normalizing snps ......" << endl;
  // normalize the snps parallel
  double *snpVector = ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * Npad);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(omp_get_max_threads() * 256 * sizeof(*workTable));
  double map0129[4] = {0, 1, 2, 9};

  Meanstd.resize(M);

#pragma omp parallel for
  for (uint64 m = 0; m < M; m++) {
    if (projMaskSnps[m]) {
      genoData.decodeSnpsVector(snpVector + (omp_get_thread_num() * Npad),
                                maskIndivs,
                                m,
                                map0129,
                                workTable + (omp_get_thread_num() << 8));
      projMaskSnps[m] = normalizeSnps(m, snpVector + (omp_get_thread_num() * Npad));
    }
  }

  cout << "Impute missing value by " << imputeMethod << endl;
  chromID.resize(M);
  // get the number of SNPs per chromsome
  const vector<SnpInfo> &snpsInfo = genoData.getSnpInfo();
  for (uint64 m = 0; m < M; m++) {
    numSnpsPerChrom[snpsInfo[m].chrom]++;
    chromID[m] = snpsInfo[m].chrom;
  }

  // get the chromsome start point
  uint64 startPoint = 0;
  for (int i = 1; i < numChrom + 1; i++) { // be carefully, this for loop should start from 1
    chromStartPoint.push_back(startPoint);
    startPoint += numSnpsPerChrom[i];
  }

  ALIGN_FREE(snpVector);
  ALIGN_FREE(workTable);
}

LMMCPU::~LMMCPU() {
  ALIGN_FREE(projMaskSnps);
  ALIGN_FREE(snpLookupTable);
  ALIGN_FREE(L2normXcols);
}

uchar LMMCPU::normalizeSnps(uint64 m, double *snpVector) {
  double sumGenoNonMissing = 0;
  int numGenoNonMissing = 0;
  for (uint64 n = 0; n < Npad; n++) {
    if (maskIndivs[n] // important! don't use masked-out values
        && snpVector[n] != 9) {
      sumGenoNonMissing += snpVector[n];
      numGenoNonMissing++;
    }
  }

  if (numGenoNonMissing == 0) return 0;

  // mean-center and replace missing values with mean (centered to 0)
  double mean = sumGenoNonMissing / numGenoNonMissing;

  for (uint64 n = 0; n < Npad; n++) {
    if (maskIndivs[n]) {
      if (snpVector[n] == 9)
        snpVector[n] = 0;
      else
        snpVector[n] -= mean;
    } else
      assert(snpVector[n] == 0); // buildMaskedSnpVector should've already zeroed
  }

  double meanCenterNorm2 = NumericUtils::norm2(snpVector, Npad);
  // normalize to Nused-1 (dimensionality of subspace with all-1s vector projected out)
  double invMeanCenterNorm = sqrt(static_cast<double>(Nused - 1) / meanCenterNorm2);

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

  Meanstd[m] = std::make_pair(mean, invMeanCenterNorm);

  return 1;
}

void LMMCPU::invnormalizeSnps() {

#pragma omp parallel for
  for (uint64 m = 0; m < M; m++) {
    double mean = Meanstd[m].first;

    snpLookupTable[m][0] = -mean;
    snpLookupTable[m][1] = 1 - mean;
    snpLookupTable[m][2] = 2 - mean;

    if (imputeMethod == "zero") {
      snpLookupTable[m][3] = -mean;
    } else if (imputeMethod == "mean") {
      snpLookupTable[m][3] = 0;
    } else {
      cout << "Warning: There is no such imputing method. Use the mean value to impute." << endl;
      snpLookupTable[m][3] = 0;
    }
  }
}

double LMMCPU::calyVKVy(const double *projectPheno) const {
  double *temp = ALIGN_ALLOCATE_DOUBLES(Npad);

  // compute A=XX^TVy
  multXXT(temp, projectPheno);

  // compute yVA
  MKL_INT n = Npad;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  double yvkvy = cblas_ddot(n, projectPheno, incx, temp, incy) / M;

#ifdef DEBUG
  cout << "yvkvy result is " << yvkvy << endl;
#endif
  ALIGN_FREE(temp);
  return yvkvy;
}

void LMMCPU::multXXT(double *out, const double *vec) const {

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
        buildMaskedSnpCovCompVec(snpBlock + mPlus * Npad, m,
                                 workTable + (omp_get_thread_num() << 8));
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

void LMMCPU::computeXXT(double *out) const {
  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(Npad * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * 256 * sizeof(*workTable));

  memset(out, 0, Npad * Npad * sizeof(double));
  // decode snps from lookup table and compute XXT
  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m])
        buildMaskedSnpCovCompVec(snpBlock + mPlus * Npad, m,
                                 workTable + (omp_get_thread_num() << 8));
      else
        memset(snpBlock + mPlus * Npad, 0, Npad * sizeof(snpBlock[0]));
    }

    // compute block XX^T
    MKL_INT m = Npad;
    MKL_INT n = Npad;
    MKL_INT k = snpsPerBLockCrop;
    double alpha = 1.0;
    MKL_INT lda = m;
    MKL_INT ldb = n;
    MKL_INT ldc = m;
    double beta = 1.0;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, snpBlock, lda, snpBlock, ldb, beta, out, ldc);

  }

  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
}

void LMMCPU::multXXTTrace(double *out, const double *vec) const {

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
        buildMaskedSnpCovCompVec(snpBlock + mPlus * Npad, m,
                                 workTable + (omp_get_thread_num() << 8));
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

void LMMCPU::calTraceMoM(double &kv, double &kvkv) const {
  // sample gaussian vector
  Timer timer;
  boost::mt19937 rng; // I don't set it on purpouse
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937 &, boost::normal_distribution<> > var_nor(rng, nd);

  uint64 numGaussian = estIteration * Npad;
  auto *gaussianVec = ALIGN_ALLOCATE_DOUBLES(numGaussian);
  auto *result = ALIGN_ALLOCATE_DOUBLES(estIteration * Npad);
  memset(result, 0, estIteration * Npad * sizeof(double));

  for (uint n = 0; n < numGaussian; n++) {
    gaussianVec[n] = var_nor();
  }

  cout << "Time for setting up compute trace is " << timer.update_time() << " sec" << endl;
  // compute A=XX^TZ for all blocks
  multXXTTrace(result, gaussianVec);

  // compute VA for all blocks
  for (uint iter = 0; iter < estIteration; iter++) {
    covarBasis.projectCovarsVec(result + iter * Npad);
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

void LMMCPU::calTraceExact(double &kv, double &kvkv) const {
  double *XXT = ALIGN_ALLOCATE_DOUBLES(Npad * Npad);

  computeXXT(XXT);

  for (uint64 n = 0; n < Npad; n++) {
    covarBasis.projectCovarsVec(XXT + n * Npad);
  }

  // compute tracek
  double tracek = 0;
  for (uint64 n = 0; n < Npad; n++) {
    tracek += XXT[n + n * Npad];
  }

  kv = tracek / M;

  // compute tracek2
  kvkv = dotVec(XXT, XXT, Npad * Npad) / static_cast<double>(M * M);

#ifdef DEBUG
  cout << "Calculate exact trace kvkv " << kvkv << endl;
  cout << "Calculate exact trace kv " << kv << endl;
#endif

}

double LMMCPU::calStandError(const double *projectPheno, double kv, double kvkv) const {
  // compute inverse S matrix column major
  double S[4] = {kvkv, kv, kv, static_cast<double>(Nused - covarBasis.getC())};

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

  multXXT(omegaY, projectPheno);
  for (uint64 n = 0; n < Npad; n++) {
    omegaY[n] = omegaY[n] * sigma2g / M + sigma2e * projectPheno[n];
  }

  double yTomegay = 2 * NumericUtils::dot(projectPheno, omegaY, Npad);

  // compute cov diagnoal
  auto* KVomegay = ALIGN_ALLOCATE_DOUBLES(Npad);

  covarBasis.projectCovarsVec(omegaY);
  multXXT(KVomegay, omegaY);
  NumericUtils::scaleElem(KVomegay, static_cast<double>(1) / M, Npad);

  double dig = 2 * NumericUtils::dot(projectPheno, KVomegay, Npad);

  // compute first elem
  auto* VKy = ALIGN_ALLOCATE_DOUBLES(Npad);
  auto* omegaVKy = ALIGN_ALLOCATE_DOUBLES(Npad);

  multXXT(VKy, projectPheno);
  NumericUtils::scaleElem(VKy, static_cast<double>(1) / M, Npad);
  covarBasis.projectCovarsVec(VKy);
  multXXT(omegaVKy, VKy);
  for (uint64 n = 0; n < Npad; n++) {
    omegaVKy[n] = omegaVKy[n] * sigma2g / M + sigma2e * VKy[n];
  }

  double elem = 2 * NumericUtils::dot(VKy, omegaVKy, Npad);
  elem += sigma2g * sigma2g * kv / estIteration;

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
  double denominator = sigma2g * kv + sigma2e * static_cast<double>(Nused - covarBasis.getC());
  double coeff = kv * static_cast<double>(Nused - covarBasis.getC()) / (denominator * denominator);
  Eigen::Vector2d deltah;
  deltah(0, 0) = coeff * sigma2g; deltah(1, 0) = coeff * (-sigma2e);

  double stdHreg = sqrt(deltah.transpose() * covTheta * deltah);

  return stdHreg;
}
//double LMMCPU::calStandError(const double *projectPheno, const double kvkv) const {
//  // compute the first part stand error
//  // compute (K-I)Y
//  double *tempVec = ALIGN_ALLOCATE_DOUBLES(Npad);
//  multXXT(tempVec, projectPheno);
//  NumericUtils::scaleElem(tempVec, 1 / static_cast<double>(M), Npad);
//  NumericUtils::subElem(tempVec, projectPheno, Npad);
//
//  // compute V(K-I)Y
//  double *tempVec2 = ALIGN_ALLOCATE_DOUBLES(Npad);
//  multXXT(tempVec2, tempVec);
//  NumericUtils::scaleElem(tempVec2, sigma2g / static_cast<double>(M), Npad);
//  NumericUtils::scaleElem(tempVec, sigma2e, Npad);
//  NumericUtils::sumElem(tempVec2, tempVec, Npad);
//
//  // compute (K-I)V(K-I)Y
//  double *tempVec3 = ALIGN_ALLOCATE_DOUBLES(Npad);
//  multXXT(tempVec3, tempVec2);
//  NumericUtils::scaleElem(tempVec3, 1 / static_cast<double>(M), Npad);
//  NumericUtils::subElem(tempVec3, tempVec2, Npad);
//  double error1 = 2 * NumericUtils::dot(tempVec3, projectPheno, Npad);
//
//  ALIGN_FREE(tempVec);
//  ALIGN_FREE(tempVec2);
//  ALIGN_FREE(tempVec3);
//
//  // compute the second part stand error
//  double error2 = (1 / static_cast<double>(estIteration)) * sigma2g * sigma2g * kvkv;
//
//  // final stand error
//  double standerror = (1 / (kvkv - static_cast<double>(Nused - covarBasis.getC() - 1))) * sqrt(error1 + error2);
//
//  cout << "number covariates in file " << covarBasis.getC() << endl;
//  cout << "error 1 " << error1 << " error 2 " << error2 << endl;
//  cout << "trace k^2 " << kvkv << " Nused " << Nused << endl;
//  return standerror;
//}

void LMMCPU::makeConjugateMask(uchar *mask) {
  uchar *origin_mask = ALIGN_ALLOCATE_UCHARS(M * numChrom);

  memset(origin_mask, 1, M * numChrom * sizeof(uchar)); // default use all snps

  uchar *curr_mask;
  for (uint64 chrom = 0; chrom < numChrom; chrom++) {
    curr_mask = origin_mask + chrom * M + chromStartPoint[chrom];
    memset(curr_mask, 0, numSnpsPerChrom[chrom + 1] * sizeof(uchar)); // set the mask from chrom start point as 0
  }

  memset(mask, 0, M * numChrom * sizeof(uchar));
  // resize the mask to fit the intermediate result M x B
  curr_mask = mask;
  uchar *start_mask = origin_mask;
  for (uint64 start = 0; start < M; start += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, start + snpsPerBlock) - start;
    // copy one section of S x B to B x S
    for (uint64 s = 0; s < snpsPerBLockCrop; s++) {
      // copy by row
      for (uint64 chrom = 0; chrom < numChrom; chrom++) {
        curr_mask[chrom] = start_mask[M * chrom];
      }
      curr_mask += numChrom;
      start_mask++;
    }
  }

  ALIGN_FREE(origin_mask);
}

void LMMCPU::multXTmatrix(double *out, const double *matrix, int col) const {
  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(Npad * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(omp_get_max_threads() * 256 * sizeof(*workTable));

  uint64 ucol = col; // convert type in case of overflow
  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m])
        buildMaskedSnpCovCompVec(snpBlock + mPlus * Npad, m,
                                 workTable + (omp_get_thread_num() << 8));
      else
        memset(snpBlock + mPlus * Npad, 0, Npad * sizeof(snpBlock[0]));
    }

    // compute A=X^TV
    MKL_INT m = col;
    MKL_INT n = snpsPerBLockCrop;
    MKL_INT k = Npad;
    double alpha = 1.0;
    MKL_INT lda = Npad;
    MKL_INT ldb = Npad;
    double beta = 0.0;
    MKL_INT ldc = col;
    double *temp_out = out + m0 * ucol;

    const char transA = 'T';
    const char transB = 'N';
    dgemm(&transA, &transB, &m, &n, &k, &alpha, matrix, &lda, snpBlock, &ldb, &beta, temp_out, &ldc);
  }

  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
}

void LMMCPU::multXmatrix(double *out, const double *matrix, int col) const {
  double *snpBlock = ALIGN_ALLOCATE_DOUBLES(Npad * snpsPerBlock);
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_MEMORY(omp_get_max_threads() * 256 * sizeof(*workTable));

  memset(out, 0, Npad * col * sizeof(double));

  uint64 ucol = col; // convert type in case of overflow
  for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
    uint64 snpsPerBLockCrop = std::min(M, m0 + snpsPerBlock) - m0;
#pragma omp parallel for
    for (uint64 mPlus = 0; mPlus < snpsPerBLockCrop; mPlus++) {
      uint64 m = m0 + mPlus;
      if (projMaskSnps[m])
        buildMaskedSnpCovCompVec(snpBlock + mPlus * Npad, m,
                                 workTable + (omp_get_thread_num() << 8));
      else
        memset(snpBlock + mPlus * Npad, 0, Npad * sizeof(snpBlock[0]));
    }

    MKL_INT m = Npad;
    MKL_INT n = col;
    MKL_INT k = snpsPerBLockCrop;
    double alpha = 1.0;
    MKL_INT lda = m;
    MKL_INT ldb = n;
    MKL_INT ldc = m;
    double beta = 1.0;
    const double *temp = matrix + m0 * ucol;

    const char transA = 'N';
    const char transB = 'T';
    dgemm(&transA, &transB, &m, &n, &k, &alpha, snpBlock, &lda, temp, &ldb, &beta, out, &ldc);
  }
  ALIGN_FREE(snpBlock);
  ALIGN_FREE(workTable);
}

void LMMCPU::multXXTConjugate(double *out, const double *matrix, const uchar *mask) {

  double *XtransMatrix = ALIGN_ALLOCATE_DOUBLES(M * numChrom);

  multXTmatrix(XtransMatrix, matrix, numChrom); // compute X^TY

  // apply mask to the result for LOCO stragety
  for (uint64 n = 0; n < M * numChrom; n++)
    XtransMatrix[n] *= mask[n];

  multXmatrix(out, XtransMatrix, numChrom); // compute XX^TY

  for (uint64 chrom = 0, m = 0; chrom < numChrom; chrom++) {
    double invM = sigma2g / (double) (M - numSnpsPerChrom[chrom + 1]);
    for (uint64 n = 0; n < Npad; n++, m++)
      out[m] = invM * out[m] + sigma2e * matrix[m];
  }

  ALIGN_FREE(XtransMatrix);
}

void LMMCPU::calConjugateBatch(double *VinvChromy, const double *inputMatrix) {

  double *p = ALIGN_ALLOCATE_DOUBLES(Npad * numChrom); // store the batch result
  double *r = ALIGN_ALLOCATE_DOUBLES(Npad * numChrom);
  double *p_proj = ALIGN_ALLOCATE_DOUBLES(Npad * numChrom);

  double *VmultCovCompVecs = ALIGN_ALLOCATE_DOUBLES(Npad * numChrom);

  uchar *mask = ALIGN_ALLOCATE_UCHARS(M * numChrom);

  cout << "Making chromsome mask for conjugate gradient " << endl;
  makeConjugateMask(mask);

  // assume x = 0, so p = r
  memcpy(p, inputMatrix, Npad * numChrom * sizeof(double));
  memcpy(r, inputMatrix, Npad * numChrom * sizeof(double));

  // initialize the rsold to the same value for all chromsome
  double rsoldOrigin = dotVec(p, p, Npad); // we only need to compute the first Nstride inner product
  vector<double> rsold(numChrom, rsoldOrigin), rsnew(numChrom);

  Timer timer;
  Timer timer1;

  for (int iter = 0; iter < maxIterationConj; iter++) {

    // compute XX^T * P * sigma2g / M + sigma2e * I * p
    memcpy(p_proj, p, Npad * numChrom * sizeof(double));

    timer1.update_time();
    for (uint64 chrom = 0; chrom < numChrom; chrom++)
      covarBasis.projectCovarsVec(p_proj + chrom * Npad);

    multXXTConjugate(VmultCovCompVecs, p, mask);

    for (uint64 chrom = 0; chrom < numChrom; chrom++)
      covarBasis.projectCovarsVec(VmultCovCompVecs + chrom * Npad);

    // todo: multithreading to this section of code
    for (uint64 chrom = 0, m = 0; chrom < numChrom; chrom++) {
      double *p_temp = p + chrom * Npad;
      double *Vp_temp = VmultCovCompVecs + chrom * Npad;

      double alpha = rsold[chrom] / NumericUtils::dot(p_temp, Vp_temp, Npad);
      for (uint64 n = 0; n < Npad; n++, m++) {
        VinvChromy[m] += alpha * p[m];
        r[m] -= alpha * VmultCovCompVecs[m];
      }
    }

    // compute rsnew for each batch
    for (uint64 chrom = 0; chrom < numChrom; chrom++) {
      double *r_temp = r + chrom * Npad;
      rsnew[chrom] = NumericUtils::norm2(r_temp, Npad);
    }

    // check convergence condition
    bool converged = true;
    for (int chrom = 0; chrom < numChrom; chrom++) {
      if (sqrt(rsnew[chrom] / rsoldOrigin) > 5e-4) {
        converged = false;
      }
    }

    // output intermediate result
    double maxRatio = 0;
    double minRatio = 1e9;
    for (int chrom = 0; chrom < numChrom; chrom++) {
      double currRatio = sqrt(rsnew[chrom] / rsoldOrigin);
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
      ALIGN_FREE(mask);
      break;
    }

    for (uint64 chrom = 0, m = 0; chrom < numChrom; chrom++) {
      double r2ratio = rsnew[chrom] / rsold[chrom];
      for (uint64 n = 0; n < Npad; n++, m++)
        p[m] = r[m] + r2ratio * p[m];
    }

    rsold = rsnew;

  }
}

void LMMCPU::calBeta(const double *VinvChromy) {
  invnormalizeSnps();

  cout << endl << "Calculating beta " << endl;
  double *snpVec = ALIGN_ALLOCATE_DOUBLES(Npad * omp_get_max_threads());
  double (*workTable)[4] = (double (*)[4]) ALIGN_ALLOCATE_DOUBLES(omp_get_max_threads() * 256 * sizeof(workTable[0]));

  projGenoStd.reserve(M);
  // todo: test the nested multithread
#pragma omp parallel for
  for (uint64 m = 0; m < M; m++) {
    buildMaskedSnpCovCompVec(snpVec + omp_get_thread_num() * Npad,
                             m,
                             workTable + (omp_get_thread_num() << 8)); // decode m snp
    covarBasis.projectCovarsVec(snpVec + omp_get_thread_num() * Npad);
    const double *temp_VinvY = VinvChromy + (chromID[m] - 1) * Npad;
    z[m] = dotVec(snpVec + omp_get_thread_num() * Npad, temp_VinvY, Npad);
    projGenoStd[m] = NumericUtils::meanStdDev(snpVec + omp_get_thread_num() * Npad, Npad).second;
  }

  ALIGN_FREE(snpVec);
  ALIGN_FREE(workTable);

}

void LMMCPU::estimateFixEff(const double *Pheno, bool useApproximate) {
  // solve the conjugate gradient omegia^-1y (without leaving one out strategy) together
  const double *covarMatrix = covarBasis.getCovarMatrix(); // get covarMatrix

  // combine covarMatrix and projectPheno
  int batchsize = covarBasis.getC() + 1;
  double *inputMatrix = ALIGN_ALLOCATE_DOUBLES(Npad * batchsize);
  memcpy(inputMatrix, covarMatrix, covarBasis.getC() * Npad * sizeof(double)); // copy covarMatrix part
  memcpy(inputMatrix + covarBasis.getC() * Npad, Pheno, Npad * sizeof(double)); // copy projectPheno part

  // compute Z^TVinvChromy
  conjugateResultFixEff = ALIGN_ALLOCATE_DOUBLES(Npad * batchsize);
  memset(conjugateResultFixEff, 0, Npad * batchsize * sizeof(double));

  if (!useApproximate) {
    // compute the conjugate gradient to get the exact result
    cout << endl << "Solve conjugate gradient in estimating fix effect " << endl;
    calConjugateWithoutMask(conjugateResultFixEff, inputMatrix, batchsize); // get two parts conjugate gradient result
  } else {
    // use the approximate fix effect to boost the performance
//    cout << endl << "Solve conjugate gradient in estimating fix effect " << endl;
//    double* oinvy = ALIGN_ALLOCATE_DOUBLES(Npad);
//    calConjugateWithoutMask(oinvy, Pheno, 1);
    memcpy(conjugateResultFixEff, inputMatrix, Npad * batchsize * sizeof(double));
//    memcpy(conjugateResultFixEff + covarBasis.getC() * Npad, oinvy, Npad * sizeof(double));
//    ALIGN_FREE(oinvy);
  }

  // compute Z^Tomegia^-1Z
  double *ZTOinvZ = ALIGN_ALLOCATE_DOUBLES(covarBasis.getC() * covarBasis.getC());

  // parameters for clbas_dgemm
  MKL_INT m = covarBasis.getC();
  MKL_INT n = covarBasis.getC();
  MKL_INT k = Npad;
  const double alpha = 1.0;
  MKL_INT lda = k;
  MKL_INT ldb = k;
  const double beta = 0.0;
  const MKL_INT ldc = m;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, covarMatrix, lda, conjugateResultFixEff, ldb,
              beta, ZTOinvZ, ldc);

  // compute inverse of matrix CxC
  compInverse(ZTOinvZ, covarBasis.getC());

  // compute ZTOinvy
  double *ZTOinvy = ALIGN_ALLOCATE_DOUBLES(covarBasis.getC());
  memset(ZTOinvy, 0, covarBasis.getC() * sizeof(double));

  // parameters for clbas_dgemv (reuse the old parameters delcaration)
  const MKL_INT m1 = Npad;
  const MKL_INT n1 = covarBasis.getC();
  const double alpha1 = 1.0;
  const double beta1 = 0.0;
  const MKL_INT lda1 = m1;
  const MKL_INT incx = 1;
  const MKL_INT incy = 1;
  double *vec = conjugateResultFixEff + covarBasis.getC() * Npad;

  cblas_dgemv(CblasColMajor, CblasTrans, m1, n1, alpha1, covarMatrix, lda1, vec, incx, beta1, ZTOinvy, incy);

  // compute the final fixed effect
  fixEffect.reserve(covarBasis.getC());

  m = covarBasis.getC();
  n = covarBasis.getC();
  lda = m;

  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, ZTOinvZ, lda, ZTOinvy, incx, beta, fixEffect.data(), incy);

  ALIGN_FREE(inputMatrix);
  ALIGN_FREE(ZTOinvy);
  ALIGN_FREE(ZTOinvZ);

}

void LMMCPU::multXXTConjugateWithoutMask(double *out, const double *matrix, unsigned int batchsize) {
  double *XtransMatrix = ALIGN_ALLOCATE_DOUBLES(M * batchsize);

  multXTmatrix(XtransMatrix, matrix, batchsize); // compute X^TY

  multXmatrix(out, XtransMatrix, batchsize); // compute XX^TY

  double invM = sigma2g / static_cast<double>(M);
  for (uint64 numbatch = 0, m = 0; numbatch < batchsize; numbatch++) {
    for (uint64 n = 0; n < Npad; n++, m++)
      out[m] = out[m] * invM + sigma2e * matrix[m];
  }

  ALIGN_FREE(XtransMatrix);
}

void LMMCPU::calConjugateWithoutMask(double *Viny, const double *inputMatrix, int batchsize) {
  double *p = ALIGN_ALLOCATE_DOUBLES(Npad * batchsize); // store the batch result
  double *r = ALIGN_ALLOCATE_DOUBLES(Npad * batchsize);

  double *VmultCovCompVecs = ALIGN_ALLOCATE_DOUBLES(Npad * batchsize);

  // assume x = 0, so p = r
  memcpy(p, inputMatrix, Npad * batchsize * sizeof(inputMatrix[0]));
  memcpy(r, inputMatrix, Npad * batchsize * sizeof(inputMatrix[0]));

  // initialize the rsold to the same value for all chromsome
  vector<double> rsold(batchsize), rsnew(batchsize);
  for (int numbatch = 0; numbatch < batchsize; numbatch++) {
    double *temp_p = p + numbatch * Npad;
    rsold[numbatch] = dotVec(temp_p, temp_p, Npad);
  }
  vector<double> rsoldOrigin = rsold;

  Timer timer;
  Timer timer1;

  for (int iter = 0; iter < maxIterationConj; iter++) {

    // compute XX^T * P * sigma2g / M + sigma2e * I * p
    timer1.update_time();

    multXXTConjugateWithoutMask(VmultCovCompVecs, p, batchsize);

    // todo: multithreading to this section of code
    for (uint64 numbatch = 0, m = 0; numbatch < batchsize; numbatch++) {
      double *p_temp = p + numbatch * Npad;
      double *Vp_temp = VmultCovCompVecs + numbatch * Npad;

      double alpha = rsold[numbatch] / NumericUtils::dot(p_temp, Vp_temp, Npad);
//      cout << "Alpha " << alpha << endl;
      for (uint64 n = 0; n < Npad; n++, m++) {
        Viny[m] += alpha * p[m];
        r[m] -= alpha * VmultCovCompVecs[m];
      }
    }

    // compute rsnew for each batch
    for (uint64 numbatch = 0; numbatch < batchsize; numbatch++) {
      double *r_temp = r + numbatch * Npad;
      rsnew[numbatch] = NumericUtils::norm2(r_temp, Npad);
    }

    // check convergence condition
    bool converged = true;
    for (int numbatch = 0; numbatch < batchsize; numbatch++) {
      if (sqrt(rsnew[numbatch] / rsoldOrigin[numbatch]) > 5e-4) {
        converged = false;
      }
    }

    // output intermediate result
    double maxRatio = 0;
    double minRatio = 1e9;
    for (int numbatch = 0; numbatch < batchsize; numbatch++) {
      double currRatio = sqrt(rsnew[numbatch] / rsoldOrigin[numbatch]);
      maxRatio = std::max(maxRatio, currRatio);
      minRatio = std::min(minRatio, currRatio);
    }

    printf(" Iter: %d, time = %.2f, maxRatio = %.4f, minRatio = %.4f, convergeRatio = %.4f \n",
           iter + 1, timer.update_time(), maxRatio, minRatio, 5e-4);
//    cout << "Residual " << sqrt(rsnew[0]) << endl;

    if (converged) {
      cout << "Conjugate gradient reaches convergence at " << iter + 1 << " iteration" << endl;
      ALIGN_FREE(p);
      ALIGN_FREE(r);
      ALIGN_FREE(VmultCovCompVecs);
      break;
    }

    for (uint64 numbatch = 0, m = 0; numbatch < batchsize; numbatch++) {
      double r2ratio = rsnew[numbatch] / rsold[numbatch];
      for (uint64 n = 0; n < Npad; n++, m++)
        p[m] = r[m] + r2ratio * p[m];
    }

    rsold = rsnew;

  }

}

void LMMCPU::normalizeSingleSnp(uchar *genoLine, double *normalizedSnp, uint64 numSamples, uint64 numUsed) {
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

void LMMCPU::computeSinglePosteriorMean(const vector<string> &bimFiles, const vector<string> &bedFiles,
                                             const double* phenoData) {
  // set variables according to different datasets
  uint64 numSamples, numUsed, numPad;
  numPad = genoData.getNpad();
  numUsed = genoData.getNused();
  numSamples = genoData.getN();

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
      posteriorMean[mbed] += NumericUtils::dot(normalizedSnp, phenoData, numPad);
      mbed++;
    }
  }

  ALIGN_FREE(genoLine);
  ALIGN_FREE(bedLineIn);
  ALIGN_FREE(normalizedSnp);
}

void LMMCPU::computePosteriorMean(const vector <string> &bimFiles, const vector <string> &bedFiles,
                                  const double* pheno, bool useApproximate) {
  // note the phenotype here is original
  double *phenoData;
  if (!useApproximate) {
    // compute yhat = omega-1ZW
    cout << endl << "compute posterior by exact way" << endl;
    double *yhat = ALIGN_ALLOCATE_DOUBLES(Npad);
    MKL_INT m = Npad;
    MKL_INT n = covarBasis.getC();
    double alpha = 1.0;
    MKL_INT lda = m;
    double beta = 0.0;
    MKL_INT incx = 1;
    MKL_INT incy = 1;
    cblas_dgemv(CblasColMajor,
                CblasNoTrans,
                m,
                n,
                alpha,
                conjugateResultFixEff,
                lda,
                fixEffect.data(),
                incx,
                beta,
                yhat,
                incy);

    // compute omega-1y - yhat
    double *omegaInvy = conjugateResultFixEff + covarBasis.getC() * Npad;
    phenoData = ALIGN_ALLOCATE_DOUBLES(Npad);

    for (uint64 n = 0; n < Npad; n++) {
      phenoData[n] = omegaInvy[n] - yhat[n];
    }
    ALIGN_FREE(yhat);
  } else {
    cout << endl << "compute posterior by approximate way " << endl;
    // compute Zw_hat
    double *zw_hat = ALIGN_ALLOCATE_DOUBLES(Npad);
    MKL_INT m = Npad;
    MKL_INT n = covarBasis.getC();
    double alpha = 1.0;
    MKL_INT lda = m;
    double beta = 0.0;
    MKL_INT incx = 1;
    MKL_INT incy = 1;
    cblas_dgemv(CblasColMajor,
                CblasNoTrans,
                m,
                n,
                alpha,
                covarBasis.getCovarMatrix(),
                lda,
                fixEffect.data(),
                incx,
                beta,
                zw_hat,
                incy);

    // pheno vector reduce the zw_hat
    for (uint64 n = 0; n < Npad; n++) {
      zw_hat[n] = pheno[n] - zw_hat[n];
    }

    // solve a single conjugate gradient vector
    phenoData = ALIGN_ALLOCATE_DOUBLES(Npad);
    memset(phenoData, 0, Npad * sizeof(double));
    calConjugateWithoutMask(phenoData, zw_hat, 1);
    ALIGN_FREE(zw_hat);
  }

  // solve single conjugate gradient get result A
  scalVec(phenoData, sigma2g, Npad); // scale the result of conjugate gradient (refer to the document)

  // compute mu = X^TA, where A is a vector
  posteriorMean.resize(M);
//  multXTmatrix(posteriorMean.data(), phenoData, 1);
  computeSinglePosteriorMean(bimFiles, bedFiles, phenoData);

  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] /= sqrt(M); // scale X
  }

  // rescale the posterior mean
  subIntercept = 0;
  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] /= sqrt(M) / Meanstd[m].second;
    subIntercept += posteriorMean[m] * Meanstd[m].first;
  }

  fixEffect[0] -= subIntercept; // subtract the intercept

  // save fixeffect
  FileUtils::SafeOfstream fout;
  fout.open(outputFile + "_fixeff.txt");
  for (int i = 0; i < covarBasis.getC(); i++) {
    fout << fixEffect[i] << "\n";
  }
  fout.close();

  fout.open(outputFile + "_posteriorMean.txt");
  for (uint64 m = 0; m < M; m++) {
    fout << posteriorMean[m] << "\n";
  }
  fout.close();

  ALIGN_FREE(conjugateResultFixEff);
  ALIGN_FREE(phenoData);
}

void LMMCPU::computePosteriorMean(const double* pheno, bool useApproximate) {
  // note the phenotype here is original
  double *phenoData;
  if (!useApproximate) {
    // compute yhat = omega-1ZW
    cout << endl << "compute posterior by exact way" << endl;
    double *yhat = ALIGN_ALLOCATE_DOUBLES(Npad);
    MKL_INT m = Npad;
    MKL_INT n = covarBasis.getC();
    double alpha = 1.0;
    MKL_INT lda = m;
    double beta = 0.0;
    MKL_INT incx = 1;
    MKL_INT incy = 1;
    cblas_dgemv(CblasColMajor,
                CblasNoTrans,
                m,
                n,
                alpha,
                conjugateResultFixEff,
                lda,
                fixEffect.data(),
                incx,
                beta,
                yhat,
                incy);

    // compute omega-1y - yhat
    double *omegaInvy = conjugateResultFixEff + covarBasis.getC() * Npad;
    phenoData = ALIGN_ALLOCATE_DOUBLES(Npad);

    for (uint64 n = 0; n < Npad; n++) {
      phenoData[n] = omegaInvy[n] - yhat[n];
    }
    ALIGN_FREE(yhat);
  } else {
    cout << endl << "compute posterior by approximate way " << endl;
    // compute Zw_hat
    double *zw_hat = ALIGN_ALLOCATE_DOUBLES(Npad);
    MKL_INT m = Npad;
    MKL_INT n = covarBasis.getC();
    double alpha = 1.0;
    MKL_INT lda = m;
    double beta = 0.0;
    MKL_INT incx = 1;
    MKL_INT incy = 1;
    cblas_dgemv(CblasColMajor,
                CblasNoTrans,
                m,
                n,
                alpha,
                covarBasis.getCovarMatrix(),
                lda,
                fixEffect.data(),
                incx,
                beta,
                zw_hat,
                incy);

    // pheno vector reduce the zw_hat
    for (uint64 n = 0; n < Npad; n++) {
      zw_hat[n] = pheno[n] - zw_hat[n];
    }

    // solve a single conjugate gradient vector
    phenoData = ALIGN_ALLOCATE_DOUBLES(Npad);
    memset(phenoData, 0, Npad * sizeof(double));
    calConjugateWithoutMask(phenoData, zw_hat, 1);
    ALIGN_FREE(zw_hat);
  }

  // solve single conjugate gradient get result A
  scalVec(phenoData, sigma2g, Npad); // scale the result of conjugate gradient (refer to the document)

  // compute mu = X^TA, where A is a vector
  posteriorMean.resize(M);
  multXTmatrix(posteriorMean.data(), phenoData, 1);

  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] /= sqrt(M); // scale X
  }

  // rescale the posterior mean
  subIntercept = 0;
  for (uint64 m = 0; m < M; m++) {
    posteriorMean[m] /= sqrt(M) / Meanstd[m].second;
    subIntercept += posteriorMean[m] * Meanstd[m].first;
  }

  fixEffect[0] -= subIntercept; // subtract the intercept

  // save fixeffect
  FileUtils::SafeOfstream fout;
  fout.open(outputFile + "_fixeff.txt");
  for (int i = 0; i < covarBasis.getC(); i++) {
    fout << fixEffect[i] << "\n";
  }
  fout.close();

  fout.open(outputFile + "_posteriorMean.txt");
  for (uint64 m = 0; m < M; m++) {
    fout << posteriorMean[m] << "\n";
  }
  fout.close();

  ALIGN_FREE(conjugateResultFixEff);
  ALIGN_FREE(phenoData);
}

void LMMCPU::predict(double *output, const GenoData &predictData, const CovarBasis<GenoData> &predictCov) const {
  uint64 numPredict = predictData.getNpad(); // get number of prediction samples

  // build SNPs lookup table for prediction data (impute missing data by mode)
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
  for (uint64 n = 0; n < predictData.getNused(); n++) {
    fout << predictedRandomEff[n] << "\t" << output[n] << endl;
  }
  fout.close();

  ALIGN_FREE(predictedRandomEff);
  ALIGN_FREE(predictProjMaskSnps);
  ALIGN_FREE(snpVector);
  ALIGN_FREE(workTable);
  ALIGN_FREE(predictionSnpLookupTable);
}

void LMMCPU::buildPredictSnpsLookupTable(uint64 m,
                                         uint64 numPredict,
                                         double *snpVector,
                                         double (*predictionSnpLookupTable)[4]) const {
  // get the mode of each snp
  vector<uint64> snpsCounts;
  snpsCounts.reserve(3);
  for (uint64 n = 0; n < numPredict; n++) {
    if (snpVector[n] != 9)
      snpsCounts[(int) snpVector[n]]++;
  }
  // find the mode index in this case, which is one of 0, 1, 2
  int mode = std::distance(snpsCounts.begin(), std::max_element(snpsCounts.begin(), snpsCounts.end()));

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

void LMMCPU::predictRandomEff(uint64 numPredict,
                              double *predictMaskIndivs,
                              double *randomEff,
                              double (*predictionSnpLookupTable)[4],
                              const GenoData &predictData) const {
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

void LMMCPU::predictFixEff(uint64 numPredict, double *fixEff, const double *predictCovarMatrix) const {
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
  // we only use the former part of fixeffect, we do not need to deal with
}

std::vector<uint64> LMMCPU::selectProSnps(int numCalibSnps) {
  // Todo:: do we need to add the chisquare statistics standard to select the snps

  vector<uint64> selectSnpsIndex(numCalibSnps, M);

  // divide snps up into numCalibSnps blocks in order to select snps randomly
  vector<uint64> mFirst(numCalibSnps + 1, M);
  uint64 blockWidth = M / numCalibSnps;
  for (int snp = 0; snp < numCalibSnps + 1; snp++) {
    mFirst[snp] = blockWidth;
    blockWidth *= 2;
  }

  // setup random number generator
  boost::mt19937 rng(321);
  boost::uniform_int<> unif(0, 1 << 30);
  boost::variate_generator<boost::mt19937 &, boost::uniform_int<> > randu(rng, unif);

  for (int j = 0; j < numCalibSnps; j++) {
    uint64 m = mFirst[j] + randu() % (mFirst[j + 1] - mFirst[j]);
    if (projMaskSnps[m]) {
      selectSnpsIndex[m] = m;
    }
  }

  cout << "Selected " << numCalibSnps << " SNPs for computation of prospective stat" << endl;

  return selectSnpsIndex;
}

void LMMCPU::calHeritability(const double *projectPheno) {
  double yvy = NumericUtils::dot(projectPheno, projectPheno, Npad);
  double yvkvy = calyVKVy(projectPheno);

  double kv, kvkv;

  if (!useExactTrace) {
    cout << "Calculate trace by using random algorithm" << endl;
    calTraceMoM(kv, kvkv);
  } else {
    cout << "Calculate trace by using exact algorithm" << endl;
    calTraceExact(kv, kvkv);
  }

  auto temp = static_cast<double>(Nused - covarBasis.getC());
  // the solution we slove the liner equation directly
  sigma2g = (yvkvy * temp - yvy * kv) / (temp * kvkv - kv * kv);
  sigma2e = (yvy - kv * sigma2g) / temp;

  cout << "sigma2g " << sigma2g << endl;
  cout << "sigma2e " << sigma2e << endl;

  cout << "heritability is " << sigma2g / (sigma2e + sigma2g) << endl;

  double stderror = calStandError(projectPheno, kv, kvkv);
  cout << "The stand error of heritability is: " << stderror << endl;

}

void LMMCPU::calCalibrationFactor(const double *projectPheno, bool sampleCalibrationFactor) {

  // make conjugate gradient input CalibSnps + phenotype
  double *yMatrix = ALIGN_ALLOCATE_DOUBLES(Npad * numChrom);
  for (uint64 chrom = 0; chrom < numChrom; chrom++) {
    double *y_temp = yMatrix + chrom * Npad;
    memcpy(y_temp, projectPheno, Npad * sizeof(double));
  }

  double *VinvChromy = ALIGN_ALLOCATE_DOUBLES(Npad * numChrom);
  memset(VinvChromy, 0, Npad * numChrom * sizeof(double));

  Timer timer;
  // calculate batch conjugate gradient for V_locoY
  calConjugateBatch(VinvChromy, yMatrix);

  cout << "Time for solve batch conjugate gradient is " << timer.update_time() << " sec" << endl;

  z.reserve(M);
  zsquare.reserve(M);

#ifdef DEBUG
  cout << "std list " << endl;
  for (int i = 0; i < 10; i++)
      cout << projGenoStd[i] << endl;
  cout << endl << "Ainvb " << endl;
  for (int i = 0; i < 10; i++)
      cout << Ainvb[i] << endl;
  pair <double, double> muSigma = NumericUtils::meanStdDev(Ainvb, Nstride);
      cout << "Ainvb " << ":   N = " << Nstride << "   mean = " << muSigma.first
           << "   std = " << muSigma.second << endl;
#endif
  calBeta(VinvChromy);

  cout << "Time for compuate beta is " << timer.update_time() << " sec" << endl;
  NumericUtils::divideElem(z.data(), projGenoStd.data(), M);

#ifdef DEBUG
  cout << endl << "Z value" << endl;
  for (int i = 0; i < 10; i++)
      cout << z[i] << endl;
#endif
  memcpy(zsquare.data(), z.data(), M * sizeof(double));
  NumericUtils::multipElem(zsquare.data(), z.data(), M); // elementwise multiply

  // compute Vchromy l2 norm
  VinvyNorm.reserve(numChrom);
  for (int chrom = 0; chrom < numChrom; chrom++) {
    double *temp_VinvY = VinvChromy + chrom * Npad;
    VinvyNorm.push_back(NumericUtils::norm2(temp_VinvY, Npad));
  }

  // normalize Vchromy and compute the mean of statistics
  double sumStatics = 0;
  for (uint64 m = 0; m < M; m++) {
    zsquare[m] *= 1 / VinvyNorm[chromID[m] - 1];
    sumStatics += zsquare[m];
  }

  // compute calibration factor  mean(zVy.^2) / lambda
  double meanStatics = (double) sumStatics / M;
  double hg = sigma2g / (sigma2g + sigma2e); // I think the hg2 is the square of heritability
  double hg2 = hg * hg;
  double rsquare = (double) Nused * hg2 / M;
  double lambda = 1 + rsquare / (1 - rsquare * hg2);

  calibrationFactor = meanStatics / lambda;
//        cout << "lambda is " << lambda << endl;
//        cout << "calibration factor calculated is " << calibrationFactor << endl;
//        calibrationFactor = 1.0;
  cout << "Calibration Factor is " << calibrationFactor << endl;
  scalVec(zsquare.data(), 1 / calibrationFactor, M);

}

void LMMCPU::computeStatistics(std::string &outputFile) const {

  double *zval = ALIGN_ALLOCATE_DOUBLES(M);
  double *beta = ALIGN_ALLOCATE_DOUBLES(M);
  double *pvalue = ALIGN_ALLOCATE_DOUBLES(M);

  memcpy(zval, z.data(), M * sizeof(double));
  memcpy(beta, z.data(), M * sizeof(double));

  scalVec(zval, sqrt(1 / calibrationFactor), M);
  scalVec(beta, 1 / calibrationFactor, M);

  for (uint64 m = 0; m < M; m++)
    beta[m] *= 1 / VinvyNorm[chromID[m] - 1];

  NumericUtils::divideElem(beta, projGenoStd.data(), M);

  for (uint64 m = 0; m < M; m++)
    pvalue[m] = gsl_cdf_chisq_Q(zsquare[m], 1);

  // save the result to outputFile
  FileUtils::SafeOfstream fout;
  fout.open(outputFile);

  const vector<SnpInfo> &outSnpInfo = genoData.getSnpInfo();

  fout << std::scientific; // use scientific format output
  fout << "chrom" << "\t" << "snpID" << "\t" << "allele1" << "\t" << "allele2" << "\t";
  fout << "beta" << "\t" << "zvalue" << "\t" << "chisq" << "\t" << "pvalue" << "\n";

  for (uint64 m = 0; m < M; m++) {
    fout << outSnpInfo[m].chrom << "\t" << outSnpInfo[m].ID << "\t" << outSnpInfo[m].allele1 << "\t"
         << outSnpInfo[m].allele2 << "\t";
    fout << beta[m] << "\t" << zval[m] << "\t" << zsquare[m] << "\t" << pvalue[m] << "\n";
  }
  fout.close();

  cout << "Finish writing association test result to " << outputFile << endl;

}

// ********** overload blas inferface **********

double LMMCPU::dotVec(const double *vec1, const double *vec2, uint64 N) const {
  MKL_INT n = N;
  MKL_INT incx = 1;
  MKL_INT incy = 1;
  return cblas_ddot(n, vec1, incx, vec2, incy);
}

void LMMCPU::scalVec(double *vec, double alpha, uint64 elem) const {
  MKL_INT n = elem;
  MKL_INT incx = 1;
  cblas_dscal(n, alpha, vec, incx);
}

void LMMCPU::compInverse(double *matrix, const unsigned int row) const {

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

