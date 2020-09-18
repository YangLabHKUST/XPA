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
#include <iostream>
#include <sstream>
#include <set>
#include <assert.h>
#include <string.h>

#include "GenoData.h"
#include "NumericUtils.h"
//#include "CovarBasis.h"
#include "MemoryUtils.h"
#include "AuxGenoData.h"

namespace LMMNET {

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::cerr;

template<typename T>
CovarBasis<T>::CovarBasis(const std::string &covarFile, const T &genodata,
                          const std::vector<std::string> &covarCols, std::vector<double> &_maskIndivs) :
    dataBasis<T>(covarFile, genodata), maskIndivs(_maskIndivs) {
  initialize(covarCols);
}

template<typename T>
CovarBasis<T>::~CovarBasis() {
  ALIGN_FREE(covarMatirx);
}

template<typename T>
void CovarBasis<T>::initialize(const std::vector<std::string> &covarCols) {
  this->missing_dbl = -9;
  this->missing_str = "NA";
  string ALL_ONES_NAME = "ONES";

  this->rowNames = vector<string>(1, ALL_ONES_NAME); // always include all-1s vector
  FileUtils::SafeIfstream fin;

  // read covarFile col name
  if (!this->filePath.empty()) {
    fin.open(this->filePath);
    string line;
    getline(fin, line);
    std::istringstream iss(line);
    string FID, IID, covarName;
    iss >> FID >> IID;
    if (FID != "FID" || IID != "IID") {
      cerr << "ERROR: Phenotype/covariate file must start with header: FID IID" << endl;
      exit(1);
    }
    while (iss >> covarName) this->rowNames.push_back(covarName);
  }

  this->nrows = this->rowNames.size();
  this->ncols = this->genoData.getNpad();
  Nused = this->genoData.getNused();

  this->data_str = vector<vector<string> >(this->nrows, vector<string>(this->ncols, this->missing_str));
  this->data_str[0] = vector<string>(this->ncols, "1");

  std::set<uint64> indivsSeen;
  std::vector<double> maskForCov(this->genoData.getNpad(), 0.);

  if (!this->filePath.empty()) {
    string line, FID, IID;
    int numLines = 0, numIgnore = 0;
    while (getline(fin, line)) {
      numLines++;
      std::istringstream iss(line);
      iss >> FID >> IID;
      uint64 n = this->genoData.getIndivInd(FID, IID);
      if (n == GenoData::IND_MISSING) {
        if (numIgnore < 5)
          cerr << "WARNING: Ignoring indiv not in genotype data: FID=" << FID << ", IID=" << IID
               << endl;
        numIgnore++;
      } else {
        if (indivsSeen.count(n)) {
          cerr << "WARNING: Duplicate entry for indiv FID=" << FID << ", IID=" << IID << endl;
        } else indivsSeen.insert(n);

        maskForCov[n] = 1.; // update the mask for covariate file
        string covarValue;
        vector<string> covars;
        while (iss >> covarValue) covars.push_back(covarValue);
        if (covars.size() != this->nrows - 1) { // nrows-1, not nrows, because all-1s vector was added
          cerr << "ERROR: Wrong number of entries in data row:" << endl;
          cerr << line << endl;
          cerr << "Expected " << this->nrows - 1 << " fields after FID, IID cols" << endl;
          cerr << "Parsed " << covars.size() << " fields" << endl;
          exit(1);
        }
        for (uint64 iCovar = 0; iCovar < covars.size(); iCovar++)
          this->data_str[iCovar + 1][n] = covars[iCovar]; // offset by 1 because of all-1s vector
      }
    }
    fin.close();
    if (numIgnore)
      cout << "WARNING: Ignoring indiv not in genotype data " << numIgnore << " indivs" << endl;

    // check if there is missing individual in covariate file
    if (numLines != Nused) {
      cout << "WARNING: there are missing "<< Nused - numLines << " individuals in covariate file" << endl;
      cout << "Finally, there are " << numLines << " individuals in analysis" << endl;
    }
  } else {
    // in this case, we do not have covariate and we will only use a column of one
    // We cannot set mask index according to provided covarfile. Here we keep all indivs
    for (uint64 n = 0; n < this->genoData.getN(); n++) {
      maskForCov[n] = 1.;
    }
  }

  Npad = this->genoData.getNpad();

  // select the covariate used and convert in double type
  if (!covarCols.empty()) {
    C = covarCols.size() + 1; // plus the all 1 vector
    covarMatirx = ALIGN_ALLOCATE_DOUBLES(C * Npad); // here we also need to pad the covariate matrix
    // fill the first column with one
    for (uint64 n = 0; n < Npad; n++) {
      covarMatirx[n] = 1;
    }
    int totalRows = this->rowNames.size();
    int store_index = 1;
    for (int row = 0; row < totalRows; row++) {
      bool found = false;
      for (int col = 0; col < covarCols.size(); col++)
        if (this->rowNames[row + 1] == covarCols[col]) {
          cout << "Use covariate " << this->rowNames[row + 1] << " in analysis " << endl;
          found = true;
          break;
        }

      if (found) {
        int decode_index = row + 1;
        // store covariate matrix in double for further computatoin
        for (uint64 n = 0; n < Npad; n++) {
          if (this->getEntryDbl(decode_index, n) == this->missing_dbl) {
            covarMatirx[store_index * Npad + n] = 0; // deal with the missing covariate
          } else {
            covarMatirx[store_index * Npad + n] = this->getEntryDbl(decode_index, n);
          }
        }
        store_index++;
      }
    }
  } else { // default use all columns in covariate
    C = this->rowNames.size(); // plus the all 1 vector
    covarMatirx = ALIGN_ALLOCATE_DOUBLES(C * Npad); // here we also need to pad the covariate matrix
    cout << "Use all columns in covariate file in analysis" << endl;
    for (int c = 0; c < C; c++) {
      for (uint64 n = 0; n < Npad; n++) {
        if (this->getEntryDbl(c, n) == this->missing_dbl) {
          covarMatirx[c * Npad + n] = 0; // deal with the missing covariate
        } else {
          covarMatirx[c * Npad + n] = this->getEntryDbl(c, n);
        }
      }
    }
  }

  // set the padding part of covarmatrix to zeros
  for (uint64 n = Nused; n < Npad; n++) {
    for (uint64 col = 0; col < C; col++) {
      covarMatirx[n + col * Npad] = 0;
    }
  }

  // update the final mask according to the covariate file mask
  for (uint64 n = 0; n < Npad; n++) {
    maskIndivs[n] *= maskForCov[n];
  }

  // deal with the missing individual and missing covariate cases
  for (uint64 n = 0; n < Npad; n++) {
    if (!maskIndivs[n]) {
      for (uint64 c = 0; c < C; c++) {
        covarMatirx[c * Npad + n] = 0;
      }
    }
  }

  // precompute WTW
  computeWTW();
}

template<typename T>
void CovarBasis<T>::computeWTW() {
  WTW = ALIGN_ALLOCATE_DOUBLES(C * C);

  MKL_INT m1 = C;
  MKL_INT n1 = C;
  MKL_INT k1 = Npad;
  double alpha1 = 1.0;
  MKL_INT lda1 = Npad;
  MKL_INT ldb1 = Npad;
  MKL_INT ldc1 = C;
  double beta1 = 0.0;
  cblas_dgemm(CblasColMajor,
              CblasTrans,
              CblasNoTrans,
              m1,
              n1,
              k1,
              alpha1,
              covarMatirx,
              lda1,
              covarMatirx,
              ldb1,
              beta1,
              WTW,
              ldc1);
}

template<typename T>
void CovarBasis<T>::projectCovarsVec(double *vec) const {
  double *temp = ALIGN_ALLOCATE_DOUBLES(C);
  double *temp1 = ALIGN_ALLOCATE_DOUBLES(Npad);
  double *temp2 = ALIGN_ALLOCATE_DOUBLES(C);

  // first compute A=W^TV
  MKL_INT m = Npad;
  MKL_INT n = C;
  double alpha = 1.0;
  MKL_INT lda = Npad;
  MKL_INT incx = 1;
  double beta = 0.0;
  MKL_INT incy = 1;
  cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, covarMatirx, lda, vec, incx, beta, temp, incy);

  MKL_INT n1 = C;
  MKL_INT lda1 = n1;
  MKL_INT ldb1 = n1;
  lapack_int nrhs = 1;
  lapack_int ldx1 = n1;
  lapack_int ipiv[n1];
  lapack_int iter = -1;
  lapack_int info1 = LAPACKE_dsgesv(LAPACK_COL_MAJOR, n1, nrhs, WTW, lda1, ipiv, temp, ldb1, temp2, ldx1, &iter);

  if (info1 != 0) {
    cerr << "Failed to compute LU decomposition " << endl;
    cerr << "Cannot compute the inverse matrix " << endl;
    exit(1);
  }

  // compute WB
  MKL_INT m3 = Npad;
  MKL_INT n3 = C;
  MKL_INT lda3 = Npad;

  cblas_dgemv(CblasColMajor, CblasNoTrans, m3, n3, alpha, covarMatirx, lda3, temp2, incx, beta, temp1, incy);

  // compute V - WB
  NumericUtils::subElem(vec, temp1, Npad);

  ALIGN_FREE(temp);
  ALIGN_FREE(temp1);
  ALIGN_FREE(temp2);
}

template<typename T>
uint64 CovarBasis<T>::getC() const {
  return C;
}

template<typename T>
const double *CovarBasis<T>::getCovarMatrix() const {
  return covarMatirx;
}

template<typename T>
void CovarBasis<T>::writeMask(std::vector<double>& maskIndiv) const {
  assert(maskIndiv.size() == this->maskIndivs.size()); // mask vector should have the same length
  uint64 length = maskIndiv.size();
  for (uint64 n = 0; n < length; n++) {
    maskIndiv[n] *= this->maskIndivs[n];
  }
}

}