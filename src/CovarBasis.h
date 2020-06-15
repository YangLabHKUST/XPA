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

#ifndef LMMNET_COVARBASIS_H
#define LMMNET_COVARBASIS_H

#include "dataBasis.h"

namespace LMMNET {

template<typename T>
class CovarBasis : public dataBasis<T> {

 private:

  uint64 Nused, Npad, C; // Dimension of covariate matrix

  double *covarMatirx; // Store the column major of double type covariate matrix
  double *WTW; // Store the matrix of W^TW and we can reuse the result

  std::vector<double> maskIndivs; // Store the mask of removed individuals

  /**
   * Read the column major and double type of covariate matrix and add
   * addition column of ones to covariate matrix
   * 
   * @param covarCols the path of covariate files
  */
  void initialize(const std::vector<std::string> &covarCols);
  void computeWTW();

 public:
  CovarBasis(const std::string &covarFile,
             const T &genodata,
             const std::vector<std::string> &covarCols,
             std::vector<double> &_maskIndivs);
  CovarBasis(const CovarBasis &) = delete;
  ~CovarBasis();

  /**
   * Project the covariates from given input
   * 
   * @param vec the given input vectors
  */
  void projectCovarsVec(double *vec) const;

  uint64 getC() const;
  const double *getCovarMatrix() const;
  void writeMask(std::vector<double>& maskIndiv) const;
};
}

#include "CovarBasis_impl.h"
#endif //LMMNET_COVARBASIS_H
