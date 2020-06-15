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
#ifndef LMMNET_DATABASIS_H
#define LMMNET_DATABASIS_H

#include <vector>
#include <string>
#include <mkl.h>

#include "TypeDef.h"
#include "GenoData.h"

namespace LMMNET {
template<typename T>
class dataBasis {
 protected:
  const T &genoData;
  const std::string &filePath;

  // size of phenotype data, now we assume only one row
  uint64 nrows, ncols;

  // phenotype data rownames
  std::vector<std::string> rowNames;

  // maskindivs
  std::vector<double> maskIndivs;

  // vector to store phenotype data
  std::vector<std::vector<std::string> > data_str;

  // missing key of type str and double
  std::string missing_str;
  double missing_dbl;

 public:

  dataBasis(const std::string &covarFile, const T &_genoData);

  double parseDouble(const std::string &strValue) const;
  uint64 getRowIndex(const std::string &rowName) const;
  std::vector<std::string> getRowStr(const std::string &rowName) const;
  std::vector<double> getRowDbl(const std::string &rowName) const;
  double getEntryDbl(uint64 r, uint64 c) const;
};
}

#include "dataBasis_impl.h"
#endif //LMMNET_DATABASIS_H
