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

//#include "dataBasis.h"

namespace LMMNET {

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::cerr;

template<typename T>
dataBasis<T>::dataBasis(const std::string &_filePath, const T &_genoData):
    filePath(_filePath), genoData(_genoData) {
}

template<typename T>
double dataBasis<T>::parseDouble(const string &strValue) const {
  if (strValue == missing_str)
    return missing_dbl;
  else {
    double d;
    int success = sscanf(strValue.c_str(), "%lf", &d);
    if (success)
      return d;
    else {
      cerr << "ERROR: Could not parse DataMatrix field to floating-point: " << strValue << endl;
      exit(1);
    }
  }
}

template<typename T>
uint64 dataBasis<T>::getRowIndex(const std::string &rowName) const {
  for (uint64 r = 0; r < nrows; r++)
    if (rowNames[r] == rowName)
      return r;
  cerr << "ERROR: Unable to find field named " << rowName << endl;
  exit(1);
}

template<typename T>
std::vector<std::string> dataBasis<T>::getRowStr(const std::string &rowName) const {
  uint64 r = getRowIndex(rowName);
  vector<string> rowDataStr(ncols);
  for (uint64 i = 0; i < ncols; i++)
    rowDataStr[i] = data_str[r][i];
  return rowDataStr;
}

template<typename T>
std::vector<double> dataBasis<T>::getRowDbl(const std::string &rowName) const {
  vector<string> rowDataStr = getRowStr(rowName);
  vector<double> rowDataDbl(ncols);
  for (uint64 i = 0; i < ncols; i++)
    rowDataDbl[i] = parseDouble(rowDataStr[i]);
  return rowDataDbl;
}

template<typename T>
double dataBasis<T>::getEntryDbl(uint64 r, uint64 c) const {
  return parseDouble(data_str[r][c]);
}


}