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

#include "GenoData.h"
//#include "PhenoBasis.h"
#include "NumericUtils.h"
#include "AuxGenoData.h"

namespace LMMNET {

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::cerr;

template<typename T>
PhenoBasis<T>::PhenoBasis(const std::string &phenoFile, const T &_genoData,
                          const std::vector<std::string> &phenoCols, vector<double> &_maskIndivs,
                          bool _phenoUseFame): dataBasis<T>(phenoFile, _genoData), phenoUseFam(_phenoUseFame) {
  initialize(phenoCols, _maskIndivs);
}

template<typename T>
void PhenoBasis<T>::initialize(const std::vector<std::string> &phenoCols, vector<double> &_maskIndivs) {
  // set missing key for further use
  this->missing_dbl = -9;
  this->missing_str = "NA";

  if (!phenoUseFam) {
    // read phenotype file from disk
    FileUtils::SafeIfstream fin;

    if (!this->filePath.empty()) {
      fin.open(this->filePath);
      string line;
      getline(fin, line); // get the header line
      std::istringstream iss(line);
      string FID, IID, phenoName;
      iss >> FID >> IID;
      if (FID != "FID" || IID != "IID") {
        cerr << "ERROR: Phenotype file must start with header: FID IID" << endl;
        exit(1);
      }
      while (iss >> phenoName) this->rowNames.push_back(phenoName);
    } else {
      cerr << "ERROR: Please provide the avaliable phenotype file path or use 6th column in fam file" << endl;
      exit(1);
    }

    // get phenotype data size
    this->nrows = this->rowNames.size();
    this->ncols = this->genoData.getNpad();

    this->data_str = vector<vector<string> >(this->nrows, vector<string>(this->ncols, this->missing_str));

    std::set<uint64> indivsSeen;

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

          string phenoValue;
          vector<string> phenos;
          while (iss >> phenoValue) phenos.push_back(phenoValue);
          if (phenos.size() != this->nrows) { // nrows-1, not nrows, because all-1s vector was added
            cerr << "ERROR: Wrong number of entries in data row:" << endl;
            cerr << line << endl;
            cerr << "Expected " << this->nrows << " fields after FID, IID cols" << endl;
            cerr << "Parsed " << phenos.size() << " fields" << endl;
            exit(1);
          }
          for (uint64 ipheno = 0; ipheno < phenos.size(); ipheno++)
            this->data_str[ipheno][n] = phenos[ipheno];
        }
      }
      fin.close();
    }

    // convert phenotype str to double type
    assert(phenoCols.size() == 1 && "The model can only analysis one phenotype one time.");
    phenotype_dbl.resize(phenoCols.size());
    for (uint p = 0; p < phenotype_dbl.size(); p++) {
      const string &phenoCol = phenoCols[p];
      cout << "Use phenotype " << phenoCol << " in analysis" << endl;
      phenotype_dbl[p] = this->getRowDbl(phenoCol); // assign pheno from column of file
      if (phenotype_dbl[p].empty()) {
        cerr << "ERROR: Phenotype data matrix does not contain column " << phenoCol << endl;
        exit(1);
      }
    }
  } else { // in this case, the number of phenotype must be equal to the number of individuals
    phenotype_dbl.resize(1);
    phenotype_dbl[0] = this->genoData.getFamPhenos();
  }

  if (!phenotype_dbl.empty()) {
    int numGoodIndivs = 0;
    for (uint64 n = 0; n < phenotype_dbl[0].size(); n++) {
      for (uint p = 0; p < phenotype_dbl.size(); p++) {
        if (phenotype_dbl[p][n] == this->missing_dbl) {
          _maskIndivs[n] = 0;
          phenotype_dbl[p][n] = 0;
        }
        numGoodIndivs += (int) _maskIndivs[n];
      }
    }
    cout << "Number of indivs with no missing phenotype(s) to use: " << numGoodIndivs << endl;
  }

  // update local maskindivs to global maskindivs
  this->maskIndivs = _maskIndivs;
}

template<typename T>
void PhenoBasis<T>::padPhenodbl() {
  uint64 Npad = this->genoData.getNpad();
  for (int col = 0; col < phenotype_dbl.size(); col++) {
    while (phenotype_dbl[col].size() < Npad)
      phenotype_dbl[col].push_back(0);
  }
}

template<typename T>
std::vector<std::vector<double> > PhenoBasis<T>::getPhenodbl() const {
  return phenotype_dbl;
}

}