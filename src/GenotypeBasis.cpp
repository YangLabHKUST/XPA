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


#include <cstring>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>

#include <emmintrin.h>
#include <random>

#include "TypeDef.h"
#include "GenoData.h"
#include "IOUtils.h"
#include "MemoryUtils.h"
#include "InfoStructure.h"
#include "GenotypeBasis.h"

namespace LMMNET {

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using FileUtils::getline;

const uint64 GenoBasis::IND_MISSING = (uint64) -1;

void GenoBasis::processIndivs(const std::string &famFile, const std::vector<std::string> &removeIndivsFiles) {
  string line;

  vector<IndivInfo> bedIndivs;
  FileUtils::SafeIfstream fin;
  fin.open(famFile);
  while (getline(fin, line)) {
    std::istringstream iss(line);
    IndivInfo indiv;
    if (!(iss >> indiv.famID >> indiv.indivID >> indiv.paternalID >> indiv.maternalID
              >> indiv.sex >> indiv.pheno)) {
      cerr << "ERROR: Incorrectly formatted fam file: " << famFile << endl;
      cerr << "Line " << bedIndivs.size() + 1 << ":" << endl;
      cerr << line << endl;
      cerr << "Unable to input 6 values (4 string, 1 int, 1 double)" << endl;
      exit(1);
    }
    string combined_ID = indiv.famID + " " + indiv.indivID;
    if (FID_IID_to_ind.find(combined_ID) != FID_IID_to_ind.end()) {
      cerr << "ERROR: Duplicate individual in fam file at line " << bedIndivs.size() + 1 << endl;
      exit(1);
    }
    FID_IID_to_ind[combined_ID] = bedIndivs.size();
    bedIndivs.push_back(indiv);
  }
  fin.close();
  Nbed = bedIndivs.size();

  cout << "Total indivs in PLINK data: Nbed = " << Nbed << endl;

  // process individuals to remove
  vector<bool> useIndiv(Nbed, true);
  for (uint f = 0; f < removeIndivsFiles.size(); f++) {
    const string &removeIndivsFile = removeIndivsFiles[f];
    cout << "Reading remove file (indivs to remove): " << removeIndivsFile << endl;
    fin.open(removeIndivsFile);
    int lineCtr = 0;
    int numRemoved = 0;
    int numAbsent = 0;
    while (getline(fin, line)) {
      lineCtr++;
      std::istringstream iss(line);
      string FID, IID;
      if (!(iss >> FID >> IID)) {
        cerr << "ERROR: Incorrectly formatted remove file: " << removeIndivsFile << endl;
        cerr << "Line " << lineCtr << ":" << endl;
        cerr << line << endl;
        cerr << "Unable to input FID and IID" << endl;
        exit(1);
      }
      string combined_ID = FID + " " + IID;
      if (FID_IID_to_ind.find(combined_ID) == FID_IID_to_ind.end()) {
        if (numAbsent < 5)
          cerr << "WARNING: Unable to find individual to remove: " << combined_ID << endl;
        numAbsent++;
      } else if (useIndiv[FID_IID_to_ind[combined_ID]]) {
        useIndiv[FID_IID_to_ind[combined_ID]] = false;
        numRemoved++;
      }
    }
    fin.close();
    cout << "Removed " << numRemoved << " individual(s)" << endl;
    if (numAbsent)
      cerr << "WARNING: " << numAbsent << " individual(s) not found in data set" << endl;
  }

  bedIndivToRemoveIndex.resize(Nbed);
  FID_IID_to_ind.clear();
  for (uint64 nbed = 0; nbed < Nbed; nbed++) {
    if (useIndiv[nbed]) {
      bedIndivToRemoveIndex[nbed] = indivs.size();
      FID_IID_to_ind[bedIndivs[nbed].famID + " " + bedIndivs[nbed].indivID] = indivs.size();
      indivs.push_back(bedIndivs[nbed]);
    } else
      bedIndivToRemoveIndex[nbed] = -1;
  }
  N = indivs.size();
  cout << "Total indivs stored in memory: N = " << N << endl;

  // allocate and initialize maskIndivs to all good (aside from filler at end)
  Nstride = (N + 3) & ~3;
  maskIndivs = ALLOCATE_DOUBLES(Nstride);
  for (uint64 n = 0; n < N; n++) maskIndivs[n] = 1;
  for (uint64 n = N; n < Nstride; n++) maskIndivs[n] = 0;
}

void GenoBasis::storeBedLine(uchar *bedLineOut, const uchar *genoLine) {
  const int genoToBed[10] = {3, 2, 0, 0, 0, 0, 0, 0, 0, 1};
  memset(bedLineOut, 0, (Nstride >> 2) * sizeof(bedLineOut[0]));
  for (uint64 n = 0; n < N; n++)
    bedLineOut[n >> 2] = (uchar) (bedLineOut[n >> 2] | genoToBed[genoLine[n]] << ((n & 3) << 1));
}

GenoBasis::GenoBasis(const std::string &famFile, const std::vector<std::string> &_bimFiles,
                     const std::vector<std::string> &_bedFiles,
                     const std::vector<std::string> &removeSNPsFiles,
                     const std::vector<std::string> &removeIndivsFiles, double _maxMissingPerSnp,
                     double _maxMissingPerIndiv) : bimFiles(_bimFiles), bedFiles(_bedFiles),
                                                   maxMissingPerIndiv(_maxMissingPerIndiv),
                                                   maxMissingPerSnp(_maxMissingPerSnp) {
  processIndivs(famFile, removeIndivsFiles);
}

vector<SnpInfo> GenoBasis::readBimFile(const std::string &bimFile) {
  vector<SnpInfo> ret;
  string line;
  FileUtils::SafeIfstream fin;
  fin.open(bimFile);
  while (getline(fin, line)) {
    std::istringstream iss(line);
    SnpInfo snp;
    string chrom_str;
    if (!(iss >> snp.chrom >> snp.ID >> snp.genpos >> snp.physpos >> snp.allele1 >> snp.allele2)) {
      cerr << "ERROR: Incorrectly formatted bim file: " << bimFile << endl;
      cerr << "Line " << ret.size() + 1 << ":" << endl;
      cerr << line << endl;
      cerr << "Unable to input 6 values (2 string, 1 double, 1 int, 2 string)" << endl;
      exit(1);
    }
    ret.push_back(snp);
  }
  fin.close();
  return ret;
}

void GenoBasis::readBedLine(uchar *genoLine, uchar *bedLineIn, FileUtils::SafeIfstream &fin) const {
  fin.read((char *) bedLineIn, (Nbed + 3) >> 2);
  const int bedToGeno[4] = {2, 9, 1, 0};
  for (uint64 nbed = 0; nbed < Nbed; nbed++)
    if (bedIndivToRemoveIndex[nbed] != -1) {
      int genoValue = bedToGeno[(bedLineIn[nbed >> 2] >> ((nbed & 3) << 1)) & 3];
      genoLine[bedIndivToRemoveIndex[nbed]] = (uchar) genoValue;
    }
}

double GenoBasis::computeAlleleFreq(const uchar genoLine[], const double subMaskIndivs[]) const {
  double sum = 0;
  int num = 0;
  for (size_t n = 0; n < N; n++)
    if (subMaskIndivs[n] && genoLine[n] != 9) {
      sum += genoLine[n];
      num++;
    }
  return 0.5 * sum / num;
}

double GenoBasis::computeMAF(const uchar *genoLine, const double *subMaskIndivs) const {
  double alleleFreq = computeAlleleFreq(genoLine, subMaskIndivs);
  return std::min(alleleFreq, 1.0 - alleleFreq);
}

double GenoBasis::computeSnpMissing(const uchar *genoLine, const double subMaskIndivs[]) const {
  double sum = 0;
  int num = 0;
  for (uint64 n = 0; n < N; n++)
    if (subMaskIndivs[n]) {
      sum += (genoLine[n] == 9);
      num++;
    }
  return sum / num;
}

void GenoBasis::buildLookupTable(double (*workTable)[4], const double lookupBedTable[4]) const {
  for (int byte4 = 0; byte4 < 256; byte4 += 4) {
    for (int k = 0; k < 4; k++) // fill 4 values for first of 4 consecutive bytes
      workTable[byte4][k] = lookupBedTable[(byte4 >> (k + k)) & 3];
    for (int k = 1; k < 4; k++) {
      memcpy(workTable[byte4 + k], workTable[byte4], sizeof(workTable[0]));
      workTable[byte4 + k][0] = lookupBedTable[k];
    }
  }
}

uint64 GenoBasis::getIndivInd(std::string &FID, std::string &IID) const {
  std::map<string, uint64>::const_iterator it = FID_IID_to_ind.find(FID + " " + IID);
  if (it != FID_IID_to_ind.end())
    return it->second;
  else
    return IND_MISSING;
}

const vector<SnpInfo> &GenoBasis::getSnpInfo() const {
  return snps;
}

void GenoBasis::updateNused() {
  Nused = 0;
  for (uint64 n = 0; n < N; n++)
    Nused += maskIndivs[n];
}

uint64 GenoBasis::getNpad() const {
  return Nstride;
}

uint64 GenoBasis::getM() const {
  return M;
}

uint64 GenoBasis::getNused() const {
  return Nused;
}

vector<double> GenoBasis::getFamPhenos() const {
  vector<double> phenos(N);
  for (uint64 n = 0; n < N; n++) phenos[n] = indivs[n].pheno;
  return phenos;
}

void GenoBasis::writeMaskSnps(uchar *out) const {
  memcpy(out, maskSnps, M * sizeof(maskSnps[0]));
}

void GenoBasis::writeMaskIndivs(double *out) const {
  memcpy(out, maskIndivs, Nstride * sizeof(maskIndivs[0]));
}

GenoBasis::~GenoBasis() {
  free(maskSnps);
}
}