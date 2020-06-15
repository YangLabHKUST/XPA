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

namespace LMMNET {

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using FileUtils::getline;

vector<SnpInfo> GenoData::processSnps(std::vector<uint64> &Mfiles, const std::vector<std::string> &bimFiles,
                                      const std::vector<std::string> &removeSNPsFiles) {
  FileUtils::SafeIfstream fin;
  string line;

  vector<SnpInfo> bedSnps;
  // read bim files
  for (uint i = 0; i < bimFiles.size(); i++) {
    cout << "Reading bim file #" << (i + 1) << ": " << bimFiles[i] << endl;
    vector<SnpInfo> snps_i = readBimFile(bimFiles[i]);
    bedSnps.insert(bedSnps.end(), snps_i.begin(), snps_i.end());
    Mfiles.push_back(snps_i.size());
    cout << "    Read " << Mfiles.back() << " snps" << endl;
  }
  Mbed = bedSnps.size();

  cout << "Total snps in PLINK data: Mbed = " << Mbed << endl;

  // marker for storing snps
  const int excludeVCnum = -1;
  const int defaultVCnum = 1;

  vector<int> snpVCnum(Mbed, (char) defaultVCnum);
  // check the duplicate snps
  std::set<string> rsIDs;
  for (uint64 mbed = 0; mbed < Mbed; mbed++) {
    if (rsIDs.find(bedSnps[mbed].ID) != rsIDs.end()) {
      cerr << "WARNING: Duplicate snp ID " << bedSnps[mbed].ID
           << " -- masking duplicate" << endl;
      snpVCnum[mbed] = excludeVCnum;
    } else
      rsIDs.insert(bedSnps[mbed].ID);
  }

  if (!removeSNPsFiles.empty()) {
    // create dictionary rsID -> index in full bed snp list
    std::map<string, uint64> rsID_to_ind;
    for (uint64 mbed = 0; mbed < Mbed; mbed++)
      if (rsID_to_ind.find(bedSnps[mbed].ID) == rsID_to_ind.end()) // only use first of dupe IDs
        rsID_to_ind[bedSnps[mbed].ID] = mbed;

    // exclude snps
    for (uint f = 0; f < removeSNPsFiles.size(); f++) {
      const string &excludeFile = removeSNPsFiles[f];
      cout << "Reading exclude file (SNPs to exclude): " << excludeFile << endl;
      fin.open(excludeFile);
      int numExcluded = 0;
      int numAbsent = 0;
      while (getline(fin, line)) {
        std::istringstream iss(line);
        string rsID;
        iss >> rsID;
        if (rsID_to_ind.find(rsID) == rsID_to_ind.end()) {
          if (numAbsent < 5)
            cerr << "WARNING: Unable to find SNP to exclude: " << rsID << endl;
          numAbsent++;
        } else if (snpVCnum[rsID_to_ind[rsID]] != excludeVCnum) {
          snpVCnum[rsID_to_ind[rsID]] = excludeVCnum;
          numExcluded++;
        }
      }
      fin.close();
      cout << "Excluded " << numExcluded << " SNP(s)" << endl;
      if (numAbsent)
        cerr << "WARNING: " << numAbsent << " SNP(s) not found in data set" << endl;
    }
  }

  // determine number of snps remaining post-exclusion and set up index
  M = 0; // note: M will be updated later after further QC
  uint64 Mexclude = 0;
  for (uint64 mbed = 0; mbed < Mbed; mbed++) {
    int vcNum = snpVCnum[mbed];
    bedSnps[mbed].vcNum = vcNum;
    if (vcNum == defaultVCnum) {
      M++;
    } else if (vcNum == excludeVCnum) {
      Mexclude++;
    }
  }

  cout << endl;
  cout << "Breakdown of SNP pre-filtering results:" << endl;
  cout << "  " << M << " SNPs to include in model " << endl;
  cout << "  " << Mexclude << " excluded SNPs" << endl;

//        M = Mbed;
  return bedSnps;
}

GenoData::GenoData(const std::string &_famFile, const std::vector<std::string> &_bimFiles,
                   const std::vector<std::string> &_bedFiles,
                   const std::vector<std::string> &_removeSNPsFiles,
                   const std::vector<std::string> &_removeIndivsFiles, double _maxMissingPerSnp,
                   double _maxMissingPerIndiv) : GenoBasis(_famFile, _bimFiles, _bedFiles, _removeSNPsFiles,
                                                           _removeIndivsFiles, _maxMissingPerSnp, _maxMissingPerIndiv) {

  // process snp file and update the snpID_position map
  vector<uint64> Mfiles;
  vector<SnpInfo> bedSnps = processSnps(Mfiles, bimFiles, _removeSNPsFiles);

  // allocate memory for row genotype data
  cout << "Allocating " << M << " x " << Nstride << "/4 bytes to store genotypes" << endl;
  genotypes = ALLOCATE_UCHARS(M * Nstride / 4); // note: M will be reduced after QC
  numIndivsQC = N;

  cout << "Reading genotypes and performing QC filtering on snps and indivs..." << endl;

  vector<int> numMissingPerIndiv(N);
  uchar *bedLineOut = genotypes;
  uint64 mbed = 0;

  for (uint i = 0; i < bedFiles.size(); i++) {
    if (Mfiles[i] == 0) continue;
    uint64 bytesInFile = Mfiles[i] * (uint64) ((Nbed + 3) >> 2);
    cout << "Reading bed file #" << (i + 1) << ": " << bedFiles[i] << endl;
    cout << "    Expecting " << bytesInFile << " (+3) bytes for "
         << Nbed << " indivs, " << Mfiles[i] << " snps" << endl;
    FileUtils::SafeIfstream fin;
    fin.open(bedFiles[i], std::ios::in | std::ios::binary);
    uchar header[3];
    fin.read((char *) header, 3);
    if (!fin || header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01) {
      cerr << "ERROR: Incorrect first three bytes of bed file: " << bedFiles[i] << endl;
      exit(1);
    }

    // read genotypes
    uchar *genoLine = ALLOCATE_UCHARS(N);
    uchar *bedLineIn = ALLOCATE_UCHARS((Nbed + 3) >> 2);
    int numSnpsFailedQC = 0;
    for (uint64 mfile = 0, goodSnp = 0; mfile < Mfiles[i]; mfile++, mbed++) {
      readBedLine(genoLine, bedLineIn, fin);
      if (bedSnps[mbed].vcNum != -1) { // if the snp is not excluded
        double snpMissing = computeSnpMissing(genoLine, maskIndivs);
        bool snpPassQC = snpMissing <= maxMissingPerSnp;
        if (snpPassQC) {
          storeBedLine(bedLineOut, genoLine);
          bedLineOut += Nstride >> 2;
          snps.push_back(bedSnps[mbed]);
          snps.back().MAF = computeMAF(genoLine, maskIndivs);
          // update indiv QC info
          for (uint64 n = 0; n < N; n++) {
            if (genoLine[n] == 9) {
              numMissingPerIndiv[n]++;
            }
          }

          // extract the good snps information (pass the QC)
          snpID_position[bedSnps[mbed].ID] = goodSnp;
          goodSnp++;
        } else {
          if (numSnpsFailedQC < 5)
            cout << "Filtering snp " << bedSnps[mbed].ID << ": "
                 << snpMissing << " missing" << endl;
          numSnpsFailedQC++;
        }
      }
    }
    free(genoLine);
    free(bedLineIn);

    if (numSnpsFailedQC)
      cout << "Filtered " << numSnpsFailedQC << " SNPs with > " << maxMissingPerSnp << " missing"
           << endl;

    if (!fin || fin.get() != EOF) {
      cerr << "ERROR: Wrong file size or reading error for bed file: "
           << bedFiles[i] << endl;
      exit(1);
    }
    fin.close();
  }

  M = snps.size();

  maskSnps = ALLOCATE_UCHARS(M);
  memset(maskSnps, 1, M * sizeof(maskSnps[0]));

  // QC of indivs for missing
  int numIndivsFailedQC = 0;
  for (uint64 n = 0; n < N; n++)
    if (maskIndivs[n] && numMissingPerIndiv[n] > maxMissingPerIndiv * M) {
      maskIndivs[n] = 0;
      numIndivsQC--;
      if (numIndivsFailedQC < 5)
        cout << "Filtering indiv " << indivs[n].famID << " " << indivs[n].indivID << ": "
             << numMissingPerIndiv[n] << "/" << M << " missing" << endl;
      numIndivsFailedQC++;
    }
  if (numIndivsFailedQC)
    cout << "Filtered " << numIndivsFailedQC << " indivs with > " << maxMissingPerIndiv
         << " missing" << endl;

  // update number of used indivs, here we assume that we do not change this value with respect to
  // covar. TODO: Add consideration for the missing indvi in covariate
  updateNused();

  cout << "Total indivs after QC: " << numIndivsQC << endl;
  cout << "Total post-QC SNPs: M = " << M << endl;

}

void GenoData::decodeSnpsVector(double *out, const double *submaskIndivs, uint64 m,
                                const double *map0129, double(*workTable)[4]) const {
  double lookBedTable[4] = {map0129[2], map0129[3], map0129[1], map0129[0]};
  buildLookupTable(workTable, lookBedTable);

  uchar *currGenotype = genotypes + m * (Nstride >> 2); // point to the current genotype
  for (uint64 n4 = 0; n4 < Nstride; n4 += 4) { // decode snps vector and add mask by sse2
    __m128d x01 = _mm_load_pd(&workTable[*currGenotype][0]);
    __m128d x23 = _mm_load_pd(&workTable[*currGenotype][2]);
    __m128d mask01 = _mm_load_pd(&submaskIndivs[n4]);
    __m128d mask23 = _mm_load_pd(&submaskIndivs[n4 + 2]);
    _mm_store_pd(&out[n4], _mm_mul_pd(x01, mask01));
    _mm_store_pd(&out[n4 + 2], _mm_mul_pd(x23, mask23));
    currGenotype++;
  }
}

GenoData::~GenoData() {
  free(genotypes);
}

const std::map<std::string, uint64> &GenoData::getsnpRef() const {
  return snpID_position;
}
}