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

#ifndef LMMNET__GENOTYPEBASIS_H_
#define LMMNET__GENOTYPEBASIS_H_

#include <vector>
#include <string>
#include <map>
#include <boost/utility.hpp>

#include "TypeDef.h"
#include "IOUtils.h"
#include "InfoStructure.h"

namespace LMMNET {
class GenoBasis {
 public:
  static const uint64 IND_MISSING;

 protected:
  uint64 Mbed, Nbed; // PLINK data dimensions
  std::vector<int> bedIndivToRemoveIndex; // index of removing individuals

  uint64 M, N, Nstride; // PLINK data dimensions after data processing
  uint64 Nused;

  uchar *maskSnps; // mask of snps failed QC
  double *maskIndivs; // mask of indivs failed QC
  uint64 numIndivsQC; // number of indivs remaining after QC

  std::map<std::string, uint64> FID_IID_to_ind; // map FID+IID to ind

  double maxMissingPerIndiv; // max missing rate per individual
  double maxMissingPerSnp; // max missing rate per snp

  // path for bim and bed files
  std::vector<std::string> bimFiles;
  std::vector<std::string> bedFiles;

  // container for indivs and snps information
  std::vector<SnpInfo> snps;
  std::vector<IndivInfo> indivs;

  /**
   * Store the data into bedLineOut memory with binary format
   * Convert the {0, 1, 2, 9} to {00, 01, 10, 11}
   * 
   * @param bedLineOut destination address contains the current data line
   * @param genoLine original address contains the current data line
  */
  void storeBedLine(uchar bedLineOut[], const uchar genoLine[]);
  /**
   * Process the individual information from readed file
   * 1. check the format of provided file and read data
   * 2. set mask for unused individuals
   * 3. pad the number of individuals to multiplier of 4 (for storing in binary format)
   * 
   * @param famFile the path of Fam file
   * @param removeIndivsFile the path of file which contains the index of removing individuals
  */
  void processIndivs(const std::string &famFile, const std::vector<std::string> &removeIndivsFiles);

 public:
  GenoBasis(const std::string &famFile, const std::vector<std::string> &bimFiles,
            const std::vector<std::string> &bedFiles,
            const std::vector<std::string> &removeSNPsFiles,
            const std::vector<std::string> &removeIndivsFiles, double _maxMissingPerSnp,
            double _maxMissingPerIndiv);

  GenoBasis(const GenoBasis &) = delete; // disable copy constructor
  ~GenoBasis();
  /**
   * Read bimfile from disk and process the contained information
   * 
   * @param bimFile the path of Bim file
   * @return The vector contains the SNP information
  */
  static std::vector<SnpInfo> readBimFile(const std::string &bimFile);

  /**
   * Read the SNPs data from disk and convert the binary formate to {0, 1, 2, 9}
   * Set the SNPs of removed individuals as zero
   * 
   * @param genoLine the binary format of genetic data
   * @param bedLineIn the converted format of genetic data ({0, 1, 2, 9})
   * @param fin the filestream binded with SNPs file
   * @param loadGenoLine whether load current genoline into memory
  */
  void readBedLine(uchar genoLine[], uchar bedLineIn[], FileUtils::SafeIfstream &fin, bool loadGenoLine=true) const;

  /**
   * Compute the allele frequency without masked individuals
   * 
   * @param genoLine contains one SNP data
   * @param subMaskIndivis the mask of removed individuals
   * @return allele frequency 
  */
  double computeAlleleFreq(const uchar genoLine[], const double subMaskIndivs[]) const;

  /**
   * Compute the minor allele frequency without masked individuals
   * 
   * @param genoLine contains one SNP data 
   * @param subMaskIndivis the mask of removed individuals 
   * @return minor allele frequency
  */
  double computeMAF(const uchar genoLine[], const double subMaskIndivs[]) const;

  /**
   * Compute the rate of missing SNPs without masked individuals
   * 
   * @param genoLine contains one SNP data
   * @param subMaskIndivis the mask of removed individuals 
   * @return SNP missing rate 
  */
  double computeSnpMissing(const uchar genoLine[], const double subMaskIndivs[]) const;

  /**
   * Build the look up table for decoding genotype data from binary format
   * to {0, 1, 2, missing}
   * 
   * @param workTable decode table of size 256 x 4 which contains the permutation in 8 bits
   * @param lookupBedTable the reference table of size 4 x 1 which contains the normalized SNPs
  */
  void buildLookupTable(double (*workTable)[4], const double lookupBedTable[4]) const;

  // interface of get and set
  uint64 getIndivInd(std::string &FID, std::string &IID) const;
  const std::vector<SnpInfo> &getSnpInfo() const;
  void updateNused();

  uint64 getNpad() const;
  uint64 getM() const;
  uint64 getNused() const;
  uint64 getN() const;
  std::vector<double> getFamPhenos() const;
  // output individual and snps masks to file
  void writeMaskSnps(uchar out[]) const;
  void writeMaskIndivs(double out[]) const;
};
}

#endif //LMMNET__GENOTYPEBASIS_H_
