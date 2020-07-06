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

#ifndef LMMNET_DATAUTILS_H
#define LMMNET_DATAUTILS_H

#include <vector>
#include <string>
#include <map>
#include <boost/utility.hpp>

#include "TypeDef.h"
#include "IOUtils.h"
#include "InfoStructure.h"
#include "GenotypeBasis.h"

namespace LMMNET {
class GenoData : public GenoBasis {

 private:
  uchar *genotypes; // M x Nstride / 4 genotype data

  std::map<std::string, uint64> snpID_position; // to store the snpID and position
  std::vector<std::string> modelSnpsFiles;

  /**
   * Read SNPs information from Bim files and set mask for removed SNPs
   * 
   * @param Mfiles the vector contains number of SNPs from different Bim files (mainly for distributed computation)
   * @param bbimFiles the path of Bim Files
   * @param removeSnpsFiles the path of index file which contains the removed index of SNPs
   * @return the vector contains SNPs information
  */
  std::vector<SnpInfo> processSnps(std::vector<uint64> &Mfiles,
                                   const std::vector<std::string> &bimFiles,
                                   const std::vector<std::string> &removeSNPsFiles);

 public:
  GenoData(const std::string &_famFile, const std::vector<std::string> &_bimFiles,
           const std::vector<std::string> &_bedFiles,
           const std::vector<std::string> &_removeSNPsFiles,
           const std::vector<std::string> &_removeIndivsFiles,
           const std::vector<std::string> &_modelSnpsFiles,
           double _maxMissingPerSnp,
           double _maxMissingPerIndiv);

  GenoData(const GenoData &) = delete; // disable copy constructor
  ~GenoData();

  /**
   * Decode the SNPs from binary format to normalized double-precision floating number 
   * and SNPs of masked individuals will be zero
   * 
   * @param out output of decoded SNPs data
   * @param maskIndivs the index of masked individuals
   * @param m index of current decoded SNPs
   * @param map0129 reference table of size 4 x 1 to store the normalized version of {0, 1, 2, 9}
   * @param workTable work table of size 256 x 1 to store the all permutations in 8 bits (4 SNPs)
  */
  void decodeSnpsVector(double out[], const double maskIndivs[], uint64 m,
                        const double map0129[4], double (*workTable)[4]) const;
  const std::map<std::string, uint64> &getsnpRef() const;
};
}
#endif //LMMNET_DATAUTILS_H
