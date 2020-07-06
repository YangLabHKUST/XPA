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

#ifndef LMMNET__AUXGENODATA_H_
#define LMMNET__AUXGENODATA_H_

#include <vector>
#include <string>
#include <map>
#include <boost/utility.hpp>

#include "TypeDef.h"
#include "IOUtils.h"
#include "InfoStructure.h"
#include "GenotypeBasis.h"

namespace LMMNET {
class AuxGenoData : public GenoBasis {
 private:
  uchar *genotypes; // M x Nstride / 4 genotype data

  std::map<std::string, uint64> snpID_position; // to store the snpID and position

  std::vector<uint64> snpIndex; // map the original snp position to the reference snp position

  std::vector<SnpInfo> snpOrderInfo;

  std::vector<std::string> modelSnpsFiles; // store the path of modelSnpsFiles

  std::vector<SnpInfo> processSnps(std::vector<uint64> &Mfiles,
                                   const std::vector<std::string> &bimFiles);
 public:
  AuxGenoData(const std::string &_famFile, const std::vector<std::string> &_bimFiles,
              const std::vector<std::string> &_bedFiles,
              const std::vector<std::string> &_removeIndivsFiles, double _maxMissingPerSnp,
              double _maxMissingPerIndiv, const std::map<std::string, uint64> &_snpRef);

  AuxGenoData(const AuxGenoData &) = delete; // disable copy constructor
  ~AuxGenoData();

  void decodeSnpsVector(double out[], const double maskIndivs[], uint64 m,
                        const double map0129[4], double (*workTable)[4]) const;

  std::vector<uint64> getSnpIndex() const;
  const std::vector<SnpInfo> &getSnpFilterInfo() const;
};
}

#endif //LMMNET__AUXGENODATA_H_
