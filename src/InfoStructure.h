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

#ifndef LMMNET_INFOSTRUCTURE_H
#define LMMNET_INFOSTRUCTURE_H

#include <string>

// Snps information structure
struct SnpInfo {
  int chrom;
  std::string ID;
  double genpos;
  int physpos;
  std::string allele1, allele2;
  double MAF;
  int vcNum;
};

// Individual information structure
struct IndivInfo {
  std::string famID;
  std::string indivID;
  std::string paternalID;
  std::string maternalID;
  int sex; // (1=male; 2=female; other=unknown)
  double pheno;
};

#endif //LMMNET_INFOSTRUCTURE_H
