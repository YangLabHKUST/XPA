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

#ifndef LMMNET_PHENOBASIS_H
#define LMMNET_PHENOBASIS_H

#include "dataBasis.h"

namespace LMMNET {
template<typename T>
class PhenoBasis : public dataBasis<T> {
 private:

  std::vector<std::vector<double> > phenotype_dbl; // store the double type phenotype data

  bool phenoUseFam; // whether use fam as the phenotype data
  void initialize(const std::vector<std::string> &phenoCols, std::vector<double> &_maskIndivs);

 public:
  PhenoBasis(const std::string &phenoFile, const T &_genoData,
             const std::vector<std::string> &phenoCols, std::vector<double> &_maskIndivs, bool _phenoUseFam);
  PhenoBasis(const PhenoBasis &) = delete;

  void padPhenodbl(); // pad phenotype to fit the pad genotype data
  std::vector<std::vector<double> > getPhenodbl() const;
};
}

#include "PhenoBasis_impl.h"
#endif //LMMNET_PHENOBASIS_H
