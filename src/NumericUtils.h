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

#ifndef LMMNET_NUMERICUTILS_H
#define LMMNET_NUMERICUTILS_H

#include <cstdlib>
#include <vector>
#include <utility>

#include "TypeDef.h"

namespace NumericUtils {

inline double sq(double x) { return x * x; }
double sum(const double x[], uint64 N);
double mean(const std::vector<double> &x);
double median(std::vector<double> x);

// takes into account that some 0 values may indicate missing/ignored: divide out by Nused, not N
double mean(const double x[], uint64 N, uint64 Nused);

double dot(const double x[], const double y[], uint64 N);
double norm2(const double x[], uint64 N);
void normalize(double x[], uint64 N);

void divideElem(double *vec1, const double *vec2, uint64 N);
void multipElem(double *vec1, const double *vec2, uint64 N);
void subElem(double *vec1, const double *vec2, uint64 N);
void scaleElem(double *vec1, const double alpha, uint64 N);
void sumElem(double *vec1, const double *vec2, uint64 N);

std::pair<double, double> meanStdDev(const double x[], uint64 N);
std::pair<double, double> meanStdErr(const double x[], uint64 N);
std::pair<double, double> meanStdDev(const std::vector<double> &x);
std::pair<double, double> meanStdErr(const std::vector<double> &x);
}

#endif //LMMNET_NUMERICUTILS_H
