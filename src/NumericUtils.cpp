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

#include <cmath>
#include <cstdlib>
#include <utility>
#include <algorithm>

#include "NumericUtils.h"

namespace NumericUtils {

double sum(const double x[], uint64 N) {
  double ans = 0;
  for (uint64 n = 0; n < N; n++)
    ans += x[n];
  return ans;
}

double mean(const std::vector<double> &x) {
  uint64 N = x.size();
  return sum(&x[0], N) / N;
}

double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  int n = x.size();
  return (x[n / 2] + x[(n - 1) / 2]) / 2;
}

// takes into account that some 0 values may indicate missing/ignored: divide out by Nused, not N
double mean(const double x[], uint64 N, uint64 Nused) {
  return sum(x, N) / Nused;
}

double dot(const double x[], const double y[], uint64 N) {
  double ans = 0;
  for (uint64 n = 0; n < N; n++)
    ans += x[n] * y[n];
  return ans;
}

double norm2(const double x[], uint64 N) {
  double ans = 0;
  for (uint64 n = 0; n < N; n++)
    ans += sq(x[n]);
  return ans;
}

void normalize(double x[], uint64 N) {
  double scale = 1.0 / sqrt(norm2(x, N));
  for (uint64 n = 0; n < N; n++)
    x[n] *= scale;
}

void divideElem(double *vec1, const double *vec2, uint64 N) {
//#pragma omp parallel for
  for (uint64 n = 0; n < N; n++) {
    vec1[n] /= vec2[n];
  }
}

void multipElem(double *vec1, const double *vec2, uint64 N) {
//#pragma omp parallel for
  for (uint64 n = 0; n < N; n++) {
    vec1[n] *= vec2[n];
  }
}

void subElem(double *vec1, const double *vec2, uint64 N) {
  for (uint64 n = 0; n < N; n++) {
    vec1[n] -= vec2[n];
  }
}

void sumElem(double *vec1, const double *vec2, uint64 N) {
  for (uint64 n = 0; n < N; n++) {
    vec1[n] += vec2[n];
  }
}

void scaleElem(double *vec1, const double alpha, uint64 N) {
  for (uint64 n = 0; n < N; n++) {
    vec1[n] *= alpha;
  }
}

std::pair<double, double> meanStdDev(const double x[], uint64 N) {
  double mu = 0, s2 = 0;
  for (uint64 n = 0; n < N; n++) mu += x[n];
  mu /= N;
  for (uint64 n = 0; n < N; n++) s2 += sq(x[n] - mu);
  s2 /= (N - 1);
  return std::make_pair(mu, sqrt(s2));
}

std::pair<double, double> meanStdErr(const double x[], uint64 N) {
  std::pair<double, double> ret = meanStdDev(x, N);
  ret.second /= sqrt((double) N);
  return ret;
}

std::pair<double, double> meanStdDev(const std::vector<double> &x) {
  return meanStdDev(&x[0], x.size());
}

std::pair<double, double> meanStdErr(const std::vector<double> &x) {
  return meanStdErr(&x[0], x.size());
}
}