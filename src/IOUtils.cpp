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
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "IOUtils.h"

namespace FileUtils {

using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

void requireEmptyOrReadable(const std::string &file) {
  if (file.empty()) return;
  std::ifstream fin;
  fin.open(file.c_str());
  if (!fin) {
    cerr << "Error: Unable to open file: " << file << endl;
    exit(1);
  }
  fin.close();
}

void requireEachEmptyOrReadable(const std::vector<std::string> &fileList) {
  for (uint i = 0; i < fileList.size(); i++)
    requireEmptyOrReadable(fileList[i]);
}

void requireEmptyOrWriteable(const std::string &file) {
  if (file.empty()) return;
  std::ofstream fout;
  fout.open(file.c_str(), std::ios::out | std::ios::app);
  if (!fout) {
    cerr << "Error: Output file is not writeable: " << file << endl;
    exit(1);
  }
  fout.close();
}

// implement for SaveIfstream
void SafeIfstream::open(const std::string &file, std::ios_base::openmode mode) {
  fin.open(file.c_str(), mode);
  if (!fin) {
    cerr << "Error: Unable to open file: " << file << endl;
    exit(1);
  }
}

void SafeIfstream::close() {
  fin.close();
}

SafeIfstream::operator bool() const {
  return !fin.eof();
}

int SafeIfstream::get() {
  return fin.get();
}

SafeIfstream &SafeIfstream::read(char *s, std::streamsize n) {
  fin.read(s, n);
  return *this;
}

SafeIfstream &getline(SafeIfstream &in, std::string &s) {
  std::getline(in.fin, s);
  return in;
}

void SafeOfstream::open(const std::string &file, std::ios_base::openmode mode) {
  fout.open(file.c_str(), mode);
  if (!fout) {
    cerr << "ERROR: Unable to open file: " << file << endl;
    exit(1);
  }
}

void SafeOfstream::close() {
  fout.close();
}

SafeOfstream &SafeOfstream::operator<<(std::ostream &(*manip)(std::ostream &)) {
  manip(fout);
  return *this;
}
}