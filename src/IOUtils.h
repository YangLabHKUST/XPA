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

#ifndef LMMNET_IOUTILS_H
#define LMMNET_IOUTILS_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

namespace FileUtils {

void requireEmptyOrReadable(const std::string &file);

void requireEachEmptyOrReadable(const std::vector<std::string> &fileList);

void requireEmptyOrWriteable(const std::string &file);

class SafeIfstream {
  std::ifstream fin;

 public:
  void open(const std::string &file, std::ios_base::openmode mode = std::ios::in);
  void close();
  operator bool() const;
  SafeIfstream &read(char *s, std::streamsize n);
  int get();
  friend SafeIfstream &getline(SafeIfstream &in, std::string &s);
};

SafeIfstream &getline(SafeIfstream &in, std::string &s);

class SafeOfstream {
  std::ofstream fout;

 public:
  void open(const std::string &file, std::ios_base::openmode mode = std::ios::out);
  void close();

  template<class T>
  SafeOfstream &operator<<(const T &x) {
    fout << x;
    if (fout.fail()) {
      std::cerr << "ERROR: File write failed" << std::endl;
      exit(1);
    }
    return *this;
  }

  SafeOfstream &operator<<(std::ostream &(*manip)(std::ostream &));
};
}

#endif //LMMNET_IOUTILS_H
