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

#include <iostream>
#include <stdlib.h>

#include "MemoryUtils.h"
#include "TypeDef.h"

void *ALLOCATE_MEMORY(uint64 size) {
  void *p = malloc(size);

  if (p == NULL) {
    std::cerr << "ERROR: Failed to allocate " << size << " bytes" << std::endl;
    exit(1);
  }

  return p;
}

void *ALIGN_ALLOCATE_MEMORY(uint64 size) {
  void *p = mkl_malloc(size, MEM_ALIGNMENT);

  if (p == NULL) {
    std::cerr << "ERROR: Failed to allocate " << size << " bytes" << std::endl;
    exit(1);
  }

  return p;
}