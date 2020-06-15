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

#ifndef LMMNET_MEMORYUTILS_H
#define LMMNET_MEMORYUTILS_H

#include <mkl.h>

#include "TypeDef.h"

#define MEM_ALIGNMENT 64

void *ALLOCATE_MEMORY(uint64 size);
void *ALIGN_ALLOCATE_MEMORY(uint64 size);

#define ALLOCATE_DOUBLES(numDoubles) (double *) ALLOCATE_MEMORY(numDoubles*sizeof(double))
#define ALLOCATE_FLOATS(numFloats) (float *) ALLOCATE_MEMORY(numFloats*sizeof(float))
#define ALLOCATE_UCHARS(numUchars) (uchar *) ALLOCATE_MEMORY(numUchars*sizeof(uchar))

#define ALIGN_ALLOCATE_DOUBLES(numDoubles) (double *) ALIGN_ALLOCATE_MEMORY(numDoubles*sizeof(double))
#define ALIGN_ALLOCATE_FLOATS(numFloats) (float *) ALIGN_ALLOCATE_MEMORY(numFloats*sizeof(float))
#define ALIGN_ALLOCATE_UCHARS(numUchars) (uchar *) ALIGN_ALLOCATE_MEMORY(numUchars*sizeof(uchar))

#define ALIGN_FREE mkl_free

#endif //LMMNET_MEMORYUTILS_H
