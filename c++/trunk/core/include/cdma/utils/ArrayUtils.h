//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_ARRAYUTILS_H__
#define __CDMA_ARRAYUTILS_H__

#include <list>
#include <vector>

#include "cdma/exception/Exception.h"
#include "cdma/array/Array.h"
#include "cdma/array/Range.h"

namespace cdma
{

class CDMA_DECL ArrayUtils 
{
public:
  /// destructor
  ~ArrayUtils() {};
  
  Array& getArray();
    
  /// Copy the contents of this array to another array. The two arrays must
  /// have the same size.
  /// 
  /// @param newArray
  ///            an existing array
  /// @throw  Exception
  ///             wrong shape
  ///
  void copyTo(const Array& newArray) throw ( Exception );

  /// Check if the shape matches with another Array object.
  /// 
  /// @param newArray
  ///            another Array object
  /// @throw  Exception
  ///             shape not match
  ///
  void checkShape(const Array& newArray) throw ( Exception );
    
  /// Concatenate with another array. The array need to be equal of less in
  /// rank.
  /// 
  /// @param array
  ///            Array object
  /// @return new Array
  /// @throw  Exception
  ///             mismatching shape
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> concatenate(const Array& array) throw ( Exception );

  /// Create a new Array using same backing store as this Array, by eliminating
  /// any dimensions with length one.
  /// 
  /// @return the new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> reduce();

  /// Create a new Array using same backing store as this Array, by eliminating
  /// the specified dimension.
  /// 
  /// @param dim
  ///            dimension to eliminate: must be of length one, else
  ///            Exception
  /// @return the new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> reduce(int dim);
    
  /// Reduce the array to at least certain rank. The dimension with only 1 bin
  /// will be reduced.
  /// 
  /// @param rank
  ///            in int type
  /// @return GDM Array with the same storage Created on 10/11/2008
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> reduceTo(int rank);
    
  /// Create a new Array, with the given shape, that references the same backing store as this Array.
  /// 
  /// @param shape
  ///            the new shape
  /// @return the new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> reshape(std::vector<int> shape) throw ( Exception );

    
  /// Create a new Array as a subsection of this Array, with rank reduction. No
  /// data is moved, so the new Array references the same backing store as the
  /// original.
  /// <p>
  /// 
  /// @param origin
  ///            int array specifying the starting index. Must be same rank as
  ///            original Array.
  /// @param shape
  ///            int array specifying the extents in each dimension. This
  ///            becomes the shape of the returned Array. Must be same rank as
  ///            original Array. If shape[dim] == 1, then the rank of the
  ///            resulting Array is reduced at that dimension.
  /// @return Array object
  /// @throw  Exception
  ///             invalid range
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> section(const std::vector<int> origin, const std::vector<int> shape) throw ( Exception );
    
  /// Create a new Array as a subsection of this Array, with rank reduction. No
  /// data is moved, so the new Array references the same backing store as the
  /// original.
  /// 
  /// @param origin
  ///            int array specifying the starting index. Must be same rank as
  ///            original Array.
  /// @param shape
  ///            int array specifying the extents in each dimension. This
  ///            becomes the shape of the returned Array. Must be same rank as
  ///            original Array. If shape[dim] == 1, then the rank of the
  ///            resulting Array is reduced at that dimension.
  /// @param stride
  ///            int array specifying the strides in each dimension. If null,
  ///            assume all ones.
  /// @return the new Array
  /// @throw  Exception
  ///             invalid range
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> section(std::vector<int> origin, std::vector<int> shape, std::vector<int> stride) throw ( Exception );

  /// Create a new ArrayUtils as a subsection of this Array, without rank reduction.
  /// No data is moved, so the new Array references the same backing store as
  /// the original.
  /// 
  /// @param origin
  ///            int array specifying the starting index. Must be same rank as
  ///            original Array.
  /// @param shape
  ///            int array specifying the extents in each dimension. This
  ///            becomes the shape of the returned Array. Must be same rank as
  ///            original Array.
  /// @param stride
  ///            int array specifying the strides in each dimension. If null,
  ///            assume all ones.
  /// @return the new Array
  /// @throw  Exception
  ///             invalid range
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> sectionNoReduce(std::vector<int> origin, std::vector<int> shape, std::vector<int> stride) throw ( Exception );

  /// Create a new ArrayUtils as a subsection of this Array, without rank reduction.
  /// No data is moved, so the new Array references the same backing store as
  /// the original.
  /// 
  /// @param ranges
  ///            list of Ranges that specify the array subset. Must be same
  ///            rank as original Array. A particular Range: 1) may be a
  ///            subset, or 2) may be null, meaning use entire Range.
  /// @return the new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> sectionNoReduce(std::list<Range> ranges) throw ( Exception );

  /// Create a new Array using same backing store as this Array, by fixing the
  /// specified dimension at the specified index value. This reduces rank by 1.
  /// 
  /// @param dim
  ///            which dimension to fix
  /// @param value
  ///            at what index value
  /// @return a new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> slice(int dim, int value);

  /// Create a new Array using same backing store as this Array, by transposing
  /// two of the indices.
  /// 
  /// @param dim1
  ///            transpose these two indices
  /// @param dim2
  ///            transpose these two indices
  /// @return the new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> transpose(int dim1, int dim2);
            
  /// Check if the two arrays are conformable. They must have exactly the same
  /// shape (excluding dimensions of length 1)
  /// 
  /// @param array
  ///            in Array type
  /// @return Array with new storage Created on 14/07/2008
  ///
  bool isConformable(Array& array);

  /// Element-wise apply a bool map to the array. The values of the Array
  /// will get updated. The map's rank must be smaller or equal to the rank of
  /// the array. If the rank of the map is smaller, apply the map to subset of
  /// the array in the lowest dimensions iteratively. For each element, if the
  /// AND map value is true, return itself, otherwise return NaN.
  /// 
  /// @param boolMap
  ///            bool Array
  /// @return Array itself
  /// @throw  Exception
  ///             Created on 04/08/2008
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> eltAnd(Array& boolMap) throw ( Exception );

  /// Integrate on given dimension. The result array will be one dimensional
  /// reduced from the given array.
  /// 
  /// @param dimension
  ///            integer value
  /// @param isVariance
  ///            true if the array serves as variance
  /// @return new Array object
  /// @throw  Exception
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> integrateDimension(int dimension, bool isVariance) throw ( Exception );
    
  /// Integrate on given dimension. The result array will be one dimensional
  /// reduced from the given array.
  /// 
  /// @param dimension   integer value
  /// @param isVariance  true if the array serves as variance
  /// @return new Array object
  /// @throw  Exception
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> enclosedIntegrateDimension(int dimension, bool isVariance) throw ( Exception );

  /// Create a new Array using same backing store as this Array, by flipping
  /// the index so that it runs from shape[index]-1 to 0.
  /// 
  /// @param dim dimension to flip
  /// @return the new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> flip(int dim);

  /// Create a new Array using same backing store as this Array, by permuting
  /// the indices.
  /// 
  /// @param dims
  ///            the old index dims[k] becomes the new kth index.
  /// @return the new Array
  ///
  yat::SharedPtr<ArrayUtils, yat::Mutex> permute(std::vector<int> dims);
};

} //namespace CDMACore
#endif //__CDMA_ARRAYUTILS_H__
