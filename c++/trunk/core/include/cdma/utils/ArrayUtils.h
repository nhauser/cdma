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
#include "cdma/IObject.h"

namespace cdma
{

class CDMA_DECL ArrayUtils
{
public:
  /// constructor
  ArrayUtils( const ArrayPtr& array );

  /// destructor
  ~ArrayUtils();
  
  ArrayPtr getArray();

  /// Check if the shape matches with another Array object.
  /// 
  /// @param newArray
  ///            another Array object
  ///
  /// @return false if shapes do not match
  //
  bool checkShape(const ArrayPtr& newArray);
    
  /// Create a new Array using same backing store as this Array, by eliminating
  /// any dimensions with length one.
  /// 
  /// @return the new Array
  //
  ArrayUtilsPtr reduce();

  /// Create a new Array using same backing store as this Array, by eliminating
  /// the specified dimension.
  /// 
  /// @param dim
  ///            dimension to eliminate: must be of length one, else
  ///            Exception
  /// @return the new Array
  //
  ArrayUtilsPtr reduce(int dim);
    
  /// Reduce the array to at least certain rank. Only dimension with length of 1
  /// will be reduced.
  /// 
  /// @param rank
  ///            in int type
  /// @return GDM Array with the same storage Created on 10/11/2008
  //
  ArrayUtilsPtr reduceTo(int rank);
    
  /// Create a new Array, with the given shape, that references the same backing store as this Array.
  /// 
  /// @param shape
  ///            the new shape
  /// @return the new Array
  //
  ArrayUtilsPtr reshape(std::vector<int> shape) throw ( Exception );

  /// Create a new Array using same backing store as this Array, by fixing the
  /// specified dimension at the specified index value. This reduces rank by 1.
  /// 
  /// @param dim
  ///            which dimension to fix
  /// @param value
  ///            at what index value
  /// @return a new Array
  //
  ArrayUtilsPtr slice(int dim, int value);

  /// Create a new Array using same backing store as this Array, by transposing
  /// two of the indices.
  /// 
  /// @param dim1 transpose these two indices
  /// @param dim2 transpose these two indices
  /// @return the new Array
  //
  ArrayUtilsPtr transpose(int dim1, int dim2);
            
  /// Integrate on given dimension. The result array will be one dimensional
  /// reduced from the given array.
  /// 
  /// @param dimension
  ///            integer value
  /// @param isVariance
  ///            true if the array serves as variance
  /// @return new Array object
  /// @throw  Exception
  //TODO method imported from JAVA: its implementation is obscure to me... Please complete it if you understand it.
  //ArrayUtilsPtr integrateDimension(int dimension, bool isVariance) throw ( Exception );

  /// Create a new Array using same backing store as this Array, by flipping
  /// the index so that it runs from shape[index]-1 to 0.
  /// 
  /// @param dim dimension to flip
  /// @return the new Array
  //
  ArrayUtilsPtr flip(int dim);

  /// Create a new Array using same backing store as this Array, by permuting
  /// the indices.
  /// 
  /// @param dims
  ///            the old index dims[k] becomes the new kth index.
  /// @return the new Array
  //
  ArrayUtilsPtr permute(std::vector<int> dims);

private:
  ArrayPtr m_array;   // SharedPtr on array
};

} //namespace CDMACore
#endif //__CDMA_ARRAYUTILS_H__
