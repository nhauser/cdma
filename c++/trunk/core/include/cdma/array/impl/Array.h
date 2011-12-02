// *****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez Clément
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// *****************************************************************************

#ifndef __CDMA_ARRAY_H__
#define __CDMA_ARRAY_H__

#include <stdio.h>
#include <iostream>

#include <yat/utils/String.h>

#include <cdma/exception/Exception.h>
#include <cdma/array/IArray.h>
#include <cdma/math/IArrayMath.h>
#include <cdma/utils/IArrayUtils.h>
#include <cdma/array/impl/internal/ArrayData.h>
#include <cdma/array/impl/Index.h>

namespace cdma
{

//==============================================================================
/// Array default implementation
//==============================================================================
class Array : public cdma::IArray
{
private:
  yat::SharedPtr<Data, yat::Mutex> m_data;    // Memory storage of the matrix
  std::vector<int>                 m_shape;   // Shape of the matrix
  cdma::IIndexPtr                  m_index;   // Viewable part of the matrix
  yat::String                      m_factory; // name of the factory

public:
  ~Array();
  Array(const Array& src, IIndexPtr view);
  Array(const yat::String& plugin_id, const std::type_info& type, std::vector<int> shape, void* pData = NULL);
  template<typename T> Array(const yat::String& factory, T* values, std::vector<int> shape);

  //@{plugin methods
    
 //   static const type_info& detectType(NexusDataType type);
 //   static void* allocate(NexusDataType type, unsigned int length);
 
    //template<typename T> static void* allocate(T* values, unsigned int length);
    
  //@}

  //@{ IArray interface
  IArrayPtr copy();
  IArrayPtr copy(bool data);
  IArrayUtilsPtr getArrayUtils();
  IArrayMathPtr getArrayMath();
  yat::Any get(IIndexPtr& index);
  yat::Any get();
  void set(const IIndexPtr& ima, const yat::Any& value);
  const std::type_info& getElementType();
  IIndexPtr getIndex();
  IArrayIteratorPtr getIterator();
  int getRank();
  IArrayIteratorPtr getRegionIterator(std::vector<int> start, std::vector<int> shape) throw ( cdma::Exception );
  std::vector<int> getShape();
  long getSize();
  std::string shapeToString();
  void setIndex(const IIndexPtr& index);
  ISliceIteratorPtr getSliceIterator(int rank) throw ( cdma::Exception);
  void releaseStorage() throw ( Exception );
  long getRegisterId();
  void lock();
  void unlock();
  bool isDirty();
  void* getStorage() { return m_data->getStorage(); };
  //@} IArray interface
  //@{ IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::Array; };
  std::string getFactoryName() const { return m_factory; };
  //@}
};

}

#include "Array.hpp"
#endif // __CDMA_ARRAY_H__
