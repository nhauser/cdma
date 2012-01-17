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
#include <string>
#include <vector>

#include <cdma/exception/Exception.h>
#include <cdma/math/ArrayMath.h>
#include <cdma/utils/ArrayUtils.h>
#include <cdma/array/impl/ArrayStorage.h>
#include <cdma/array/View.h>

namespace cdma
{

//==============================================================================
/// Array default implementation
//==============================================================================
class Array
{
private:
  IArrayStoragePtr m_data_impl;    // Memory storage of the matrix
  std::vector<int> m_shape;   // Shape of the matrix
  ViewPtr          m_view;   // Viewable part of the matrix
  std::string      m_factory; // name of the factory

public:
  ~Array();
  Array( const Array& array );
  Array( const std::string& factory, const IArrayStoragePtr& data_ptr, const ViewPtr view );
  Array( const Array& src, ViewPtr view );
  Array( const ArrayPtr& src, ViewPtr view );
  Array( const std::string& plugin_id, const std::type_info& type, std::vector<int> shape, void* pData = NULL );
  template<typename T> Array(const std::string& factory, T* values, std::vector<int> shape);

  ArrayPtr copy(bool data);
  ArrayUtilsPtr getArrayUtils();
  ArrayMathPtr getArrayMath();
  yat::Any& get( const ViewPtr& ind, std::vector<int> position );
//  yat::Any& get();
  void set(const ViewPtr& ima, std::vector<int> position, const yat::Any& value);
  const std::type_info& getElementType();
  ViewPtr getView();
  ArrayIterator begin();
  ArrayIterator end();
  int getRank();
  ArrayPtr getRegion(std::vector<int> start, std::vector<int> shape) throw ( cdma::Exception );
  std::vector<int> getShape();
  long getSize();
  void setView(const ViewPtr& view);
  SliceIteratorPtr getSliceIterator(int rank) throw ( cdma::Exception);
  bool dirty();
  const IArrayStoragePtr& getStorage() { return m_data_impl; };

  //@{ IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::Array; };
  std::string getFactoryName() const { return m_factory; };
  //@}
};

}

  

#include "Array.hpp"
#endif // __CDMA_ARRAY_H__
