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
#ifndef __CDMA_INDEX_H__
#define __CDMA_INDEX_H__

#include <string>
#include <vector>
#include <list>

#include <yat/utils/String.h>

#include <cdma/array/IIndex.h>
#include <cdma/array/impl/Range.h>

namespace cdma
{

//==============================================================================
/// Index default implementation
/// Indexes for Multidimensional Arrays. An Index refers to a particular
/// element of an Array.
//==============================================================================
class Index : public cdma::IIndex
{
private:
  int                m_rank;       // Rank of the index
  std::vector<int>   m_vCurPos;    // Current position pointed by this index
  std::vector<Range> m_pRanges;    // Ranges that constitute the index global view in each dimension
  bool               m_upToDate;   // Does the overall shape has changed
  int                m_lastIndex;
  yat::String        m_factory;    // Name of the factory

public:
  // Constructors
  Index(cdma::IIndexPtr index );
  Index(const std::string factoryName, int rank, int shape[], int start[]);
  ~Index();

  //@{ IIndex interface
  int                  getRank();
  std::vector<int>     getShape();
  std::vector<int>     getOrigin();
  long                 getSize();
  std::vector<int>     getStride();
  int                  currentElement();
  int                  lastElement();
  void                 set(std::vector<int> index);
  void                 setDim(int dim, int value);
  void                 setOrigin(std::vector<int> origin);
  void                 setShape(std::vector<int> shape);
  void                 setStride(std::vector<int> stride);
  std::vector<int>     getCurrentCounter();
  void                 setIndexName(int dim, const std::string& indexName);
  std::string          getIndexName(int dim);
  IIndexPtr            reduce();
  IIndexPtr            reduce(int dim) throw ( Exception );
  //@} IIndex interface
  
  //@{ IObject interface
  std::string          getFactoryName() const { return m_factory; };
  CDMAType::ModelType  getModelType() const   { return CDMAType::Other; };
  //@} IObject interface
};

}
#endif
