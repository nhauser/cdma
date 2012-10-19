//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

//-----------------------------------------------------------------------------
// DEPENDENCIES
//-----------------------------------------------------------------------------
#include <cdma/dictionary/plugin/SimpleDataItem.h>

namespace cdma
{
//=============================================================================
//
// SimpleDataItem
//
//=============================================================================
//---------------------------------------------------------------------------
// c-tor
//---------------------------------------------------------------------------
SimpleDataItem::SimpleDataItem(IDataset* dataset_ptr, IArrayPtr ptrArray, const std::string &name):
m_dataset_ptr(dataset_ptr), m_name(name), m_array_ptr(ptrArray)
{
  CDMA_FUNCTION_TRACE("SimpleDataItem::SimpleDataItem");
}

//---------------------------------------------------------------------------
// SimpleDataItem::findAttributeIgnoreCase
//---------------------------------------------------------------------------
cdma::IAttributePtr SimpleDataItem::findAttributeIgnoreCase( const std::string& )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::findAttributeIgnoreCase");
}

//---------------------------------------------------------------------------
// SimpleDataItem::findDimensionView
//---------------------------------------------------------------------------
int SimpleDataItem::findDimensionView( const std::string& )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::findDimensionView");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getASlice
//---------------------------------------------------------------------------
cdma::IDataItemPtr SimpleDataItem::getASlice( int /*dimension*/, int /*value*/ ) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getASlice");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getParent
//---------------------------------------------------------------------------
cdma::IGroupPtr SimpleDataItem::getParent()
{
  return NULL;
}

//---------------------------------------------------------------------------
// SimpleDataItem::getRoot
//---------------------------------------------------------------------------
cdma::IGroupPtr SimpleDataItem::getRoot()
{
  return NULL;
}

//---------------------------------------------------------------------------
// SimpleDataItem::getData
//---------------------------------------------------------------------------
cdma::IArrayPtr SimpleDataItem::getData(std::vector<int> position) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("SimpleDataItem::getData(vector<int> position)");
  int node_rank = m_array_ptr->getRank();
  
  std::vector<int> origin;
  std::vector<int> shape;

  for( int dim = 0; dim < node_rank; dim++ )
  {
    if( dim < (int)position.size() )
    {
      origin.push_back( position[dim] );
      shape.push_back( m_array_ptr->getShape()[dim] - position[dim] );
    }
    else
    {
      origin.push_back( 0 );
      shape.push_back( m_array_ptr->getShape()[dim] );
    }
  }
  cdma::IArrayPtr array_ptr = getData(origin, shape);
  return array_ptr;
}

//---------------------------------------------------------------------------
// SimpleDataItem::getData
//---------------------------------------------------------------------------
cdma::IArrayPtr SimpleDataItem::getData(std::vector<int> origin, std::vector<int> shape) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("SimpleDataItem::getData(vector<int> origin, vector<int> shape)");

  int rank = m_array_ptr->getRank();
  int* iShape = new int[rank];
  int* iStart = new int[rank];
  for( int i = 0; i < rank; i++ )
  {
    iStart[i] = origin[i];
    iShape[i] = shape[i];
  }
  cdma::IViewPtr view = new cdma::View( rank, iShape, iStart );
  //## Should pass the shared pointer rather than a pointer to the referenced object
  cdma::IArrayPtr array_ptr = new cdma::Array( m_array_ptr, view );
  return array_ptr;
}

//---------------------------------------------------------------------------
// SimpleDataItem::getDescription
//---------------------------------------------------------------------------
std::string SimpleDataItem::getDescription()
{
  return std::string("");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getDimensions
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> SimpleDataItem::getDimensions( int )
{
  return std::list<cdma::IDimensionPtr>();
}

//---------------------------------------------------------------------------
// SimpleDataItem::getDimensionList
//---------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> SimpleDataItem::getDimensionList()
{
  return std::list<cdma::IDimensionPtr>();
}

//---------------------------------------------------------------------------
// SimpleDataItem::getDimensionsString
//---------------------------------------------------------------------------
std::string SimpleDataItem::getDimensionsString()
{
  return std::string("");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getElementSize
//---------------------------------------------------------------------------
int SimpleDataItem::getElementSize()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getNameAndDimensions");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getNameAndDimensions
//---------------------------------------------------------------------------
std::string SimpleDataItem::getNameAndDimensions()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getNameAndDimensions");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getNameAndDimensions
//---------------------------------------------------------------------------
std::string SimpleDataItem::getNameAndDimensions( bool /*useFullName*/, bool /*showDimLength*/ )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getNameAndDimensions");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getRangeList
//---------------------------------------------------------------------------
std::list<cdma::RangePtr> SimpleDataItem::getRangeList()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getRangeList");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getRank
//---------------------------------------------------------------------------
int SimpleDataItem::getRank()
{
  return m_array_ptr->getRank();
}

//---------------------------------------------------------------------------
// SimpleDataItem::getSection
//---------------------------------------------------------------------------
cdma::IDataItemPtr SimpleDataItem::getSection( std::list<cdma::RangePtr> /*section*/ ) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getSection");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getSectionRanges
//---------------------------------------------------------------------------
std::list<cdma::RangePtr> SimpleDataItem::getSectionRanges()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getSectionRanges");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getShape
//---------------------------------------------------------------------------
std::vector<int> SimpleDataItem::getShape()
{
  return m_array_ptr->getShape();
}

//---------------------------------------------------------------------------
// SimpleDataItem::getSize
//---------------------------------------------------------------------------
long SimpleDataItem::getSize()
{
  return m_array_ptr->getSize();
}

//---------------------------------------------------------------------------
// SimpleDataItem::getSizeToCache
//---------------------------------------------------------------------------
int SimpleDataItem::getSizeToCache()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getSizeToCache");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getSlice
//---------------------------------------------------------------------------
cdma::IDataItemPtr SimpleDataItem::getSlice(int /*dim*/, int /*value*/) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getSlice");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getType
//---------------------------------------------------------------------------
const std::type_info& SimpleDataItem::getType()
{
  return m_array_ptr->getValueType();
}

//---------------------------------------------------------------------------
// SimpleDataItem::getUnit
//---------------------------------------------------------------------------
std::string SimpleDataItem::getUnit()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getUnit");
}

//---------------------------------------------------------------------------
// SimpleDataItem::hasCachedData
//---------------------------------------------------------------------------
bool SimpleDataItem::hasCachedData()
{
  return false;
}

//---------------------------------------------------------------------------
// SimpleDataItem::findAttributeIgnoreCase
//---------------------------------------------------------------------------
//##int SimpleDataItem::hashCode()

//---------------------------------------------------------------------------
// SimpleDataItem::invalidateCache
//---------------------------------------------------------------------------
void SimpleDataItem::invalidateCache()
{
  // Do nothing
}

//---------------------------------------------------------------------------
// SimpleDataItem::isCaching
//---------------------------------------------------------------------------
bool SimpleDataItem::isCaching()
{
  return false;
}

//---------------------------------------------------------------------------
// SimpleDataItem::isScalar
//---------------------------------------------------------------------------
bool SimpleDataItem::isScalar()
{
  return m_array_ptr->getRank() == 0;
}

//---------------------------------------------------------------------------
// SimpleDataItem::isUnlimited
//---------------------------------------------------------------------------
bool SimpleDataItem::isUnlimited()
{
  return false;
}

//---------------------------------------------------------------------------
// SimpleDataItem::isUnsigned
//---------------------------------------------------------------------------
bool SimpleDataItem::isUnsigned()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::isUnsigned");
}

//---------------------------------------------------------------------------
// SimpleDataItem::removeAttribute
//---------------------------------------------------------------------------
bool SimpleDataItem::removeAttribute( const cdma::IAttributePtr& )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::removeAttribute");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setCachedData
//---------------------------------------------------------------------------
//##void SimpleDataItem::setCachedData(Array& cacheData, bool isMetadata) throw ( cdma::Exception )

//---------------------------------------------------------------------------
// SimpleDataItem::setCaching
//---------------------------------------------------------------------------
void SimpleDataItem::setCaching( bool )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::setCaching");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setDataType
//---------------------------------------------------------------------------
void SimpleDataItem::setDataType( const std::type_info& )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::setDataType");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setDimension
//---------------------------------------------------------------------------
void SimpleDataItem::setDimension( const cdma::IDimensionPtr&, int /*ind*/) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::setDimension");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setElementSize
//---------------------------------------------------------------------------
void SimpleDataItem::setElementSize( int )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::setElementSize");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setSizeToCache
//---------------------------------------------------------------------------
void SimpleDataItem::setSizeToCache( int )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::setSizeToCache");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setUnit
//---------------------------------------------------------------------------
void SimpleDataItem::setUnit( const std::string& )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::setUnit");
}

//---------------------------------------------------------------------------
// SimpleDataItem::clone
//---------------------------------------------------------------------------
IDataItemPtr SimpleDataItem::clone()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::clone");
}

//---------------------------------------------------------------------------
// SimpleDataItem::addAttribute
//---------------------------------------------------------------------------
void SimpleDataItem::addAttribute( const cdma::IAttributePtr& )
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::addAttribute");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getAttribute
//---------------------------------------------------------------------------
IAttributePtr SimpleDataItem::getAttribute(const std::string&)
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getAttribute");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getAttributeList
//---------------------------------------------------------------------------
AttributeList SimpleDataItem::getAttributeList()
{
  return m_attr_list;
}

//---------------------------------------------------------------------------
// SimpleDataItem::getLocation
//---------------------------------------------------------------------------
std::string SimpleDataItem::getLocation() const
{
  return m_name;
}

//---------------------------------------------------------------------------
// SimpleDataItem::getName
//---------------------------------------------------------------------------
std::string SimpleDataItem::getName() const
{
  return m_name;
}

//---------------------------------------------------------------------------
// SimpleDataItem::getShortName
//---------------------------------------------------------------------------
std::string SimpleDataItem::getShortName() const
{
  return m_name;
}

//---------------------------------------------------------------------------
// SimpleDataItem::hasAttribute
//---------------------------------------------------------------------------
bool SimpleDataItem::hasAttribute(const std::string&)
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::hasAttribute");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setName
//---------------------------------------------------------------------------
void SimpleDataItem::setName(const std::string& name)
{
  m_name = name;
}

//---------------------------------------------------------------------------
// SimpleDataItem::setShortName
//---------------------------------------------------------------------------
void SimpleDataItem::setShortName(const std::string& /*name*/)
{
}

//---------------------------------------------------------------------------
// SimpleDataItem::setParent
//---------------------------------------------------------------------------
void SimpleDataItem::setParent(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::setParent");
}

//---------------------------------------------------------------------------
// SimpleDataItem::getDataset
//---------------------------------------------------------------------------
cdma::IDatasetPtr SimpleDataItem::getDataset()
{
  THROW_NOT_IMPLEMENTED("SimpleDataItem::getDataset");
}

//---------------------------------------------------------------------------
// SimpleDataItem::setData
//---------------------------------------------------------------------------
void SimpleDataItem::setData(const cdma::IArrayPtr& array)
{
  m_array_ptr = array;
}

} // namespace
