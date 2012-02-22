//*****************************************************************************
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
//*****************************************************************************

//-----------------------------------------------------------------------------
// DEPENDENCIES
//-----------------------------------------------------------------------------
#include <cdma/exception/Exception.h>

#include <cdma/array/Array.h>
#include <cdma/array/ArrayIterator.h>

namespace cdma
{
//-----------------------------------------------------------------------------
// ArrayIterator::ArrayIterator
//-----------------------------------------------------------------------------
ArrayIterator::ArrayIterator(const cdma::ArrayPtr& array, const cdma::ViewPtr& view, std::vector<int> position) 
{
//  CDMA_FUNCTION_TRACE("ArrayIterator::ArrayIterator");
  m_array = array;
  m_view = view;
  m_position = position;
}

//-----------------------------------------------------------------------------
// ArrayIterator::ArrayIterator
//-----------------------------------------------------------------------------
ArrayIterator::ArrayIterator(const ArrayIterator& iter) : m_view ( iter.m_view )
{
//  CDMA_FUNCTION_TRACE("ArrayIterator::ArrayIterator");
  m_array = iter.m_array;
  m_position = iter.m_position;
}

//---------------------------------------------------------------------------
// ArrayIterator::~ArrayIterator
//---------------------------------------------------------------------------
ArrayIterator::~ArrayIterator()
{
//  CDMA_FUNCTION_TRACE("ArrayIterator::~ArrayIterator");
};

//-----------------------------------------------------------------------------
// ArrayIterator::getPosition
//-----------------------------------------------------------------------------
std::vector<int> ArrayIterator::getPosition() const
{
  return m_position; 
}

//-----------------------------------------------------------------------------
// ArrayIterator::currentElement
//-----------------------------------------------------------------------------
long ArrayIterator::currentElement() const
{
  return m_view->getElementOffset(m_position);
}

//-----------------------------------------------------------------------------
// ArrayIterator::operator++ prefix operator
//-----------------------------------------------------------------------------
ArrayIterator& ArrayIterator::operator++(void) 
{
  this->incrementPosition(m_view, m_position);
  return *this;
}

//-----------------------------------------------------------------------------
// ArrayIterator::operator++ suffix operator
//-----------------------------------------------------------------------------
ArrayIterator ArrayIterator::operator++(int)
{
  ArrayIterator it(*this);
  operator++();
  return it;
}

//-----------------------------------------------------------------------------
// ArrayIterator::operator-- prefix operator
//-----------------------------------------------------------------------------
ArrayIterator& ArrayIterator::operator--(void) 
{
  this->decrementPosition(m_view, m_position);
  return *this;
}

//-----------------------------------------------------------------------------
// ArrayIterator::operator-- suffix operator
//-----------------------------------------------------------------------------
ArrayIterator ArrayIterator::operator--(int)
{
  ArrayIterator it(*this);
  operator--();
  return it;
}

//-----------------------------------------------------------------------------
// ArrayIterator::operator++ suffix operator
//-----------------------------------------------------------------------------
bool ArrayIterator::operator==(const ArrayIterator& it)
{
  return m_view->getElementOffset(m_position) == it.m_view->getElementOffset(it.m_position);
}

//-----------------------------------------------------------------------------
// ArrayIterator::operator++ suffix operator
//-----------------------------------------------------------------------------
bool ArrayIterator::operator!=(const ArrayIterator& it)
{
  return m_view->getElementOffset(m_position) != it.m_view->getElementOffset(it.m_position);
}

//-----------------------------------------------------------------------------
// ArrayIterator::incrementPosition
//-----------------------------------------------------------------------------
std::vector<int>& ArrayIterator::incrementPosition(const ViewPtr& view, std::vector<int>& position)
{
  std::vector<int> shape = view->getShape();

  if( position[0] < shape[0] )
  {
    for( int i = int(position.size()) - 1; i >= 0; i-- )
    {
      if( position[i] + 1 >= shape[i] && i > 0)
      {
        position[i] = 0;
      }
      else
      {
        position[i]++;
        break;
      }
    }
  }
  return position;
}

//-----------------------------------------------------------------------------
// ArrayIterator::decrementPosition
//-----------------------------------------------------------------------------
std::vector<int>& ArrayIterator::decrementPosition(const ViewPtr& view, std::vector<int>& position)
{
  std::vector<int> shape = view->getShape();

  // Check if this == begin iterator then reset position
  if( !isBeginPosition( shape, position ) )
  {
    // Check the position is not out of range
    if( position[0] >= 0 )
    {
      for( int i = int(position.size()) - 1; i >= 0; i-- )
      {
        if( position[i] - 1 < 0 && i > 0  )
        {
          position[i] = shape[i] - 1;
        }
        else
        {
          position[i]--;
          break;
        }
      }
    }
  }
  return position;
}

//-----------------------------------------------------------------------------
// ArrayIterator::isEndPosition
//-----------------------------------------------------------------------------
bool ArrayIterator::isEndPosition(std::vector<int> shape, std::vector<int> position)
{
  bool result = true;

  if( position[0] == shape[0] )
  {
    for( unsigned int i = 1; i < shape.size(); i++ )
    {
      if( position[i] != 0 )
      {
        result = false;
        break;
      }
    }
  }
  else
  {
    result = false;  
  }
  return result;
}

//-----------------------------------------------------------------------------
// ArrayIterator::isBeginPosition
//-----------------------------------------------------------------------------
bool ArrayIterator::isBeginPosition(std::vector<int> shape, std::vector<int> position)
{
  bool result = true;

  if( position[0] == -1 )
  {
    for( unsigned int i = 1; i < shape.size(); i++ )
    {
      if( position[i] != shape[i] - 1 )
      {
        result = false;
        break;
      }
    }
  }
  else
  {
    result = false;  
  }
  return result;
}


}


