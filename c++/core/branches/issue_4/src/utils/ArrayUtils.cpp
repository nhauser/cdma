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
#include <typeinfo>

#include <cdma/exception/impl/ExceptionImpl.h>
#include <cdma/utils/ArrayUtils.h>
#include <cdma/array/impl/View.h>
#include <cdma/array/IArray.h>
#include <cdma/array/ArrayIterator.h>
#include <cdma/array/SliceIterator.h>
#include <cdma/array/Slicer.h>

#include <string>
#include <iostream>
namespace cdma
{
//---------------------------------------------------------------------------
// ArrayUtils::ArrayUtils
//---------------------------------------------------------------------------
ArrayUtils::ArrayUtils( const IArrayPtr& array )
{
  m_array = array;
}

//---------------------------------------------------------------------------
// ArrayUtils::~ArrayUtils
//---------------------------------------------------------------------------
ArrayUtils::~ArrayUtils() { }

//---------------------------------------------------------------------------
// ArrayUtils::getArray
//---------------------------------------------------------------------------
IArrayPtr ArrayUtils::getArray()
{
  CDMA_FUNCTION_TRACE("ArrayUtils::getArray");
  return m_array;
}

//---------------------------------------------------------------------------
// ArrayUtils::checkShape
//---------------------------------------------------------------------------
bool ArrayUtils::checkShape(const IArrayPtr& newArray)
{
  bool result = true;
  if( m_array && newArray )
  {
    if( m_array->getRank() == newArray->getRank() )
    {
      std::vector<int> shapeA = m_array->getShape();
      std::vector<int> shapeB = newArray->getShape();    
      
      for( unsigned int i = 0; i < shapeA.size(); i++ )
      {
        if( shapeA[i] != shapeB[i] )
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
  }
  else
  {
    result = false;
  }

  return result;  
}

//---------------------------------------------------------------------------
// ArrayUtils::reduce
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::reduce()
{
  CDMA_FUNCTION_TRACE("ArrayUtils::reduce");

  if( getArray() )
  {
    IViewPtr view(new View( m_array->getView() ));
    view->reduce();

    return ArrayUtilsPtr(
           new ArrayUtils(
               IArrayPtr(new Array(getArray(), view))
               )
           );
  }
  else
  {
    return ArrayUtilsPtr(new ArrayUtils( getArray() ));
  }
  
}

//---------------------------------------------------------------------------
// ArrayUtils::reduce
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::reduce(int dim)
{
  IArrayPtr thisArray = m_array;
  if( thisArray )
  {
    IViewPtr view(new View( thisArray->getView() ));
    view->reduce(dim);

    thisArray = IArrayPtr(new Array(thisArray, view));
  }
  return ArrayUtilsPtr(new ArrayUtils(thisArray));
}

//---------------------------------------------------------------------------
// ArrayUtils::reduceTo
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::reduceTo(int rank)
{
  if( m_array )
  {
    IViewPtr view(new View( m_array->getView() ));
    std::vector<int> shape = view->getShape();
    for( unsigned int i = 0; i < shape.size(); i++ )
    {
      if( view->getRank() == rank )
      {
        break;
      }
      else
      {
        if( shape[i] == 1 )
        {
          view->reduce(i);
          shape = view->getShape();
        }
      }
    }

    return ArrayUtilsPtr(new ArrayUtils(IArrayPtr(new Array(m_array, view))));
  }
  return ArrayUtilsPtr(new ArrayUtils(m_array));
}

//---------------------------------------------------------------------------
// ArrayUtils::reshape
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::reshape(std::vector<int> shape) throw ( Exception )
{
  IArrayPtr thisArray = m_array;
  if( thisArray )
  {
    std::vector<int> start ( shape.size() );
    IViewPtr view(new View( shape, start ));
    view->compose( thisArray->getView() );

    thisArray = IArrayPtr(new Array(thisArray, view));
  }
  return ArrayUtilsPtr(new ArrayUtils(thisArray));
}

//---------------------------------------------------------------------------
// ArrayUtils::slice
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::slice(int dim, int value)
{
  IArrayPtr thisArray = m_array;
  if( thisArray )
  {
    IViewPtr view(new View( thisArray->getView() ));

    if( dim < view->getRank() ) 
    {
      // Get the view's properties
      std::vector<int> shape  = view->getShape();
      std::vector<int> origin = view->getOrigin();
      std::vector<int> stride = view->getStride();
      origin[dim] = value;
      shape[dim] = 1;

      // update the view
      view->setStride(stride);
      view->setOrigin(origin);
      view->setShape(shape);

      // Reduce the array
      if( view->getRank() > 0 )
      {
        view->reduce(dim);
      }

      // Create a new array
      thisArray = IArrayPtr(new Array(thisArray, view));
    }
  }
  return ArrayUtilsPtr(new ArrayUtils(thisArray));
}

//---------------------------------------------------------------------------
// ArrayUtils::transpose
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::transpose(int dim1, int dim2)
{
  IArrayPtr thisArray = m_array;
  if( thisArray )
  {
    // Get current view of the array
    IViewPtr view(new View( thisArray->getView() ));

    std::vector<int> shape  = view->getShape();
    std::vector<int> origin = view->getOrigin();
    std::vector<int> stride = view->getStride();

    int sha  = shape[dim1];
    int ori  = origin[dim1];
    long str = stride[dim1];

    // Swapping dim1 and dim2
    shape[dim1]  = shape[dim2];
    origin[dim1] = origin[dim2];
    stride[dim1] = stride[dim2];

    shape[dim2]  = sha;
    origin[dim2] = ori;
    stride[dim2] = str;

    // Construct new view
    view = IViewPtr(new View( shape, origin, stride ));

    shape  = view->getShape();
    origin = view->getOrigin();
    stride = view->getStride();

    thisArray = IArrayPtr(new Array(thisArray, view));
  }
  return ArrayUtilsPtr(new ArrayUtils(thisArray));
}

//---------------------------------------------------------------------------
// ArrayUtils::flip
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::flip(int dim)
{
  IArrayPtr thisArray = m_array;
  if( thisArray )
  {
    // Get current view of the array
    IViewPtr view(new View( thisArray->getView() ));
    std::vector<int> shape  = view->getShape();
    std::vector<int> origin = view->getOrigin();
    std::vector<int> stride = view->getStride();

    // Invert the requested dimension (i.e: starts at last and finish at first, stride becomes negative)
    int length  = shape[dim];
    stride[dim] = stride[dim] * (-1);
    origin[dim] = origin[dim] + length - 1;

    thisArray = IArrayPtr(new Array(thisArray, IViewPtr(new View( shape, origin, stride
                    ))));
  }
  return ArrayUtilsPtr(new ArrayUtils(thisArray));
}

//---------------------------------------------------------------------------
// ArrayUtils::permute
//---------------------------------------------------------------------------
ArrayUtilsPtr ArrayUtils::permute(std::vector<int> dims)
{
  IArrayPtr thisArray = m_array;
  if( thisArray )
  {
    // Get current view of the array
    IViewPtr view(new View( thisArray->getView() ));
    int rank = view->getRank();
    
    std::vector<int> shape  = view->getShape();
    std::vector<int> origin = view->getOrigin();
    std::vector<int> stride = view->getStride();

    std::vector<int> newShape  (rank);
    std::vector<int> newOrigin (rank);
    std::vector<int> newStride (rank);

    for( int i = 0; i < rank; i++ ) 
    {
      newShape[i]    = shape[ dims[i] ];
      newOrigin[i]   = origin[ dims[i] ];
      newStride[i]   = stride[ dims[i] ];
    }

    view = IViewPtr(new View( newShape, newOrigin, newStride ));

    thisArray = IArrayPtr(new Array(thisArray, view));
  }
  return ArrayUtilsPtr(new ArrayUtils(thisArray));
}

};
