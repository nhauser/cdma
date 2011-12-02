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
#include <TypeDetector.h>

namespace cdma
{
  
  //-----------------------------------------------------------------------------
  // TypeDetector::detectType
  //-----------------------------------------------------------------------------
  const std::type_info& TypeDetector::detectType( NexusDataType type )
  {
    switch( type )
    {
      case NX_INT16:
        return typeid(short);
        break;
      case NX_UINT16:
        return typeid(unsigned short);
        break;
      case NX_UINT32:
        return typeid(unsigned long);
        break;
      case NX_INT32:
        return typeid(long);
        break;
      case NX_FLOAT32:
        return typeid(float);
        break;
      case NX_INT64:
        return typeid(yat::int64);
        break;
      case NX_UINT64:
        return typeid(unsigned long);
        break;
      case NX_FLOAT64:
        return typeid(double);
        break;
      default:  // CHAR, NX_INT8, NX_UINT8
        return typeid(char);
    }
  }

  //-----------------------------------------------------------------------------
  // TypeDetector::detectType
  //-----------------------------------------------------------------------------
  const NexusDataType& TypeDetector::detectType( const std::type_info& type )
  {
    NexusDataType result;
    if( type == typeid( short ) )
    {
      result = NX_INT16;
    }
    else if( type == typeid( unsigned short ) )
    {
      result = NX_UINT16;
    }
    else if( type == typeid( unsigned long ) )
    {
      result = NX_UINT32;
    }
    else if( type == typeid( long ) )
    {
      result = NX_INT32;
    }
    else if( type == typeid( float ) )
    {
      result = NX_FLOAT32;
    }
    else if( type == typeid( yat::int64 ) )
    {
      result = NX_INT64;
    }
    else if( type == typeid( yat::uint64 ) )
    {
      result = NX_INT64;
    }
    else if( type == typeid( double ) )
    {
      result = NX_FLOAT64;
    }
    else
    {
      result = NX_CHAR;
    }
    return result;
  }
  
} // namespace