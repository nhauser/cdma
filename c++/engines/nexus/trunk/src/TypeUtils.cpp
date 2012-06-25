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
#include <TypeUtils.h>

namespace cdma
{
namespace nexus
{
//-----------------------------------------------------------------------------
// TypeUtils::toCType
//-----------------------------------------------------------------------------
NexusDataType TypeUtils::toNexusDataType( const std::type_info& type )
{
  if( type == typeid(short) )
  {
    return NX_INT16;
  }
  else if( type == typeid(unsigned short) )
  {
    return NX_UINT16;
  }
  else if( type == typeid(unsigned long) )
  {
    return NX_UINT64;
  }
  else if( type == typeid(long) )
  {
    return NX_INT32;
  }
  else if( type == typeid(float) )
  {
    return NX_FLOAT32;
  }
  else if( type == typeid(yat::int64) )
  {
    return NX_INT64;
  }
  else if( type == typeid(double) )
  {
    return NX_FLOAT64;
  }
  else
  {
    return NX_CHAR;
  }
}


//-----------------------------------------------------------------------------
// TypeUtils::toCType
//-----------------------------------------------------------------------------
const std::type_info& TypeUtils::toCType( NexusDataType type )
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
// TypeUtils::sizeOf
//-----------------------------------------------------------------------------
int TypeUtils::sizeOf(NexusDataType type)
{
  switch( type )
  {
    case NX_CHAR:
    case NX_INT8:
    case NX_UINT8:
      return sizeof(char);
      break;
    case NX_INT16:
    case NX_UINT16:
      return sizeof(short);
      break;
    case NX_UINT32:
    case NX_INT32:
      return sizeof(unsigned long);
      break;
    case NX_INT64:
    case NX_UINT64:
      return sizeof(yat::int64);
      break;
    case NX_FLOAT32:
      return sizeof(float);
      break;
    case NX_FLOAT64:
      return sizeof(double);
      break;
    default:
      return 0;
  }
}

} // namespace nexus
} // namespace cdma
