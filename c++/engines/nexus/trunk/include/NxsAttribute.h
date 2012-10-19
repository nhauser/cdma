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
#ifndef __CDMA_NEXUS_ATTRIBUTE_H__
#define __CDMA_NEXUS_ATTRIBUTE_H__

/// CDMA Attribute, with name and value. The metadata for data items and groups.

#include <yat/memory/MemBuf.h>

#include <nxfile.h>

#include <internal/common.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{
namespace nexus
{

//==============================================================================
/// IAttribute implementation for NeXus engine
/// See IAttribute definition for more explanation
//==============================================================================
  class CDMA_NEXUS_DECL Attribute : public IAttribute
{
private:
  std::string    m_name;
  NexusDataType  m_datatype;
  IArrayPtr       m_array_ptr;    // Array object

public:
  Attribute();
  Attribute( const NexusFilePtr& file_ptr, const NexusAttrInfo& info );

  //Attribute( const string& name, const string value ) { m_name = name; m_value = new string(value); };

  std::string getName();
  const std::type_info& getType();
  bool isArray();
  int getSize();
  IArrayPtr getData();
  void setData(const IArrayPtr&);
  void setName(const std::string& name);
  };

} // namespace nexus
} // namespace cdma

#endif
