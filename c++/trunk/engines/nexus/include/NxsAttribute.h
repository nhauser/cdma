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
#ifndef __CDMA_NXSATTRIBUTE_H__
#define __CDMA_NXSATTRIBUTE_H__

/// CDMA Attribute, with name and value. The metadata for data items and groups.

#include <string>
#include <yat/utils/String.h>
#include <yat/memory/MemBuf.h>

#include <nxfile.h>

#include <internal/common.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{

//==============================================================================
/// IAttribute implementation for NeXus engine
/// See IAttribute definition for more explanation
//==============================================================================
class CDMA_NEXUS_DECL NxsAttribute : public IAttribute
{
private:
  std::string    m_name;
  NexusDataType  m_datatype;
  yat::MemBuf    m_value_buf;

public:
  NxsAttribute();
  NxsAttribute( const NexusFilePtr& file_ptr, const NexusAttrInfo& info );

  //NxsAttribute( const string& name, const string value ) { m_name = name; m_value = new string(value); };

  std::string getName();
  const std::type_info& getType();
  bool isString();
  bool isArray();
  int getLength();
  std::string getStringValue();
  std::string toString();
  void setStringValue(const std::string& val);
  void setName(const std::string& name);
  long getIntValue();
  double getFloatValue();
  void setIntValue(int value);
  void setFloatValue(float value);
  };
}
#endif
