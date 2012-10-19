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

#ifndef __CDMA_NXSDIMENSION_H__
#define __CDMA_NXSDIMENSION_H__

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IDimension.h>
#include <cdma/array/IArray.h>
#include <NxsDataItem.h>

#include <internal/common.h>

namespace cdma
{
namespace nexus
{

//=============================================================================
/// IDimension implementation
//=============================================================================
class CDMA_NEXUS_DECL Dimension : public IDimension
{
typedef std::map<std::string, IAttributePtr> AttributeMap;

private:
  DataItemPtr m_item;   // DataItem of the dimension
  bool        m_shared; // Is this dimension shared

public:
  // Constructors
    Dimension( DataItemPtr item );
    
  //@{ IDimension interface
    std::string getName();
    int getSize();
    bool isUnlimited();
    bool isVariableLength();
    bool isShared();
    IArrayPtr getCoordinateVariable();
    int getDimensionAxis();
    int getDisplayOrder();
    void setUnlimited(bool b);
    void setVariableLength(bool b);
    void setShared(bool b);
    void setSize(int n);
    void setName(const std::string& name);
    void setCoordinateVariable(const IArrayPtr& array) throw ( Exception );
    void setDimensionAxis(int index);
    void setDisplayOrder(int order);
    std::string getUnit();
  //@}
 };

} // namespace nexus
} // namespace cdma

#endif
