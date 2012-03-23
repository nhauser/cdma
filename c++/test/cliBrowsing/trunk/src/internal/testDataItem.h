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

#ifndef __TEST_DATAITEM_H__
#define __TEST_DATAITEM_H__

#include <list>
#include <vector>
#include <string>

#include <cdma/navigation/IGroup.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/navigation/IDimension.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{

class TestDataItem
{
public:
  enum CommandDataItem
  {
    findAttributeIgnoreCase = 0,
    findDimensionView,
    getParent,
    getRoot,
    getData,
    getDescription,
    getDimensions,
    getDimensionList,
    getDimensionsString,
    getElementSize,
    getRank,
    getShape,
    getSize,
    getSlice,
    getType,
    getUnitsString,
    isMemberOfStructure,
    isMetadata,
    isScalar,
    isUnlimited,
    isUnsigned,
    readScalarByte,
    readScalarDouble,
    readScalarFloat,
    readScalarInt,
    readScalarLong,
    readScalarShort,
    readString,
    getAttribute,
    getAttributeList,
    getContainerType,
    getLocation,
    getName,
    getShortName,
    hasAttribute,
    test,
    help,
    list,
    exit,
    display,
    back
  };

  struct CommandItem
  {
  public:
    CommandItem() {};
    CommandDataItem command;
    std::vector<yat::String> args;
  };
  
  static CommandItem getCommandItem(const std::string& entry);
  static void execute(const IDataItemPtr& item, CommandItem command, IDataItemPtr& out_item, IGroupPtr& out_group);

private:  
  static std::vector<std::string>     s_commandNames;
  static std::vector<TestDataItem::CommandDataItem> s_commandItems;
  
};

}

#endif
