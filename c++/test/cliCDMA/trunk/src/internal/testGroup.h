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

#ifndef __TEST_GROUP_H__
#define __TEST_GROUP_H__

#include <list>
#include <vector>
#include <string>

#include <cdma/navigation/IGroup.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/navigation/IDimension.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{

class TestGroup
{
public:
  enum CommandGrp
  {
    getParent = 0,
    getRoot,
    getDimensionList,
    getAttribute,
    getAttributeList,
    getContainerType,
    getLocation,
    getName,
    getShortName,
    hasAttribute,
    isRoot,
    isEntry,
    getDataItemList,
    getDataItem, 
    getDataItemWithAttribute,
    getDimension,
    getGroupList,
    getGroup,
    getGroupWithAttribute,
    display,
    help,
    list,
    exit,
    back,
    open
  };

  struct CommandGroup
  {
  public:
    CommandGroup() {};
    CommandGrp command;
    std::vector<yat::String> args;
  };
  
  static CommandGroup getCommandGroup(const std::string& entry);
  static void execute(const IGroupPtr& group, CommandGroup command, IDataItemPtr& out_item, IGroupPtr& out_group);
  
  static std::vector<std::string>     s_commandNames;
  static std::vector<TestGroup::CommandGrp> s_commandGroup;
  
};

}

#endif
