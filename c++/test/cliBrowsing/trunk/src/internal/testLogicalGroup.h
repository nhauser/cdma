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

#ifndef __TEST_LOGICAL_GROUP_H__
#define __TEST_LOGICAL_GROUP_H__

#include <list>
#include <vector>
#include <string>

#include <cdma/dictionary/LogicalGroup.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/navigation/IDimension.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{

class TestLogicalGroup
{


public:
  enum CommandLogicalGrp
  {
    getParent = 0,
    getName,
    getShortName,
    getLocation,
    getDataItem,
    getDataItemList,
    getGroup,
    getKeyNames,
    getKeys,
    display,
    help,
    open,
    list,
    exit,
    back
  };

  struct CommandLogicalGroup
  {
  public:
    CommandLogicalGroup() {};
    CommandLogicalGrp command;
    std::vector<yat::String> args;
  };
  
  static CommandLogicalGroup getCommandLogicalGroup(const std::string& entry);
  static void execute(const LogicalGroupPtr& group, CommandLogicalGroup command, IDataItemPtr& out_item, LogicalGroupPtr& out_group);
  
  static std::vector<std::string>                         s_commandLogicalNames;
  static std::vector<TestLogicalGroup::CommandLogicalGrp> s_commandLogicalGroup;
  
};

}

#endif
