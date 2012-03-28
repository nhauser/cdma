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


#include <internal/testLogicalGroup.h>
#include <string>

#include <tools.h>

namespace cdma
{
using namespace std;


const char*                         cmd_lname_init[] = { "getParent", "getName", "getShortName", "getLocation", "getDataItem", "getDataItemList", "getGroup", "getKeyNames", "getKeys", "help", "list", "exit", "back", "open", "display" };
TestLogicalGroup::CommandLogicalGrp cmd_lgrou_init[] = { TestLogicalGroup::getParent, TestLogicalGroup::getName, TestLogicalGroup::getShortName, TestLogicalGroup::getLocation, TestLogicalGroup::getDataItem, TestLogicalGroup::getDataItemList, TestLogicalGroup::getGroup, TestLogicalGroup::getKeyNames, TestLogicalGroup::getKeys, TestLogicalGroup::help, TestLogicalGroup::list, TestLogicalGroup::exit, TestLogicalGroup::back, TestLogicalGroup::open, TestLogicalGroup::display };

std::vector<std::string>                         TestLogicalGroup::s_commandLogicalNames ( cmd_lname_init, Tools::end( cmd_lname_init ) ) ;
std::vector<TestLogicalGroup::CommandLogicalGrp> TestLogicalGroup::s_commandLogicalGroup ( cmd_lgrou_init, Tools::end( cmd_lgrou_init ) );

//---------------------------------------------------------------------------
// TestLogicalGroup::getCommandLogicalGroup
//---------------------------------------------------------------------------
TestLogicalGroup::CommandLogicalGroup TestLogicalGroup::getCommandLogicalGroup(const std::string& entry )
{
  CommandLogicalGroup cmd;
  
  vector<yat::String> keys;
  
  yat::String keyStr = entry;

  if( entry != "" )
  {
    // Split string to determine name of command and args  
    keyStr.split(' ', &keys );
    string name = keys[0];
    
    cmd.command = open;
    for(unsigned int i = 0; i < s_commandLogicalNames.size(); i++ )
    {
      if( name == s_commandLogicalNames[i] )
      {
        cmd.command = s_commandLogicalGroup[i];
      }
    }
    
    if( name == "exit" || name == "quit" || name == "q" )
    {
      cmd.command = exit;
    }
    else if( name == "back" || name == "b" || name == "prev" )
    {
      cmd.command = back;
    }
    else if( name == "help" || name == "h" )
    {
      cmd.command = help;
    }
    else if( cmd.command == open && name != "open" )
    {
      cmd.args.push_back( name );
    }
    
    for(unsigned int i = 1; i < keys.size(); i++ )
    {
      if( keys[i] != "" )
      {
        cmd.args.push_back( keys[i] );
      }
    }
  } 
  return cmd;
}

//---------------------------------------------------------------------------
// TestLogicalGroup::execute
//---------------------------------------------------------------------------
void TestLogicalGroup::execute(const LogicalGroupPtr& group, CommandLogicalGroup cmd, IDataItemPtr& out_item, LogicalGroupPtr& out_group)
{
  switch( cmd.command )
  {
    case getParent:
      out_group = group->getParent();
    case getLocation:
      cout<<"Location: "<< group->getLocation()<<endl;
    break;
    case getName:
      cout<<"Name: "<< group->getName()<<endl;
    break;
    case getShortName:
      cout<<"Short name: "<< group->getShortName()<<endl;
    break;
    case getDataItemList:
    {
      if( cmd.args.size() > 0 )
      {
        std::list<IDataItemPtr> lst = group->getDataItemList( cmd.args[0] );
        cout<<"Group: "<<lst.size()<<" were found"<<endl;
        for( std::list<IDataItemPtr>::iterator it = lst.begin(); it != lst.end(); it++ )
        {
          cout<<"  - "<<( (*it)->getShortName() )<<" =>> "<< ( (*it)->getName() ) <<endl;
        }
      }
    }
    break;
    case getDataItem:
    {
      if( cmd.args.size() > 0 )
      {
        out_item = group->getDataItem( cmd.args[0] );
        if( out_item )
        {
          cout<<"Item found: "<< out_item->getName()<<endl;
        }
        else
        {
          cout<<"Item not found!"<<endl;
        }
      }
    }
    break;
    case getGroup:
    {
      if( cmd.args.size() > 0 )
      {
        out_group = group->getGroup( cmd.args[0] );
        if( out_group )
        {
          cout<<"Group found: "<< out_group->getName()<<endl;
        }
        else
        {
          cout<<"Group not found!"<<endl;
        }
      }
    }
    break;
    case open:
    {
      if( cmd.args.size() > 0 )
      {
        // Create requested key
        KeyPtr key = Tools::getKey(group, cmd.args[0]);
        // Ask for its target
        if( key )
        {
          cout<<"key not null"<<endl;
          if( key->getType() == Key::ITEM )
          {
            cout<<"Opened item: "<<endl;
            out_item = group->getDataItem( key );
            cout<<key->getName()<<endl;
          }
          else
          {
            cout<<"Opened group: "<<endl;
            out_group = group->getGroup( key );
            cout<<key->getName()<<endl;
          }
        }
        else
        {
          cout<<"No item found for key '"<<cmd.args[0]<<"'"<<endl;
        }
      }
    }
    break;
    case display:
      cout<<Tools::displayLogicalGroup( group )<<endl;
    break;
    case list:
    {
      cout<<"=============================="<<endl;
      cout<<"Available commands seen as method signatures:"<<endl;
      cout<<"    IDataItemPtr getDataItem(const std::string& key)"<<endl;
      cout<<"    std::list<IDataItemPtr> getDataItemList(const std::string& key)"<<endl;
      cout<<"    LogicalGroupPtr getGroup(const std::string& key)"<<endl;
      cout<<"    std::list<KeyPtr> getKeys()"<<endl;
      cout<<"    LogicalGroupPtr getParent()"<<endl;
      cout<<"    std::string getLocation()"<<endl;
      cout<<"    std::string getName()"<<endl;
      cout<<"    std::string getShortName()"<<endl;
      cout<<"=============================="<<endl;
    }
    break;
    case help:
      cout<<"=============================="<<endl;
      cout<<"Usage:"<<endl;
      cout<<"  enter 'name_of_key' to select the corresponding node"<<endl;
      cout<<"  enter 'display' to show all properties of the current node"<<endl;
      cout<<"  enter 'list' to list all available commands"<<endl;
      cout<<"  enter 'back' to leave current level and return to the parent one"<<endl;
      cout<<"  enter 'exit' to quit the program"<<endl;
      cout<<"  enter a command and its paramters (optional) to execute it"<<endl;
      cout<<"=============================="<<endl;
    break;
    case back:
      cout<<"Steping back"<<endl;
    break;
    case exit:
      cout<<"Exiting program"<<endl;
    break;
  }
  
  
}

}
