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


#include <internal/testGroup.h>
#include <internal/tools.h>
#include <string>

namespace cdma
{
using namespace std;

const char*           cmd_name_inir[] = { "parent", "open", "display", "help", "exit", "back"};
TestGroup::CommandGrp cmd_grou_init[] = { TestGroup::parent, TestGroup::open, TestGroup::display, TestGroup::help, TestGroup::exit, TestGroup::back };

std::vector<std::string>           TestGroup::s_commandNames ( cmd_name_inir, Tools::end( cmd_name_inir ) ) ;
std::vector<TestGroup::CommandGrp> TestGroup::s_commandGroup ( cmd_grou_init, Tools::end( cmd_grou_init ) );

//---------------------------------------------------------------------------
// TestGroup::getCommandGroup
//---------------------------------------------------------------------------
std::string TestGroup::listChild(const IGroupPtr& group)
{
  stringstream res;
  std::list<IGroupPtr> list_grp = group->getGroupList();
  if( list_grp.size() > 0 )
  {
    res<<"Groups: ";
    for( std::list<IGroupPtr>::iterator iter = list_grp.begin(); iter != list_grp.end(); )
    {
      res<<(*iter)->getShortName();
      iter++;
      if( iter != list_grp.end() )
        res<<", ";
    }
    res<<std::endl;
  }
  std::list<IDataItemPtr> list_item = group->getDataItemList();
  if( list_item.size() > 0 )
  {
    res<<"DataItems: ";
    for( std::list<IDataItemPtr>::iterator iter = list_item.begin(); iter != list_item.end(); )
    {
      res<<(*iter)->getShortName();
      iter++;
      if( iter != list_item.begin() )
        res<<", ";
    }
    res<<std::endl;
  }
  std::list<IDimensionPtr> list_dim = group->getDimensionList();
  if( list_dim.size() > 0 )
  {
    res<<"Dimensions: ";
    for( std::list<IDimensionPtr>::iterator iter = list_dim.begin(); iter != list_dim.end(); )
    {
      res<<(*iter)->getName();
      iter++;
      if( iter != list_dim.end() )
        res<<", ";
    }
  }
  return res.str();
}

//---------------------------------------------------------------------------
// TestGroup::getCommand
//---------------------------------------------------------------------------
std::string TestGroup::getCommand(const IGroupPtr& cnt)
{
  yat::String entry = "";
  cout<<endl<<endl<<"=============================="<<endl;
  if( cnt != NULL )
  {
    cout<<"Current node: "<<cnt->getName()<<endl;
    cout<<"Current location: "<<cnt->getLocation()<<endl;
  }
  cout<<"=============================="<<endl;
  cout<<"Available entries:"<<endl;
  cout<<TestGroup::listChild(cnt)<<endl;
  cout<<"Please enter a command or 'h' for help: "<<endl<<"> ";
  while( entry == "" )
  {
    getline(cin, entry, '\n');
    entry.trim();
  }
  cout<<"=============================="<<endl;
  return entry;
}

//---------------------------------------------------------------------------
// TestGroup::getCommandGroup
//---------------------------------------------------------------------------
TestGroup::CommandGroup TestGroup::getCommandGroup(const std::string& entry )
{
  CommandGroup cmd;
  
  vector<yat::String> keys;
  
  yat::String keyStr = entry;
  
  if( entry != "" )
  {
    keyStr.split(' ', &keys );
    string name = keys[0];
    
    cmd.command = open;
    for(unsigned int i = 0; i < s_commandNames.size(); i++ )
    {
      if( name == s_commandNames[i] )
      {
        cmd.command = s_commandGroup[i];
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
    else if( name == "h" || name == "help" )
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
// TestGroup::execute
//---------------------------------------------------------------------------
void TestGroup::execute(const IGroupPtr& group, CommandGroup cmd, IDataItemPtr& out_item, IGroupPtr& out_group)
{
  switch( cmd.command )
  {
    case parent:
      out_group = group->getParent();
    break;
    case open:
      if( cmd.args.size() > 0 )
      {
        yat::String name = cmd.args[0], itmName;
        name.to_lower();

        {
          std::list<IGroupPtr> lst = group->getGroupList();
          for( std::list<IGroupPtr>::iterator it = lst.begin(); it != lst.end(); it++ )
          {
            itmName = (*it)->getShortName();
            itmName.to_lower();
            if( itmName == name )
            {
              out_group = (*it);
              break;
            }
          }
        }
        
        if( ! out_group )
        {
          std::list<IDataItemPtr> lst = group->getDataItemList();
          for( std::list<IDataItemPtr>::iterator it = lst.begin(); it != lst.end(); it++ )
          {
            itmName = (*it)->getShortName();
            itmName.to_lower();
            if( itmName == name )
            {
              out_item = (*it);
              break;
            }
          }          
        }
      }
    break;
    case display:
      cout<<"Group: "<<Tools::displayGroup(group)<<endl;
    break;
    case help:
    {
      cout<<"=============================="<<endl;
      cout<<"Usage:"<<endl;
      cout<<"  enter 'display' to show all properties of the current node"<<endl;
      cout<<"  enter 'exit' to quit the program"<<endl;
      cout<<"  enter 'back' to leave current level and return to the parent one"<<endl;
      cout<<"  enter 'name_of_a_node' to open it"<<endl;
    }
    break;
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
