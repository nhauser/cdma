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
#include <internal/tools.h>
#include <string>


namespace cdma
{
using namespace std;


const char*                         cmd_lname_init[] = { "parent", "help", "exit", "back", "open", "display" };
TestLogicalGroup::CommandLogicalGrp cmd_lgrou_init[] = { TestLogicalGroup::parent, TestLogicalGroup::help, TestLogicalGroup::exit, TestLogicalGroup::back, TestLogicalGroup::open, TestLogicalGroup::display };

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
// TestLogicalGroup::getCommand
//---------------------------------------------------------------------------
std::string TestLogicalGroup::getCommand(const LogicalGroupPtr& group)
{
  yat::String entry = "";
  std::list<IDataItemPtr> list_of_items;

  // Display available keys
  cout<<endl<<endl<<"=============================="<<endl;
  cout<<"Current logical path: "<< group->getLocation() << endl;
  cout<<"Available keys:"<<endl;
  cout<<Tools::iterate_over_keys( group, list_of_items );

  // Enter a key
  cout<<"=============================="<<endl;
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
// TestLogicalGroup::execute
//---------------------------------------------------------------------------
void TestLogicalGroup::execute(const LogicalGroupPtr& group, CommandLogicalGroup cmd, IDataItemPtr& out_item, LogicalGroupPtr& out_group)
{
  switch( cmd.command )
  {
    case parent:
      out_group = group->getParent();
    case open:
    {
      if( cmd.args.size() > 0 )
      {
        // Create requested key
        KeyPtr key = Tools::getKey(group, cmd.args[0]);
        // Ask for its target
        if( key )
        {
          if( key->getType() == Key::ITEM )
          {
            out_item = group->getDataItem( key );
          }
          else
          {
            out_group = group->getGroup( key );
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
    case help:
      cout<<"=============================="<<endl;
      cout<<"Usage:"<<endl;
      cout<<"  enter 'name_of_key' to select the corresponding node"<<endl;
      cout<<"  enter 'display' to show all properties of the current node"<<endl;
      cout<<"  enter 'back' to leave current level and return to the parent one"<<endl;
      cout<<"  enter 'exit' to quit the program"<<endl;
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
