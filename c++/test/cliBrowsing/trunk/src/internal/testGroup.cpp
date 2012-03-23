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
#include <string>

#include <tools.h>

namespace cdma
{
using namespace std;

const char*           cmd_name_inir[] = {"getParent", "getRoot", "getDimensionList", "getAttribute", "getAttributeList", "getContainerType", "getLocation", "getName", "getShortName", "hasAttribute", "isRoot", "isEntry", "getDataItemList", "getDataItem", "getDataItemWithAttribute", "getDimension", "getGroupList", "getGroup", "getGroupWithAttribute", "display", "help", "list", "exit", "back"};
TestGroup::CommandGrp cmd_grou_init[] = { TestGroup::getParent, TestGroup::getRoot, TestGroup::getDimensionList, TestGroup::getAttribute, TestGroup::getAttributeList, TestGroup::getContainerType, TestGroup::getLocation, TestGroup::getName, TestGroup::getShortName, TestGroup::hasAttribute, TestGroup::isRoot, TestGroup::isEntry, TestGroup::getDataItemList, TestGroup::getDataItem,  TestGroup::getDataItemWithAttribute, TestGroup::getDimension, TestGroup::getGroupList, TestGroup::getGroup, TestGroup::getGroupWithAttribute, TestGroup::display, TestGroup::help, TestGroup::list, TestGroup::exit, TestGroup::back };

std::vector<std::string>           TestGroup::s_commandNames ( cmd_name_inir, Tools::end( cmd_name_inir ) ) ;
std::vector<TestGroup::CommandGrp> TestGroup::s_commandGroup ( cmd_grou_init, Tools::end( cmd_grou_init ) );

//---------------------------------------------------------------------------
// TestGroup::getCommandGroup
//---------------------------------------------------------------------------
TestGroup::CommandGroup TestGroup::getCommandGroup(const std::string& entry )
{
  CommandGroup cmd;
  
  vector<yat::String> keys;
  
  yat::String keyStr = entry;
  
  keyStr.split(' ', &keys );
  string name = keys[0];
  
  for(unsigned int i = 1; i < keys.size(); i++ )
  {
    cmd.args.push_back( keys[i] );
  }
  
  cmd.command = help;
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

  return cmd;
}

//---------------------------------------------------------------------------
// TestGroup::execute
//---------------------------------------------------------------------------
void TestGroup::execute(const IGroupPtr& group, CommandGroup cmd, IDataItemPtr& out_item, IGroupPtr& out_group)
{
  switch( cmd.command )
  {
    case getParent:
      out_group = group->getParent();
    break;
    case getRoot:
    {
      out_group = group->getRoot();
      if( out_group )
      {
        cout<<"Root: "<<out_group->getLocation()<<std::endl;
      }
      else
      {
        cout<<"Root is NULL"<<endl;
      }
    }
    break;
    case getDimensionList:
    {
      std::list<IDimensionPtr> dims = group->getDimensionList( );
      for( std::list<IDimensionPtr>::iterator it = dims.begin(); it != dims.end(); it++ )
      {
        cout<<Tools::displayDimension(*it)<<endl;
      }
    }
    break;
    case getAttribute:
    {
      if( cmd.args.size() > 0 )
      {
        IAttributePtr attr = group->getAttribute( cmd.args[0] );
        if( attr )
        {
        
          cout<<Tools::displayAttribute(attr)<<endl;
        }
        else
        {
          cout<<"Attribute '"<<cmd.args[0]<<"' not found !!"<<endl;
        }
      }
    }
    break;
    case getAttributeList:
    {
      std::list<IAttributePtr> attrList = group->getAttributeList();
      cout<<"Attribute: "<<attrList.size()<<" were found"<<endl;
      for( std::list<IAttributePtr>::iterator it = attrList.begin(); it != attrList.end(); it ++ )
      {
        cout<<"  - "<<Tools::displayAttribute( (*it) )<<endl;
      }
    }
    break;
    case getContainerType:
      cout<<"Container type: "<< (group->getContainerType() == IContainer::DATA_GROUP ? "DATA_GROUP" : "not a group" ) <<endl;
    break;
    case getLocation:
      cout<<"Location: "<< group->getLocation()<<endl;
    break;
    case getName:
      cout<<"Name: "<< group->getName()<<endl;
    break;
    case getShortName:
      cout<<"Short name: "<< group->getShortName()<<endl;
    break;
    case hasAttribute:
    if( cmd.args.size() > 0 )
      cout<<"hasAttribute '"<< cmd.args[0] <<"' : "<< ( group->hasAttribute(cmd.args[0]) ? "true" : "false" ) <<endl;
    break;
    case isRoot:
      cout<<"isRoot: "<< ( group->isRoot() ?  "true" : "false" ) <<endl;
    break;
    case isEntry:
      cout<<"isEntry: "<< ( group->isEntry() ?  "true" : "false" ) <<endl;
     break;
    case getDataItemList:
    {
      std::list<IDataItemPtr> lst = group->getDataItemList();
      cout<<"Group: "<<lst.size()<<" were found"<<endl;
      for( std::list<IDataItemPtr>::iterator it = lst.begin(); it != lst.end(); it++ )
      {
        cout<<"  - "<<( (*it)->getShortName() )<<" =>> "<< ( (*it)->getName() ) <<endl;
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
    case getDataItemWithAttribute:
    {
      if( cmd.args.size() > 1 )
      {
        out_item = group->getDataItemWithAttribute( cmd.args[0], cmd.args[1] );
        if( out_item )
        {
          cout<<"Item found with attribute ('"<<cmd.args[0]<<"' = '"<<cmd.args[1]<<"'  ): "<< out_item->getName()<<endl;
        }
        else
        {
          cout<<"Item not found with attribute ('"<<cmd.args[0]<<"' = '"<<cmd.args[1]<<"'  )!"<<endl;
        }
      }
    }
    break;
    case getDimension:
    {
      if( cmd.args.size() > 0 )
      {
        IDimensionPtr dim = group->getDimension( cmd.args[0] );
        if( dim )
        {
          cout<<"Dimension found:\n"<<Tools::displayDimension( dim )<<endl;
        }
        else
        {
          cout<<"Dimension not found ('"<<cmd.args[0]<<"')!"<<endl;
        }
      }
    }
    break;
    case getGroupList:
    {
      std::list<IGroupPtr> lst = group->getGroupList();
      cout<<"Group: "<<lst.size()<<" were found"<<endl;
      for( std::list<IGroupPtr>::iterator it = lst.begin(); it != lst.end(); it++ )
      {
        cout<<"  - "<<( (*it)->getShortName() )<<" =>> "<< ( (*it)->getName() ) <<endl;
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
    case getGroupWithAttribute:
    {
      if( cmd.args.size() > 1 )
      {
        out_group = group->getGroupWithAttribute( cmd.args[0], cmd.args[1] );
        if( out_group )
        {
          cout<<"Group found with attribute ('"<<cmd.args[0]<<"' = '"<<cmd.args[1]<<"'  ): "<< out_group->getName()<<endl;
        }
        else
        {
          cout<<"Group not found with attribute ('"<<cmd.args[0]<<"' = '"<<cmd.args[1]<<"'  )!"<<endl;
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
      cout<<"  enter 'list' to list all available commands"<<endl;
      cout<<"  enter 'exit' to quit the program"<<endl;
      cout<<"  enter 'back' to leave current level and return to the parent one"<<endl;
      cout<<"  enter a command and its paramters (optional) to execute it"<<endl;
      cout<<"=============================="<<endl;
    }
    break;
    case list:
    {
      cout<<"=============================="<<endl;
      cout<<"Available commands seen as method signatures:"<<endl;
      cout<<"    IGroupPtr getParent()"<<endl;
      cout<<"    bool isRoot()"<<endl;
      cout<<"    bool isEntry()"<<endl;
      cout<<"    IGroupPtr getRoot()"<<endl;
      cout<<"    IGroupPtr getParent()"<<endl;
      cout<<"    IDataItemPtr getDataItem(const std::string& short_name)"<<endl;
      cout<<"    IDataItemPtr getDataItemWithAttribute(const std::string& name, const std::string& value)"<<endl;
      cout<<"    IDimensionPtr getDimension(const std::string& name)"<<endl;
      cout<<"    IAttributePtr getAttribute(const std::string&)"<<endl;
      cout<<"    IGroupPtr getGroup(const std::string& short_name)"<<endl;
      cout<<"    IGroupPtr getGroupWithAttribute(const std::string& attributeName, const std::string& value)"<<endl;
      cout<<"    list<cdma::IAttributePtr> getAttributeList()"<<endl;
      cout<<"    list<cdma::IDataItemPtr> getDataItemList()"<<endl;
      cout<<"    list<cdma::IDimensionPtr> getDimensionList()"<<endl;
      cout<<"    list<cdma::IGroupPtr> getGroupList()"<<endl;
      cout<<"    string getLocation()"<<endl;
      cout<<"    string getName()"<<endl;
      cout<<"    string getShortName()"<<endl;
      cout<<"    bool hasAttribute(const std::string&)"<<endl;
      cout<<"    IContainer::Type getContainerType()"<<endl;
      cout<<"=============================="<<endl;
    }
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
