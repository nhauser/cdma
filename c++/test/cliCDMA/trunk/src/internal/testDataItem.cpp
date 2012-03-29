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


#include <internal/testDataItem.h>
#include <internal/testArrayUtils.h>
#include <string>

#include <internal/tools.h>

namespace cdma
{
using namespace std;

const char*                   cmd_name_init[] = {"findAttributeIgnoreCase", "findDimensionView", "getParent", "getRoot", "getData", "getDescription", "getDimensions", "getDimensionList", "getDimensionsString", "getElementSize", "getRank", "getShape", " getSize", "getSlice", "getType", "getUnitsString", "isMemberOfStructure", "isMetadata", "isScalar", "isUnlimited", "isUnsigned", "readScalarByte", "readScalarDouble", "readScalarFloat", "readScalarInt", "readScalarLong", "readScalarShort", "readString", "getAttribute", "getAttributeList", "getContainerType", "getLocation", "getName", "getShortName", "hasAttribute", "help", "list", "exit", "display", "back", "test"};
TestDataItem::CommandDataItem cmd_item_init[] = { TestDataItem::findAttributeIgnoreCase, TestDataItem::findDimensionView, TestDataItem::getParent, TestDataItem::getRoot, TestDataItem::getData, TestDataItem::getDescription, TestDataItem::getDimensions, TestDataItem::getDimensionList, TestDataItem::getDimensionsString, TestDataItem::getElementSize, TestDataItem::getRank, TestDataItem::getShape, TestDataItem:: getSize, TestDataItem::getSlice, TestDataItem::getType, TestDataItem::getUnitsString, TestDataItem::isMemberOfStructure, TestDataItem::isMetadata, TestDataItem::isScalar, TestDataItem::isUnlimited, TestDataItem::isUnsigned, TestDataItem::readScalarByte, TestDataItem::readScalarDouble, TestDataItem::readScalarFloat, TestDataItem::readScalarInt, TestDataItem::readScalarLong, TestDataItem::readScalarShort, TestDataItem::readString, TestDataItem::getAttribute, TestDataItem::getAttributeList, TestDataItem::getContainerType, TestDataItem::getLocation, TestDataItem::getName, TestDataItem::getShortName, TestDataItem::hasAttribute, TestDataItem::help, TestDataItem::list, TestDataItem::exit, TestDataItem::display, TestDataItem::back, TestDataItem::test };

std::vector<std::string>                   TestDataItem::s_commandNames ( cmd_name_init, Tools::end(cmd_name_init) ) ;
std::vector<TestDataItem::CommandDataItem> TestDataItem::s_commandItems ( cmd_item_init, Tools::end( cmd_item_init ) );
//---------------------------------------------------------------------------
// TestDataItem::getCommandItem
//---------------------------------------------------------------------------
TestDataItem::CommandItem TestDataItem::getCommandItem(const std::string& entry )
{
  CommandItem cmd;
  
  vector<yat::String> keys;
  
  yat::String keyStr = entry;
  
  keyStr.split(' ', &keys );
  string name = keys[0];
  
  for(unsigned int i = 1; i < keys.size(); i++ )
  {
    if( keys[i] != "" )
    {
      cmd.args.push_back( keys[i] );
    }
  }
  
  cmd.command = help;
  for( unsigned int i = 0; i < s_commandNames.size(); i++ )
  {
    if( name == s_commandNames[i] )
    {
      cmd.command = s_commandItems[i];
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
// TestDataItem::execute
//---------------------------------------------------------------------------
void TestDataItem::execute(const IDataItemPtr& item, CommandItem cmd, IDataItemPtr& out_item, IGroupPtr& out_group)
{
  switch( cmd.command )
  {
    case findAttributeIgnoreCase:
    {
      if( cmd.args.size() > 0 )
      {
        IAttributePtr attr = item->findAttributeIgnoreCase( cmd.args[0] );
        if( attr )
        {
          cout<<Tools::displayAttribute(attr)<<endl;
        }
        else
        {
          cout<<"No attribute found: "<<cmd.args[0]<<endl;
        }
      }
    }
    break;
    case findDimensionView:
      if( cmd.args.size() > 0 )
      {
        cout<<"Dimension "<<cmd.args[0]<<" is applied on axis: "<<item->findDimensionView( cmd.args[0] )<<endl;
      }
    break;
    case getParent:
    {
      out_group = item->getParent();
      if( out_group )
      {
        cout<<"Parent: "<<out_group->getLocation()<<std::endl;
      }
      else
      {
        cout<<"Parent is NULL"<<endl;
      }
    }
    break;
    case getRoot:
    {
      out_group = item->getRoot();
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
    case getData:
    {
      ArrayPtr array = item->getData();
      cout<<Tools::displayArray( array )<<endl;
    }
    break;
    case getDescription:
      cout<<"Description: "<<item->getDescription()<<endl;
    break;
    case getDimensions:
    {
      if( cmd.args.size() > 0 )
      {
        int dim = atoi( cmd.args[0].c_str() );
        std::list<IDimensionPtr> dims = item->getDimensions( dim );
        for( std::list<IDimensionPtr>::iterator it = dims.begin(); it != dims.end(); it++ )
        {
          cout<<Tools::displayDimension(*it)<<endl;
        }
      }
    }
    break;
    case getDimensionList:
    {
      std::list<IDimensionPtr> dims = item->getDimensionList( );
      for( std::list<IDimensionPtr>::iterator it = dims.begin(); it != dims.end(); it++ )
      {
        cout<<Tools::displayDimension(*it)<<endl;
      }
    }
    break;
    case getDimensionsString:
      cout<<item->getDimensionsString()<<endl;
    break;
    case getElementSize:
      cout<<"Element size: "<<item->getElementSize()<<endl;
    break;
    case getRank:
      cout<<"Rank: "<<item->getRank()<<endl;
    break;
    case getShape:
      cout<<"Shape: "<< Tools::displayArray( item->getShape() ) <<endl;
    break;
    case getSize:
      cout<<"Size: "<<item->getSize()<<endl;
    break;
    case getSlice:
    {
      if( cmd.args.size() > 1 )
      {
        cout<<"Slice on dimensions "<<cmd.args[0]<<" at position "<<cmd.args[1]<<endl;
        int dim = atoi( cmd.args[0].c_str() );
        int val = atoi( cmd.args[1].c_str() );
        IDataItemPtr slice = item->getSlice(dim, val);
        if( slice )
        {
          cout<<Tools::displayDataItem( slice )<<endl;
        }
        else
        {
          cout<<"An error occured!!!"<<endl;
        }
        out_item = slice;
      }
    }
    break;
    case getType:
    {
      cout<<"Type: "<<item->getType().name()<<endl;
    }
    break;
    case getUnitsString:
    {
      cout<<"Units: "<<item->getUnitsString()<<endl;
    }
    break;
    case isMemberOfStructure:
    {
      cout<<"isMemberOfStructure: "<<( item->isMemberOfStructure() ? "true" : "false" )<<endl;
    }
    break;
    case isMetadata:
    {
      cout<<"isMetadata: "<<( item->isMetadata() ? "true" : "false" )<<endl;
    }
    break;
    case isScalar:
    {
      cout<<"isScalar: "<<( item->isScalar() ? "true" : "false" )<<endl;
    }
    break;
    case isUnlimited:
    {
      cout<<"isUnlimited: "<<( item->isUnlimited() ? "true" : "false" )<<endl;
    }
    break;
    case isUnsigned:
    {
      cout<<"isUnsigned: "<<( item->isUnsigned() ? "true" : "false" )<<endl;
    }
    break;
    case readScalarByte:
    {
      cout<<"readScalarByte: "<<( item->readScalarByte() )<<endl;
    }
    break;
    case readScalarDouble:
    {
      cout<<"readScalarDouble: "<<( item->readScalarDouble() )<<endl;
    }
    break;
    case readScalarFloat:
    {
      cout<<"readScalarFloat: "<<( item->readScalarFloat() )<<endl;
    }
    break;
    case readScalarInt:
    {
      cout<<"readScalarInt: "<<( item->readScalarInt() )<<endl;
    }
    break;
    case readScalarLong:
    {
      cout<<"readScalarLong: "<<( item->readScalarLong() )<<endl;
    }
    break;
    case readScalarShort:
    {
      cout<<"readScalarShort: "<<( item->readScalarShort() )<<endl;
    }
    break;
    case readString:
    {
      cout<<"readString: "<<( item->readString() )<<endl;
    }
    break;
    case getAttribute:
    {
      if( cmd.args.size() > 0 )
      {
        IAttributePtr attr = item->getAttribute( cmd.args[0] );
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
      std::list<IAttributePtr> attrList = item->getAttributeList();
      cout<<"Attribute: "<<attrList.size()<<" were found"<<endl;
      for( std::list<IAttributePtr>::iterator it = attrList.begin(); it != attrList.end(); it ++ )
      {
        cout<<"  - "<<Tools::displayAttribute( (*it) )<<endl;
      }
    }
    break;
    case getContainerType:
      cout<<"Container type: "<< (item->getContainerType() == IContainer::DATA_ITEM ? "DATA_ITEM" : "not an item")<<endl;
    break;
    case getLocation:
      cout<<"Location: "<< item->getLocation()<<endl;
    break;
    case getName:
      cout<<"Name: "<< item->getName()<<endl;
    break;
    case getShortName:
      cout<<"Short name: "<< item->getShortName()<<endl;
    break;
    case hasAttribute:
      if( cmd.args.size() > 0 )
      {
        cout<<"hasAttribute '"<< cmd.args[0] <<"' : "<< ( item->hasAttribute(cmd.args[0]) ? "true" : "false" )<<endl;
      }
    break;
    case test:
    {
      if( item->getRank() > 0 )
      {
        TestArrayUtils test_utils( item->getData() );
        test_utils.run_test();
      }
    }
    break;
    case display:
      cout<<"Item: "<<Tools::displayDataItem(item)<<endl;
    break;
    case help:
    {
      cout<<"=============================="<<endl;
      cout<<"Usage:"<<endl;
      cout<<"  enter 'test' to execute some tests on the underlying array"<<endl;
      cout<<"  enter 'display' to show all properties of the current node"<<endl;
      cout<<"  enter 'list' to list all available commands"<<endl;
      cout<<"  enter 'back' to leave current level and return to the parent one"<<endl;
      cout<<"  enter 'exit' to quit the program"<<endl;
      cout<<"  enter a command and its paramters (optional) to execute it"<<endl;
      cout<<"=============================="<<endl;
    }
    break;
    case list:
    {
      cout<<"=============================="<<endl;
      cout<<"Available commands seen as method signatures:"<<endl;
      cout<<"    int findDimensionView(const std::string& name)"<<endl;
      cout<<"    IGroupPtr getParent()"<<endl;
      cout<<"    IGroupPtr getRoot()"<<endl;
      cout<<"    ArrayPtr getData()"<<endl;
      cout<<"    ArrayPtr getData(std::vector<int> origin, std::vector<int> shape)"<<endl;
      cout<<"    std::string getDescription()"<<endl;
      cout<<"    std::list<IDimensionPtr> getDimensions(int i)"<<endl;
      cout<<"    std::list<IDimensionPtr> getDimensionList()"<<endl;
      cout<<"    std::string getDimensionsString()"<<endl;
      cout<<"    int getElementSize()"<<endl;
      cout<<"    int getRank()"<<endl;
      cout<<"    std::vector<int> getShape()"<<endl;
      cout<<"    long getSize()"<<endl;
      cout<<"    IDataItemPtr getSlice(int dim, int value)"<<endl;
      cout<<"    const std::type_info& getType()"<<endl;
      cout<<"    std::string getUnitsString()"<<endl;
      cout<<"    bool isMemberOfStructure()"<<endl;
      cout<<"    bool isMetadata()"<<endl;
      cout<<"    bool isScalar()"<<endl;
      cout<<"    bool isUnlimited()"<<endl;
      cout<<"    bool isUnsigned()"<<endl;
      cout<<"    unsigned char readScalarByte()"<<endl;
      cout<<"    double readScalarDouble()"<<endl;
      cout<<"    float readScalarFloat()"<<endl;
      cout<<"    int readScalarInt()"<<endl;
      cout<<"    long readScalarLong()"<<endl;
      cout<<"    short readScalarShort()"<<endl;
      cout<<"    std::string readString()"<<endl;
      cout<<"    IAttributePtr getAttribute(const std::string&)"<<endl;
      cout<<"    AttributeList getAttributeList()"<<endl;
      
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
