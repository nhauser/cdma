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


#include <internal/testNavigation.h>
#include <string>
#include <iostream>

#include <cdma/utils/ArrayUtils.h>
#include <cdma/array/Array.h>

#include <internal/tools.h>

namespace cdma
{
using namespace std;
//=============================================================================
//
// TestNavigation
//
//=============================================================================
//---------------------------------------------------------------------------
// TestNavigation::TestNavigation
//---------------------------------------------------------------------------
TestNavigation::TestNavigation( const IDatasetPtr& dataset )
{
  m_dataset = dataset;
  m_current = dataset->getRootGroup();
}

//---------------------------------------------------------------------------
// TestNavigation::run_test
//---------------------------------------------------------------------------
bool TestNavigation::run_physical()
{
  cout<<"Performing physical navigation tests:..."<<endl;
  return test_group(m_dataset->getRootGroup());
}

//---------------------------------------------------------------------------
// TestNavigation::run_logical
//---------------------------------------------------------------------------
bool TestNavigation::run_logical()
{
  cout<<"Performing logical navigation tests:..."<<endl;

  // Accessing its logical root
  LogicalGroupPtr log_root = m_dataset->getLogicalRoot();
  LogicalGroupPtr group = log_root;

  return test_logical_group( m_dataset->getLogicalRoot() );
}

//---------------------------------------------------------------------------
// TestNavigation::display_all
//---------------------------------------------------------------------------
void TestNavigation::display_all(string indent)
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::scan_group");
  IGroupPtr grp = m_current;
  
  cout<<Tools::displayGroup( m_current, indent );
  cout<<indent<<" |\\  "<<endl;
  // Scan all sub group
  std::list<IGroupPtr> list_grp = m_current->getGroupList();
  for( std::list<IGroupPtr>::iterator iter = list_grp.begin(); iter != list_grp.end(); iter++ )
  {
    m_current = (*iter);
    display_all( indent + " | | ");
  }

  // Scan all dataitem
  std::list<IDataItemPtr> list_itm = m_current->getDataItemList();
  for( std::list<IDataItemPtr>::iterator iter = list_itm.begin(); iter != list_itm.end(); iter++ )
  {
    IDataItemPtr item = (*iter);
    cout<<Tools::displayDataItem( item, indent + " | | ");
    
    std::list<IDimensionPtr> list_dim = m_current->getDimensionList();
    if( list_dim.size() > 0) 
    {
      cout<<indent<<" | |\\  "<<endl;
      for( std::list<IDimensionPtr>::iterator iter_d = list_dim.begin(); iter_d != list_dim.end(); iter_d++ )
      {
        cout<<Tools::displayDimension( (*iter_d), indent + " | | | ")<<endl;
      }
      cout<<indent<<" | |/  "<<endl;    
    }   
  }
  cout<<indent<<" |/  "<<endl;
  
  m_current = grp;
}

//---------------------------------------------------------------------------
// TestNavigation::test_dataitem
//---------------------------------------------------------------------------
bool TestNavigation::test_dataitem(const IDataItemPtr& item)
{
  CDMA_FUNCTION_TRACE("TestNavigation::test_dataitem");
  bool doContinue = true;
  
  TestDataItem::CommandItem cmd = TestDataItem::getCommandItem( getCommand(item.get()) );
  if( cmd.command != TestDataItem::exit )
  {
    while( cmd.command != TestDataItem::exit && cmd.command != TestDataItem::back && doContinue)
    {
      if( cmd.command == TestDataItem::exit )
      {
        doContinue = false;
      }
      else
      {
        IDataItemPtr itm (NULL);
        IGroupPtr grp (NULL);
        TestDataItem::execute(item, cmd, itm, grp);
        
        if( grp )
        {
          doContinue = test_group( grp );
        }
        else if( itm )
        {
          doContinue = test_dataitem(itm);
        }
        
        if( doContinue )
        {
          cmd = TestDataItem::getCommandItem( getCommand(item.get()) );
          if( cmd.command == TestDataItem::exit )
          {
            doContinue = false;      
          }
        }
      }
    }
  }  
  return doContinue;
}

//---------------------------------------------------------------------------
// TestNavigation::test_group
//---------------------------------------------------------------------------
bool TestNavigation::test_group(const IGroupPtr& group)
{
  CDMA_FUNCTION_TRACE("TestNavigation::test_dataitem");
  bool doContinue = true;
  
  TestGroup::CommandGroup cmd = TestGroup::getCommandGroup( getCommand(group.get()) );
  if( cmd.command != TestGroup::exit )
  {
    while( doContinue && cmd.command != TestGroup::back)
    {
      IDataItemPtr itm (NULL);
      IGroupPtr grp (NULL);
      TestGroup::execute(group, cmd, itm, grp);
      
      if( grp )
      {
        doContinue = test_group( grp );
      }
      else if( itm )
      {
        doContinue = test_dataitem(itm);
      }
      
      if( doContinue )
      {
        cmd = TestGroup::getCommandGroup( getCommand(group.get()) );
        if( cmd.command == TestGroup::exit )
        {
          doContinue = false;      
        }
      }
    }
  }
  return doContinue;
}

//---------------------------------------------------------------------------
// TestNavigation::test_logical_group
//---------------------------------------------------------------------------
bool TestNavigation::test_logical_group(const LogicalGroupPtr& group)
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_logical_group");
  bool doContinue = true;
  
  TestLogicalGroup::CommandLogicalGroup cmd = TestLogicalGroup::getCommandLogicalGroup( getCommand(group) );
  if( cmd.command != TestLogicalGroup::exit )
  {
    while( doContinue && cmd.command != TestLogicalGroup::back)
    {
      IDataItemPtr itm (NULL);
      LogicalGroupPtr grp (NULL);
      TestLogicalGroup::execute(group, cmd, itm, grp);
      
      if( grp )
      {
        doContinue = test_logical_group( grp );
      }
      else if( itm )
      {
        doContinue = test_dataitem(itm);
      }
      
      if( doContinue )
      {
        cmd = TestLogicalGroup::getCommandLogicalGroup( getCommand(group) );
        if( cmd.command == TestLogicalGroup::exit )
        {
          doContinue = false;
        }
      }
    }
  } 
  return doContinue;
}

//---------------------------------------------------------------------------
// TestNavigation::getCommand
//---------------------------------------------------------------------------
std::string TestNavigation::getCommand(IContainer* cnt)
{
  yat::String entry = "";
  cout<<endl<<endl<<"=============================="<<endl;
  if( cnt != NULL )
  {
    cout<<"Current node: "<<cnt->getName()<<endl;
    cout<<"Current location: "<<cnt->getLocation()<<endl;
  }
  cout<<"=============================="<<endl;
  cout<<"Please enter a command or 'h' for help: ";
  while( entry == "" )
  {
    getline(cin, entry, '\n');
    entry.trim();
  }
  cout<<"=============================="<<endl;
  return entry;
}

//---------------------------------------------------------------------------
// TestNavigation::getCommand
//---------------------------------------------------------------------------
std::string TestNavigation::getCommand(const LogicalGroupPtr& group)
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
  cout<<"Please enter a command or 'h' for help: ";
  while( entry == "" )
  {
    getline(cin, entry, '\n');
    entry.trim();
  }
  cout<<"=============================="<<endl;
  
  return entry;
}

}
