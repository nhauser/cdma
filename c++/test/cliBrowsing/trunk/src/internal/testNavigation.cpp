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
#include <string>
#include <iostream>

#include <cdma/utils/ArrayUtils.h>
#include <cdma/array/Array.h>

#include <internal/testNavigation.h>
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
  
  return test_group( m_dataset->getRootGroup() );
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
// TestNavigation::test_group
//---------------------------------------------------------------------------
bool TestNavigation::test_group(const IGroupPtr& group)
{
  bool doContinue = true;
  
  TestGroup::CommandGroup cmd;
  cmd.command = TestGroup::open;

  while( doContinue && cmd.command != TestGroup::back)
  {
    TestGroup::listChild( group );
    
    cmd = TestGroup::getCommandGroup( TestGroup::getCommand(group) );    

    try
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
        cout<<Tools::displayDataItem( itm )<<endl;
        cout<<Tools::displayArray( itm->getData() )<<endl;
      }
    }
    catch ( yat::Exception& e )
    {
      cout<<"=======> Exception !! <======="<<endl;
      cout<<"Orig.: "<<e.errors[0].origin<<endl;
      cout<<"Desc.: "<<e.errors[0].desc<<endl;
      cout<<"Reas.: "<<e.errors[0].reason<<endl;
      cout<<"=============================="<<endl;
    }
    
    if( doContinue && cmd.command == TestGroup::exit )
    {
      doContinue = false;
    }
  }
  return doContinue;
}

//---------------------------------------------------------------------------
// TestNavigation::test_logical_group
//---------------------------------------------------------------------------
bool TestNavigation::test_logical_group(const LogicalGroupPtr& group)
{
  bool doContinue = true;
  
  TestLogicalGroup::CommandLogicalGroup cmd;
  cmd.command = TestLogicalGroup::open;
  
  while( doContinue && cmd.command != TestLogicalGroup::back)
  {
    cmd = TestLogicalGroup::getCommandLogicalGroup( TestLogicalGroup::getCommand(group) );    

    try
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
        cout<<Tools::displayDataItem( itm )<<endl;
        cout<<Tools::displayArray( itm->getData() )<<endl;
      }
    }
    catch ( yat::Exception& e )
    {
      cout<<"=======> Exception !! <======="<<endl;
      cout<<"Orig.: "<<e.errors[0].origin<<endl;
      cout<<"Desc.: "<<e.errors[0].desc<<endl;
      cout<<"Reas.: "<<e.errors[0].reason<<endl;
      cout<<"=============================="<<endl;
    }
   
    if( doContinue && cmd.command == TestLogicalGroup::exit )
    {
        doContinue = false;
    }
  }
  return doContinue;
}

}
