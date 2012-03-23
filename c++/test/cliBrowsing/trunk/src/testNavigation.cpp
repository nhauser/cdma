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

#include <yat/utils/String.h>
#include <testNavigation.h>
#include <cdma/utils/ArrayUtils.h>
#include <cdma/array/Array.h>

#include <tools.h>
using namespace std;
namespace cdma
{

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
  m_depth = 0;
}

//---------------------------------------------------------------------------
// TestNavigation::run_test
//---------------------------------------------------------------------------
void TestNavigation::run_test()
{
  m_log = "";
  m_testOk = 0;
  m_total = 0;

  std::cout<<"Performing navigation tests: please wait..."<<std::endl;

/*  scan_group();
  std::cout<<"---------------------------------"<<std::endl;
  std::cout<<"---------------------------------"<<std::endl;
  std::cout<<"---------------------------------"<<std::endl;
*/  scan_dataitem();


//  test_dataitem();
  
//  test_attribute();

//  test_dimension();
 
  std::cout<<"Result test: "<<m_testOk<<" / " <<m_total<<std::endl;
  std::cout<<m_log<<std::endl;
}


//---------------------------------------------------------------------------
// TestNavigation::scan_group
//---------------------------------------------------------------------------
void TestNavigation::scan_group(std::string indent)
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::scan_group");
  IGroupPtr grp = m_current;
  std::cout<<Tools::displayGroup( m_current, indent );
  
  std::list<IGroupPtr> list = m_current->getGroupList();
  for( std::list<IGroupPtr>::iterator iter = list.begin(); iter != list.end(); iter++ )
  {
    m_current = (*iter);
    scan_group( indent + "  " );
  }
  m_current = grp;
}

//---------------------------------------------------------------------------
// TestNavigation::scan_dataitem
//---------------------------------------------------------------------------
void TestNavigation::scan_dataitem(std::string indent)
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::scan_group");
  IGroupPtr grp = m_current;
  
  std::cout<<Tools::displayGroup( m_current, indent );
  std::cout<<indent<<" |\\  "<<std::endl;
  // Scan all sub group
  std::list<IGroupPtr> list_grp = m_current->getGroupList();
  for( std::list<IGroupPtr>::iterator iter = list_grp.begin(); iter != list_grp.end(); iter++ )
  {
    m_current = (*iter);
    scan_dataitem( indent + " | | ");
  }

  // Scan all dataitem
  std::list<IDataItemPtr> list_itm = m_current->getDataItemList();
  for( std::list<IDataItemPtr>::iterator iter = list_itm.begin(); iter != list_itm.end(); iter++ )
  {
    IDataItemPtr item = (*iter);
    std::cout<<Tools::displayDataItem( item, indent + " | | ");
    
    std::list<IDimensionPtr> list_dim = m_current->getDimensionList();
    if( list_dim.size() > 0) 
    {
      std::cout<<indent<<" | |\\  "<<std::endl;
      for( std::list<IDimensionPtr>::iterator iter_d = list_dim.begin(); iter_d != list_dim.end(); iter_d++ )
      {
        std::cout<<Tools::displayDimension( (*iter_d), indent + " | | | ")<<std::endl;
      }
      std::cout<<indent<<" | |/  "<<std::endl;    
    }   
  }
  std::cout<<indent<<" |/  "<<std::endl;
  
  m_current = grp;
}

//---------------------------------------------------------------------------
// TestNavigation::test_dataitem
//---------------------------------------------------------------------------
bool TestNavigation::test_dataitem(const IDataItemPtr& )
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_dataitem");
  bool result = false;
  

  if( result )
    m_testOk++;
  m_total++;
  m_log += "\n- TestNavigation::test_dataitem: ";
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}

//---------------------------------------------------------------------------
// TestNavigation::test_attribute
//---------------------------------------------------------------------------
bool TestNavigation::test_attribute(const IAttributePtr& )
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_attribute");
  bool result = false;
  

  if( result )
    m_testOk++;
  m_total++;
  m_log += "\n- TestNavigation::test_attribute: ";
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}

//---------------------------------------------------------------------------
// TestNavigation::test_group
//---------------------------------------------------------------------------
bool TestNavigation::test_dimension(const IDimensionPtr& )
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_dimension");
  bool result = false;
  

  if( result )
    m_testOk++;
  m_total++;
  m_log += "\n- TestNavigation::test_dimension: ";
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}


}
