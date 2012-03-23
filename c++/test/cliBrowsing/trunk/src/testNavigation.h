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

#ifndef __NXS_TEST_NAVIGATION_H__
#define __NXS_TEST_NAVIGATION_H__

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <cdma/navigation/IGroup.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/navigation/IDimension.h>
#include <cdma/navigation/IAttribute.h>

namespace cdma
{

class TestNavigation
{
public:
  TestNavigation( const IDatasetPtr& array );
  
  void run_test();
  
protected:
  void scan_group(std::string indent = "");

  void scan_dataitem(std::string indent = "");

  bool test_group( const IGroupPtr& );

  bool test_dataitem( const IDataItemPtr& );
  
  bool test_attribute( const IAttributePtr& );

  bool test_dimension( const IDimensionPtr& );

private:
  IGroupPtr m_current;
  IDatasetPtr m_dataset;
  std::string m_log;
  int m_total;
  int m_testOk;
  int m_depth;
};

}

#endif // __NXS_TEST_NAVIGATION_H__
