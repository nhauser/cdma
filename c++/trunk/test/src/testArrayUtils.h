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

#ifndef __NXS_TEST_ARRAYUTILS_H__
#define __NXS_TEST_ARRAYUTILS_H__

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <cdma/array/Array.h>

namespace cdma
{

class TestArrayUtils
{
public:
  TestArrayUtils( const ArrayPtr& array );
  
  void run_test();
  
protected:
  bool test_checkShape();
    
  bool test_reduce();

  bool test_reduce_int();
    
  bool test_reduceTo();
    
  bool test_reshape();

  bool test_slice();

  bool test_transpose();

  bool test_flip();

  bool test_permute();
  
private:
  ArrayPtr m_array;
  std::string m_log;
  int m_total;
  int m_testOk;
};

}

#endif // __NXS_TEST_ARRAYUTILS_H__
