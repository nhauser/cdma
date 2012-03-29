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
#include <internal/testArrayUtils.h>
#include <cdma/utils/ArrayUtils.h>
#include <cdma/array/Array.h>

#include <internal/tools.h>
using namespace std;
namespace cdma
{

//=============================================================================
//
// TestArrayUtils
//
//=============================================================================
//---------------------------------------------------------------------------
// TestArrayUtils::TestArrayUtils
//---------------------------------------------------------------------------
TestArrayUtils::TestArrayUtils( const ArrayPtr& array )
{
  m_array = array;
}

//---------------------------------------------------------------------------
// TestArrayUtils::run_test
//---------------------------------------------------------------------------
void TestArrayUtils::run_test()
{
  m_log = "";
  m_testOk = 0;
  m_total = 0;

  std::cout<<"Performing test on arrays: please wait..."<<std::endl;

  test_checkShape();
  
  test_reduce();
  
  test_reduce_int();
  
  test_reduceTo();
  
  test_reshape();
  
  test_slice();
  
  test_transpose();

  test_flip();
  
  test_permute();
 
  std::cout<<"Result test: "<<m_testOk<<" / " <<m_total<<std::endl;
  std::cout<<m_log<<std::endl;
}

//---------------------------------------------------------------------------
// TestArrayUtils::test_checkShape
//---------------------------------------------------------------------------
bool TestArrayUtils::test_checkShape()
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_checkShape");
  bool result = false;
  
  ArrayUtilsPtr util = new ArrayUtils( m_array );
  ArrayPtr array = new Array( m_array, m_array->getView() );
  
  std::vector<int> start;
  std::vector<int> shape = array->getShape();
  for(int i = 0; i < m_array->getRank(); i++ ) {
    start.push_back(0);
  }
  shape[ m_array->getRank() - 1 ] = shape[ m_array->getRank() - 1 ] - 1;
  ArrayPtr slice = m_array->getRegion(start, shape);
  
  result = util->checkShape( array );
  result = ( result && (! util->checkShape( slice )) );

  if( result )
    m_testOk++;
  m_total++;
  m_log += "\n- ArrayUtils::checkShape: ";
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}
  
//---------------------------------------------------------------------------
// TestArrayUtils::test_reduce
//---------------------------------------------------------------------------
bool TestArrayUtils::test_reduce()
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_reduce");
  bool result = false;
  
  std::vector<int> start;
  std::vector<int> shape = m_array->getShape();
  for(int i = 0; i < m_array->getRank(); i++ ) 
  {
    start.push_back(0);
    if( i < m_array->getRank() - 1 )
    {
      shape[i] = 1;
    }
  }

  ArrayPtr slice = m_array->getRegion(start, shape);
  ArrayUtilsPtr util = new ArrayUtils(slice);
  util = util->reduce();
  slice = util->getArray();
  
  result = ( slice->getRank() == 1 && slice->getShape()[0] == m_array->getShape()[m_array->getRank() - 1] );

  m_log += "\n- ArrayUtils::reduce(): ";
  if( result )
  {
    ArrayIterator iterSrc = m_array->begin();
    ArrayIterator iterDst = slice->begin();
    ArrayIterator iterDstEnd = slice->end();    
    m_log += "View ok => checking values ";
    
    if( iterSrc == iterDstEnd )
    {
      result = false;
    }
    else
    {
      while( iterDst != iterDstEnd )
      {
        if( iterSrc.getValue<double>() != iterDst.getValue<double>() )
        {
          result = false;
          break;
        }
        ++iterDst;
        ++iterSrc;
      }
    }
  }
  
  if( result )
    m_testOk++;
  m_total++;

  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}

//---------------------------------------------------------------------------
// TestArrayUtils::test_reduce_int
//---------------------------------------------------------------------------
bool TestArrayUtils::test_reduce_int()
{
CDMA_FUNCTION_TRACE("TestArrayUtils::test_reduce_int");
// reduce(int dim)
  bool result = false;
  int dimReduced = -1;
  std::vector<int> start;
  std::vector<int> shape = m_array->getShape();
  
  for(int i = 0; i < m_array->getRank(); i++ )
  {
    start.push_back(0);
    if( i < m_array->getRank() - 1 )
    {
      if( i == m_array->getRank() - 2 ) {
        dimReduced = i;
      }
      shape[i] = 1;
    }
  }

  m_log += "\n- ArrayUtils::reduce_int(): ";
  if( dimReduced >=0 ) 
  {
    ArrayPtr slice = m_array->getRegion(start, shape);
    ArrayUtilsPtr util = new ArrayUtils(slice);
    util = util->reduce(dimReduced);
    slice = util->getArray();

    result = ( slice->getRank() == m_array->getRank() - 1 && slice->getShape()[slice->getRank() - 1] == m_array->getShape()[m_array->getRank() - 1] );

    
    if( result )
    {
      ArrayIterator iterSrc = m_array->begin();
      ArrayIterator iterDst = slice->begin();
      ArrayIterator iterDstEnd = slice->end();    
      
      m_log += "View ok => checking values ";
      
      if( iterDst == iterDstEnd )
      {
        result = false;
      }
      else
      {
        while( iterDst != iterDstEnd )
        {
          if( iterSrc.getValue<double>() != iterDst.getValue<double>() )
          {
            result = false;
            break;
          }
          ++iterDst;
          ++iterSrc;
        }
      }
    }
  }
  else 
  {
    m_log += "UNABLE to procede to test rank is too low! ";
  }
  if( result )
  {
    m_testOk++;
  }
  m_total++;

  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}
  
//---------------------------------------------------------------------------
// TestArrayUtils::test_reduceTo
//---------------------------------------------------------------------------
bool TestArrayUtils::test_reduceTo()
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_reduceTo");
  bool result = false;
  
  std::vector<int> start;
  std::vector<int> shape = m_array->getShape();

  m_log += "\n- ArrayUtils::reduceTo(int rank): ";
  
  if( m_array->getRank() > 1 )
  {
    for(int i = 0; i < m_array->getRank(); i++ )
    {
      start.push_back(0);
      if( i < m_array->getRank() - 1 )
      {
        shape[i] = 1;
      }
    }

    ArrayPtr slice = m_array->getRegion(start, shape);
    ArrayUtilsPtr util = new ArrayUtils( slice );
    result = ( util->reduceTo(1)->getArray()->getRank() == 1 );
    
    shape[m_array->getRank() - 1] = 1;
    slice = m_array->getRegion(start, shape);
    util  = new ArrayUtils( slice );
    result = ( util->reduceTo(0)->getArray()->getRank() == 1 && result );
  }
  else 
  {
    m_log += "UNABLE to procede to test rank is too low! ";
  }
  if( result )
  {
    m_testOk++;
  }
  m_total++;
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}
  
//---------------------------------------------------------------------------
// TestArrayUtils::test_reshape
//---------------------------------------------------------------------------
bool TestArrayUtils::test_reshape()
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_reshape");
// reshape(std::vector<int> shape)
  bool result = true;
  std::vector<int> shape = m_array->getShape();
  if( m_array->getRank() >= 2 )
  {
    bool split = true;
    for( int i = 0; i < m_array->getRank(); i++ )
    {
      if( i % 2 == 0 )
      {
        if( split && i == m_array->getRank() - 1)
        {
          int la = shape[i] / 2;
          shape[i] = 2;
          shape.push_back( la );
          split = false;
        }
        else
        {
          shape[i] = shape[i] / 2;
        }
      }
      else
        shape[i] = shape[i] * 2;
      
    }
  }
  
  ArrayUtilsPtr util = new ArrayUtils( m_array );
  ArrayPtr array = util->reshape( shape )->getArray();
  ArrayIterator beginSrc = array->begin();
  ArrayIterator endSrc   = array->end();
  ArrayIterator beginDst = m_array->begin();
  ArrayIterator endDst   = m_array->end();

  while( beginSrc != endSrc && beginDst != endDst )
  {
    if( beginSrc.getValue<double>() != beginDst.getValue<double>() )
    {
      result = false;
      break;
    }
  
    ++beginSrc;
    ++beginDst;
  }
  
  if( beginSrc != endSrc || beginDst != endDst )
    result = false;
  
  if( result )
    m_testOk++;
  else
  {
    std::cout<<beginSrc.getValue<double>()<<" != "<<beginDst.getValue<double>()<<std::endl;
    std::cout<<"at pos: " << beginSrc.currentElement() <<"  ||  "<< beginDst.currentElement() <<std::endl;
    std::cout<<"array: "<< Tools::displayArray( beginSrc.getPosition() )<<"   |   m_array: "<< Tools::displayArray( beginDst.getPosition() )<<std::endl; 
  }
  m_total++;
  m_log += "\n- ArrayUtils::reshape(std::vector<int> shape): ";
  m_log += "before reshape: " + Tools::displayArray( m_array->getShape() ) + "  after: " + Tools::displayArray( shape ) + "   "; 
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}

//---------------------------------------------------------------------------
// TestArrayUtils::test_slice
//---------------------------------------------------------------------------
bool TestArrayUtils::test_slice()
{
CDMA_FUNCTION_TRACE("TestArrayUtils::test_slice");
// slice(int dim, int value)
  bool result = true;

  m_log += "\n- ArrayUtils::slice(int dim, int value): ";  

  if( m_array->getRank() >= 2 )
  {
    std::vector<int> start;
    std::vector<int> shape = m_array->getShape();
    for(int i = 0; i < m_array->getRank(); i++ )
    {
      if( i == m_array->getRank() - 2 )
      {
        start.push_back(1);
        shape[i] = 1;
      }
      else
      {
        start.push_back(0);
      }
    }

    ArrayUtilsPtr util = new ArrayUtils( m_array );
    ArrayPtr region = m_array->getRegion(start, shape);
    ArrayPtr slice  = util->slice(0, 1)->getArray();
    
    ArrayIterator beginSrc = slice->begin();
    ArrayIterator endSrc   = slice->end();
    ArrayIterator beginDst = region->begin();
    ArrayIterator endDst   = region->end();

    if( beginSrc == endSrc )
    {
      result = false;
    }
    else
    {
      while( beginSrc != endSrc && beginDst != endDst )
      {
        if( beginSrc.getValue<double>() != beginDst.getValue<double>() )
        {
          result = false;
          break;
        }
      
        ++beginSrc;
        ++beginDst;
      }
    }
  }
  else
  {
    result = false;
    m_log += "UNABLE to procede to test rank is too low! ";
  }
  
  if( result )
    m_testOk++;
  m_total++;

  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}

//---------------------------------------------------------------------------
// TestArrayUtils::test_transpose
//---------------------------------------------------------------------------
bool TestArrayUtils::test_transpose()
{
CDMA_FUNCTION_TRACE("TestArrayUtils::test_transpose");
// transpose(int dim1, int dim2)
  bool result = true;
  m_log += "\n- ArrayUtils::transpose(int dim1, int dim2): ";

  if( m_array->getRank() >= 2 )
  {
    std::vector<int> start;
    std::vector<int> shape = m_array->getShape();
    start.push_back(0);
    for(int i = 1; i < m_array->getRank(); i++ )
    {
      shape[i] = 1;
      start.push_back(0);
    }

    ArrayUtilsPtr util = new ArrayUtils( m_array );
    ArrayPtr region = m_array->getRegion(start, shape);
    ArrayPtr transpose  = util->transpose(0, m_array->getRank() - 1)->getArray();

    ArrayIterator beginSrc = transpose->begin();
    ArrayIterator endSrc   = transpose->end();
    ArrayIterator beginDst = region->begin();
    ArrayIterator endDst   = region->end();


    if( beginSrc == endSrc )
    {
      result = false;
    }
    else
    {
      while( beginSrc != endSrc && beginDst != endDst )
      {
        if( beginSrc.getValue<double>() != beginDst.getValue<double>() )
        {
          result = false;
          break;
        }
      
        ++beginSrc;
        ++beginDst;
      }
    }
    
    if( ! result )
    {
      std::cout<<"============= transpose ==============" <<std::endl;
      std::cout<<beginSrc.getValue<double>()<<" != "<<beginDst.getValue<double>()<<std::endl;
      std::cout<<"at pos: " << beginSrc.currentElement() <<"  ||  "<< beginDst.currentElement() <<std::endl;
      std::cout<<"array: "<< Tools::displayArray( beginSrc.getPosition() )<<"   |   m_array: "<< Tools::displayArray( beginDst.getPosition() )<<std::endl;
      std::cout<<"======================================" <<std::endl;
    }
    else
      m_testOk++;
  }
  else
  {
    result = false;
    m_log += "UNABLE to procede to test rank is too low! ";
  }
  
  m_total++;
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}
          
//---------------------------------------------------------------------------
// TestArrayUtils::test_flip
//---------------------------------------------------------------------------
bool TestArrayUtils::test_flip()
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_flip");
  bool result = false;
  
  ArrayUtilsPtr util = new ArrayUtils( m_array );
  for( int i = 0; i < m_array->getRank(); i++ )
  {
    util = util->flip(i);
  }
  ArrayPtr flip  = util->getArray();

  ArrayIterator beginSrc = flip->begin();
  ArrayIterator endSrc   = flip->end();
  ArrayIterator beginDst = m_array->end();
  ArrayIterator endDst   = m_array->begin();
  --beginDst;
  --endDst;

  if( beginSrc != endSrc && beginDst != endDst )
  {
    result = true;
    while( beginSrc != endSrc && beginDst != endDst )
    {
      if( beginSrc.getValue<double>() != beginDst.getValue<double>() )
      {
        result = false;
        break;
      }
    
      ++beginSrc;
      --beginDst;
    }
  }

  if( beginSrc != endSrc )
  {
    result = false;
  }

  if( result )
    m_testOk++;
  if( ! result )
  {
    std::cout<<"================ flip ================" <<std::endl;
    std::cout<<beginSrc.getValue<double>()<<" != "<<beginDst.getValue<double>()<<std::endl;
    std::cout<<"at pos: " << beginSrc.currentElement() <<"  ||  "<< beginDst.currentElement() <<std::endl;
    std::cout<<"end pos: " << endSrc.currentElement() <<"  ||  "<< endDst.currentElement() <<std::endl;
    std::cout<<"beg iter array: "<< Tools::displayArray( beginSrc.getPosition() )<<"   ||   beg iter m_array: "<< Tools::displayArray( beginDst.getPosition() )<<std::endl;
    std::cout<<"end iter array: "<< Tools::displayArray( endSrc.getPosition() )<<"   ||   end iter m_array: "<< Tools::displayArray( endDst.getPosition() )<<std::endl;
    std::cout<<"======================================" <<std::endl;
  }
  
  m_total++;
  m_log += "\n- ArrayUtils::flip(int dim): ";
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}

//---------------------------------------------------------------------------
// TestArrayUtils::test_permute
//---------------------------------------------------------------------------
bool TestArrayUtils::test_permute()
{
  CDMA_FUNCTION_TRACE("TestArrayUtils::test_permute");
  bool result = true;
  m_log += "\n- ArrayUtils::permute(std::vector<int> dims): ";

  if( m_array->getRank() >= 2 )
  {
    ArrayUtilsPtr util  = new ArrayUtils( m_array );

    std::vector<int> permute_vec;
    for(int i = m_array->getRank() - 1; i >= 0 ; i-- )
    {
      if( i >= m_array->getRank() / 2 )
      {
        util = util->transpose(i, m_array->getRank() - 1 - i);
      }
      permute_vec.push_back(i);
    }
    
    ArrayPtr transpose  = util->getArray();
    util = new ArrayUtils( m_array );
    ArrayPtr permute    = util->permute( permute_vec )->getArray();

    ArrayIterator beginSrc = permute->begin();
    ArrayIterator endSrc   = permute->end();
    ArrayIterator beginDst = transpose->begin();
    ArrayIterator endDst   = transpose->end();
    
    if( beginSrc == endSrc )
    {
      result = false;
    }
    else
    {
      while( beginSrc != endSrc && beginDst != endDst )
      {
        if( beginSrc.getValue<double>() != beginDst.getValue<double>() )
        {
          result = false;
          break;
        }
      
        ++beginSrc;
        ++beginDst;
      }
    }

    if( ! result )
    {
      std::cout<<"============== permute ==============" <<std::endl;
      std::cout<<beginSrc.getValue<double>()<<" != "<<beginDst.getValue<double>()<<std::endl;
      std::cout<<"permute at pos: " << beginSrc.currentElement() <<"  ||  reference: "<< beginDst.currentElement() <<std::endl;
      std::cout<<"permute: "<< Tools::displayArray( beginSrc.getPosition() )<<"   |   reference: "<< Tools::displayArray( beginDst.getPosition() )<<std::endl;
      std::cout<<"======================================" <<std::endl;
    }
    else
      m_testOk++;
  }
  else
  {
    result = false;
    m_log += "UNABLE to procede to test rank is too low! ";
  }
  
  m_total++;
  m_log += ( result ? "Ok" : "Ko !!!" );
  
  return result;
}

}
