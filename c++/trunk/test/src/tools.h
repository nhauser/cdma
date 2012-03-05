//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __NXS_TEST_TOOLS_H__
#define __NXS_TEST_TOOLS_H__

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <yat/any/Any.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/dictionary/LogicalGroup.h>
#include <cdma/dictionary/Key.h>

namespace cdma
{
  class Tools
  {
    public:
      // Convert an int into string
      static std::string convertInt(int number);

      template<typename T> static std::string displayArray(T* array, int length);

      // display a template array into a string
      template<typename T> static std::string displayArray(std::vector<T> array);

      // display Array's content
      static std::string displayArray( const cdma::ArrayPtr& array, int maxCell = 15 );

      // display properties of a DataItem
      static std::string displayDataItem( const cdma::IDataItemPtr& item);

      
      //static std::string iterate_over_keys( cdma::LogicalGroupPtr group, const std::string& indent, std::list<cdma::IDataItemPtr>& items );

      // Get key
      static cdma::KeyPtr getKey( const cdma::LogicalGroupPtr& group, const std::string& key );
      
      // Display values using iterators
      static std::string displayValues(const ArrayPtr& array);

      // Display values of an iterator
      static std::string scanValues(ArrayIterator& begin, ArrayIterator& end, int maxCell = 15);
      
      // display all keys in a structured string representation
      static std::string iterate_over_keys( LogicalGroupPtr group, std::list<IDataItemPtr>& items );
      
    private:
      static std::string iterate_over_keys( LogicalGroupPtr group, const std::string& indent, std::list<IDataItemPtr>& items );
  };

  template<typename T> std::string Tools::displayArray(std::vector<T> array)
  {
    std::stringstream ss;
    ss << "[";
    for( unsigned int i = 0; i < array.size(); i++ )
    {
      ss<< array[i];
      if( i < array.size() - 1 )
      {
        ss <<", ";
      }
    }
    ss<<"]";
    return ss.str();
  }

template<typename T> std::string Tools::displayArray(T* array, int length)
  {
    std::stringstream ss;
    ss << "[";
    for( unsigned int i = 0; i < length; i++ )
    {
      ss<< array[i];
      if( i < length - 1 )
      {
        ss <<", ";
      }
    }
    ss<<"]";
    return ss.str();
  }

} //namespace CDMACore
#endif //__NXS_TEST_TOOLS_H__

