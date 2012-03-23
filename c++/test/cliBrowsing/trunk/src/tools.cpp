#include <stdio.h>
#include <typeinfo>

#include <cdma/navigation/IGroup.h>
#include <cdma/array/View.h>
#include <cdma/array/Array.h>
#include <cdma/array/ArrayIterator.h>
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/LogicalGroup.h>
#include <cdma/Common.h>
#include <tools.h>

using namespace std;
using namespace cdma;

namespace cdma
{
  //-----------------------------------------------------------------------------
  // Tools::convertInt
  //-----------------------------------------------------------------------------
  std::string Tools::convertInt(int number)
  {
    stringstream ss;
    ss << number;
    return ss.str();
  }

  //-----------------------------------------------------------------------------
  // Tools::displayDataItem
  //-----------------------------------------------------------------------------
  std::string Tools::displayDataItem( const IDataItemPtr& item, std::string indent)
  {
    stringstream res;

    res<<indent<<"DataItem name: "<<item->getName()<<endl;
    res<<indent<<"  short name: "<<item->getShortName()<<endl;
    res<<indent<<"  Physical location: "<<item->getLocation()<<endl;
    res<<indent<<"  Rank: "<<item->getRank();
    res<<" Shape: "<<displayArray( item->getShape() );
    res<<" Nb element: "<<item->getSize();
    res<<" Type: "<<item->getType().name()<<std::endl;
    std::list<IAttributePtr> list_attr = item->getAttributeList();
    if( list_attr.size() > 0 )
    {
      res<<indent<<"   - getAttributeList: ";
      int i = 0;
      for( std::list<IAttributePtr>::iterator iter = list_attr.begin(); iter != list_attr.end(); iter++)
      {
        if( iter != list_attr.begin() )
          res<<", ";
        if( i > 0 && i % 10 == 0 )
          res<<std::endl<<indent<<"                       ";
        res<<(*iter)->getName();
        i++;
      }
      res<<std::endl;
    }
    std::list<IDimensionPtr> list_dim = item->getDimensionList();
    if( list_dim.size() > 0 )
    {
      res<<indent<<"   - getDimensionList: ";
      int i = 0;
      for( std::list<IDimensionPtr>::iterator iter = list_dim.begin(); iter != list_dim.end(); iter++)
      {
        if( iter != list_dim.begin() )
          res<<", ";
        if( i > 0 && i % 10 == 0 )
          res<<std::endl<<indent<<"                       ";
        res<<(*iter)->getName();
        i++;
      }
      res<<std::endl;
    }
    return res.str();
  }

  //-----------------------------------------------------------------------------
  // Tools::displayGroup
  //-----------------------------------------------------------------------------
  std::string Tools::displayDimension( const cdma::IDimensionPtr& dim, std::string indent)
  {
    stringstream res;
    
    res<<indent<<"Dimension: "<<dim->getName()<<std::endl;
    res<<indent<<"  Length: "<<dim->getLength()<<"  Unlimited: "<<(dim->isUnlimited() ? "true" : "false");
    res<<"  Variable length:"<<(dim->isVariableLength() ? "true" : "false");
    res<<"  Shared: "<<(dim->isShared() ? "true" : "false")<<std::endl;
    res<<indent<<"  Axis: "<<dim->getDimensionAxis();
    res<<"  Display order: "<<dim->getDisplayOrder();
    res<<"  Units: "<<dim->getUnitsString()<<std::endl;
    ArrayPtr arr = dim->getCoordinateVariable();
    res << Tools::displayArray(arr, 15);
    return res.str();
  }

  //-----------------------------------------------------------------------------
  // Tools::displayLogicalGroup
  //-----------------------------------------------------------------------------
  std::string Tools::displayLogicalGroup( const cdma::LogicalGroupPtr& group, std::string indent )
  {
    stringstream res;
    res<<indent<<"Logical group: "<<group->getShortName()<<std::endl;
    res<<indent<<" - getName='"<<group->getName()<<"'"<<std::endl;
    res<<indent<<" - getLocation='"<<group->getLocation()<<"'"<<std::endl;
    
    std::list<KeyPtr> list_keys = group->getKeys();
    if( list_keys.size() > 0 )
    { 
      res<<indent<<" - getKeys:"<<std::endl;   
      for( std::list<KeyPtr>::iterator it = list_keys.begin(); it != list_keys.end(); it++ )
      {
        if( (*it)->getType() != Key::ITEM )
        {
          res<<indent<<"     Group: '"<< (*it)->getName() <<"'"<<std::endl;
        }
      }
      for( std::list<KeyPtr>::iterator it = list_keys.begin(); it != list_keys.end(); it++ )
      {
        if( (*it)->getType() == Key::ITEM )
        {
          res<<indent<<"     Item: '"<< (*it)->getName() <<"'"<<std::endl;
        }
      }
    }
    return res.str();  
  }

  //-----------------------------------------------------------------------------
  // Tools::displayGroup
  //-----------------------------------------------------------------------------
  std::string Tools::displayAttribute( const cdma::IAttributePtr& attr, std::string indent )
  {
    stringstream res;
    
    res<<indent<<"Attribute: '"<<attr->getName();
    
    if( attr->isString() )
    {
      res<<"'  value: '"<<attr->getStringValue()<<"'";
    }
    else 
    {
      res<<"'  value: '"<<attr->getFloatValue()<<"'";
    }
    
    return res.str();
  }

  //-----------------------------------------------------------------------------
  // Tools::displayGroup
  //-----------------------------------------------------------------------------
  std::string Tools::displayGroup( const cdma::IGroupPtr& group, std::string indent )
  {
    stringstream res;
    res<<indent<<"Group: "<<group->getShortName()<<std::endl;
    res<<indent<<" - getName='"<<group->getName()<<"'"<<std::endl;
    res<<indent<<" - getLocation='"<<group->getLocation()<<"'"<<std::endl;
    res<<indent<<" - isRoot='"<<( group->isRoot() ? "true": "false" )<<"'"<<std::endl;
    res<<indent<<" - isEntry='"<<( group->isEntry() ? "true": "false" )<<"'"<<std::endl;
    std::list<IGroupPtr> list_grp = group->getGroupList();
    if( list_grp.size() > 0 )
    {
      res<<indent<<" - getGroupList: ";
      int i = 0;
      for( std::list<IGroupPtr>::iterator iter = list_grp.begin(); iter != list_grp.end(); iter++)
      {
        if( iter != list_grp.begin() )
          res<<", ";
        if( i > 0 && i % 10 == 0 )
          res<<std::endl<<indent<<"                 ";
        res<<(*iter)->getShortName();
        i++;
      }
      res<<std::endl;
    }
    std::list<IDataItemPtr> list_item = group->getDataItemList();
    if( list_item.size() > 0 )
    {
      res<<indent<<" - getDataItemList: ";
      int i = 0;
      for( std::list<IDataItemPtr>::iterator iter = list_item.begin(); iter != list_item.end(); iter++)
      {
        if( iter != list_item.begin() )
          res<<", ";
        if( i > 0 && i % 10 == 0 )
          res<<std::endl<<indent<<"                    ";
        res<<(*iter)->getShortName();
        i++;
      }
      res<<std::endl;
    }
    std::list<IAttributePtr> list_attr = group->getAttributeList();
    if( list_attr.size() > 0 )
    {
      res<<indent<<" - getAttributeList: ";
      int i = 0;
      for( std::list<IAttributePtr>::iterator iter = list_attr.begin(); iter != list_attr.end(); iter++)
      {
        if( iter != list_attr.begin() )
          res<<", ";
        if( i > 0 && i % 10 == 0 )
          res<<std::endl<<indent<<"                     ";
        res<<(*iter)->getName();
        i++;
      }
      res<<std::endl;
    }
    std::list<IDimensionPtr> list_dim = group->getDimensionList();
    if( list_dim.size() > 0 )
    {
      res<<indent<<" - getDimensionList: ";
      int i = 0;
      for( std::list<IDimensionPtr>::iterator iter = list_dim.begin(); iter != list_dim.end(); iter++)
      {
        if( iter != list_dim.begin() )
          res<<", ";
        if( i > 0 && i % 10 == 0 )
          res<<std::endl<<indent<<"                     ";
        res<<(*iter)->getName();
        i++;
      }
      res<<std::endl;
    }
    return res.str();
  }

  //-----------------------------------------------------------------------------
  // Tools::displayArray
  //-----------------------------------------------------------------------------
  std::string Tools::displayArray( const ArrayPtr& arr, int maxCell )
  {
    ArrayIterator begin = arr->begin();
    ArrayIterator end   = arr->end();
    
    return Tools::scanValues( begin, end, maxCell );
  }

  //-----------------------------------------------------------------------------
  // Tools::getKey
  //-----------------------------------------------------------------------------
  cdma::KeyPtr Tools::getKey( const cdma::LogicalGroupPtr& group, const string& key )
  {
    list<KeyPtr> keys = group->getKeys();
    list<KeyPtr>::iterator key_it;

    for( key_it = keys.begin(); key_it != keys.end(); key_it++ )
    {
      if( (*key_it)->getName() == key )
      {
        return *key_it;
      }
    }
    return NULL;
  }
  
  //-----------------------------------------------------------------------------
  // Tools::displayValues
  //-----------------------------------------------------------------------------
  std::string Tools::displayValues(const ArrayPtr& array) 
  {
    std::cout<<"Entering Tools::displayValues"<<std::endl;
    std::string str;
    ArrayIterator begin = array->begin();
    ArrayIterator end = array->end();
    str = scanValues(begin, end, 15);
    std::cout<<"Leaving Tools::displayValues"<<std::endl;
    return str;
  }
  
  //-----------------------------------------------------------------------------
  // Tools::scanValues
  //-----------------------------------------------------------------------------
  std::string Tools::scanValues(ArrayIterator& iterator, ArrayIterator& end, int maxCell) 
  {
    std::cout<<"Entering Tools::scanValues"<<std::endl;

    stringstream res;
    int j = 0;
    vector<int> current = iterator.getPosition();
    double value;

    while( iterator != end )
    {
      current = iterator.getPosition();
      if( current[0] < maxCell )
      {
        if( current[current.size() - 1] == 0 )
        {
          if( current[0] != 0 )
          {
            res << "...\n";
          }
          res << "raw" << Tools::displayArray( iterator.getPosition() ) << "= ";
          j++;
        }
        value = iterator.getValue<double>();
        if( current[current.size() - 1] < maxCell )
        {
          stringstream tmp_ss;
          tmp_ss << value;
          if( tmp_ss.str().size() > 9 )
          {
            res << tmp_ss.str().substr(0, 5) << "... ";
          }
          else
          {
            int size = tmp_ss.str().size();
            for( int i = 0; i < 9 - size; i++ )
            {
              tmp_ss << " ";
            }
            res << tmp_ss.str().c_str();
          }
        }
      }
      ++iterator;
    }
    res << "...";
    std::string str = res.str();
    std::cout<<"Leaving Tools::scanValues"<<std::endl;
    return str;
  }
  
  //-----------------------------------------------------------------------------
  // Tools::scanValues
  //-----------------------------------------------------------------------------
  string Tools::iterate_over_keys( LogicalGroupPtr group, list<IDataItemPtr>& items )
  {
    return Tools::iterate_over_keys( group, "", items );
  }
  
  string Tools::iterate_over_keys( LogicalGroupPtr group, const string& indent, list<IDataItemPtr>& items )
  {
    // Get all keys under given group
    list<KeyPtr> keys = group->getKeys();

    // Iterate over keys
    stringstream res;
    list<KeyPtr>::iterator keys_it;
    for( keys_it = keys.begin(); keys_it != keys.end(); keys_it++ )
    {
      // If key matches a LogicalGroup
      if( (*keys_it)->getType() == Key::GROUP )
      {
        // Get the group
        LogicalGroupPtr tmpGroup = group->getGroup( *keys_it );
        res<<indent<<"|--> Key group: "<<tmpGroup->getShortName()<<endl;

        // Start recursion
        res<<iterate_over_keys( tmpGroup, "| " + indent, items );
      }
      // If keys matches a DataItem
      else if( (*keys_it)->getType() == Key::ITEM )
      {
        // Get that item
        IDataItemPtr tmpData = group->getDataItem( *keys_it );
        res<<indent<<"|--> Key item: "<<tmpData->getShortName()<<endl;

        // Store it in in/out list
        items.push_back(tmpData);
      }
    }
    // Return a structure string representation
    return res.str();
  }

} // namespace
