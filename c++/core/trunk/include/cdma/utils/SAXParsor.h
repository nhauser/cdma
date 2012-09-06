//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

#ifndef __CDMA_SAX_PARSOR_H__
#define __CDMA_SAX_PARSOR_H__

#include <cdma/Common.h>

#include <map>
#include <stack>
#include <string>
#include <stdio.h>
#include <yat/utils/String.h>
#include <yat/utils/Singleton.h>

namespace cdma
{

/// @cond internal

// !! SAXParsor class is strictly for internal purpose !!

//=============================================================================
/// Purpose : Reading XML files using libxml2 library then parsing the document
/// to construct a configuration
//=============================================================================
class SAXParsor : public yat::Singleton<SAXParsor>
{
public:

  typedef std::map<std::string, std::string> Attributes;
  typedef std::map<std::string, std::string>::const_iterator AttributesConstIterator;

  /// Interface of all object than can analyse a XML element
  class INodeAnalyser
  {
  public:
    virtual INodeAnalyser* on_element(const yat::String& element_name, const Attributes& attrs, const yat::String& current_file)=0;
    virtual void on_element_content(const yat::String& element_name, const yat::String& element_content, const yat::String& current_file)=0;
    virtual void on_end_element(const yat::String& element_name)=0;
    virtual void release() {}
  };

protected:
  /// Parse a XML node
  void parse_node(void *_pNode, INodeAnalyser* pAnalyser);

public :
  SAXParsor();
  ~SAXParsor();

  /// Start the SAX parsor
  static void start(const std::string& document_path, INodeAnalyser* pRootAnalyser);

  /// helper method to check an attribute by its name
  static bool has_attribute(const Attributes& attrs, const yat::String& name)
  {
    if( attrs.find(name) != attrs.end() )
      return true;
    return false;
  }

  /// helper method to get a the value of an attribute
  static yat::String attribute_value(const Attributes& attrs, const yat::String& name)
  {
    cdma::SAXParsor::AttributesConstIterator cit = attrs.find(name);
    if( cit == attrs.end() )
      throw yat::Exception("NOT_FOUND", 
                      PSZ_FMT("Unable to get value for attribute '%s'", PSZ(name)),
                      "cdma::SAXParsor");
    return cit->second;
  }
};

/// Macro helper to retrieve attribute value
#define FIND_ATTR_VALUE(attrs, name, value_string) \
    cdma::SAXParsor::AttributesConstIterator cit = attrs.find(name); \
    if( cit == attrs.end() ) \
    throw yat::Exception("NOT_FOUND", yat::String::str_format("Unable to get value for attribute '%s'", name), "cdma::SAXParsor"); \
    value_string = cit->second

/// Macro helper to retrieve attribute value. nothrow
#define FIND_ATTR_VALUE_NO_THROW(attrs, name, value_string) \
    cdma::SAXParsor::AttributesConstIterator cit = attrs.find(name); \
    if( cit != attrs.end() ) \
      value_string = cit->second

/// Macro helper to test attribute presence
/// usage:
/// IF_HAS_ATTR(attrs, name)
/// {
/// ...
/// }
#define IF_HAS_ATTR(attrs, name) \
    cdma::SAXParsor::AttributesConstIterator cit = attrs.find(name); \
    if( cit != attrs.end() )

/// @endcond internal

} // namespace

#endif
