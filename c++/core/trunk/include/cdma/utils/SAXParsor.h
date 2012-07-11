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

namespace cdma
{

/// @cond internal

// !! SAXParsor class is strictly for internal purpose !!

//=============================================================================
/// Purpose : Reading a XML file using libxml2 library then parsing the document
/// to construct a configuration
//=============================================================================
class SAXParsor
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

private:
  yat::String m_current_parsed_source;

protected:
  /// Parse a XML node
  void parse_node(void *_pNode, INodeAnalyser* pAnalyser);

public :
  SAXParsor();
  ~SAXParsor();

  /// Start the SAX parsor
  void start(const std::string& document_path, INodeAnalyser* pRootAnalyser);
};

/// Macro helper to retrive attribute value
#define FIND_ATTR_VALUE(attrs, name, value_string) \
    cdma::SAXParsor::AttributesConstIterator cit = attrs.find(name); \
    if( cit == attrs.end() ) \
      throw Exception("NOT_FOUND", yat::String::str_format("Unable to get value for attribute '%s'", name), "cdma::SAXParsor"); \
    value_string = cit->second

/// Macro helper to retrive attribute value. nothrow
#define FIND_ATTR_VALUE_NO_THROW(attrs, name, value_string) \
    cdma::SAXParsor::AttributesConstIterator cit = attrs.find(name); \
    if( cit != attrs.end() ) \
      value_string = cit->second

/// @endcond internal

} // namespace

#endif
