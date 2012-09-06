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

//-----------------------------------------------------------------------------
// DEPENDENCIES
//-----------------------------------------------------------------------------
#include <iostream>

#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xinclude.h>

#include <yat/utils/Logging.h>

#include <cdma/exception/Exception.h>
#include <cdma/utils/SAXParsor.h>

#define THROW_LIBXML2_EXCEPTION(m,o) \
  throw cdma::Exception("LIBXML2_ERROR", m, o)

namespace cdma
{
//----------------------------------------------------------------------------
// SAXParsor::SAXParsor
//----------------------------------------------------------------------------
SAXParsor::SAXParsor()
{

}

//----------------------------------------------------------------------------
// SAXParsor::~SAXParsor
//----------------------------------------------------------------------------
SAXParsor::~SAXParsor()
{
}

//----------------------------------------------------------------------------
// SAXParsor::start
//----------------------------------------------------------------------------
void SAXParsor::start(const std::string& document_path, INodeAnalyser* pRootAnalyser)
{
  CDMA_STATIC_FUNCTION_TRACE("SAXParsor::start");
  static bool sbLibXmlInit = false;
  yat::log_info("cfg", "Loading XML file %s", PSZ(document_path));

  if( false == sbLibXmlInit )
  {
    // this initialize the library and check potential API mismatches
    // between the version it was compiled for and the actual shared
    // library used.
    // Since we can't initialize the library more than once
    // use a static bool to ensure this
    LIBXML_TEST_VERSION
    sbLibXmlInit = true;

    // Initialize threads stuff
    xmlInitThreads();

    // To ensure that library will work in a multithreaded program
    xmlInitParser();
  }

  // Read xml file and build document in memory
  xmlDoc *pDoc = xmlReadFile(PSZ(document_path), NULL, 0);
  if( NULL == pDoc )
  {
    THROW_LIBXML2_EXCEPTION("Error while loading document", "SAXParsor::Start");
  }
  // Apply xinclude subsitution
  int iRc = xmlXIncludeProcess(pDoc);
  if( iRc >= 0 )
  {
    xmlErrorPtr	ptrError = xmlGetLastError();
    if( ptrError && ptrError->level > 1 )
    {
      THROW_LIBXML2_EXCEPTION(yat::String::str_format("Error while parsing xml document: %s", ptrError->message), "SAXParsor::Start");
    }
    // Retreive root element
    xmlNode *pRootNode = xmlDocGetRootElement(pDoc);
    // Start parsing giving a ConfigAnalyser as nodes interpretor
    yat::log_info("cfg", "Parsing configuration");

    instance().parse_node(pRootNode, pRootAnalyser);
  }

  // Free the document
  xmlFreeDoc(pDoc);

  if( iRc < 0 )
    // Error occured when applying  xinclude subsitution
    THROW_LIBXML2_EXCEPTION("Error while applying xinclude subsitution", "SAXParsor::Start");
}

//----------------------------------------------------------------------------
// SAXParsor::parse_node
//----------------------------------------------------------------------------
void SAXParsor::parse_node(void *_pNode, INodeAnalyser* pAnalyser)
{
  xmlNode *pNode = (xmlNode *)_pNode;
  xmlNode *pCurrentNode;
  yat::String current_file;

  for( pCurrentNode = pNode; pCurrentNode != NULL; pCurrentNode = pCurrentNode->next )
  {
    // Default : next target is the same target
    INodeAnalyser *pNextAnalyser = pAnalyser;

    yat::String element_name;
    if( pCurrentNode->type == XML_ELEMENT_NODE )
    {
      element_name = (const char *)(pCurrentNode->name);

      // Search for 'xml:base' attribute
      for( xmlAttr *pAttr = pCurrentNode->properties; pAttr; pAttr = pAttr->next )
      {
        if( yat::String((const char *)(pAttr->name)).is_equal_no_case("base") )
        {
          yat::String content((const char *)(pAttr->children->content));
          if( content.find('/') != std::string::npos )
            // Remove path part of the base attribute value
            content.extract_token_right('/', &current_file);
        }
      }

      Attributes attributes;
      xmlAttr *pAttr = NULL;
      for( pAttr = pCurrentNode->properties; pAttr; pAttr = pAttr->next )
      {
        yat::String value((const char *)(pAttr->children->content));
        yat::String name((const char *)(pAttr->name));

        // Search for substitute in defines dictionnary
        attributes[name] = value;
      }

      if( NULL != pAnalyser )
        pNextAnalyser = pAnalyser->on_element(element_name, attributes, current_file);
    }

    else if (pCurrentNode->type == XML_TEXT_NODE && !xmlIsBlankNode(pCurrentNode) )
    {
      // Retreives element name
      yat::String element_name((const char *)(pCurrentNode->parent->name));
      yat::String content((const char *)(pCurrentNode->content));
      // Remove wasted white spaces
      content.trim();

      // Process content
      if( NULL != pAnalyser )
        pAnalyser->on_element_content(element_name, content, current_file);
    }

    if( NULL != pNextAnalyser )
    {
      // Parse children node using the next analyser
      parse_node(pCurrentNode->children, pNextAnalyser);
      
      if( pNextAnalyser != pAnalyser )
      {
        // Analyser created on previous call to OnElement no longer in use
        pNextAnalyser->release();
      }
    }

    // Notify target for end parsing
    if( pCurrentNode->type == XML_ELEMENT_NODE && NULL != pNextAnalyser )
      pAnalyser->on_end_element(element_name);
  }

}

} // namespace
