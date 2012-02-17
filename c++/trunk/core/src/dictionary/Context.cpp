//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************

// Include CDMA
#include <cdma/dictionary/Context.h>

namespace cdma
{

//==============================================================================
// Context
//==============================================================================
//----------------------------------------------------------------------------
// Context::getTopDataItem
//----------------------------------------------------------------------------
IDataItemPtr Context::getTopDataItem() const
{
  CDMA_FUNCTION_TRACE("Context::getTopDataItem");
  if( m_dataitems.empty() )
  {
    CDMA_TRACE("No data items");
    return IDataItemPtr(NULL);
  }
  return m_dataitems.front();
}

//----------------------------------------------------------------------------
// Context::pushDataItem
//----------------------------------------------------------------------------
void Context::pushDataItem(const IDataItemPtr& dataitem_ptr) 
{ 
  CDMA_FUNCTION_TRACE("Context::pushDataItem");
  m_dataitems.push_back(dataitem_ptr); 
}

}
