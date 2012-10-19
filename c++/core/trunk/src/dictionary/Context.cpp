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
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/dictionary/impl/Key.h>
#include <cdma/dictionary/plugin/Context.h>

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
  m_dataitems.push_back(dataitem_ptr); 
}

}
