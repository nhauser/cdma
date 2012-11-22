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
#include <cdma/exception/impl/ExceptionImpl.h>
#include <cdma/dictionary/impl/Key.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/dictionary/plugin/Context.h>
#include <cdma/dictionary/plugin/PluginMethods.h>

namespace cdma
{

//==============================================================================
// KeyPath
//==============================================================================
//---------------------------------------------------------------------------
// KeyPath::solve
//---------------------------------------------------------------------------
void KeyPath::solve( Context& context ) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("KeyPath::solve");

  // Retreive a reference on a physical data item
  IDataItemPtr item_ptr = context.getDataset()->getItemFromPath( m_path );

  // Sets some properties
  item_ptr->setShortName( context.getKey()->getName() );

  // Push the result on the context
  context.pushDataItem( item_ptr );
}

//==============================================================================
// KeyMethod
//==============================================================================
//---------------------------------------------------------------------------
// KeyMethod::solve
//---------------------------------------------------------------------------
void KeyMethod::solve( Context& context ) throw ( cdma::Exception )
{
  CDMA_FUNCTION_TRACE("KeyMethod::solve");

  m_method_ptr->execute(context);
}

} // namespace cdma
