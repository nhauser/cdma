// ****************************************************************************
// Copyright (c) 2011-2012 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ****************************************************************************

// Yat
#include <yat/plugin/PlugInSymbols.h>

// CDMA core
#include <cdma/IObject.h>
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Context.h>
#include <cdma/dictionary/PluginMethods.h>
#include <cdma/dictionary/SimpleDataItem.h>

// NeXus Engine 
#include <NxsDataset.h>

// Soleil plugin
#include <SoleilNxsFactory.h>
#include <SoleilNxsDataSource.h>
#include <SoleilNxsDataset.h>

namespace cdma
{

//==============================================================================
/// Test method
//==============================================================================
class CDMA_DECL TestMethod : public cdma::IPluginMethod
{
public:
  ~TestMethod() 
  {
    CDMA_TRACE("TestMethod::~TestMethod");
  }

  void execute(Context& context) throw (cdma::Exception);
};

// Export the method
EXPORT_PLUGIN_METHOD(TestMethod);

//==============================================================================
// TestMethod::execute
//==============================================================================
void TestMethod::execute(Context& ctx) throw (cdma::Exception)
{
  CDMA_FUNCTION_TRACE("TestMethod::execute");

  IDataItemPtr dataitem_ptr = ctx.getTopDataItem();
  double value = dataitem_ptr->readScalarDouble();

  IDataItemPtr new_data_item_ptr = new SimpleDataItem(ctx.getDataset(),
                                                        new cdma::Array(PlugInID, value * 2),
                                                        "2yBin");

  ctx.clearDataItems();
  ctx.pushDataItem(new_data_item_ptr);
}


} // namespace cdma
