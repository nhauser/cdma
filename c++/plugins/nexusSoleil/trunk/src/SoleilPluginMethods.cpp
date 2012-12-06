// ****************************************************************************
// Copyright (c) 2011-2012 Synchrotron Soleil.
// The cdma-core library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ****************************************************************************

// Yat
#include <yat/plugin/PlugInSymbols.h>

// CDMA core
#include <cdma/Common.h>
#include <cdma/dictionary/IKey.h>
#include <cdma/dictionary/plugin/Context.h>
#include <cdma/dictionary/plugin/PluginMethods.h>
#include <cdma/dictionary/plugin/SimpleDataItem.h>

// NeXus Engine 
#include <NxsDataset.h>

// Soleil plugin
#include <SoleilNxsFactory.h>
#include <SoleilNxsDataSource.h>
#include <SoleilNxsDataset.h>

namespace cdma
{
namespace soleil
{
namespace nexus
{

//==============================================================================
/// Test method
//==============================================================================
class TestMethod : public cdma::IPluginMethod
{
public:
  ~TestMethod() 
  {
    FUNCTION_TRACE("TestMethod::~TestMethod");
  }

  void execute(Context& context) throw (cdma::Exception);
};

// Exports the method
EXPORT_PLUGIN_METHOD(TestMethod);

//==============================================================================
// TestMethod::execute
//==============================================================================
void TestMethod::execute(Context& ctx) throw (cdma::Exception)
{
  FUNCTION_TRACE("TestMethod::execute");

  IDataItemPtr dataitem_ptr = ctx.getTopDataItem();
  double value = dataitem_ptr->getValue<double>();

  IDataItemPtr new_data_item_ptr(new SimpleDataItem(ctx.getDataset(),
          cdma::IArrayPtr(new cdma::Array(value * 2)),ctx.getKey()->getName()));

  ctx.clearDataItems();
  ctx.pushDataItem(new_data_item_ptr);
}

} // namespace nexus
} // namespace soleil
} // namespace cdma
