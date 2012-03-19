
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/Dictionary.h>
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Context.h>
#include <cdma/dictionary/PluginMethods.h>

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
