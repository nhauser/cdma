#include <list>
#include <string>
#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include <yat/utils/URI.h>

// Core
#include <cdma/navigation/IDataItem.h>
#include <cdma/IDataSource.h>
#include <cdma/Factory.h>
#include <cdma/IFactory.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/dictionary/LogicalGroup.h>
#include <cdma/dictionary/Key.h>
#include <cdma/array/Array.h>
#include <cdma/array/ArrayIterator.h>
#include <cdma/array/SliceIterator.h>
#include <cdma/array/Slicer.h>
#include <cdma/navigation/IGroup.h>
#include <cdma/utils/ArrayUtils.h>
// Test
#include <tools.h>
#include <testArrayUtils.h>

#include <exception>


using namespace std;
using namespace cdma;
using namespace yat;

 string FILEPATH = "";
 string VIEW = "";
 string PLUGIN_PATH = "";

bool init(int argc, char* argv[] )
{
    // initializing globals
    if( argc >= 3 )
    {
      // File to be opened
      FILEPATH = string( argv[1] );

      // View to use
      VIEW = string ( argv[2] );

      // Path where to find plugins
      PLUGIN_PATH = string( argv[3] );
      
      return true;
    }
    else
    {
      cout<<"=============================="<<endl;
      cout<<"Error 3 parameters are requested:"<<endl;
      cout<<"   - 1/ Target file to be used as datasource"<<endl;
      cout<<"   - 2/ View to be used: flat, hierarchical..."<<endl;
      cout<<"   - 3/ Location of compiled plug-ins\n"<<endl;
      cout<<"Usage: CDMATest [file_path] [view_to_use] [path_where_to_find_plugin]"<<endl;
      cout<<"Note: make sure you have declared environment variable \"CDMA_DICTIONARY_PATH=path_of_view_files\" "<<endl;
      cout<<"Example: CDMATest samples/demo.nxs flat /tmp/CDMA/"<<endl;
      cout<<"=============================="<<endl;
      
      return false;
    }
}

string getCommand(String path, LogicalGroupPtr group, list<IDataItemPtr> list_of_items)
{
      string keyStr;

      // Display available keys
      cout<<"=============================="<<endl;
      cout<<"Current logical path: "<< path << endl;
      cout<<"Available keys:"<<endl;
      cout<<"=============================="<<endl;
      cout<<Tools::iterate_over_keys( group, list_of_items );
      cout<<"'help' for list of commands"<<endl;
      cout<<"=============================="<<endl;

      // Enter a key
     
      cout<<"Please enter a key name: ";
      cin >> keyStr;
      
      if( keyStr == "help" || keyStr == "h" )
      {
        cout<<"=============================="<<endl;
        cout<<"Usage:"<<endl;
        cout<<"  enter 'name_of_key' to select the corresponding node"<<endl;
        cout<<"  enter 'root' to get back to the root of the file"<<endl;
        cout<<"  enter 'exit' to quit the program"<<endl;
        cout<<"=============================="<<endl;
        keyStr = "";
      }
      
      return keyStr;
}

int main(int argc,char *argv[])
{
  // init values
  if( ! init( argc, argv ) )
  {
    return 1;
  }
    

  // Create URI
  stringstream uri_str;
  uri_str << "file:" << FILEPATH;
  yat::URI uri ( uri_str.str() );

  cout<<"Plugin path: "<< PLUGIN_PATH<<endl;
  cout<<"File path: "<< FILEPATH <<endl;
  cout<<"View: "<< VIEW <<endl;
  
  try
  {
    // Initialize plugins
    Factory::init( PLUGIN_PATH );
    
    // Setting the view on data
    Factory::setActiveView( VIEW );

    // Instantiating a plugin factory
    IFactoryPtr ptrFactory = Factory::detectPluginFactory( uri );

    // Checking the source is readable    
    IDataSourcePtr dts = ptrFactory->getPluginURIDetector();

    if( dts->isProducer( uri ) ) 
    {
      cout<<">>> Dictionary enabled"<<endl;
      
      // Getting a handle on file
      IDatasetPtr dataset = ptrFactory->openDataset( uri.get() );

      // Accessing its logical root
      LogicalGroupPtr log_root = dataset->getLogicalRoot();
      LogicalGroupPtr group = log_root;

      // Iterate over keys structure
      list<IDataItemPtr> list_of_items;
      KeyPtr key;
      string path = "/";
      
      // Get the first command
      string keyStr = getCommand( path, group, list_of_items );
      
      // While no 'exit' asked continue;
      while( keyStr != "exit" )
      {
        
        if( keyStr != "" )
        {
          if( keyStr == "root" )
          {
            group = log_root;
            path = "/";
          }
          else if( keyStr == "exit" || keyStr == "quit" )
          {
            break;
          }
          else 
          {
            // Create requested key
            key = Tools::getKey(group, keyStr);
            
            // Ask for its target
            if( ! key.is_null() )
            {
              if( key->getType() == Key::ITEM )
              {
                IDataItemPtr item = group->getDataItem( key );
                cout<<endl<<"=============================="<<endl;
                cout<<"Displaying data item informations"<<endl;
                cout<<"Logical path: "<< path <<keyStr<<endl;
                cout<<Tools::displayDataItem( item )<<endl;
                cout<<"=============================="<<endl;
                cout<<"=============================="<<endl; 
                cout<<"Display values"<<endl;
                ArrayPtr array = item->getData();
                cout<<Tools::displayArray( array )<<endl;
                cout<<"=============================="<<endl;
                if( array->getRank() > 0 )
                {
                  cout<<"=============================="<<endl; 
                  cout<<"Testing array"<<endl;
                  TestArrayUtils test_utils( array );
                  test_utils.run_test();
                  cout<<"=============================="<<endl;
                }
                group = log_root;
                path = "/";
              }
              else if( key->getType() == Key::GROUP )
              {
                group = group->getGroup( key );
                path += keyStr + "/";
              }
            }
            keyStr = "";
          }
        }

        cout<<endl<<endl;

        // Get the next command
        keyStr = getCommand( path, group, list_of_items );
      }
    }
    else
    {
      cout<<"Plugin isn't producer >>> Dictionary disabled"<<endl;
      cout<<"Leaving program!!"<<endl;
    }
  
  }
  catch ( yat::Exception& e )
  {
    cout<<"=======> Exception !! <======="<<endl;
    cout<<"Orig.: "<<e.errors[0].origin<<endl;
    cout<<"Desc.: "<<e.errors[0].desc<<endl;
    cout<<"Reas.: "<<e.errors[0].reason<<endl;
    cout<<"=============================="<<endl;
  }
  catch( std::exception& e1 )
  {
     cout << e1.what() << endl;
  }
  catch ( ... )
  {
    cout<<"=======> Unknown Exception !! <======="<<endl;
  }
}


