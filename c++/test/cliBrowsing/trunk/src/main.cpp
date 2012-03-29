#include <list>
#include <string>
#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include <yat/utils/URI.h>

// Core
#include <cdma/IDataSource.h>
#include <cdma/Factory.h>
#include <cdma/IFactory.h>
#include <cdma/navigation/IDataset.h>
// Test
#include <internal/tools.h>
#include <internal/testNavigation.h>
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

// Return int value: 
//     - result > 0 implies logical mode
//     - result = 0 implies physical mode
//     - result < 0 implies exit
int isLogicalMode()
{
  string entry = "";
  cout<<"=============================="<<endl;
  cout<<"Available navigation modes:"<<endl;
  cout<<" - 1: logical"<<endl;
  cout<<" - 2: physical"<<endl;
  cout<<" - 3: quit"<<endl;
  cout<<"=============================="<<endl;
  cout<<"Please select a mode: ";
  while( entry == "" ) 
  {
    getline(cin, entry, '\n');
  }
  cout<<"=============================="<<endl;
  
  if( entry == "quit" || entry == "exit" || entry == "q" || entry == "e" || entry == "3" )
  {
    return -1;
  }
  else if( entry != "1" && entry != "2" && entry != "logical" && entry != "physical" )
  {
    return isLogicalMode();
  }
  else
  {
    return ( entry == "1" || entry == "logical" ) ? 1 : 0;
  }
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
    std::pair<IDatasetPtr, IFactoryPtr> plugin_pair = Factory::openDataset( uri );
    IDatasetPtr dataset    = plugin_pair.first;
    IFactoryPtr ptrFactory = plugin_pair.second;

    TestNavigation navigation ( dataset );

    // Checking the source is readable    
    IDataSourcePtr dts = ptrFactory->getPluginURIDetector();
    if( dts->isProducer( uri ) )
    {
      cout<<">>> Dictionary enabled"<<endl;
      bool doContinue = true;
      
      // While not exit is required
      while( doContinue )
      {
        int action = isLogicalMode();
        if( action > 0 )
        {
          doContinue = navigation.run_logical();
        }
        else if( action == 0 )
        {
          doContinue = navigation.run_physical();
        }
        else 
        {
          doContinue = false;
        }
      }
    }
    else
    {
      cout<<"Plugin isn't producer >>> Dictionary disabled"<<endl;
      navigation.run_physical();
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
  Factory::cleanup();
}


