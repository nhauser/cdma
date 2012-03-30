package org.gumtree.data.test;

import java.io.File;
import java.io.IOException;
import java.util.Vector;

import org.gumtree.data.Benchmarker;
import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IGroup;
//import org.gumtree.data.soleil.navigation.NxsDataset;
//import org.gumtree.data.util.configuration.ConfigManager;

public class MainTest {
	protected static int iiii = 0;
	protected static String plugin;
	protected static IDataset dataset;
	protected static String sample_path;
	private static Vector<Thread> poolThread = new Vector<Thread>();
	private static boolean useThreads = false;
	
    /**
     * Main process
     * @param args path to reach file we want to parse
     * @throws IOException 
     * @throws FileAccessException
     * args[0] = folder of application's dictionary view
     * args[1] = plugin's name
     * args[2] = folder where to find file
     */
    public static void main(String[] args) throws Exception
    {
    	
        final long time = System.currentTimeMillis();
        final String dico_path;
        int nbFiles = 0;
        plugin = "org.gumtree.data.soleil.NxsFactory";
        
        // Dictionary path
        if( args.length > 0 ) {
            dico_path = args[0];
            Factory.setDictionariesFolder(dico_path);
        }
        
        // Setting plug-in to use 
        if( args.length > 1 ) {
        	plugin = args[1];
        }
        
        if( args.length > 2 ) {
        	sample_path = args[2];
        }
        
        IFactory factory = null;
        File folder = new File(sample_path);
        if( folder.exists() ) {
	        for( File sample : folder.listFiles() ) {

 	        	String name = sample.getName();

	        	// Instantiate Dataset (Gumtree meaning) on file to be read (open it then)
		        dataset = null;
		        try {
		        	// Forcing plugin
		        	//factory = Factory.getFactory(plugin);
		        	
		        	// Auto detecting factory
		        	factory = Factory.getFactory(sample.toURI());

		        	if( factory == null ) {
		        		System.out.println( "Failed auto-detecting plugin: " + sample.getAbsolutePath() );
		        		continue;
		        	}

		        	// Create dataset according URI
		        	dataset = factory.createDatasetInstance(sample.toURI());
		            

		            
		        }
		        catch(IOException ioe) {
		        	ioe.printStackTrace();
		        	continue;
		        }
		        catch(Exception e) {
		        	e.printStackTrace();
		        	return;
		        }
		        
		        // File will be processed
		        nbFiles++;
		        
		        IDataset DATASET = dataset;
				String pluginName = plugin;
				
	            // open the targeted file
				try {
					DATASET.open();
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
					return;
				}
				
				// Get the file root
		        ILogicalGroup root = null;
		        IGroup phy_root = null;

		        // Execute tests
		        // Displaying physical structure of the datasource
		        setActiveView(DATASET, "NONE: physical parsing!!", pluginName);
		        phy_root = DATASET.getRootGroup();
		        TestStructure.displayStructure(phy_root);
		        
		        // Displaying 2 kind of view for data
		        setActiveView(DATASET, "FLAT", pluginName);
		        
		        TestDictionaries.testKeys(root);
		        
		        setActiveView(DATASET, "HIERARCHICAL", pluginName);
		        root = DATASET.getLogicalRoot();
		        TestDictionaries.testKeys(root);

		        // Displaying values associated to keys 
		        TestDictionaries.testValues(root);
		        
		        // Testing CubeData
		        IDataItem item = root.getDataItem("images");
		        if( item != null ) {
		        	TestCubeData.testCube(item);
		        }
		        
		        // Testing some datas
				TestData.testData(root);

				// Testing arrays iteration, copy, slice...
				TestArray.TestArray(plugin, sample.getAbsolutePath());

				System.out.println(Benchmarker.print());
		        
		        try {
		        	DATASET.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
        	}
        }
        factory = Factory.getFactory(plugin);
        //TestArrayManual.testManuallyCreatedArrays(factory);
        System.out.println("Total execution time: " + ((System.currentTimeMillis() - time) / 1000.) + "s");
        System.out.println("Nb files tested: " + nbFiles );
    }
    
    public static void setActiveView(String expName) {
    	setActiveView(dataset, expName, plugin);
    }
    
    public static void setActiveView(IDataset dset, String expName, String pluginName ) {
    	Factory.setActiveView(expName);
        System.out.println("------------------------------------------------------------------");
        System.out.println("----------------             " + expName + "               ---------------");
        System.out.println("------------------------------------------------------------------");
        System.out.println("Active plugin: "   + pluginName);
        System.out.println("Active view: "     + Factory.getActiveView() );
        System.out.println("View dict: "       + Factory.getKeyDictionaryPath() );
        System.out.println("Dict. folder: "    + Factory.getMappingDictionaryFolder( Factory.getFactory(pluginName) ) );
        System.out.println("Working on file: " + dset.getLocation() );
//        if( dset instanceof NxsDataset ) {
//    		System.out.println( "Using configuration mode: " + ((NxsDataset) dset).getConfiguration().getLabel() );
//    	}
    }
    

}






