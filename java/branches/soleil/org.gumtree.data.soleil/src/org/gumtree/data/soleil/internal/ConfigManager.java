package org.gumtree.data.soleil.internal;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Vector;

import org.gumtree.data.Factory;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.soleil.navigation.NxsDataset;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.JDOMException;
import org.jdom.input.SAXBuilder;

public class ConfigManager {
	private static ConfigManager mSingleton;
	private File mFile;                            // XML file containing the configuration
	private Vector<ConfigDataset> mConfigurations; // found configurations
	private boolean mInitialized;
	
	public static ConfigManager getInstance() {
        synchronized (ConfigManager.class ) {
            if( mSingleton == null ) {
            	mSingleton  = new ConfigManager();
            }
        }
        return mSingleton;
	}
	
	public ConfigDataset getConfig( NxsDataset dataset ) throws FileAccessException {
		synchronized (ConfigManager.class ) {
			if( ! mInitialized ) {
				load();
			}
		}

		ConfigDataset result = null;
		boolean found = false;
		for( ConfigDataset conf: mConfigurations ) {
			if( conf.getCriteria().match(dataset) ) {
				result = conf;
				found = true;
				break;
			}
		}
		if( !found ) {
			System.out.println("NO matching configuration has been found!!");
		}
		
		return result;
	}
	
	
	// ---------------------------------------------------------
	/// Private methods
	// ---------------------------------------------------------
	// Private constructor
	private ConfigManager() {
		mConfigurations = new Vector<ConfigDataset>();
		
		// Get the mapping folder that should contains the config file
		String folder = Factory.getMappingDictionaryFolder( NxsFactory.getInstance() );
		
		// Construct the file 
		mFile = new File( folder + "/cdma_nexussoleil_config.xml");
		mInitialized = false;
	}
	
	private void load() throws FileAccessException {
		// Check config file existence
		if( mFile != null ) {
			// Determine the experiment dictionary according to given path
	        if (!mFile.exists()) {
	            throw new FileAccessException(NxsFactory.LABEL + " configuration file does not exist:\n" + mFile.getAbsolutePath());
	        }
			
	        // Parse the XML configuration file
	        SAXBuilder xmlFile = new SAXBuilder();
	        Document config;
	        try {
	        	config = xmlFile.build(mFile);
	        }
	        catch (JDOMException e1) {
	            throw new FileAccessException("error while to parsing the configuration!\n" + mFile.getAbsolutePath() + "\n", e1);
	        }
	        catch (IOException e1) {
	            throw new FileAccessException("an I/O error prevent parsing configuration!\n" + mFile.getAbsolutePath() + "\n", e1);
	        }
	        
	        Element root = config.getRootElement();
			
			List<?> nodes = root.getChildren("dataset_model");
			Element elem;

			for( Object node : nodes ) {
				elem = (Element) node;
				mConfigurations.add( new ConfigDataset(elem) );
			}
			mInitialized = true;
		}
	}
}
