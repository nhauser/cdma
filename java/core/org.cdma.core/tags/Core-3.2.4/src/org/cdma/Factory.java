// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
//
// Contributors:
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    Tony Lam (nxi@Bragg Institute) - initial API and implementation
//    Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
// ****************************************************************************
package org.cdma;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.cdma.exception.FileAccessException;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.internal.dictionary.readers.DictionaryReader;
import org.cdma.utils.FactoryManager;
import org.cdma.utils.IFactoryManager;

/**
 * @brief The Core factory is the entry point of the CDMA API
 * 
 *        The Factory class in common data model is a tools to create CDMA objects.. It manages all
 *        plug-ins instances.
 *        <p>
 *        According to an URI, it will detect which plug-in is relevant to that data source. It can
 *        take an URI as a parameter to instantiate a plug-in in order to get an access of the
 *        targeted data source using CDMA objects.
 *        <p>
 *        Abbreviation: Common Data Model Access -- CDMA
 * 
 * @author XIONG Norman
 * @contributor RODRIGUEZ Clément
 * @version 1.1
 */
public final class Factory {
    // Version of the current CDMA API
    private static final String CDMA_VERSION = "3_2_0";

    // Plugin manager
    private static volatile IFactoryManager manager;

    // Dictionary view
    private static String CDM_VIEW = "";
    private static final String DICO_PATH_PROP       = "CDM_DICTIONARY_PATH";

    // Files' suffix / prefix
    private static final String FILE_CONCEPT_NAME    = "concepts";
    private static final String FILE_CONCEPT_CORE    = "core_";
    private static final String FILE_VIEW_SUFFIX     = "view";

    // Dictionaries
    private static final String PATH_FOLDER_MAPS     = "mappings";
    private static final String PATH_FOLDER_VIEWS    = "views";
    private static final String PATH_FOLDER_CONCEPTS = "concepts";

    // Global logger for the CDMA
    private static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    /**
     * Create a CDMA Dataset that can read the given URI.
     * 
     * @param uri URI object
     * @return CDMA Dataset
     * @throws Exception
     */
    public static IDataset openDataset(URI uri ) throws Exception {
        return openDataset(uri, false);
    }

    /**
     * Create a CDMA Dataset that can read the given URI and use optionally the Extended Dictionary
     * mechanism.
     * 
     * @param uri URI object
     * @param useProducer only
     * @return CDMA Dataset
     * @throws Exception
     */
    public static IDataset openDataset(URI uri, boolean useDictionary) throws Exception {
        IFactory factory = getFactory(uri);
        IDataset dataset = null;
        if( factory != null ) {
            IDatasource source = factory.getPluginURIDetector();
            if( ! useDictionary || source.isProducer( uri ) ) {
                dataset = factory.openDataset(uri);
            }
        }
        return dataset;
    }

    /**
     * Returns the logger that will be used by the CDMA.
     * 
     * @return java.util.logging.Logger
     */
    public static Logger getLogger() {
        return LOGGER;
    }

    /**
     * Set the logger that the CDMA have to use for messages.
     * 
     * @param logger
     */
    public static void setLogger( Logger logger ) {
        LOGGER = logger;
    }

    /**
     * Set the name of the current view (e.q application) that will be active for
     * the Extended Dictionary mechanism.
     * 
     * @param view name
     */
    public static void setActiveView(String application) {
        CDM_VIEW = application;
    }

    /**
     * Returns the name of the current view that is active for
     * the Extended Dictionary mechanism.
     * 
     * @return view name
     */
    public static String getActiveView() {
        return CDM_VIEW;
    }

    /**
     * Returns the list of all available views for the dictionary mechanism.
     * 
     * @return views' names
     */
    public static List<String> getAvailableViews() {
    	List<String> result = new ArrayList<String>();
    	
    	// Get the fictionary folder
        String path = getDictionariesFolder();
        if( path != null ) {
        	File folder = new File( path + File.separator + PATH_FOLDER_VIEWS );
        	if( folder.exists() && folder.isDirectory() ) {
        		// List folder's files
        		File[] files = folder.listFiles();
        		String name, fileName;
        		for( File file : files ) {
        			fileName = file.getPath();
        			// Check files name
        			if( fileName.endsWith("_" + FILE_VIEW_SUFFIX + ".xml" ) ) {
        				try {
        					// Get the view name
							name = DictionaryReader.getDictionaryName(fileName);
							result.add( name );
						} catch (FileAccessException e) {
							Factory.getLogger().log(Level.INFO, "Unable to get the view name for: " + fileName, e);
						}
        			}
        		}
        	}
        }

    	return result;
    }
    
    /**
     * According to the currently defined experiment, this method will return the path
     * to reach the declarative dictionary. It means the file where
     * is defined what should be found in a IDataset that fits the experiment.
     * It's a descriptive file.
     * 
     * @return the path to the standard declarative file
     */
    public static String getPathKeyDictionary() {
        String file  = null;
        String sDict = getDictionariesFolder();
        String view  = getActiveView();

        if( ! view.trim().isEmpty() ) {
            String vFile = ( view + "_" + FILE_VIEW_SUFFIX + ".xml" ).toLowerCase();
            file = sDict + File.separator + PATH_FOLDER_VIEWS + File.separator + vFile;
        }

        return file;
    }

    /**
     * This method will return the path to reach the <b>common concept dictionary</b>.
     * It means the file where is defined every physical concepts.
     * 
     * @return the path to the common concept file or null if not found
     * @note The file name is as following: getDictionariesFolder() +  "concepts.xml"
     * @see Factory.getConceptViewDictionaryPath()
     */
    public static String getPathCommonConceptDictionary() {
        String sDict = getDictionariesFolder();
        String sFile = FILE_CONCEPT_CORE + FILE_CONCEPT_NAME + ".xml".toLowerCase();
        String sPath = sDict + File.separator + PATH_FOLDER_CONCEPTS + File.separator + sFile;

        // Check the concept dictionary corresponding to the view exist
        File file = new File(sPath);
        if( ! file.exists() ) {
            sPath = null;
        }

        return sPath;
    }

    /**
     * According to the given factory this method will return the path to reach
     * the folder containing mapping dictionaries. This file associate entry
     * keys to paths that are plug-in dependent.
     * 
     * @param factory of the plug-in instance from which we want to load the dictionary
     * @return the path to the plug-in's mapping dictionaries folder
     */
    public static String getPathMappingDictionaryFolder(IFactory factory) {
        String sDict = getDictionariesFolder();

        return sDict + File.separator + PATH_FOLDER_MAPS + File.separator + factory.getName() + File.separator;
    }

    /**
     * This method will return the path to reach the <b>specific concept dictionary</b>.
     * It means the file where is defined every specific physical concepts defined by the view.
     * 
     * @return the path to the specific concept folder or null if not found
     */
    public static String getPathConceptDictionaryFolder() {
        String sDict = getDictionariesFolder();
        String sPath = sDict + File.separator + PATH_FOLDER_CONCEPTS + File.separator;

        // Check the concept dictionary corresponding to the view exist
        File file = new File(sPath);
        if( ! file.exists() ) {
            sPath = null;
        }

        return sPath;
    }

    /**
     * Set the folder path where to search for key dictionary files.
     * This folder should contains all dictionaries that the above application needs.
     * 
     * @param path targeting a folder
     */
    public static void setDictionariesFolder(String path) {
        if (path != null) {
            System.setProperty(DICO_PATH_PROP, path);
        }
    }

    /**
     * Get the folder path where to search for key dictionary files (e.q: view or experiment).
     * This folder should contains all dictionaries that the above application needs.
     * 
     * @return path targeting a folder
     */
    public static String getDictionariesFolder() {
        return System.getProperty(DICO_PATH_PROP, System.getenv(DICO_PATH_PROP));
    }

    /**
     * Return the singleton instance of the plug-ins factory manager
     * @return IFactoryManager unique instance
     */
    public static IFactoryManager getManager() {
        if (manager == null) {
            synchronized (Factory.class) {
                if (manager == null) {
                    manager = new FactoryManager(CDMA_VERSION);
                }
            }
        }
        return manager;
    }

    /**
     * Return the IFactory of the first available plug-in that was loaded
     * @return first loaded IFactory
     */
    public static IFactory getFactory() {
        return getManager().getFactory();
    }

    /**
     * Return the plug-in's factory having the given name
     * @param name of the requested factory
     * @return IFactory instance
     */
    public static IFactory getFactory(String name) {
        return getManager().getFactory(name);
    }

    /**
     * Return a plug-in IFactory that is the most relevant for the given URI.
     * Try to detect factories according the following:
     * if a plug-in declares itself as the owner of the targeted data source returns its factory
     * else returns the first plug-in that is compatible with given data format
     * no plug-in is compatible returns null
     * 
     * @param uri of the data source
     * @return IFactory instance
     */
    public static IFactory getFactory(URI uri) {
        List<String> reader = new ArrayList<String>();
        IFactory result = null;

        // Get the list of data source detector
        List<IDatasource> sources = getDatasources();
        // For each check if it can read the given source
        for ( IDatasource source : sources ) {
            // Can read ?
        	boolean canRead = source.isReadable(uri);
            if( canRead ) {
                reader.add( source.getFactoryName() );

                // Does it have the ownership on the source
                boolean isProd = source.isProducer(uri);
                if( isProd ) {
                    result = getFactory( source.getFactoryName() );
                    break;
                }
            }
        }

        // No ownership detected, so return the first reader
        if( result == null && reader.size() > 0 ) {
            result = getFactory( reader.get(0) );
        }
        return result;
    }

    /**
     * Returns the list of all available IDataSource implementations.
     * 
     * @return list of found IDataSource
     */
    public static List<IDatasource> getDatasources() {
        List<IDatasource> result = new ArrayList<IDatasource>();
        IDatasource source;

        // Ensure a factory manager has been loaded
        IFactoryManager mngr = getManager();

        // Get the registry of factories
        Map<String, IFactory> registry = mngr.getFactoryRegistry();
        for( Entry<String, IFactory> entry : registry.entrySet() ) {
            source = entry.getValue().getPluginURIDetector();
            if( source != null ) {
                result.add( source );
            }
        }

        return result;
    }


    /**
     * Hide default constructor.
     */
    private Factory() {
    }


}
