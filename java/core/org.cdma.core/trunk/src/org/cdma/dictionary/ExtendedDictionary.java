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
package org.cdma.dictionary;

// JAVA imports
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.exception.FileAccessException;
import org.cdma.interfaces.IKey;
import org.cdma.internal.IModelObject;
import org.cdma.internal.dictionary.readers.DataConcepts;
import org.cdma.internal.dictionary.readers.DataMapping;
import org.cdma.internal.dictionary.readers.DataView;
import org.cdma.internal.dictionary.readers.DictionaryReader;
import org.cdma.internal.dictionary.solvers.ItemSolver;


/**
 * @brief ExtendedDictionary class is the logical representation of a IDataset.
 * 
 * It defines how data is logically structured and permits a standardized browsing what
 * ever the plug-in, the data source format or its structure is.
 * <br/>
 * The dictionary is compound of two element a key file that defines the representation
 * of the dataset and a mapping file that associates 
 * Association of objects is the following:
 * <br/> - IKey and Path for a IDataItem, 
 * <br/> - IKey and ExtendedDictionary for a LogicalGroup.  
 */


public final class ExtendedDictionary implements IModelObject, Cloneable{
    
    private IFactory mFactory;     // Name of the plug-in's factory that created this object 
    private String   mView;        // View matching that dictionary
//    private String   mVersion;     // Version of the read dictionary
    private String   mKeyFile;     // Path to reach the key file (containing the view)
    private String   mMapFile;     // Path to reach the mapping file
    private IKey[]   mKeyPath;

    private DictionaryReader mReader;
    
    private Map<IKey, String>    mKeys;
    private Map<String, ItemSolver> mMaps;

    /**
     * Create an empty dictionary
     * @param factory
     */
    public ExtendedDictionary(IFactory factory) {
        mFactory      = factory; 
        mView         = Factory.getActiveView();
        mKeyFile      = null;
        mMapFile      = null;
        mKeyPath      = new IKey[] {};
    }
    
    public ExtendedDictionary(IFactory factory, String keyFile, String mapFile) {
        mFactory      = factory; 
        mView         = null;
        mKeyFile      = keyFile;
        mMapFile      = mapFile;
        mKeyPath      = new IKey[] {};
    }

    /**
     * Add an entry of key and item solver.
     * 
     * @param keyName key's name in string
     * @param solver Solver that will be used to resolve the key when asked 
     */
    public void addEntry(String keyName, ItemSolver solver) {
    	mKeys.put( mFactory.createKey(keyName), keyName );
    	mMaps.put( keyName, solver );
    }
    
    /**
     * Add an entry of key and path.
     * 
     * @param keyName key's name in string
     * @param path where data can be found
     */
    public void addEntry(String keyName, Path path) {
        ItemSolver solver = new ItemSolver(mFactory, path);
    	mKeys.put( mFactory.createKey(keyName), keyName );
    	mMaps.put( keyName, solver );
    }
    
    /**
     * Add a concept to the dictionary.
     * 
     * @param concept to be added
     */
    public void addConcept(Concept concept) {
    	try {
			mReader.getConcepts().addConcept(concept);
		} catch (FileAccessException e) {
			Factory.getLogger().log( Level.SEVERE, e.getMessage() );
		}
    }
    

    /**
     * Returns true if the given key is in this dictionary
     * 
     * @param key key object
     * @return true or false
     */
    public boolean containsKey(String keyName) {
    	IKey key = mFactory.createKey(keyName);
        return mKeys.containsKey( key );
    }

    /**
     * Return all keys referenced in the dictionary.
     * 
     * @return a list of String objects
     */
    public List<IKey> getAllKeys() {
        return new ArrayList<IKey>(mKeys.keySet());
    }

    /**
     * Get the item solver referenced by the key or null if not found.
     * 
     * @param key key object
     * @return ItemSolver object
     */
    public ItemSolver getItemSolver(IKey key)
    {
        ItemSolver solver = null;
        if( mKeys.containsKey( key ) )
        {
            String keyName = mKeys.get(key);
            if( mMaps.containsKey(keyName) ) {
                solver = mMaps.get(keyName);
            }
        }
        return solver;
    }
    
    /**
     * Remove an entry from the dictionary.
     * 
     * @param key key object
     */
    public void removeEntry(String keyName) {
        IKey key = Factory.getFactory().createKey(keyName);
        mKeys.remove(key);
        mMaps.remove(key);
    }

    public ExtendedDictionary clone() throws CloneNotSupportedException
    {
        ExtendedDictionary dict = new ExtendedDictionary(mFactory, mKeyFile, mMapFile);
        dict.mReader  = mReader;
        dict.mKeyPath = mKeyPath.clone();
        dict.mView    = mView;
        
        return dict;
    }

    /**
     * Get a sub part of this dictionary that corresponds to a key.
     * @param IKey object
     * @return IExtendedDictionary matching the key
     */
    public ExtendedDictionary getDictionary(IKey key) {
    	ExtendedDictionary subDict;
		try {
			subDict = this.clone();
			
			// Update the key path to reach sub dictionary
			int depth = mKeyPath.length + 1;
			subDict.mKeyPath = java.util.Arrays.copyOf(mKeyPath, depth);
			subDict.mKeyPath[depth - 1] = key.clone();
			
			// Calculate the sub data view
			DataView view = mReader.getView();
			for( IKey tmp : subDict.mKeyPath ) {
				view = view.getView( tmp.getName() );
			}
			
			// Get concepts and mappings
			DataConcepts concepts = mReader.getConcepts();
			DataMapping mapping = mReader.getMapping(mFactory, mMapFile);
			
			subDict.link(view, mapping, concepts);
		} catch (CloneNotSupportedException e) {
			subDict = null;
			Factory.getLogger().log( Level.SEVERE, e.getMessage() );
		} catch (FileAccessException e) {
			subDict = null;
			Factory.getLogger().log( Level.SEVERE, e.getMessage() );
		}

        return subDict;
    }

    /**
     * Get the view name matching this dictionary
     * 
     * @return the name of the experimental view
     */
    public String getView() {
        if( mView == null ) {
            try {
                readEntries();
            } catch (FileAccessException e) {
            }
        }
        return mView;
    }

    @Override
    public String getFactoryName() {
        return mFactory.getName();
    }

    /**
     * Return the path to reach the key dictionary file
     * 
     * @return the path of the dictionary key file
     */
    public String getKeyFilePath() {
        return mKeyFile;
    }

    /**
     * Return the path to reach the mapping dictionary file
     * 
     * @return the path of the plug-in's dictionary mapping file
     */
    public String getMappingFilePath() {
        return mMapFile;
    }
    
    /**
     * Return the concept object corresponding to the given key
     * @param key
     * @return the Concept 
     */
    public Concept getConcept(IKey key) {
    	Concept concept;
        try {
        	concept = mReader.getConcepts().getConcept(key.getName(), mFactory.getName() );
		} catch (FileAccessException e) {
			concept = null;
			Factory.getLogger().log( Level.SEVERE, e.getMessage() );
		}
    	return concept;
    }
    
    /**
     * Read all keys stored in the XML dictionary file
     * 
     * @throws FileAccessException in case of any problem while reading
     */
    public void readEntries() throws FileAccessException {
    	synchronized( ExtendedDictionary.class ) {
    		// Init the XML files reader: mappings, views, concepts
	    	mReader = new DictionaryReader(mKeyFile);
	    	mReader.init();
	    	
	    	// Get mappings, views, concepts
	    	DataView view = mReader.getView();
	    	DataConcepts concepts = mReader.getConcepts();
	    	DataMapping mapping = mReader.getMapping(mFactory, mMapFile);
	    	if( mView == null ) {
	    		mView = mReader.getView().getName();
	    	}
	    	
	    	// Link the data view and the concept
	    	link(view, mapping, concepts );
    	}
    }

	private void link(DataView view, DataMapping mapping, DataConcepts concepts) {
    	String keyID;

    	// Init keys map
    	if( mKeys == null ) {
    		mKeys = new HashMap<IKey, String>();
    	}
    	
    	// Get concept IDs from view item
    	for( String keyName : view.getItemKeys() ) {
    		Concept concept = concepts.getConcept(keyName);
    		if( concept == null ) {
    			concept = new Concept(keyName);
    			concepts.addConcept(concept);
    			keyID = keyName;
    		}
    		else {
    			keyID = concept.getConceptID();
    		}
    		
    		mKeys.put( mFactory.createKey(keyName), keyID);
    	}
		
    	// Init mapping map
		if( mMaps == null ) {
			mMaps = new HashMap<String, ItemSolver>();
		}
		
    	// Add all sub views to dictionary (i.e: keys and mappings)
    	for( Entry<String, DataView> entry : view.getSubViews().entrySet() ) {
    		IKey key = mFactory.createKey( entry.getKey() );
    		mMaps.put( key.getName(), new ItemSolver(mFactory, key) );
    		mKeys.put( key, key.getName() );
    	}
    	
    	// Add all items to mapping
    	for( Entry<String, ItemSolver> entry : mapping.getSolvers().entrySet() ) {
    		Concept concept = concepts.getConcept( entry.getKey() );
    		if( concept == null ) {
    			keyID = entry.getKey();
    		}
    		else {
    			keyID = concept.getConceptID();
    		}
    		mMaps.put( keyID, entry.getValue() );
    		if( mKeyFile == null ) {
    			mKeys.put( mFactory.createKey(keyID), keyID);
    		}
    	}
    }
}

