// ****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors
//    Clement Rodriguez - initial API and implementation
//    Norman Xiong
// ****************************************************************************
package org.cdma.dictionary;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.exception.FileAccessException;
import org.cdma.interfaces.IKey;
import org.cdma.internal.IModelObject;
import org.cdma.internal.dictionary.ConceptManager;
import org.cdma.internal.dictionary.ItemSolver;
import org.cdma.internal.dictionary.PluginMethodManager;
import org.cdma.internal.dictionary.Solver;
import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.jdom2.input.sax.XMLReaders;


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
    private static volatile PluginMethodManager mMethodMgr;
    private IFactory mFactory;     // Name of the plug-in's factory that created this object 
    private String   mView;        // View matching that dictionary
    private String   mVersion;     // Version of the read dictionary
    private String   mKeyFile;     // Path to reach the key file (containing the view)
    private String   mMapFile;     // Path to reach the mapping file
    private String   mConceptFile; // Path where to find all concepts

    private Map<IKey, String>               mKeyMap  = new HashMap<IKey, String>();               // Key / ID association
    //private Map<String, List<Solver> >      mPathMap = new HashMap<String, List<Solver> >();      // ID / Path association
    private Map<String, ItemSolver >        mPathMap = new HashMap<String, ItemSolver>();        // ID / ItemSolver association
    private Map<String, ExtendedDictionary> mSubDict = new HashMap<String, ExtendedDictionary>(); // ID / sub-dictionaries
    
    private ConceptManager mConcepts; // All available concepts

    public ExtendedDictionary(IFactory factory, String keyFile, String mapFile, String conceptFile) {
        mMethodMgr   = PluginMethodManager.instantiate();
        mFactory     = factory; 
        mView        = Factory.getActiveView();
        mKeyFile     = keyFile;
        mMapFile     = mapFile;
        mConceptFile = conceptFile;
        mConcepts    = null;
    }

    /**
     * Add an entry of key and path.
     * 
     * @param keyName key's name in string
     * @param solver Solver that will be used to resolve the key when asked 
     */
    public void addEntry(String keyName, ItemSolver solver) {
        IKey key = mFactory.createKey(keyName);
        mKeyMap.put(key, keyName);
        mPathMap.put(keyName, solver);
    }

    /**
     * Returns true if the given key is in this dictionary
     * 
     * @param key key object
     * @return true or false
     */
    public boolean containsKey(String keyName) {
        IKey key = mFactory.createKey(keyName);
        return mKeyMap.containsKey(key);
    }

    /**
     * Return all keys referenced in the dictionary.
     * 
     * @return a list of String objects
     */
    public List<IKey> getAllKeys()
    {
        return new ArrayList<IKey>(mKeyMap.keySet());
    }

    /**
     * Get the item solver referenced by the key or null if not found.
     * 
     * @param key key object
     * @return ItemSolver object
     */
/*
    public List<Solver> getKeySolver(IKey key)
    {
        List<Solver> solvers;
        if( mKeyMap.containsKey(key) )
        {
            String keyName = mKeyMap.get(key);
            if( mPathMap.containsKey(keyName) ) {
                solvers = mPathMap.get(keyName);
            }
            else {
                solvers = new ArrayList<Solver>();
            }
        }
        else {
            solvers = new ArrayList<Solver>();
        }
        return solvers;
    }
 */
    public ItemSolver getItemSolver(IKey key)
    {
        ItemSolver solver = null;
        if( mKeyMap.containsKey(key) )
        {
            String keyName = mKeyMap.get(key);
            if( mPathMap.containsKey(keyName) ) {
                solver = mPathMap.get(keyName);
            }
        }
        return solver;
    }
    
    /**
     * Read all keys stored in the XML dictionary file
     * 
     * @throws FileAccessException in case of any problem while reading
     */
    public void readEntries() throws FileAccessException {
        File dicFile = new File(mKeyFile);
        if (!dicFile.exists()) 
        {
            throw new FileAccessException("the target dictionary file does not exist");
        }

        // Read the pivot dictionary
        readDictionaryConcepts();
        
        // Read corresponding dictionaries
        readDictionaryKeys(null);
        readDictionaryMappings();
    }

    /**
     * Remove an entry from the dictionary.
     * 
     * @param key key object
     */
    public void removeEntry(String keyName) {
        IKey key = Factory.getFactory().createKey(keyName); 
        String keyID = mKeyMap.get(key);
        mKeyMap.remove(key);
        mPathMap.remove(keyID);
    }

    @SuppressWarnings("unchecked")
    public ExtendedDictionary clone() throws CloneNotSupportedException
    {
        ExtendedDictionary dict = new ExtendedDictionary(mFactory, mKeyFile, mMapFile, mConceptFile);
        dict.mVersion  = mVersion;
        dict.mKeyMap   = (HashMap<IKey, String>) ((HashMap<IKey, String>) mKeyMap).clone();
        dict.mPathMap  = (HashMap<String, ItemSolver >) ((HashMap<String, ItemSolver >) mPathMap).clone();
        dict.mSubDict  = (HashMap<String, ExtendedDictionary>) ((HashMap<String, ExtendedDictionary>) mSubDict).clone();
        dict.mConcepts = mConcepts;
        return dict;
    }

    /**
     * Get a sub part of this dictionary that corresponds to a key.
     * @param IKey object
     * @return IExtendedDictionary matching the key
     */
    public ExtendedDictionary getDictionary(IKey key) {
        String keyID = mKeyMap.get(key);
        ExtendedDictionary subDict = null;

        if( keyID != null ) {
            subDict = mSubDict.get(keyID);
        }

        return subDict;
    }

    /**
     * Get the version number (in 3 digits default implementation) that is plug-in
     * dependent. This version corresponds of the dictionary defining the path. It  
     * permits to distinguish various generation of IDataset for a same institutes.
     * Moreover it's required to select the right class when using a IClassLoader
     * invocation.
     * 
     * @return the string representation of the plug-in's version number
     */
    public String getVersionNum() {
        return mVersion;
    }

    /**
     * Get the view name matching this dictionary
     * 
     * @return the name of the experimental view
     */
    public String getView() {
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
        return mConcepts.getConcept(key.getName());
    }

    // ---------------------------------------------------------------
    // PROTECTED methods
    // ---------------------------------------------------------------
    private ExtendedDictionary(IFactory factory, String keyFile, String mapFile, String conceptFile, String view) {
        mMethodMgr   = PluginMethodManager.instantiate();
        mView        = view;
        mFactory     = factory;
        mKeyFile     = keyFile;
        mMapFile     = mapFile;
        mConceptFile = conceptFile;
    }
    
    // ---------------------------------------------------------------
    // PRIVATE : Reading methods
    // ---------------------------------------------------------------
    private void readDictionaryConcepts() {
        List<Concept> concepts= new ArrayList<Concept>();
        
        String commonFile = Factory.getPathCommonConceptDictionary();
        if( commonFile != null ) {
            concepts.addAll( readConceptFile(commonFile) );
        }
        
        if( mConceptFile != null ) {
            concepts.addAll( readConceptFile( mConceptFile ) );
        }
        mConcepts = new ConceptManager(concepts);
    }
    
    @SuppressWarnings("unchecked")
    private List<Concept> readConceptFile( String file ) {
        List<Concept> concepts = new ArrayList<Concept>();
        try {
            Element elem = saxBuildFile(file);

            List<?> nodes = elem.getChildren("concept");
            for (Object child : (List<Element>) nodes) {
                elem = (Element) child;
                concepts.add(new Concept(elem));
            }
        } catch (FileAccessException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return concepts;
    }
    
    private void readDictionaryKeys(Element xmlNode) throws FileAccessException {
        Element startNode;
        if( xmlNode == null ) {
            startNode = saxBuildFile(mKeyFile);

            String exp = startNode.getAttributeValue("name");
            if( ! exp.equalsIgnoreCase(mView) ) {
                throw new FileAccessException("an I/O error prevent parsing dictionary!\nThe dictionary doesn't match the experiment!");
            }
        }
        else {
            startNode = xmlNode;
        }
        List<?> nodes = startNode.getChildren();
        Element elem;
        IKey key;
        String keyName;
        String keyID;
        // For each element of the view (item or group)
        for( Object current : nodes ) {
            elem = (Element) current;
            // Get the key name
            keyName = elem.getAttributeValue("key");
            if( keyName != null && !keyName.isEmpty() ) {
                // Create key
                key = mFactory.createKey(keyName);
                
                // If the element is an item
                if( elem.getName().equals("item") ) {
                    // Search the corresponding concept ID
                    keyID = mConcepts.getConceptID(keyName);
                    if( keyID == null ) {
                        keyID = keyName;
                    }
                    mKeyMap.put(key, keyID);
                }
                // If the element is a group of keys
                else if( elem.getName().equals("group") ){
                    // Read corresponding dictionaries
                    ExtendedDictionary dict = new ExtendedDictionary(mFactory, mKeyFile, mMapFile, mConceptFile, mView);
                    dict.mConcepts = mConcepts;
                    dict.mConceptFile = dict.mConceptFile;
                    dict.readDictionaryKeys(elem);
                    dict.readDictionaryMappings();

                    // Create corresponding path
                    ItemSolver solver = new ItemSolver(mFactory, key);
                    
                    // Update maps
                    mSubDict.put(keyName, dict);
                    mKeyMap.put(key, keyName);
                    mPathMap.put(keyName, solver);
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void readDictionaryMappings() throws FileAccessException{
        Element root = saxBuildFile(mMapFile);
        mVersion = root.getAttributeValue("version");

        List<?> nodes = root.getChildren("item");
        List<Solver> solvers;
        String keyID;
        String keyName;

        // Updating the KeyID / Path map
        for (Element elem : (List<Element>) nodes) {
            // Get the key name
            keyName = elem.getAttributeValue("key");

            // Retrieve the corresponding concept ID
            keyID = mConcepts.getConceptID(keyName);
            
            if ( keyID == null ) {
                keyID = keyName;
            }
            
            if( ! mPathMap.containsKey(keyID) ) {
                //solvers = loadKeySolver(elem);
                mPathMap.put( keyID, new ItemSolver(mFactory, mMethodMgr, elem) );
            }
        }
    }
/*
    @SuppressWarnings("unchecked")
    private List<Solver> loadKeySolver(Element elem) {
        // Prepare result
        List<Solver> result = new ArrayList<Solver>();
        
        // List DOM children 
        List<?> nodes = elem.getChildren();
        
        IPluginMethod method;
        Solver current;
        
        // For each children of the mapping key item
        for( Element node : (List<Element>) nodes ) {
            // If path open the path
            if( node.getName().equals("path") )  {
                current = new Solver( mFactory.createPath(node.getText()) );
                result.add(current);
            }
            // If call on a method
            else if( node.getName().equals("call") )  {
                method = mMethodMgr.getPluginMethod(mFactory.getName(), node.getText());
                current = new Solver(method);
                result.add( current );
            }
        }

        return result;
    }
*/
    /**
     * Check the given file exists and open the root node
     * 
     * @param filePath XML file to be opened
     * @return Sax element that is the root of the XML file
     * @throws FileAccessException
     */
    private Element saxBuildFile(String filePath) throws FileAccessException {
        // Check the file path isn't empty
        if( filePath == null || filePath.isEmpty() ) {
            throw new FileAccessException("empty file path for XML file: unable to open it!");
        }
        
        // Check the file path is a valid XML file 
        File dicFile = new File(filePath);
        if ( !dicFile.exists() ) {
            throw new FileAccessException("the target dictionary file does not exist:\n" + filePath);
        }
        else if( dicFile.isDirectory() ) {
            throw new FileAccessException("the target dictionary is a folder not a XML file:\n" + filePath);
        }

        // Open the XML file to get its root element
        SAXBuilder xmlFile = new SAXBuilder(XMLReaders.NONVALIDATING);
        Document dictionary;
        try {
            dictionary = xmlFile.build(dicFile);
        }
        catch (JDOMException e1) {
            throw new FileAccessException("error while to parsing the dictionary!\n", e1);
        }
        catch (IOException e1) {
            throw new FileAccessException("an I/O error prevent parsing dictionary!\n", e1);
        }

        return dictionary.getRootElement();
    }
}

