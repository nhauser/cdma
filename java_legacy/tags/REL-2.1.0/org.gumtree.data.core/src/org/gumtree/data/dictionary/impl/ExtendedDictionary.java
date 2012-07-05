/****************************************************************************** 
 * Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 * 	  Clement Rodriguez - initial API and implementation
 *    Norman Xiong
 ******************************************************************************/
package org.gumtree.data.dictionary.impl;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IExtendedDictionary;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.IPathMethod;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IKey;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.JDOMException;
import org.jdom.input.SAXBuilder;


/**
 * @note This class is just a test and is not representative of how the real implementation should work.
 * Behaviors and algorithms of this class do not apply to the CDM dictionary's behaviour!
 * @author rodriguez
 *
 */
public class ExtendedDictionary implements IExtendedDictionary
{
	private IFactory            m_factory;     // Name of the plug-in's factory that created this object 
    private String              m_experiment;  // Experiment matching that dictionary
    private String              m_version;     // Version of the read dictionary
    private ExternalClassLoader m_classLoader; // Object that loads external classes
    
    private String m_keyFile;
    private String m_mapFile;
    
	private HashMap<IKey, String>  m_keyMap = new HashMap<IKey, String>();
	private HashMap<String, IPath> m_pathMap = new HashMap<String, IPath>();
	private HashMap<String, ExtendedDictionary> m_subDict = new HashMap<String, ExtendedDictionary>();
	
	public ExtendedDictionary(IFactory factory, String keyFile, String mapFile) { 
		m_factory    = factory; 
		m_experiment = Factory.getActiveView();
		m_keyFile    = keyFile;
		m_mapFile    = mapFile;
	}
	
	protected ExtendedDictionary(IFactory factory, String keyFile, String mapFile, String experiment) {
		m_experiment = experiment;
		m_factory    = factory;
		m_keyFile    = keyFile;
		m_mapFile    = mapFile;
	}
	
	@Override
	public void addEntry(String keyName, String entryPath)
	{
    	IPath path = null;
    	IKey key = m_factory.createKey(keyName);
		m_keyMap.put(key, keyName);
		m_pathMap.put(keyName, path);
	}
    
	@Override
	public void addEntry(String keyName, IPath path) {
    	IKey key = m_factory.createKey(keyName);
		m_keyMap.put(key, keyName);
		m_pathMap.put(keyName, path);
	}

    @Override
	public boolean containsKey(String keyName)
	{
		return m_keyMap.containsKey(keyName);
	}

	@Override
	public List<IKey> getAllKeys()
	{
		return new ArrayList<IKey>(m_keyMap.keySet());
	}

	@Override
	public List<IPath> getAllPaths(IKey key)
	{
		String keyId = m_keyMap.get(key);
		List<IPath> result = null;
		if( keyId != null ) {
			 result = new ArrayList<IPath>();
			 result.add(m_pathMap.get(keyId));
		}
		
		return result;
	}

	@Override
	public IPath getPath(IKey key)
	{
		IPath path = null;
		if( m_keyMap.containsKey(key) )
		{
			String keyName = m_keyMap.get(key);
			path = m_pathMap.get(keyName);
		}
		return path;
	}

	@Override
	public void readEntries(URI uri) throws FileAccessException
	{
		File dicFile = new File(uri);
		if (!dicFile.exists()) 
		{
			throw new FileAccessException("the target dictionary file does not exist");
		}
		try 
		{
			String filePath = dicFile.getAbsolutePath();
			readEntries(filePath);
		} 
		catch (Exception ex) 
		{
			throw new FileAccessException("failed to open the dictionary file", ex);
		}
	}

	@Override
    public void readEntries(String filePath) throws FileAccessException {
		if( filePath != null ) {
			m_keyFile = filePath;
		}

        // Read corresponding dictionaries
		readKeyDictionary();
		readMappingDictionary();
    }

	@Override
    public void readEntries() throws FileAccessException {
		readEntries((String) null);
	}
	
	@Override
	public void removeEntry(String keyName, String path) {
		IKey key = Factory.getFactory().createKey(keyName);
		String keyID = m_keyMap.get(key);
		m_keyMap.remove(key);
		m_pathMap.remove(keyID);
	}

	@Override
	public void removeEntry(String keyName) {
		IKey key = Factory.getFactory().createKey(keyName); 
		String keyID = m_keyMap.get(key);
		m_keyMap.remove(key);
		m_pathMap.remove(keyID);
	}
	
    
    @SuppressWarnings("unchecked")
	@Override
	public IDictionary clone() throws CloneNotSupportedException
	{
		ExtendedDictionary dict = new ExtendedDictionary(m_factory, m_keyFile, m_mapFile, m_experiment);
		dict.m_classLoader = m_classLoader;
		dict.m_version = m_version;
		dict.m_keyMap  = (HashMap<IKey, String>) m_keyMap.clone();
		dict.m_pathMap = (HashMap<String, IPath>) m_pathMap.clone();
		dict.m_subDict = (HashMap<String, ExtendedDictionary>) m_subDict.clone();
		return dict;
	}
    
    @Override
    public ExtendedDictionary getDictionary(IKey key) {
    	String keyID = m_keyMap.get(key);
    	ExtendedDictionary subDict = null;
    	
    	if( keyID != null ) {
    		subDict = m_subDict.get(keyID);
    	}
    	
    	return subDict;
    }
    
    @Override
	public String getVersionNum() {
		return m_version;
	}
	
	@Override
	public String getView() {
		return m_experiment;
	}
    
	@Override
	public ExternalClassLoader getClassLoader() {
		if( m_classLoader == null ) {
			m_classLoader = new ExternalClassLoader(m_factory.getName(), m_version);
		}
		
		return m_classLoader;
	}
	
	@Override
	public String getFactoryName() {
		return m_factory.getName();
	}
	
	@Override
	public String getKeyFilePath() {
		return m_keyFile;
	}

	@Override
	public String getMappingFilePath() {
		return m_mapFile;
	}

	
	// ---------------------------------------------------------------
	// PRIVATE : Reading methods
	// ---------------------------------------------------------------
	private HashMap<IKey, String> readKeyDictionary() throws FileAccessException {
		Element root = saxBuildFile(m_keyFile);
        
        String exp = root.getAttributeValue("name");
        if( ! exp.equalsIgnoreCase(m_experiment) ) {
        	throw new FileAccessException("an I/O error prevent parsing dictionary!\nThe dictionary doesn't match the experiment!");
        }
        return readKeyDictionary(root);
	}
	
	private HashMap<IKey, String> readKeyDictionary(Element xmlNode) throws FileAccessException {

		List<?> nodes = xmlNode.getChildren();
		Element elem;
		IKey key;
		String keyName;

		for( Object current : nodes ) {
			elem = (Element) current;
			keyName = elem.getAttributeValue("key");
			if( keyName != null && !keyName.isEmpty() ) {
				key = m_factory.createKey(keyName);
				
				// If the element is an entry
				if( elem.getName().equals("item") ) {
					m_keyMap.put(key, keyName);
				}
				// If the element is a group of keys
				else if( elem.getName().equals("group") ){
			        // Read corresponding dictionaries
					ExtendedDictionary dict = new ExtendedDictionary(m_factory, m_keyFile, m_mapFile, m_experiment);
			        dict.readKeyDictionary(elem);
			        dict.readMappingDictionary();
					m_subDict.put(keyName, dict);
					m_keyMap.put(key, keyName);
				}
			}
		}

		return m_keyMap;
	}
	
	@SuppressWarnings("unchecked")
	private HashMap<String, String> readMappingDictionary() throws FileAccessException {
		HashMap<String, String> mappingMap = new HashMap<String, String>();
		
		Element root = saxBuildFile(m_mapFile);
        m_version = root.getAttributeValue("version");
        
        List<?> nodes = root.getChildren();

		IPathMethod meth;
		IPath path;
		String keyID;
		IKey key;

        for( Entry<IKey, String> fullKey : m_keyMap.entrySet() ) {
        	key   = fullKey.getKey();
        	keyID = fullKey.getValue();
        	
        	// Check if a corresponding logical group exists
        	if( m_subDict.containsKey(keyID) ) {
        		// Create the corresponding path for a logical group
				meth = new PathMethod( "org.gumtree.data.Factory.createLogicalGroup" );
				path = m_factory.createPath(keyID);
				meth.pushParam(key);
				meth.isExternal(false);
				m_keyMap.put(key, keyID);
				m_pathMap.put(keyID, path);
        	}
        	// No corresponding path where found so doing key/path association
        	else if( !m_pathMap.containsKey(keyID) ) {
        		for( Element elem : (List<Element>) nodes ) {
        			if( elem.getAttributeValue("key").equals(keyID) ) {
        				path = loadPath(elem);
        				if( path == null ) {
        					throw new FileAccessException("error while associating IKey to IPath from dictionary!");
        				}
        				m_pathMap.put(keyID, path);
        				break;
        			}
        		}
        	}
        }

		return mappingMap;
	}
	
	@SuppressWarnings("unchecked")
	private IPath loadPath(Element elem) {
		IPath path = null;
		List<?> nodes = elem.getChildren();
		List<IPathMethod> methods = new ArrayList<IPathMethod>();
		PathMethod method;
		path = m_factory.createPath("");
		for( Element node : (List<Element>) nodes ) {
			if( node.getName().equals("path") )	{
				path.setValue(node.getText());
			}
			else if( node.getName().equals("call") )	{
				method = new PathMethod( node.getText() );
				method.isExternal(true);
				methods.add( method );

				// Set method calls on path
				path.setMethods(methods);
			}
		}
			
		return path;
	}
	
	private Element saxBuildFile(String filePath) throws FileAccessException {
		// Determine the experiment dictionary according to given path
        File dicFile = null;
        if( filePath != null && !filePath.isEmpty() ) {
	        dicFile = new File(filePath);
	        // file doesn't exist
	        if (!dicFile.exists()) {
	            throw new FileAccessException("the target dictionary file does not exist:\n" + filePath);
	        }

	        // Try to detect the corresponding experiment
	        String file = dicFile.getName();
	        String view = file.substring( 0, file.lastIndexOf("_") );
        }
        if( dicFile == null ) {
    		dicFile = new File( Factory.getKeyDictionaryPath() );
    		
    		if( ! dicFile.exists() ) {
                throw new FileAccessException("the target dictionary file does not exist");
            }
        }

        // Parse the XML key dictionary
        SAXBuilder xmlFile = new SAXBuilder();
        Element root;
        Document dictionary;
        try {
            dictionary = xmlFile.build(dicFile);
        }
        catch (JDOMException e1) {
            throw new FileAccessException("error while to parsing the dictionary!\n" + e1.getMessage());
        }
        catch (IOException e1) {
            throw new FileAccessException("an I/O error prevent parsing dictionary!\n" + e1.getMessage());
        }
        root = dictionary.getRootElement();
        
        return root;
	}
}

