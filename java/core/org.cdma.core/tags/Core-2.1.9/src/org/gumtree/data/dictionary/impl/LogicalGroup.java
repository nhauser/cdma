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

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IContext;
import org.gumtree.data.dictionary.IExtendedDictionary;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPathMethod;
import org.gumtree.data.dictionary.IPathParamResolver;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.exception.BackupException;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.exception.NoResultException;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.interfaces.IModelObject;
import org.gumtree.data.utils.Utilities.ModelType;

public class LogicalGroup implements ILogicalGroup {

	public static final String KEY_PATH_SEPARATOR = ":";

	// Physical structure
	IDataset              m_dataset;      // File handler
	
	// Logical structure
	IKey    	          m_key;          // IKey that populated this items (with filters eventually used)
    IExtendedDictionary	  m_dictionary;   // Dictionary that belongs to this current LogicalGroup
    ILogicalGroup         m_parent;       // Parent logical group if root then, it's null
    IFactory              m_factory;
    boolean               m_throw;        // Display debug info trace when dictionary isn't valid 

    public LogicalGroup(IKey key, IDataset dataset) {
    	this(key, dataset, false);
    }

    public LogicalGroup(IKey key, IDataset dataset, boolean exception) {
    	this( null, key, dataset, false);
    }
    public LogicalGroup(ILogicalGroup parent, IKey key, IDataset dataset ) {
    	this( parent, key, dataset, false);
    }

    public LogicalGroup(ILogicalGroup parent, IKey key, IDataset dataset, boolean exception ) {
    	if( key != null ) {
    		m_key = key.clone();
    	}
    	else {
    		m_key = null;
    	}
        m_parent  = parent;
        m_dataset = dataset;
        m_factory = Factory.getFactory( dataset.getFactoryName() );
        m_throw   = exception;
    }
    
    public ILogicalGroup clone() {
        LogicalGroup group = new LogicalGroup(m_parent, m_key, m_dataset, m_throw);
        group.m_dictionary = m_dictionary;

        return group;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LogicalGroup;
    }

    @Override
    public ILogicalGroup getParentGroup() {
        return m_parent;
    }

    @Override
    public ILogicalGroup getRootGroup() {
    	if( getParentGroup() == null ) {
    		return this;
    	}
    	else {
    		return (ILogicalGroup) m_parent.getRootGroup();
    	}
    }

    @Override
    public String getShortName() {
       	return m_key.getName();
    }

	@Override
	public String getLocation() {
		String location;
		if( m_parent == null ) {
			location = "/";
		}
		else {
			location = m_parent.getLocation();
			if ( !location.endsWith("/") ) {
				location+="/";
			}
			location += getName();
		}
		
		return location;
	}

    @Override
    public String getName() {
    	if( m_parent == null || m_key == null ) {
    		return "";
    	}
    	else {
    		return m_key.getName();
    	}
    }
    
	/**
	 * Get the dictionary belonging to the root group.
	 * 
	 * @return IDictionary
	 *            the dictionary currently applied to this group
	 */
	@Override
	public IExtendedDictionary getDictionary() {
		if( m_dictionary == null ) {
			m_dictionary = findAndReadDictionary();
		}
		return m_dictionary;
	}
	
	/**
	 * Set a dictionary to the root group.
	 * 
	 * @param dictionary
	 *            the dictionary to set
	 */
	@Override
	public void setDictionary(IDictionary dictionary) {
		m_dictionary = (IExtendedDictionary) dictionary;
	}
	
	/**
	 * Check if this is the logical root.
	 * 
	 * @return true or false
	 */
	boolean isRoot() {
		return (m_parent == null && m_key == null);
	}
	
	@Override
	public IDataItem getDataItem(IKey key) {
        IDataItem item = null;
        List<IContainer> list = new ArrayList<IContainer>();
        list = getItemByKey(key);
        
        for( IContainer object : list ) {
        	if( object.getModelType().equals(ModelType.DataItem) ) {
        		item = (IDataItem) object;
        		break;
        	}
        }
        
        return item;
	}

	@Override
	public IDataItem getDataItem(String keyPath) {
		String[] keys = keyPath.split(KEY_PATH_SEPARATOR);
		
		int i = 0;
		ILogicalGroup grp = this;
		IDataItem result = null;
		String key;
		if( keys.length >= 1 ) {
			while( i < (keys.length - 1) ) {
				key = keys[i++];
				if( key != null && !key.isEmpty() ) {
					grp = grp.getGroup( m_factory.createKey(key) );
				}
			}
			result = grp.getDataItem( m_factory.createKey(keys[i]) );
		}
		
		return result;
	}

	@Override
	public List<IDataItem> getDataItemList(IKey key) {
        List<IContainer> list = new ArrayList<IContainer>();
        List<IDataItem> result = new ArrayList<IDataItem>();
        list = getItemByKey(key);
        
        for( IContainer object : list ) {
        	if( object.getModelType().equals(ModelType.DataItem) ) {
        		result.add( (IDataItem) object);
        	}
        }
        
        return result;
	}

	@Override
	public List<IDataItem> getDataItemList(String keyPath) {
		String[] keys = keyPath.split(KEY_PATH_SEPARATOR);
		
		int i = 0;
		ILogicalGroup grp = this;
		List<IDataItem> result = null;
		if( keys.length >= 1 ) {
			while( i < (keys.length - 1) && grp != null) {
				grp = grp.getGroup( m_factory.createKey(keys[i++]) );
			}
			if( grp != null ) {
				result = grp.getDataItemList( m_factory.createKey(keys[i]) );
			}
		}
		
		return result;
	}

	public ILogicalGroup getGroup(IKey key) {
		ILogicalGroup item = null;
		
        // Get the path from the dictionary
        ExtendedDictionary dico = (ExtendedDictionary) getDictionary();
        ExtendedDictionary part = dico.getDictionary(key);
        
        // Construct the corresponding ILogicalGroup
        if( part != null ) {
        	item = new LogicalGroup(this, key, m_dataset, m_throw); 
        	item.setDictionary(part);
        }
		return item;
	}
	
	@Override
	public ILogicalGroup getGroup(String keyPath) {
		String[] keys = keyPath.split(KEY_PATH_SEPARATOR);
		
		int i = 0;
		ILogicalGroup grp = this;
		ILogicalGroup result = null;
		if( keys.length >= 1 ) {
			while( i < keys.length && grp != null) {
				grp = grp.getGroup( m_factory.createKey(keys[i++]) );
			}
			result = grp;
		}
		
		return result;
	}
	
	@Override
	public List<IPathParameter> getParameterValues(IKey key) {
		List<IPathParameter> result = new ArrayList<IPathParameter>();
		
		// Get the path
		IPath path = getDictionary().getPath(key);
		if( path != null ) {
			path.applyParameters( key.getParameterList() );
			
			// Extract first parameter (name and type)
			IPathParameter param = null;
			StringBuffer strPath = new StringBuffer();
			param = path.getFirstPathParameter(strPath);
	
			// Try to resolve parameter values
			IGroup root = m_dataset.getRootGroup();
			List<IContainer> list = new ArrayList<IContainer>();
			try {
				list.addAll( root.findAllContainerByPath(strPath.toString()) );
			} catch (NoResultException e) {}
			
			IPathParamResolver resolver = m_factory.createPathParamResolver(path);
			for( IContainer node : list ) {
				param = resolver.resolvePathParameter(node);
				if( param != null ) {
					result.add(param);
				}
			}
		}
		
		return result;
	}

	@Override
	public IDataset getDataset() {
		return m_dataset;
	}

	@Override
	public List<String> getKeyNames(ModelType model) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IKey bindKey(String bind, IKey key) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void setParent(ILogicalGroup group) {
		m_parent = group;
	}

	@Override
	public String getFactoryName() {
		return m_factory.getName();
	}
	
	@Override
    public boolean hasAttribute(String name, String value) {
        new BackupException("Object does not support this method!").printStackTrace();
        return false;
    }

    @Override
    public boolean removeAttribute(IAttribute attribute) {
        new BackupException("Object does not support this method!").printStackTrace();
        return false;
    }

    @Override
    public void setName(String name) {
        new BackupException("Object does not support this method!").printStackTrace();
    }

    @Override
    public void setParent(IGroup group) {
        new BackupException("Object does not support this method!").printStackTrace();
    }

    @Override
    public void setShortName(String name) {
        new BackupException("Object does not support this method!").printStackTrace();
    }

	@Override
	public void addOneAttribute(IAttribute attribute) {
		new BackupException("Object does not support this method!").printStackTrace();
	}

	@Override
	public void addStringAttribute(String name, String value) {
		new BackupException("Object does not support this method!").printStackTrace();
	}

	@Override
	public IAttribute getAttribute(String name) {
		new BackupException("Object does not support this method!").printStackTrace();
		return null;
	}

	@Override
	public List<IAttribute> getAttributeList() {
		new BackupException("Object does not support this method!").printStackTrace();
		return null;
	}
	
	@Override
	public IExtendedDictionary findAndReadDictionary() {
    	if( m_dictionary == null ) {
    		// Detect the key dictionary file and mapping dictionary file
    		String keyFile = Factory.getKeyDictionaryPath();
    		String mapFile = Factory.getMappingDictionaryFolder( m_factory ) + m_factory.getName().toLowerCase() + "_dictionary.xml";
   			m_dictionary = new ExtendedDictionary( m_factory, keyFile, mapFile );
            try {
                m_dictionary.readEntries();
            } catch (FileAccessException e) {
                e.printStackTrace();
            }
    	}
    	return m_dictionary;
    }
	
	// ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
    /// protected methods
    // ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
	/**
	 * Get all objects that match the given path parameterized by the given key
	 * @param key can contain some parameters
	 * @param path in string to be open
	 * @return list of IObject corresponding to path and key
	 * @throws NoResultException 
	 */
	protected List<IContainer> resolvePath(IKey key, IPath path) {
		List<IContainer> result = new ArrayList<IContainer>();
		if( path != null ) {
			List<IPathMethod> met = path.getMethods();
			
			// Apply given parameters on path
			path.applyParameters(key.getParameterList());
			path.removeUnsetParameters();
			
			if( met.size() > 0 ) {
				List<IContainer> input = new ArrayList<IContainer>();
				input.add(this);
				for( IPathMethod method : path.getMethods() ) {
					input = resolveMethod(key, method, input, path);
				}
				result = input;
			}
			else {
				try {
					result.addAll(m_dataset.getRootGroup().findAllContainerByPath(path.toString()));
				} catch (NoResultException e) {
					String message = e.getMessage() + "\nKey: " + key.getName();
					message += "\nPath: " + path.getValue();
					message += "\nData source: " + m_dataset.getLocation();
					message += "\nView: " + m_dictionary.getKeyFilePath();
					message += "\nMapping: " + m_dictionary.getMappingFilePath();
					if( m_throw ) {
						NoResultException ex = new NoResultException( message );
						ex.printStackTrace();
					}
				}
			}
			// Remove set parameters on path
			path.resetParameters();
		}
		return result;
	}

	@SuppressWarnings("unchecked")
	protected List<IContainer> resolveMethod(IKey key, IPathMethod method, List<IContainer> input, IPath path) {
		List<IContainer> output = new ArrayList<IContainer>();
		
		Object methodRes;

		int nbParam = method.getParam().length;
		IContext context = new Context(m_dataset, this, key, path);
		if( nbParam > 0 ) {
			context.setParams( method.getParam() );
		}
		
		for( IContainer current : input ) {
			try {
				if( method.isExternalCall() ) {
					methodRes = m_dictionary.getClassLoader().invoke(method.getName(), context);
				}
				else {
			    	// Extract class and method names
			    	String className     = method.getName().replaceAll("(^.*[^\\.]+)(\\.[^\\.]+$)+", "$1");
			    	String methodName    = method.getName().replaceAll("(^.*[^\\.]+\\.)([^\\.]+$)+", "$2");
			    	Class<?> classType   = java.lang.Class.forName(className);
			    	Class<?>[] paramType = new Class<?>[method.getParam().length];
			    	int i=0;
			    	for( Object obj : method.getParam() ) {
			    		if( obj instanceof IModelObject ) {
			    			paramType[i++] = obj.getClass().getInterfaces()[0];
			    		}
			    		else {
			    			paramType[i++] = obj.getClass();
			    		}
			    		
			    	}
			    	Method methodToCall  = classType.getMethod(methodName, paramType);
			    	methodRes = methodToCall.invoke( current, method.getParam() );
				}
 				
				if( methodRes == null ) {
					methodRes = current;
				}
				
				if( methodRes instanceof List ) {
					output.addAll( (List<IContainer>) methodRes );
				}
				else {
					output.add( (IContainer) methodRes );
				}
				
				 
			} catch (IllegalArgumentException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			catch (SecurityException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		return output;
	}
	
	// ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
    /// private methods
    // ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
	private List<IContainer> getItemByKey(IKey iKey) {
		// Get the path from the dictionary
        IDictionary dico = getDictionary();
        IPath iPath = dico.getPath(iKey);
        
        // Resolve the path and add result to children map
    	return resolvePath(iKey, iPath);
	}
}

