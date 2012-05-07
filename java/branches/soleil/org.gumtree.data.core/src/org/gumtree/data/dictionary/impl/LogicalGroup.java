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
package org.gumtree.data.dictionary.impl;


/**
 * @brief The LogicalGroup class is a purely <b>virtual</b> object that regroup several data.
 * 
 * <p>
 * Its existence is correlated to the ExtendedDictionary. A standard CDMA dictionary make 
 * a link between a key and a path. Now let's imagine a dictionary with keys having a tree
 * structure. This structure hierarchically organized might now have a meaning regardless
 * their physical organization. So the keys are now simple notions that can have a human
 * friendly meaning.
 * <p>
 * The LogicalGroup permits to browse simply through those different levels
 * of key. More over the key used can be filtered according to some criteria.
 * The aim is to find a really specific node by doing a search that get narrower
 * while iterating over queries.
 * 
 * @author rodriguez
 */


import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import org.gumtree.data.Factory;
import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IContext;
import org.gumtree.data.dictionary.IExtendedDictionary;
import org.gumtree.data.dictionary.ILogicalGroup;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.IPathMethod;
import org.gumtree.data.dictionary.IPathParamResolver;
import org.gumtree.data.dictionary.IPathParameter;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.exception.NoResultException;
import org.gumtree.data.exception.NotImplementedException;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.internal.IModelObject;
import org.gumtree.data.utils.Utilities.ModelType;

public class LogicalGroup implements ILogicalGroup, Cloneable {

    public static final String KEY_PATH_SEPARATOR = ":";

    // Physical structure
    private IDataset             mDataset;      // File handler

    // Logical structure
    private IKey               mKey;          // IKey that populated this items (with filters eventually used)
    private IExtendedDictionary   mDictionary;   // Dictionary that belongs to this current LogicalGroup
    private ILogicalGroup        mParent;       // Parent logical group if root then, it's null
    private IFactory             mFactory;
    private boolean              mThrow;        // Display debug info trace when dictionary isn't valid 

    public LogicalGroup(IKey key, IDataset dataset) {
        this(key, dataset, false);
    }


    public LogicalGroup(IKey key, IDataset dataset, boolean exception) {
        this( null, key, dataset, exception);
    }
    public LogicalGroup(ILogicalGroup parent, IKey key, IDataset dataset ) {
        this( parent, key, dataset, false);
    }

    public LogicalGroup(ILogicalGroup parent, IKey key, IDataset dataset,  boolean exception ) {
        if( key != null ) {
            mKey = key.clone();
        }
        else {
            mKey = null;
        }
        mParent     = parent;
        mDataset    = dataset;
        mFactory    = Factory.getFactory( dataset.getFactoryName() );
        mThrow      = exception;
        mDictionary = null;
    }

    @Override
    public ILogicalGroup clone() {
        LogicalGroup group = new LogicalGroup(
                mParent, 
                mKey.clone(), 
                mDataset, 
                mThrow
                );
        IExtendedDictionary dictionary = null;
        try {
            dictionary = (IExtendedDictionary) mDictionary.clone();
        } catch (CloneNotSupportedException e) {
        }
        group.setDictionary(dictionary);
        return group;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LogicalGroup;
    }

    @Override
    public ILogicalGroup getParentGroup() {
        return mParent;
    }

    @Override
    public ILogicalGroup getRootGroup() {
        if( getParentGroup() == null ) {
            return this;
        }
        else {
            return (ILogicalGroup) mParent.getRootGroup();
        }
    }

    @Override
    public String getShortName() {
        return mKey.getName();
    }

    @Override
    public String getLocation() {
        String location;
        if( mParent == null ) {
            location = "/";
        }
        else {
            location = mParent.getLocation();
            if ( !location.endsWith("/") ) {
                location+="/";
            }
            location += getName();
        }

        return location;
    }

    @Override
    public String getName() {
        if( mParent == null || mKey == null ) {
            return "";
        }
        else {
            return mKey.getName();
        }
    }

    /**
     * Get the dictionary belonging to this ILogicalGroup.
     * 
     * @return IDictionary the dictionary currently applied to this group
     */
    @Override
    public IExtendedDictionary getDictionary() {
        if( mDictionary == null ) {
            mDictionary = findAndReadDictionary();
        }
        return mDictionary;
    }

    /**
     * Set a dictionary to this ILogicalGroup.
     * 
     * @param dictionary the dictionary to set
     */
    @Override
    public void setDictionary(IDictionary dictionary) {
        mDictionary = (IExtendedDictionary) dictionary;
    }

    /**
     * Check if this is the logical root.
     * 
     * @return true or false
     */
    boolean isRoot() {
        return (mParent == null && mKey == null);
    }
    
    /**
     * Find the IDataItem corresponding to the given key in the dictionary.
     *  
     * @param key entry of the dictionary (can carry filters)
     * @return the first encountered IDataItem that match the key, else null
     */
    @Override
    public IDataItem getDataItem(IKey key) {
        IDataItem item = null;
        List<IContainer> list = getItemByKey(key);

        for( IContainer object : list ) {
            if( object.getModelType().equals(ModelType.DataItem) ) {
                item = (IDataItem) object;
                break;
            }
        }

        return item;
    }

    /**
     * Find the IDataItem corresponding to the given key in the dictionary.
     *  
     * @param keyPath separated entries of the dictionary (can't carry filters) 
     * @return the first encountered IDataItem that match the key, else null
     * @note keyPath can contain several keys concatenated with a plug-in's separator
     */
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
                    grp = grp.getGroup( mFactory.createKey(key) );
                }
            }
            result = grp.getDataItem( mFactory.createKey(keys[i]) );
        }

        return result;
    }

    /**
     * Find all IDataItems corresponding to the given key in the dictionary.
     *  
     * @param key entry of the dictionary (can carry filters)
     * @return a list of IDataItem that match the key
     */
    @Override
    public List<IDataItem> getDataItemList(IKey key) {
        List<IDataItem> result = new ArrayList<IDataItem>();
        List<IContainer> list = getItemByKey(key);

        for( IContainer object : list ) {
            if( object.getModelType().equals(ModelType.DataItem) ) {
                result.add( (IDataItem) object);
            }
        }

        return result;
    }

    /**
     * Find all IDataItems corresponding to the given path of key in the dictionary.
     *  
     * @param keyPath separated entries of the dictionary (can't carry filters)
     * @return a list of IDataItem that match the key
     * @note keyPath can contain several keys concatenated with a plug-in's separator
     */
    @Override
    public List<IDataItem> getDataItemList(String keyPath) {
        String[] keys = keyPath.split(KEY_PATH_SEPARATOR);

        int i = 0;
        ILogicalGroup grp = this;
        List<IDataItem> result = null;
        if( keys.length >= 1 ) {
            while( i < (keys.length - 1) && grp != null) {
                grp = grp.getGroup( mFactory.createKey(keys[i++]) );
            }
            if( grp != null ) {
                result = grp.getDataItemList( mFactory.createKey(keys[i]) );
            }
        }

        return result;
    }

    /**
     * Find the Group corresponding to the given key in the dictionary.
     *  
     * @param key entry name of the dictionary
     * @return the first encountered ILogicalGroup that matches the key, else null
     */
    public ILogicalGroup getGroup(IKey key) {
        ILogicalGroup item = null;

        // Get the path from the dictionary
        ExtendedDictionary dico = (ExtendedDictionary) getDictionary();
        ExtendedDictionary part = dico.getDictionary(key);

        // Construct the corresponding ILogicalGroup
        if( part != null ) {
            item = new LogicalGroup(this, key, mDataset, mThrow);
            item.setDictionary(part);
        }
        return item;
    }

    /**
     * Find the Group corresponding to the given key in the dictionary.
     *  
     * @param keyPath separated entries of the dictionary (can't carry filters)
     * @return the first encountered ILogicalGroup that matches the key, else null
     * @note keyPath can contain several keys concatenated with a plug-in's separator
     */
    @Override
    public ILogicalGroup getGroup(String keyPath) {
        String[] keys = keyPath.split(KEY_PATH_SEPARATOR);

        int i = 0;
        ILogicalGroup grp = this;
        ILogicalGroup result = null;
        if( keys.length >= 1 ) {
            while( i < keys.length && grp != null) {
                grp = grp.getGroup( mFactory.createKey(keys[i++]) );
            }
            result = grp;
        }

        return result;
    }

    /**
     * Return the list of parameters we can set on the given key that will
     * have an occurrence in the dataset.
     * 
     * @param key IKey for which we want the parameters values
     * @return List<IPathParameter> that can be directly applied on the key
     * @note <b>EXPERIMENTAL METHOD</b> do note use/implements
     * @note if the path that matches the key hold several different parameters 
     * the method will return the FIRST undefined parameter. To know deeper parameters,
     * user has to set some IPathParameter on the key and call again this method
     */
    @Override
    public List<IPathParameter> getParameterValues(IKey key) {
        List<IPathParameter> result = new ArrayList<IPathParameter>();

        // Get the path
        IPath path = getDictionary().getPath(key);
        if( path != null ) {
            path.applyParameters( key.getParameterList() );

            // Extract first parameter (name and type)
            StringBuffer strPath = new StringBuffer();
            IPathParameter param;
            path.getFirstPathParameter(strPath);

            // Try to resolve parameter values
            IGroup root = mDataset.getRootGroup();
            List<IContainer> list = new ArrayList<IContainer>();
            try {
                list.addAll( root.findAllContainerByPath(strPath.toString()) );
            } catch (NoResultException e) {}

            IPathParamResolver resolver = mFactory.createPathParamResolver(path);
            for( IContainer node : list ) {
                param = resolver.resolvePathParameter(node);
                if( param != null ) {
                    result.add(param);
                }
            }
        }

        return result;
    }

    /**
     * Get the IDataset that hold the current Group.
     * 
     * @return CDMA IDataset 
     */
    @Override
    public IDataset getDataset() {
        return mDataset;
    }

    /**
     * Return the list of key that match the given model type.
     * 
     * @param model which kind of keys (ie: IDataItem, Group, ILogical, Attribute...)
     * @return List of type Group; may be empty, not null.
     */
    @Override
    public List<String> getKeyNames(ModelType model) {
        // TODO Auto-generated method stub
        return null;
    }

    /**
     * Bind the given key with the given name, so the key can be accessed by the bind
     * 
     * @param bind value with which we can get the key
     * @param key key object to be mapped by the bind value 
     * @return the given key
     */
    @Override
    public IKey bindKey(String bind, IKey key) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void setParent(ILogicalGroup group) {
        mParent = group;
    }

    @Override
    public String getFactoryName() {
        return mFactory.getName();
    }

    @Override
    public boolean hasAttribute(String name, String value) {
        new NotImplementedException().printStackTrace();
        return false;
    }

    @Override
    public boolean removeAttribute(IAttribute attribute) {
        new NotImplementedException().printStackTrace();
        return false;
    }

    @Override
    public void setName(String name) {
        new NotImplementedException().printStackTrace();
    }

    /**
     * Set the given logical group as parent of this logical group
     * 
     * @param group ILogicalGroup
     */
    @Override
    public void setParent(IGroup group) {
        new NotImplementedException().printStackTrace();
    }

    @Override
    public void setShortName(String name) {
        new NotImplementedException().printStackTrace();
    }

    @Override
    public void addOneAttribute(IAttribute attribute) {
        new NotImplementedException().printStackTrace();
    }

    @Override
    public void addStringAttribute(String name, String value) {
        new NotImplementedException().printStackTrace();
    }

    @Override
    public IAttribute getAttribute(String name) {
        new NotImplementedException().printStackTrace();
        return null;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        new NotImplementedException().printStackTrace();
        return null;
    }

    /**
     * This method defines the way the IExtendedDictionary will be loaded.
     * It must manage the do the detection and loading of the key file, 
     * and the corresponding mapping file that belongs to the plug-in.
     * Once the dictionary has its paths targeting both key and mapping
     * files set, the detection work is done. It just remains the loading 
     * of those files using the IExtendedDictionary.
     * 
     * @return IExtendeddictionary instance, that has already loaded keys and paths
     * @note IExtendedDictionary.readEntries() is already implemented in the core 
     */
    @Override
    public IExtendedDictionary findAndReadDictionary() {
        if( mDictionary == null ) {
            // Detect the key dictionary file and mapping dictionary file
            String keyFile = Factory.getKeyDictionaryPath();
            String mapFile = Factory.getMappingDictionaryFolder( mFactory ) + mFactory.getName() + "_dictionary.xml";
            mDictionary = new ExtendedDictionary( mFactory, keyFile, mapFile );
            try {
                mDictionary.readEntries();
            } catch (FileAccessException e) {
                e.printStackTrace();
            }
        }
        return mDictionary;
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
                    result.addAll(mDataset.getRootGroup().findAllContainerByPath(path.toString()));
                } catch (NoResultException e) {
                    String message = e.getMessage() + "\nKey: " + key.getName();
                    message += "\nPath: " + path.getValue();
                    message += "\nData source: " + mDataset.getLocation();
                    message += "\nView: " + mDictionary.getKeyFilePath();
                    message += "\nMapping: " + mDictionary.getMappingFilePath();
                    if( mThrow ) {
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
        IContext context = new Context(mDataset, this, key, path);
        if( nbParam > 0 ) {
            context.setParams( method.getParam() );
        }

        for( IContainer current : input ) {
            try {
                if( method.isExternalCall() ) {
                    methodRes = mDictionary.getClassLoader().invoke(method.getName(), context);
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


            }
            catch (IllegalArgumentException e) {
            }
            catch (NoResultException e) {
                e.printStackTrace();
            }
            catch (SecurityException e) {
            }
            catch (Exception e) {
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

