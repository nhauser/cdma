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


import java.util.ArrayList;
import java.util.List;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.filter.IFilter;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.internal.dictionary.ItemSolver;
import org.cdma.utils.Utilities.ModelType;

public class LogicalGroup implements IContainer, Cloneable {

    public static final String KEY_PATH_SEPARATOR = ":";

    // Physical structure
    private IDataset            mDataset;      // File handler

    // Logical structure
    private IKey                mKey;          // IKey that populated this items (with filters eventually used)
    private ExtendedDictionary  mDictionary;   // Dictionary that belongs to this current LogicalGroup
    private LogicalGroup        mParent;       // Parent logical group if root then, it's null
    private IFactory            mFactory;
    private boolean             mThrow;        // Display debug info trace when dictionary isn't valid 

    public LogicalGroup(IKey key, IDataset dataset) {
        this(key, dataset, false);
    }


    public LogicalGroup(IKey key, IDataset dataset, boolean exception) {
        this( null, key, dataset, exception);
    }
    public LogicalGroup(LogicalGroup parent, IKey key, IDataset dataset ) {
        this( parent, key, dataset, false);
    }

    public LogicalGroup(LogicalGroup parent, IKey key, IDataset dataset,  boolean exception ) {
        if( key != null ) {
            mKey = key.clone();
        }
        else {
            mKey = null;
        }
        mParent  = parent;
        mDataset = dataset;
        mFactory = Factory.getFactory( dataset.getFactoryName() );
        mThrow   = exception;
        if( parent != null && parent.getDictionary() != null ) {
            mDictionary = parent.getDictionary().getDictionary(key);
        }
        else {
            mDictionary = null;
        }
    }

    @Override
    public LogicalGroup clone() {
        LogicalGroup group = new LogicalGroup(
                mParent, 
                mKey.clone(), 
                mDataset, 
                mThrow
                );
        ExtendedDictionary dictionary = null;
        try {
            dictionary = (ExtendedDictionary) mDictionary.clone();
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
    public LogicalGroup getParentGroup() {
        return mParent;
    }

    @Override
    public LogicalGroup getRootGroup() {
        if( getParentGroup() == null ) {
            return this;
        }
        else {
            return (LogicalGroup) mParent.getRootGroup();
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
     * Get the dictionary belonging to this LogicalGroup.
     * 
     * @return IDictionary the dictionary currently applied to this group
     */
    public ExtendedDictionary getDictionary() {
        if( mDictionary == null ) {
            mDictionary = findAndReadDictionary();
        }
        return mDictionary;
    }

    /**
     * Set a dictionary to this LogicalGroup.
     * 
     * @param dictionary the dictionary to set
     */
    public void setDictionary(ExtendedDictionary dictionary) {
        mDictionary = dictionary;
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
    public IDataItem getDataItem(String keyPath) {
        String[] keys = keyPath.split(KEY_PATH_SEPARATOR);

        int i = 0;
        LogicalGroup grp = this;
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
    public List<IDataItem> getDataItemList(String keyPath) {
        String[] keys = keyPath.split(KEY_PATH_SEPARATOR);

        int i = 0;
        LogicalGroup grp = this;
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
     * @return the first encountered LogicalGroup that matches the key, else null
     */
    public LogicalGroup getGroup(IKey key) {

        LogicalGroup group = null;
        List<IContainer> list = getItemByKey(key);

        for( IContainer object : list ) {
            if( object.getModelType().equals(ModelType.LogicalGroup) ) {
                group = (LogicalGroup) object;
                break;
            }
        }

        return group;
    }

    /**
     * Find the Group corresponding to the given key in the dictionary.
     *  
     * @param keyPath separated entries of the dictionary (can't carry filters)
     * @return the first encountered LogicalGroup that matches the key, else null
     * @note keyPath can contain several keys concatenated with a plug-in's separator
     */
    public LogicalGroup getGroup(String keyPath) {
        String[] keys = keyPath.split(KEY_PATH_SEPARATOR);

        int i = 0;
        LogicalGroup grp = this;
        LogicalGroup result = null;
        if( keys.length >= 1 ) {
            while( i < keys.length && grp != null) {
                grp = grp.getGroup( mFactory.createKey(keys[i++]) );
            }
            result = grp;
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
    public IKey bindKey(String bind, IKey key) {
        // TODO Auto-generated method stub
        return null;
    }

    public void setParent(LogicalGroup group) {
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
     * @param group LogicalGroup
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
     * This method defines the way the ExtendedDictionary will be loaded.
     * It must manage the do the detection and loading of the key file, 
     * and the corresponding mapping file that belongs to the plug-in.
     * Once the dictionary has its paths targeting both key and mapping
     * files set, the detection work is done. It just remains the loading 
     * of those files using the ExtendedDictionary.
     * 
     * @return ExtendedDictionary instance, that has already loaded keys and paths
     * @note ExtendedDictionary.readEntries() is already implemented in the core 
     */
    public ExtendedDictionary findAndReadDictionary() {
        // Detect the key dictionary file and mapping dictionary file
        String keyFile = Factory.getPathKeyDictionary();
        String mapFile = Factory.getPathMappingDictionaryFolder( mFactory ) + mFactory.getName() + "_dictionary.xml";
        String cntFile = Factory.getPathExperimentConceptDictionary();
        mDictionary = new ExtendedDictionary( mFactory, keyFile, mapFile, cntFile );
        try {
            mDictionary.readEntries();
        } catch (FileAccessException e) {
            e.printStackTrace();
        }
        return mDictionary;
    }

    // ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
    /// private methods
    // ------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------
    private List<IContainer> getItemByKey(IKey key) {
        // Get the working dictionary
        ExtendedDictionary dico = getDictionary();

        // Create the context of resolution
        Context context = new Context(mDataset, this, key);
        
        // Update context
        context.setConcept( dico.getConcept(key) );
        
        // Get the corresponding item solver
        ItemSolver solver = dico.getItemSolver(key);

        // Create output list
        List<IContainer> result;
        
        // Execute the solver
        if( solver != null ) {
            result = solver.solve(context);
        }
        // No solver returns empty list
        else {
            result = new ArrayList<IContainer>();
        }
   
        return result;
    }

/*
    private List<IContainer> getItemByKey(IKey key) {
        // Create the context of resolution
        List<IContainer> valid = new ArrayList<IContainer>();
        List<IContainer> found = new ArrayList<IContainer>();
        Context context = new Context(mDataset, this, key);
        
        // Get the path from the dictionary
        ExtendedDictionary dico = getDictionary();
        List<Solver> solvers = dico.getKeySolver( key );
        context.setConcept( dico.getConcept(key) );

        // Execute sequentially each solver
        for( Solver solver : solvers ) {
            found = solver.solve(context);
            
            // Check found items match what is requested by key
            valid.clear();
            for( IContainer container : found ) {
                if( isValidContainer( key, container ) ) {
                    valid.add(container);
                }
            }
            
            context.setContainers(valid);
        }

        return valid;
    }
 */
}

