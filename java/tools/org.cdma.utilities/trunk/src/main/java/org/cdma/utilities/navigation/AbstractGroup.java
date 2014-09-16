/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.utilities.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.cdma.dictionary.Path;
import org.cdma.exception.BackupException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.SignalNotAvailableException;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.utilities.navigation.internal.NodeParentAttribute;
import org.cdma.utils.Utilities.ModelType;

/**
 * Default implementation of IGroup interface. This class is abstract.
 * Only the following methods has to be implemented:<br/>
 * - public long getLastModificationDate
 * - protected void init() which does all necessary initialization for this implementation<br/>
 * - protected String getPathSeparator() which give the string representation of separator of two
 * different groups
 * 
 * @author rodriguez
 * 
 */

public abstract class AbstractGroup extends NodeParentAttribute implements IGroup {
    private boolean mInitialized;
    protected List<IDimension> mDimensions; // Dimensions of this
    protected List<IContainer> mContainers; // Containers of this (i.e: Group and DataItem)

    public AbstractGroup(String factory, IDataset dataset, IGroup parent, String name) throws BackupException {
        super(factory, dataset, parent, name);
        mInitialized = false;
        mDimensions = new ArrayList<IDimension>();
        mContainers = new ArrayList<IContainer>();
    }

    public AbstractGroup(AbstractGroup group) throws BackupException {
        this(group.getFactoryName(), group.getDataset(), group.getParentGroup(), group.getShortName());
        mInitialized = false;
    }

    /**
     * Initialize internal values: children items and groups, dimensions
     */
    abstract protected void initChildren();

    @Override
    abstract public long getLastModificationDate();

    @Override
    abstract public Map<String, String> harvestMetadata(String mdStandard) throws IOException;

    @Override
    abstract public IGroup clone();

    /**
     * Return the string representing a separator between two parent elements
     * in a hierarchical path.
     */
    @Override
    abstract protected String getPathSeparator();

    @Override
    public ModelType getModelType() {
        return ModelType.Group;
    }

    @Override
    public void addDataItem(IDataItem item) {
        mContainers.add(item);
    }

    @Override
    public void addOneDimension(IDimension dimension) {
        mDimensions.add(dimension);
    }

    @Override
    public void addSubgroup(IGroup group) {
        mContainers.add(group);
    }

    @Override
    public IDataItem getDataItem(String shortName) {
        initialize();
        IDataItem result = null;
        if (shortName != null) {
            for (IContainer container : mContainers) {
                if (container != null) {
                    if (container.getModelType().equals(ModelType.DataItem) && shortName.equals(container.getShortName())) {
                        result = (IDataItem) container;
                        break;
                    }
                }
            }
        }
        return result;
    }

    @Override
    public IDataItem getDataItemWithAttribute(String name, String value) {
        initialize();
        IDataItem result = null;
        if (name != null) {
            for (IContainer container : mContainers) {
                if (container != null) {
                    if (container.getModelType().equals(ModelType.DataItem) && container.hasAttribute(name, value)) {
                        result = (IDataItem) container;
                        break;
                    }
                }
            }
        }
        return result;
    }

    @Override
    public IDimension getDimension(String name) {
        initialize();
        IDimension result = null;
        if (name != null) {
            for (IDimension dimension : mDimensions) {
                if (name.equals(dimension.getName())) {
                    result = dimension;
                    break;
                }
            }
        }
        return result;
    }

    @Override
    public IContainer getContainer(String shortName) {
        initialize();
        IContainer result = null;
        if (shortName != null) {
            for (IContainer container : mContainers) {
                if (container != null) {
                    if (shortName.equals(container.getShortName())) {
                        result = container;
                        break;
                    }
                }
            }
        }
        return result;
    }

    @Override
    public IGroup getGroup(String shortName) {
        initialize();
        IGroup result = null;
        if (shortName != null) {
            for (IContainer container : mContainers) {
                if (container != null) {
                    if (container.getModelType().equals(ModelType.Group) && shortName.equals(container.getShortName())) {
                        result = (IGroup) container;
                        break;
                    }
                }
            }
        }
        return result;
    }

    @Override
    public IGroup getGroupWithAttribute(String name, String value) {
        initialize();
        IGroup result = null;
        if (name != null) {
            for (IContainer container : mContainers) {
                if (container != null) {
                    if (container.getModelType().equals(ModelType.Group) && container.hasAttribute(name, value)) {
                        result = (IGroup) container;
                        break;
                    }
                }
            }
        }
        return result;
    }

    @Override
    public List<IDataItem> getDataItemList() {
        initialize();
        List<IDataItem> result = new ArrayList<IDataItem>();

        for (IContainer container : mContainers) {
            if ((container != null) && container.getModelType().equals(ModelType.DataItem)) {
                result.add((IDataItem) container);
            }
        }

        return result;
    }

    @Override
    public List<IDimension> getDimensionList() {
        initialize();
        return mDimensions;
    }

    @Override
    public List<IGroup> getGroupList() {
        initialize();
        List<IGroup> result = new ArrayList<IGroup>();

        for (IContainer container : mContainers) {
            if ((container != null) && container.getModelType().equals(ModelType.Group)) {
                result.add((IGroup) container);
            }
        }

        return result;
    }

    @Override
    public boolean removeDataItem(IDataItem item) {
        initialize();
        boolean result = false;
        if (item != null) {
            result = mContainers.remove(item);
        }
        return result;
    }

    @Override
    public boolean removeDataItem(String name) {
        initialize();
        IDataItem item = getDataItem(name);
        return removeDataItem(item);
    }

    @Override
    public boolean removeDimension(String name) {
        initialize();
        IDimension dimension = getDimension(name);
        return removeDimension(dimension);
    }

    @Override
    public boolean removeGroup(IGroup group) {
        initialize();
        boolean result = false;
        if (group != null) {
            result = mContainers.remove(group);
        }
        return result;
    }

    @Override
    public boolean removeGroup(String name) {
        initialize();
        IGroup item = getGroup(name);
        return removeGroup(item);
    }

    @Override
    public boolean removeDimension(IDimension dimension) {
        initialize();
        boolean result = false;
        if (dimension != null) {
            result = mDimensions.remove(dimension);
        }
        return result;
    }

    @Deprecated
    @Override
    public void setDictionary(org.cdma.interfaces.IDictionary dictionary) {
        throw new NotImplementedException();
    }

    @Deprecated
    @Override
    public org.cdma.interfaces.IDictionary findDictionary() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isRoot() {
        return getParentGroup() == null;
    }

    @Override
    public boolean isEntry() {
        return ((getParentGroup() != null) && getParentGroup().isRoot());
    }

    @Override
    public IDataItem findDataItem(String shortName) {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IGroup findGroup(String shortName) {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IGroup findGroup(IKey key) {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IContainer findContainer(String shortName) {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IContainer findContainerByPath(String path) throws NoResultException {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<IContainer> findAllContainerByPath(String path) throws NoResultException {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<IContainer> findAllContainers(IKey key) throws NoResultException {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<IContainer> findAllOccurrences(IKey key) throws NoResultException {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IContainer findObjectByPath(Path path) {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IDataItem findDataItem(IKey key) {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IDataItem findDataItemWithAttribute(IKey key, String name, String attribute) throws NoResultException {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IGroup findGroupWithAttribute(IKey key, String name, String value) {
        initialize();
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void updateDataItem(String key, IDataItem dataItem) throws SignalNotAvailableException {
        initialize();
        // TODO Auto-generated method stub

    }

    private void initialize() {
        if (!mInitialized) {
            mInitialized = true;
            initChildren();
        }
    }

    @Override
    public String toString() {
        return getLocation() + "\n" + getName();
    }
}
