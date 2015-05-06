/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * Tony Lam (nxi@Bragg Institute) - initial API and implementation
 * Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 * Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.nexus.navigation;

// Standard import
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.Path;
import org.cdma.engine.hdf.navigation.HdfDataItem;
import org.cdma.engine.hdf.navigation.HdfGroup;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.SignalNotAvailableException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.interfaces.INode;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.cdma.plugin.soleil.nexus.dictionary.NxsDictionary;
import org.cdma.plugin.soleil.nexus.dictionary.NxsLogicalGroup;
import org.cdma.plugin.soleil.nexus.utils.NxsNode;
import org.cdma.plugin.soleil.nexus.utils.NxsPath;
import org.cdma.utils.Utilities.ModelType;

public final class NxsGroup implements IGroup, Cloneable {

    private static final String NX_CLASS = "NX_class";

    // ****************************************************
    // Members
    // ****************************************************
    private NxsDataset mDataset; // Dataset to which this group belongs to
    private HdfGroup[] mGroups; // Groups having a similar path from different files
    private IGroup mParent; // Parent group folder (mandatory)
    private List<IContainer> mChildren; // All containers that are below (physically) this one
    private List<IDimension> mDimensions; // list of dimension
    private boolean mIsChildUpdate; // is the children list up to date
    private boolean mIsMultigroup; // is this group managing aggregation of group
    private NxsNode mNode;

    // ****************************************************
    // Constructors
    // ****************************************************
    private NxsGroup() {
        mParent = null;
        mDataset = null;
        mChildren = null;
        mDimensions = null;
        mIsChildUpdate = false;
        mNode = null;
        listChildren();
    }

    public NxsGroup(final NxsDataset dataset, final String name, final String path, final NxsGroup parent) {
        this(dataset, name, path, parent.getHdfGroup());
        mParent = parent;
    }

    private NxsGroup(final NxsDataset dataset, final String name, final String path, final HdfGroup parent) {
        mGroups = new HdfGroup[1];
        mGroups[0] = new HdfGroup(NxsFactory.NAME, name, path, parent, dataset.getHdfDataset());
        mParent = null;
        mDataset = dataset;
        mChildren = null;
        mDimensions = null;
        mIsChildUpdate = false;
        mNode = null;
    }

    public NxsGroup(final HdfGroup[] groups, final IGroup parent, final NxsDataset dataset) {
        mGroups = groups.clone();
        mParent = parent;
        mDataset = dataset;
        mChildren = null;
        mDimensions = null;
        mIsChildUpdate = false;
        listChildren();
    }

    public NxsGroup(final NxsGroup original) {
        mGroups = new HdfGroup[original.mGroups.length];
        int i = 0;
        for (HdfGroup group : original.mGroups) {
            mGroups[i++] = group;
        }
        mParent = original.mParent;
        mDataset = original.mDataset;
        mChildren = null;
        mDimensions = null;
        mIsChildUpdate = false;
        mIsMultigroup = mGroups.length > 1;
        mNode = original.mNode;
        listChildren();
    }

    public NxsGroup(final IGroup parent, final NxsPath path, final NxsDataset dataset) {
        try {
            List<IContainer> list = dataset.getRootGroup().findAllContainerByPath(path.getValue());
            List<IGroup> groups = new ArrayList<IGroup>();
            for (IContainer container : list) {
                if (container.getModelType() == ModelType.Group) {
                    groups.add((IGroup) container);
                }
            }
            HdfGroup[] array = new HdfGroup[groups.size()];
            mGroups = groups.toArray(array);
        } catch (NoResultException e) {
        }
        mParent = parent;
        mDataset = dataset;
        mChildren = null;
        mDimensions = null;
        mIsChildUpdate = false;
        listChildren();
    }

    // ****************************************************
    // Methods from interfaces
    // ****************************************************
    /**
     * Return a clone of this IGroup object.
     * 
     * @return new IGroup
     */
    @Override
    public NxsGroup clone() {
        NxsGroup clone = new NxsGroup();
        clone.mGroups = new HdfGroup[mGroups.length];
        int i = 0;
        for (HdfGroup group : mGroups) {
            mGroups[i++] = group.clone();
        }
        if (mParent != null) {
            clone.mParent = mParent.clone();
        }
        clone.mDataset = mDataset;
        clone.mIsChildUpdate = false;
        return clone;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.Group;
    }

    @Override
    public IAttribute getAttribute(final String name) {
        IAttribute attr = null;
        for (IGroup group : mGroups) {
            attr = group.getAttribute(name);
            if (attr != null) {
                break;
            }
        }
        return attr;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        List<IAttribute> result = new ArrayList<IAttribute>();
        for (IGroup group : mGroups) {
            result.addAll(group.getAttributeList());
        }
        return result;
    }

    @Override
    public String getLocation() {
        // return mParent.getLocation() + "/" + getShortName();
        NxsPath path = getNxsPath();
        return String.valueOf(path == null ? path : path.toString());
    }

    @Override
    public String getName() {
        String name = "";
        if (mGroups.length > 0 && mGroups[0] != null) {
            name = mGroups[0].getName();
        }
        return name;
    }

    @Override
    public String getShortName() {
        String name = "";
        if (mGroups.length > 0) {
            name = mGroups[0].getShortName();
        }
        return name;
    }

    @Override
    public boolean hasAttribute(final String name, final String value) {
        for (IGroup group : mGroups) {
            if (group.hasAttribute(name, value)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void setName(final String name) {
        for (IGroup group : mGroups) {
            group.setName(name);
        }
    }

    @Override
    public void setShortName(final String name) {
        for (IGroup group : mGroups) {
            group.setShortName(name);
        }
    }

    @Override
    public void setParent(final IGroup group) {
        mParent = group;
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public Map<String, String> harvestMetadata(final String mdStandard) throws IOException {
        return null;
    }

    @Override
    public IGroup getParentGroup() {
        return mParent;
    }

    @Override
    public IGroup getRootGroup() {
        return mDataset.getRootGroup();
    }

    @Override
    public IDataItem getDataItem(final String shortName) {
        List<IDataItem> list = getDataItemList();
        IDataItem result = null;
        INode nodeName = NxsPath.splitStringToNode(shortName)[0];
        INode groupName;
        INode[] nodes;
        for (IDataItem item : list) {
            nodes = NxsPath.splitStringToNode(item.getName());
            groupName = nodes[nodes.length - 1];
            if (groupName.matchesNode(nodeName)) {
                result = item;
                break;
            }
        }
        return result;
    }

    @Override
    public IDataItem findDataItem(final IKey key) {
        IDataItem item = null;
        List<IContainer> list = new ArrayList<IContainer>();
        try {
            list = findAllOccurrences(key);
        } catch (NoResultException e) {
        }

        for (IContainer object : list) {
            if (object.getModelType().equals(ModelType.DataItem)) {
                item = (IDataItem) object;
                break;
            }
        }

        return item;
    }

    @Override
    public IDataItem findDataItem(final String keyName) {
        IKey key = NxsFactory.getInstance().createKey(keyName);

        return findDataItem(key);
    }

    @Override
    public IDataItem getDataItemWithAttribute(final String name, final String value) {
        List<IDataItem> list = getDataItemList();
        IDataItem result = null;
        for (IDataItem item : list) {
            if (item.hasAttribute(name, value)) {
                result = item;
                break;
            }
        }

        return result;
    }

    @Override
    public IDataItem findDataItemWithAttribute(final IKey key, final String name, final String attribute)
            throws NoResultException {
        List<IContainer> list = findAllContainers(key);
        IDataItem result = null;
        for (IContainer item : list) {
            if (item.getModelType() == ModelType.DataItem && item.hasAttribute(name, attribute)) {
                result = (IDataItem) item;
                break;
            }
        }

        return result;
    }

    @Override
    public IGroup findGroupWithAttribute(final IKey key, final String name, final String value) {
        List<IContainer> list;
        try {
            list = findAllContainers(key);
        } catch (NoResultException e) {
            list = new ArrayList<IContainer>();
        }
        IGroup result = null;
        for (IContainer item : list) {
            if (item.getModelType() == ModelType.Group && item.hasAttribute(name, value)) {
                result = (IGroup) item;
                break;
            }
        }

        return result;
    }

    @Override
    public IContainer getContainer(final String shortName) {
        List<IContainer> list = listChildren();
        IContainer result = null;

        for (IContainer container : list) {
            if (container.getShortName().equals(shortName)) {
                result = container;
                break;
            }
        }

        return result;
    }

    @Override
    public IGroup getGroup(final String shortName) {
        IGroup result = null;
        NxsNode node = new NxsNode(shortName);

        if (shortName.equals(this.getShortName())) {
            result = this;
        } else {
            if (!listContainer(node).isEmpty()) {
                IContainer cnt = listContainer(node).get(0);
                if (cnt.getModelType().equals(ModelType.Group)) {
                    result = (IGroup) cnt;
                }
            }
        }
        return result;

    }

    @Override
    public IGroup getGroupWithAttribute(final String attributeName, final String attributeValue) {
        List<IGroup> list = getGroupList();
        IGroup result = null;
        for (IGroup item : list) {
            if (item.hasAttribute(attributeName, attributeValue)) {
                result = item;
                break;
            }
        }

        return result;
    }

    @Override
    public List<IDataItem> getDataItemList() {
        listChildren();

        ArrayList<IDataItem> list = new ArrayList<IDataItem>();
        for (IContainer container : mChildren) {
            if (container.getModelType() == ModelType.DataItem) {
                list.add((IDataItem) container);
            }
        }

        return list;
    }

    @Override
    public IDataset getDataset() {
        return mDataset;
    }

    @Override
    public IGroup findGroup(final IKey key) {
        IGroup item = null;
        List<IContainer> list = new ArrayList<IContainer>();
        try {
            list = findAllOccurrences(key);
        } catch (NoResultException e) {
        }

        for (IContainer object : list) {
            if (object.getModelType().equals(ModelType.Group)) {
                item = (IGroup) object;
                break;
            }
        }

        return item;
    }

    @Override
    public IGroup findGroup(final String keyName) {
        IKey key = NxsFactory.getInstance().createKey(keyName);

        return findGroup(key);
    }

    @Override
    public List<IGroup> getGroupList() {
        listChildren();

        ArrayList<IGroup> list = new ArrayList<IGroup>();
        for (IContainer container : mChildren) {
            if (container.getModelType() == ModelType.Group) {
                list.add((IGroup) container);
            }
        }

        return list;
    }

    @Override
    public IContainer findContainer(final String shortName) {
        IKey key = NxsFactory.getInstance().createKey(shortName);
        IContainer result;
        try {
            List<IContainer> list = findAllOccurrences(key);
            if (list.size() > 0) {
                result = list.get(0);
            } else {
                result = null;
            }
        } catch (NoResultException e) {
            result = null;
        }

        return result;
    }

    @Override
    public IContainer findContainerByPath(final String path) throws NoResultException {
        List<IContainer> containers = findAllContainerByPath(path);
        IContainer result = null;

        if (containers.size() > 0) {
            result = containers.get(0);
        }

        return result;
    }

    //    @Override
    //    public List<IContainer> findAllContainerByPath(final String path) throws NoResultException {
    //        List<IContainer> list = new ArrayList<IContainer>();
    //        if (path != null) {
    //            List<IContainer> tmp = null;
    //            String location;
    //
    //            // Store in a map all different containers from all m_groups
    //            Map<String, ArrayList<IContainer>> items = new HashMap<String, ArrayList<IContainer>>();
    //            List<String> sortGrp = new ArrayList<String>();
    //            String absPath = mDataset.getRootGroup().getLocation() + path;
    //            for (IGroup group : mGroups) {
    //                try {
    //                    NxsNode[] nodes = ((NxsGroup) (mDataset.getRootGroup())).getNxsPath().getNodes();
    //                    tmp = ((HdfGroup) group).findAllContainerByPath(nodes);
    //                    for (IContainer item : tmp) {
    //                        location = item.getLocation();
    //                        if (items.containsKey(location)) {
    //                            items.get(location).add(item);
    //                        } else {
    //                            ArrayList<IContainer> tmpList = new ArrayList<IContainer>();
    //                            tmpList.add(item);
    //                            items.put(location, tmpList);
    //                            sortGrp.add(location);
    //                        }
    //                    }
    //                } catch (NoResultException e) {
    //                    // Nothing to do
    //                }
    //            }
    //            // Construct that were found
    //            for (String entry : sortGrp) {
    //
    //                // for( Entry<String, ArrayList<IContainer>> entry : items.entrySet() ) {
    //                tmp = items.get(entry);
    //                // If a Group list then construct a new Group folder
    //                if (tmp.get(0).getModelType() == ModelType.Group) {
    //                    list.add(new NxsGroup(tmp.toArray(new HdfGroup[tmp.size()]), this, mDataset));
    //                }
    //                // If a IDataItem list then construct a new compound NxsDataItem
    //                else {
    //                    ArrayList<HdfDataItem> dataItems = new ArrayList<HdfDataItem>();
    //                    for (IContainer item : tmp) {
    //                        if (item.getModelType() == ModelType.DataItem) {
    //                            dataItems.add((HdfDataItem) item);
    //                        }
    //                    }
    //                    HdfDataItem[] array = new HdfDataItem[dataItems.size()];
    //                    dataItems.toArray(array);
    //                    list.add(new NxsDataItem(array, this, mDataset));
    //                }
    //            }
    //        }
    //        return list;
    //    }

    @Override
    public List<IContainer> findAllContainerByPath(final String path) throws NoResultException {

        List<IContainer> list = new ArrayList<IContainer>();
        if (path != null) {
            List<IContainer> tmp = null;
            String location;

            // Store in a map all different containers from all m_groups
            Map<String, ArrayList<IContainer>> items = new HashMap<String, ArrayList<IContainer>>();
            List<String> sortGrp = new ArrayList<String>();

            String absPath = mDataset.getRootGroup().getLocation() + path;
            for (HdfGroup group : mGroups) {
                try {

                    tmp = group.findAllContainerByPath(NxsPath.splitStringToNode(absPath));
                    for (IContainer item : tmp) {
                        location = item.getLocation();
                        if (items.containsKey(location)) {
                            items.get(location).add(item);
                        } else {
                            ArrayList<IContainer> tmpList = new ArrayList<IContainer>();
                            tmpList.add(item);
                            items.put(location, tmpList);
                            sortGrp.add(location);
                        }
                    }
                } catch (NoResultException e) {
                    // Nothing to do
                }
            }
            // Construct that were found
            for (String entry : sortGrp) {

                // for( Entry<String, ArrayList<IContainer>> entry : items.entrySet() ) {
                tmp = items.get(entry);
                // If a Group list then construct a new Group folder
                if (tmp.get(0).getModelType() == ModelType.Group) {
                    HdfGroup hdfGroup = (HdfGroup) tmp.get(0);
                    if (hdfGroup.getAttribute(NX_CLASS).equals("NXtechnical_data")) {
                        // TODO Gestion des Technical Data à faire
                        System.out.println("Construire un DATAITEM !");
                    }
                    list.add(new NxsGroup(tmp.toArray(new HdfGroup[tmp.size()]), this, mDataset));
                }
                // If a IDataItem list then construct a new compound NxsDataItem
                else {
                    ArrayList<HdfDataItem> dataItems = new ArrayList<HdfDataItem>();
                    for (IContainer item : tmp) {
                        if (item.getModelType() == ModelType.DataItem) {
                            dataItems.add((HdfDataItem) item);
                        }
                    }
                    HdfDataItem[] array = new HdfDataItem[dataItems.size()];
                    dataItems.toArray(array);

                    list.add(new NxsDataItem(array, this, mDataset));
                }
            }
        }
        return list;
    }

    @Override
    public boolean removeDataItem(final IDataItem item) {
        return removeDataItem(item.getShortName());
    }

    @Override
    public boolean removeDataItem(final String varName) {
        boolean succeed = false;
        for (IGroup group : mGroups) {
            if (group.removeDataItem(varName)) {
                succeed = true;
            }
        }
        return succeed;
    }

    @Override
    public boolean removeGroup(final IGroup group) {
        return removeGroup(group.getShortName());
    }

    @Override
    public boolean removeGroup(final String shortName) {
        boolean succeed = false;
        for (IGroup group : mGroups) {
            if (group.removeGroup(shortName)) {
                succeed = true;
            }
        }
        return succeed;
    }

    @Deprecated
    @Override
    public void setDictionary(final IDictionary dictionary) {
        if (mGroups.length > 0) {
            mGroups[0].setDictionary(dictionary);
        }
    }

    @Deprecated
    @Override
    public IDictionary findDictionary() {
        IDictionary dictionary = null;
        if (mGroups.length > 0) {
            IFactory factory = NxsFactory.getInstance();
            dictionary = new NxsDictionary();
            try {
                dictionary.readEntries(Factory.getPathMappingDictionaryFolder(factory)
                        + NxsLogicalGroup.detectDictionaryFile((NxsDataset) getDataset()));
            } catch (FileAccessException e) {
                dictionary = null;
                Factory.getLogger().log(Level.SEVERE, e.getMessage());
            }
        }
        return dictionary;
    }

    @Override
    public boolean isRoot() {
        return (mGroups.length > 0 && mGroups[0].isRoot());
    }

    @Override
    public boolean isEntry() {
        boolean result = false;
        IGroup parent = getParentGroup();
        if (parent != null) {
            result = getParentGroup().isRoot();
        }
        return result;
    }

    @Override
    public List<IContainer> findAllContainers(final IKey key) throws NoResultException {
        return findAllOccurrences(key);
    }

    @Override
    public List<IContainer> findAllOccurrences(final IKey key) throws NoResultException {
        String pathStr = findDictionary().getPath(key).toString();
        Path path = new Path(NxsFactory.getInstance(), pathStr);
        return findAllContainerByPath(path.getValue());
    }

    @Override
    public IContainer findObjectByPath(final Path path) {
        IContainer result = null;

        try {
            result = findContainerByPath(path.getValue());
        } catch (NoResultException e) {
        }

        return result;
    }

    @Override
    public void addOneAttribute(final IAttribute attribute) {
        if (mGroups != null && mGroups.length > 0) {
            mGroups[0].addOneAttribute(attribute);
        }
    }

    @Override
    public void addStringAttribute(final String name, final String value) {
        if (mGroups != null && mGroups.length > 0 && mGroups[0] != null) {
            mGroups[0].addStringAttribute(name, value);
        }
    }

    @Override
    public boolean removeDimension(final IDimension dimension) {
        throw new NotImplementedException();
    }

    @Override
    public boolean removeDimension(final String dimName) {
        throw new NotImplementedException();
    }

    @Override
    public List<IDimension> getDimensionList() {
        listChildren();
        return mDimensions;
    }

    @Override
    public IDimension getDimension(final String name) {
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
    public void addDataItem(final IDataItem v) {
        if (v instanceof NxsDataItem) {
            NxsDataItem nxsDataItem = (NxsDataItem) v;
            HdfDataItem[] itemsToAdd = nxsDataItem.getHdfDataItems();
            for (HdfDataItem hdfDataItem : itemsToAdd) {
                mGroups[0].addDataItem(hdfDataItem);
            }
            v.setParent(this);
        }
    }

    @Override
    public boolean removeAttribute(final IAttribute attribute) {
        throw new NotImplementedException();
    }

    @Override
    public void addOneDimension(final IDimension dimension) {
        throw new NotImplementedException();
    }

    @Override
    public void addSubgroup(final IGroup group) {
        if (group instanceof NxsGroup) {
            NxsGroup nxsGroup = (NxsGroup) group;
            mGroups[0].addSubgroup(nxsGroup.getHdfGroup());
        }
        mChildren.add(group);
    }

    @Override
    public void updateDataItem(final String key, final IDataItem dataItem) throws SignalNotAvailableException {
        throw new NotImplementedException();
    }

    @Override
    public long getLastModificationDate() {
        return mDataset.getLastModificationDate();
    }

    // ------------------------------------------------------------------------
    // Protected methods
    // ------------------------------------------------------------------------
    protected void setChild(final IContainer node) {
        if (!mChildren.contains(node)) {
            mChildren.add(node);
        }
    }

    // ------------------------------------------------------------------------
    // Private methods
    // ------------------------------------------------------------------------
    private List<IContainer> listChildren() {
        List<IContainer> result;
        if (mIsMultigroup) {
            result = listChildrenMultiGroup();
        } else {
            result = listChildrenMonoGroup();
        }
        return result;
    }

    private List<IContainer> listChildrenMultiGroup() {
        if (!mIsChildUpdate) {
            List<IContainer> tmp = null;
            mChildren = new ArrayList<IContainer>();
            mDimensions = new ArrayList<IDimension>();

            // Store in a map all different containers from all m_groups
            String tmpName;
            Map<String, ArrayList<IContainer>> items = new HashMap<String, ArrayList<IContainer>>();
            for (IGroup group : mGroups) {
                tmp = new ArrayList<IContainer>();
                tmp.addAll(group.getDataItemList());
                tmp.addAll(group.getGroupList());
                for (IContainer item : tmp) {
                    tmpName = item.getShortName();
                    if (items.containsKey(tmpName)) {
                        items.get(tmpName).add(item);
                    } else {
                        ArrayList<IContainer> tmpList = new ArrayList<IContainer>();
                        tmpList.add(item);
                        items.put(tmpName, tmpList);
                    }
                }
            }

            // Construct what were found
            for (Entry<String, ArrayList<IContainer>> entry : items.entrySet()) {
                tmp = entry.getValue();
                // If a Group list then construct a new Group folder
                if (tmp.get(0).getModelType() == ModelType.Group) {
                    mChildren.add(new NxsGroup(tmp.toArray(new HdfGroup[tmp.size()]), this, mDataset));
                }
                // If a IDataItem list then construct a new compound NxsDataItem
                else {
                    ArrayList<HdfDataItem> nxsDataItems = new ArrayList<HdfDataItem>();
                    for (IContainer item : tmp) {
                        if (item.getModelType() == ModelType.DataItem) {
                            nxsDataItems.add((HdfDataItem) item);
                        }
                    }
                    HdfDataItem[] array = new HdfDataItem[nxsDataItems.size()];
                    nxsDataItems.toArray(array);
                    mChildren.add(new NxsDataItem(array, this, mDataset));
                }
            }
            mIsChildUpdate = true;
        }
        return mChildren;
    }

    private List<IContainer> listChildrenMonoGroup() {
        if (!mIsChildUpdate) {
            mChildren = new ArrayList<IContainer>();
            mDimensions = new ArrayList<IDimension>();

            // Store in a list all different containers from all m_groups
            for (IDataItem item : mGroups[0].getDataItemList()) {
                mChildren.add(new NxsDataItem((HdfDataItem) item, this, mDataset));
            }

            for (IGroup group : mGroups[0].getGroupList()) {
                // HACK GV mGroups[0]
                mChildren.add(new NxsGroup(new HdfGroup[] { (HdfGroup) group }, this, mDataset));
            }
            mIsChildUpdate = true;
        }
        return mChildren;
    }

    public List<IContainer> findAllContainerByNxsPath(final String path) throws NoResultException {
        List<IContainer> result = null;
        IGroup root = getRootGroup();

        // Try to list all nodes matching the path
        // Transform path into a NexusNode array
        NxsNode[] nodes = NxsPath.splitStringToNode(path);

        // Call recursive method
        int level = 0;
        result = findAllContainer(root, nodes, level);

        return result;
    }

    private List<IContainer> findAllContainer(final IContainer container, final NxsNode[] nodes, final int level) {
        List<IContainer> result = new ArrayList<IContainer>();
        if (container != null) {
            if (container instanceof NxsGroup) {
                NxsGroup group = (NxsGroup) container;
                if (nodes.length > level) {
                    // List current node children
                    List<IContainer> childs;

                    NxsNode current = nodes[level];
                    childs = group.listContainer(current);

                    for (IContainer node : childs) {
                        if (level < nodes.length - 1) {
                            result.addAll(findAllContainer(group.getContainer(node.getName()), nodes, level + 1));
                        }
                        // Create IContainer and add it to result list
                        else {
                            result.add(group.getContainer(node.getName()));
                        }
                    }
                }
            } else {
                result.add(container);
            }
        }
        return result;
    }

    // ****************************************************
    // Specific methods
    // ****************************************************
    public NxsPath getNxsPath() {
        NxsPath result = new NxsPath(mGroups[0].getHdfPath());
        return result;
    }

    @Override
    public String toString() {
        StringBuffer res = new StringBuffer("");

        for (IGroup group : mGroups) {
            res.append(group.getLocation() + "\n");
        }
        return res.toString();
    }

    public List<IContainer> listContainer(final NxsNode node) {
        List<IContainer> result = new ArrayList<IContainer>();
        if (node != null) {
            for (IGroup group : getGroupList()) {
                NxsNode nxNode = ((NxsGroup) group).getNxsNode();
                if (nxNode != null && nxNode.matchesPartNode(node)) {
                    result.add(group);
                }
            }
        }
        return result;
    }

    private NxsNode getNxsNode() {
        if (mNode == null) {
            String clazz = "";
            IAttribute attribute = getAttribute(NX_CLASS);
            if (attribute != null) {
                clazz = attribute.getStringValue();
            }
            mNode = new NxsNode(getShortName(), clazz);
        }
        return mNode;
    }

    public HdfGroup getHdfGroup() {
        return mGroups[0];
    }
}
