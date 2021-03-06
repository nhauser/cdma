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
 * St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.hdf.navigation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.swing.tree.DefaultMutableTreeNode;

import ncsa.hdf.hdf5lib.exceptions.HDF5Exception;
import ncsa.hdf.object.Attribute;
import ncsa.hdf.object.FileFormat;
import ncsa.hdf.object.Group;
import ncsa.hdf.object.HObject;
import ncsa.hdf.object.Metadata;
import ncsa.hdf.object.h5.H5Group;
import ncsa.hdf.object.h5.H5ScalarDS;

import org.cdma.Factory;
import org.cdma.dictionary.Path;
import org.cdma.engine.hdf.utils.HdfNode;
import org.cdma.engine.hdf.utils.HdfPath;
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
import org.cdma.utils.Utilities.ModelType;

public class HdfGroup implements IGroup, Cloneable {

    private HdfGroup parent;
    private IGroup root = null;
    private final String factoryName; // Name of the factory plugin that instantiate
    private final Map<String, IGroup> groupMap = new HashMap<String, IGroup>();
    private final Map<String, IDataItem> itemMap = new HashMap<String, IDataItem>();
    private final Map<String, IAttribute> attributeMap = new HashMap<String, IAttribute>();
    private String originalName;
    private String shortName;
    private String name;
    private HdfDataset dataset;

    public HdfGroup(final String factoryName, final String name, final String path, final HdfGroup parent,
            final HdfDataset dataset) {
        this(factoryName, new H5Group(dataset.getH5File(), name, path, null), parent, dataset);
    }

    public HdfGroup(final String factoryName, final H5Group hdfGroup, final HdfGroup parent, final IDataset dataset) {
        this.factoryName = factoryName;
        this.dataset = (HdfDataset) dataset;
        this.parent = parent;
        init(hdfGroup);
    }

    private void init(final H5Group h5Group) {
        shortName = h5Group.getName();
        originalName = getName();

        List<HObject> members = h5Group.getMemberList();
        if (members != null) {
            for (HObject hObject : members) {
                if (hObject instanceof H5ScalarDS) {
                    H5ScalarDS scalarDS = (H5ScalarDS) hObject;
                    HdfDataItem dataItem = new HdfDataItem(factoryName, dataset.getH5File(), this, scalarDS);
                    itemMap.put(scalarDS.getName(), dataItem);
                }
                if (hObject instanceof H5Group) {
                    H5Group group = (H5Group) hObject;
                    HdfGroup hdfGroup = new HdfGroup(factoryName, group, this, dataset);
                    groupMap.put(group.getName(), hdfGroup);
                }
            }

            try {
                @SuppressWarnings("unchecked")
                List<Metadata> metadatas = h5Group.getMetadata();

                for (Metadata metadata : metadatas) {
                    if (metadata instanceof Attribute) {
                        Attribute attribute = (Attribute) metadata;
                        HdfAttribute hdfAttr = new HdfAttribute(factoryName, attribute);
                        attributeMap.put(hdfAttr.getName(), hdfAttr);
                    }
                }
            } catch (HDF5Exception e) {
                Factory.getLogger().severe(e.getMessage());
            }
        }
    }

    @Override
    public HdfGroup clone() {
        return new HdfGroup(factoryName, null, parent, dataset);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.Group;
    }

    @Override
    public void addStringAttribute(final String name, final String value) {
        HdfAttribute attribute = new HdfAttribute(factoryName, name, value);
        attributeMap.put(name, attribute);
    }

    @Override
    public void addOneAttribute(final IAttribute attribute) {
        attributeMap.put(attribute.getName(), attribute);
    }

    @Override
    public IAttribute getAttribute(final String name) {
        IAttribute result = attributeMap.get(name);
        return result;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        List<IAttribute> result = new ArrayList<IAttribute>(attributeMap.values());
        return result;
    }

    @Override
    public String getLocation() {
        return getName();
    }

    @Override
    public String getName() {
        if (name == null) {
            if (parent == null) {
                name = HdfPath.PATH_SEPARATOR;
            } else {
                name = (parent.isRoot()) ? HdfPath.PATH_SEPARATOR + shortName : parent.getName()
                        + HdfPath.PATH_SEPARATOR + shortName;
            }
        }
        return name;
    }

    @Override
    public String getShortName() {
        return this.shortName;
    }

    @Override
    public boolean hasAttribute(final String name, final String value) {
        boolean result = attributeMap.containsKey(name);
        return result;
    }

    @Override
    public boolean removeAttribute(final IAttribute a) {
        boolean result = true;
        attributeMap.remove(a.getName());
        return result;
    }

    @Override
    public void setName(final String name) {
        addStringAttribute("long_name", name);
    }

    @Override
    public void setShortName(final String name) {
        try {
            this.shortName = name;
            this.name = getParentGroup().getName() + HdfPath.PATH_SEPARATOR + name;
            for (IDataItem item : itemMap.values()) {
                item.setParent(this);
            }

        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public void setParent(final IGroup group) {
        try {
            parent = (HdfGroup) group;
        } catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public long getLastModificationDate() {
        long result = 0;
        getDataset().getLastModificationDate();
        return result;
    }

    @Override
    public String getFactoryName() {
        return factoryName;
    }

    @Override
    public void addDataItem(final IDataItem item) {
        item.setParent(this);
        itemMap.put(item.getName(), item);
    }

    @Override
    public Map<String, String> harvestMetadata(final String mdStandard) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup getParentGroup() {
        IGroup result = this.parent;
        return result;
    }

    @Override
    public IGroup getRootGroup() {
        if (root == null) {
            root = getDataset().getRootGroup();
        }
        return root;
    }

    @Override
    public void addOneDimension(final IDimension dimension) {
    }

    @Override
    public void addSubgroup(final IGroup group) {
        group.setParent(this);
        groupMap.put(group.getShortName(), group);
    }

    @Override
    public IDataItem getDataItem(final String shortName) {
        IDataItem result = null;
        if (shortName != null) {
            result = itemMap.get(shortName);
        }

        return result;
    }

    @Override
    public IDataItem getDataItemWithAttribute(final String name, final String value) {
        IDataItem resItem = null;
        List<IDataItem> groups = getDataItemList();
        for (Iterator<?> iter = groups.iterator(); iter.hasNext();) {
            resItem = (IDataItem) iter.next();
            if (resItem.hasAttribute(name, value)) {
                groups.clear();
                return resItem;
            }
        }

        return null;
    }

    @Override
    public IDimension getDimension(final String name) {
        IDimension result = null;
        return result;
    }

    @Override
    public IContainer getContainer(final String shortName) {
        if (shortName != null && shortName.equals("")) {
            return this;
        }

        IGroup resultGroupItem = getGroup(shortName);
        if (resultGroupItem != null) {
            return resultGroupItem;
        }
        IDataItem resultVariableItem = getDataItem(shortName);
        if (resultVariableItem != null) {
            return resultVariableItem;
        }

        return null;
    }

    @Override
    public IGroup getGroup(final String shortName) {
        IGroup result = null;
        if (shortName != null) {
            result = groupMap.get(shortName);
        }
        return result;
    }

    @Override
    public IGroup getGroupWithAttribute(final String attributeName, final String value) {
        List<IGroup> groups = getGroupList();
        IAttribute attr;
        for (IGroup group : groups) {
            attr = group.getAttribute(attributeName);
            if (attr.getStringValue().equals(value)) {
                return group;
            }
        }

        return null;
    }

    @Override
    public List<IDataItem> getDataItemList() {
        List<IDataItem> result;
        result = new ArrayList<IDataItem>(itemMap.values());
        return result;
    }

    @Override
    public IDataset getDataset() {
        if (dataset == null) {
            if (parent != null) {
                dataset = (HdfDataset) parent.getDataset();
            }
        }
        return dataset;
    }

    @Override
    public List<IDimension> getDimensionList() {
        List<IDimension> result = new ArrayList<IDimension>();
        return result;
    }

    @Override
    public IGroup findGroup(final String shortName) {
        IGroup result = null;
        result = getGroup(shortName);
        return result;
    }

    @Override
    public List<IGroup> getGroupList() {
        List<IGroup> result;
        result = new ArrayList<IGroup>(groupMap.values());
        return result;
    }

    public HdfPath getHdfPath() {
        HdfPath result;
        List<HdfNode> parentNodes = getParentNodes();
        result = new HdfPath(parentNodes.toArray(new HdfNode[parentNodes.size()]));
        return result;
    }

    private List<HdfNode> getParentNodes() {
        List<HdfNode> nodes = new ArrayList<HdfNode>();

        HdfNode node = new HdfNode(this);

        if (parent != null) {
            nodes.addAll(parent.getParentNodes());
            nodes.add(node);
        }

        return nodes;
    }

    private List<INode> getChildNodes() {
        List<INode> nodes = new ArrayList<INode>();

        for (IDataItem item : itemMap.values()) {
            nodes.add(new HdfNode(item));
        }
        for (IGroup item : groupMap.values()) {
            nodes.add(new HdfNode(item));
        }

        return nodes;
    }

    @Override
    public IContainer findContainerByPath(final String path) throws NoResultException {
        // Split path into nodes
        String[] sNodes = HdfPath.splitStringPath(path);
        IContainer node = getRootGroup();

        // Try to open each node
        for (String shortName : sNodes) {
            if (!shortName.isEmpty() && node != null && node instanceof IGroup) {
                node = ((IGroup) node).getContainer(shortName);
            }
        }

        return node;
    }

    public List<IContainer> findAllContainerByPath(final INode[] nodes) throws NoResultException {
        List<IContainer> list = new ArrayList<IContainer>();
        IGroup root = getRootGroup();

        // Call recursive method
        int level = 0;
        list = findAllContainer(root, nodes, level);

        return list;
    }

    @Override
    public List<IContainer> findAllContainerByPath(final String path) throws NoResultException {
        // Try to list all nodes matching the path
        // Transform path into a NexusNode array
        INode[] nodes = HdfPath.splitStringToNode(path);

        List<IContainer> result = findAllContainerByPath(nodes);

        return result;
    }

    private List<IContainer> findAllContainer(final IContainer container, final INode[] nodes, final int level) {
        List<IContainer> result = new ArrayList<IContainer>();
        if (container != null) {
            if (container instanceof HdfGroup) {
                HdfGroup group = (HdfGroup) container;
                if (nodes.length > level) {
                    // List current node children
                    List<INode> childs = group.getChildNodes();

                    INode current = nodes[level];

                    for (INode node : childs) {

                        if (node.matchesPartNode(current)) {
                            if (level < nodes.length - 1) {
                                result.addAll(findAllContainer(group.getContainer(node.getName()), nodes, level + 1));
                            }
                            // Create IContainer and add it to result list
                            else {
                                result.add(group.getContainer(node.getName()));
                            }
                        }
                    }
                }
            } else {
                HdfDataItem dataItem = (HdfDataItem) container;
                result.add(dataItem);
            }
        }
        return result;
    }

    @Override
    public boolean removeDataItem(final IDataItem item) {
        return removeDataItem(item.getShortName());

    }

    @Override
    public boolean removeDataItem(final String varName) {
        boolean result = true;
        itemMap.remove(varName);
        return result;
    }

    @Override
    public boolean removeDimension(final String name) {
        return false;
    }

    @Override
    public boolean removeDimension(final IDimension dimension) {
        return false;
    }

    @Override
    public boolean removeGroup(final IGroup group) {
        return removeGroup(group.getShortName());
    }

    @Override
    public boolean removeGroup(final String name) {
        groupMap.remove(name);
        return true;
    }

    @Override
    public void updateDataItem(final String key, final IDataItem dataItem) throws SignalNotAvailableException {
        throw new NotImplementedException();
    }

    @Override
    public boolean isRoot() {
        return parent == null;
    }

    @Override
    public boolean isEntry() {
        boolean result = false;
        result = getParentGroup().isRoot();
        return result;
    }

    @Override
    @Deprecated
    public void setDictionary(final IDictionary dictionary) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IDictionary findDictionary() {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public List<IContainer> findAllContainers(final IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public List<IContainer> findAllOccurrences(final IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public IContainer findObjectByPath(final Path path) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IContainer findContainer(final String shortName) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IGroup findGroup(final IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem findDataItemWithAttribute(final IKey key, final String name, final String value)
            throws NoResultException {
        List<IContainer> found = findAllOccurrences(key);
        IDataItem result = null;
        for (IContainer item : found) {
            if (item.getModelType().equals(ModelType.DataItem) && item.hasAttribute(name, value)) {
                result = (IDataItem) item;
                break;
            }

        }
        return result;
    }

    @Override
    @Deprecated
    public IGroup findGroupWithAttribute(final IKey key, final String name, final String value) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IDataItem findDataItem(final String shortName) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IDataItem findDataItem(final IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        IGroup parent = getParentGroup();
        buffer.append("Group " + getShortName() + " (parent = " + ((parent == null) ? "Nobody" : parent.getName())
                + ")");
        return buffer.toString();
    }

    public void save(final FileFormat fileToWrite, final Group parent) throws Exception {
        Group theGroup = null;

        boolean copyToNewFile = !(fileToWrite.getAbsolutePath().equals(dataset.getH5File().getAbsolutePath()));
        boolean isRoot = isRoot();
        if (!isRoot) {

            // New file or new group
            boolean isNew = fileToWrite.get(originalName) == null;

            if (isNew || copyToNewFile) {
                theGroup = fileToWrite.createGroup(getShortName(), parent);
            }
            // Group has been renamed
            else if (this.originalName != null && !this.originalName.equals(name)) {
                theGroup = (Group) fileToWrite.get(originalName);
                theGroup.setName(shortName);
                this.originalName = this.name;
            } else {
                theGroup = (Group) fileToWrite.get(name);
            }

        } else {
            DefaultMutableTreeNode theRoot = (DefaultMutableTreeNode) fileToWrite.getRootNode();
            H5Group rootObject = (H5Group) theRoot.getUserObject();
            theGroup = rootObject;
        }

        List<IAttribute> attribute = getAttributeList();
        for (IAttribute iAttribute : attribute) {
            HdfAttribute attr = (HdfAttribute) iAttribute;
            attr.save(theGroup, copyToNewFile);
        }

        List<IDataItem> dataItems = getDataItemList();
        for (IDataItem dataItem : dataItems) {
            HdfDataItem hdfDataItem = (HdfDataItem) dataItem;
            hdfDataItem.save(fileToWrite, theGroup);
        }

        List<IGroup> groups = getGroupList();
        for (IGroup iGroup : groups) {
            HdfGroup hdfGroup = (HdfGroup) iGroup;
            hdfGroup.save(fileToWrite, theGroup);
        }
    }
}
