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
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.engine.nexus.navigation;

// Standard import
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.Path;
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
import org.cdma.utils.Utilities.ModelType;
import org.nexusformat.NexusException;

import fr.soleil.nexus.Attribute;
import fr.soleil.nexus.DataItem;
import fr.soleil.nexus.NexusFileWriter;
import fr.soleil.nexus.NexusNode;
import fr.soleil.nexus.PathData;
import fr.soleil.nexus.PathGroup;
import fr.soleil.nexus.PathNexus;

public final class NexusGroup implements IGroup, Cloneable {
    public long MINIMUM_REFRESH_TIME = 1000;

    // / Members
    private final String mFactory; // Name of the factory plugin that instantiate

    // API CDMA tree need
    private final NexusDataset mDataset; // File handler
    private IGroup mParent = null; // Parent group
    private final List<IContainer> mChild; // Children nodes (group, dataitem...)

    // Internal members
    private PathNexus mN4TCurPath; // Current path
    @Deprecated
    private IDictionary mDictionary; // Group dictionary
    private final List<IAttribute> mAttributes; // Attributes belonging to this
    private final List<IDimension> mDimensions; // Dimensions direct child of this
    private final boolean readAttributes; // have attributes been read
    private long mLastLoadChild;

    // / Constructors
    public NexusGroup(String factoryName, IGroup parent, PathNexus from, NexusDataset dataset) {
        mFactory = factoryName;
        mDictionary = null;
        mN4TCurPath = from;
        mDataset = dataset;
        mChild = new ArrayList<IContainer>();
        mAttributes = new ArrayList<IAttribute>();
        mDimensions = new ArrayList<IDimension>();
        readAttributes = false;
        mLastLoadChild = 0L;
    }

    public NexusGroup(String factoryName, PathNexus from, NexusDataset dataset) {
        mFactory = factoryName;
        mDictionary = null;
        mN4TCurPath = from;
        mDataset = dataset;
        mChild = new ArrayList<IContainer>();
        mAttributes = new ArrayList<IAttribute>();
        mDimensions = new ArrayList<IDimension>();
        readAttributes = false;
        mLastLoadChild = 0L;
        if (from != null && dataset != null) {
            createFamilyTree();
        }
    }

    public NexusGroup(NexusGroup group) {
        mFactory = group.mFactory;
        mN4TCurPath = group.mN4TCurPath.clone();
        mDataset = group.mDataset;
        mParent = group.mParent;
        mChild = new ArrayList<IContainer>(group.mChild);
        mAttributes = new ArrayList<IAttribute>(group.mAttributes);
        mDimensions = new ArrayList<IDimension>(group.mDimensions);
        readAttributes = group.readAttributes;
        mLastLoadChild = group.mLastLoadChild;
        try {
            if (group.mDictionary != null) {
                mDictionary = group.mDictionary.clone();
            }
        } catch (CloneNotSupportedException e) {
            mDictionary = null;
        }
    }

    /**
     * Return a clone of this IGroup object.
     * 
     * @return new IGroup
     */
    @Override
    public NexusGroup clone() {
        return new NexusGroup(this);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.Group;
    }

    @Override
    public boolean isEntry() {
        return (mParent.getParentGroup().getParentGroup() == null);
    }

    @Override
    public void addDataItem(IDataItem v) {
        v.setParent(this);
        setChild(v);
    }

    @Override
    public void addOneAttribute(IAttribute attribute) {
        mAttributes.add(attribute);
    }

    @Override
    public void addOneDimension(IDimension dimension) {
        mDimensions.add(dimension);
    }

    @Override
    public void addStringAttribute(String name, String value) {
        IAttribute attr = new NexusAttribute(mFactory, name, value);
        mAttributes.add(attr);
    }

    @Override
    public void addSubgroup(IGroup g) {
        g.setParent(this);
    }

    @Override
    public IAttribute getAttribute(String name) {
        if (!readAttributes) {
            readAttributes();
        }

        for (IAttribute attr : mAttributes) {
            if (attr.getName().equals(name)) {
                return attr;
            }
        }

        return null;
    }

    @Override
    public IDataItem findDataItem(String keyName) {
        IKey key = Factory.getFactory(mFactory).createKey(keyName);

        return findDataItem(key);
    }

    @Override
    public IDimension getDimension(String name) {
        IDimension result = null;
        for (IDimension dim : mDimensions) {
            if (dim.getName().equals(name)) {
                result = dim;
                break;
            }
        }

        return result;
    }

    @Override
    public IGroup findGroup(String keyName) {
        IKey key = Factory.getFactory(mFactory).createKey(keyName);

        return findGroup(key);
    }

    @Override
    public IGroup getGroupWithAttribute(String attributeName, String value) {
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
    public IGroup findGroupWithAttribute(IKey key, String name, String value) {
        List<IContainer> found = new ArrayList<IContainer>();

        try {
            found = findAllOccurrences(key);
        } catch (NoResultException e) {
        }

        IGroup result = null;
        for (IContainer item : found) {
            if (item.getModelType().equals(ModelType.Group) && item.hasAttribute(name, value)) {
                result = (IGroup) item;
                break;
            }
        }
        return result;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        List<IAttribute> attributes = new ArrayList<IAttribute>();

        if (!readAttributes) {
            readAttributes();
        }

        for (IAttribute attr : mAttributes) {
            attributes.add(attr);
        }

        return attributes;
    }

    @Override
    public IDataItem getDataItemWithAttribute(String name, String value) {
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
    public IDataItem findDataItemWithAttribute(IKey key, String name, String value) throws NoResultException {
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
    public IDataItem getDataItem(String shortName) {
        List<IDataItem> items = getDataItemList();
        for (IDataItem item : items) {
            if (item.getShortName().equals(shortName)) {
                return item;
            }
        }

        return null;
    }

    @Override
    public IDataItem findDataItem(IKey key) {
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
    public List<IDataItem> getDataItemList() {
        initChildren();

        List<IDataItem> result = new ArrayList<IDataItem>();
        for (IContainer variable : mChild) {
            if (variable.getModelType().equals(ModelType.DataItem)) {
                result.add((IDataItem) variable);
            }
        }
        return result;
    }

    @Override
    public IDataset getDataset() {
        return mDataset;
    }

    @Override
    public List<IDimension> getDimensionList() {
        return mDimensions;
    }

    @Override
    public IGroup findGroup(IKey key) {
        IGroup group = null;
        List<IContainer> list = new ArrayList<IContainer>();
        try {
            list = findAllOccurrences(key);
        } catch (NoResultException e) {
            Factory.getLogger().log(Level.WARNING, e.getMessage());
        }

        for (IContainer object : list) {
            if (object.getModelType().equals(ModelType.Group)) {
                group = (IGroup) object;
                break;
            }
        }

        return group;
    }

    @Override
    public IGroup getGroup(String shortName) {
        NexusNode node = PathNexus.splitStringToNode(shortName)[0];
        List<IGroup> groups = getGroupList();
        for (IGroup group : groups) {
            if (group.getShortName().equals(shortName)
                    || (node.getNodeName().isEmpty() && node.getClassName().equals(((NexusGroup) group).getClassName()))) {
                return group;
            }
        }

        return null;
    }

    @Override
    public List<IGroup> getGroupList() {
        initChildren();

        List<IGroup> result = new ArrayList<IGroup>();
        for (IContainer variable : mChild) {
            if (variable.getModelType().equals(ModelType.Group)) {
                result.add((IGroup) variable);
            }
        }
        return result;
    }

    @Override
    public String getLocation() {
        return mN4TCurPath.toString(false);
    }

    @Override
    public String getName() {
        return (isRoot() ? "" : mParent.getName()) + "/" + getShortName();
    }

    @Override
    public IContainer getContainer(String shortName) {
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
    public IContainer findContainerByPath(String path) throws NoResultException {
        // !!!!
        // Split path into nodes
        String[] sNodes = PathNexus.splitStringPath(path);
        IContainer node = mDataset.getRootGroup();

        // Try to open each node
        for (String shortName : sNodes) {
            if (!shortName.isEmpty() && node != null && node instanceof IGroup) {
                node = ((IGroup) node).getContainer(shortName);
            }
        }

        return node;
    }

    @Override
    public IGroup getParentGroup() {
        if (mParent == null) {
            PathNexus parentPath = mN4TCurPath.getParentPath();
            if (parentPath != null) {
                mParent = new NexusGroup(mFactory, parentPath, mDataset);
                return mParent;
            } else {
                return null;
            }
        } else {
            return mParent;
        }
    }

    @Override
    public IGroup getRootGroup() {
        return mDataset.getRootGroup();
    }

    @Override
    public String getShortName() {
        NexusNode nnNode = mN4TCurPath.getCurrentNode();
        if (nnNode != null) {
            return nnNode.getNodeName();
        } else {
            return "";
        }
    }

    private String getClassName() {
        NexusNode nnNode = mN4TCurPath.getCurrentNode();
        if (nnNode != null) {
            return nnNode.getClassName();
        } else {
            return "";
        }
    }

    @Override
    public Map<String, String> harvestMetadata(String md_standard) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public boolean hasAttribute(String name, String value) {
        IAttribute attr;
        List<IAttribute> listAttr = getAttributeList();

        Iterator<IAttribute> iter = listAttr.iterator();
        while (iter.hasNext()) {
            attr = iter.next();
            if (attr.getStringValue().equals(value)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean isRoot() {
        return mParent == null;
    }

    @Override
    public boolean removeAttribute(IAttribute attribute) {
        return mAttributes.remove(attribute);
    }

    @Override
    public boolean removeDataItem(IDataItem item) {
        return mChild.remove(item);
    }

    @Override
    public boolean removeDataItem(String varName) {
        IDataItem item = getDataItem(varName);
        if (item == null) {
            return false;
        }

        return mChild.remove(item);
    }

    @Override
    public boolean removeDimension(String dimName) {
        IDimension dimension = getDimension(dimName);
        if (dimension == null) {
            return false;
        }

        return mChild.remove(dimension);
    }

    @Override
    public boolean removeDimension(IDimension dimension) {
        return mChild.remove(dimension);
    }

    @Override
    public boolean removeGroup(IGroup group) {
        return mChild.remove(group);
    }

    @Override
    public boolean removeGroup(String shortName) {
        IGroup group = getGroup(shortName);
        if (group == null) {
            return false;
        }

        return mChild.remove(group);
    }

    @Override
    public void setName(String name) {
        addStringAttribute("long_name", name);
    }

    @Override
    public void setShortName(String name) {
        NexusNode nnNode = mN4TCurPath.popNode();
        if (nnNode != null) {
            nnNode.setNodeName(name);
            mN4TCurPath.pushNode(nnNode);
        }
    }

    @Deprecated
    @Override
    public void setDictionary(IDictionary dictionary) {
        mDictionary = dictionary;
    }

    @Override
    public void setParent(IGroup group) {
        mParent = group;
        if (group != null) {
            ((NexusGroup) group).setChild(this);
        }
    }

    @Override
    public List<IContainer> findAllContainers(IKey key) throws NoResultException {
        return findAllOccurrences(key);
    }

    @Override
    public List<IContainer> findAllOccurrences(IKey key) throws NoResultException {
        String path = mDictionary.getPath(key).toString();
        return findAllContainerByPath(path);
    }

    @Override
    public List<IContainer> findAllContainerByPath(String path) throws NoResultException {
        List<IContainer> list = new ArrayList<IContainer>();
        NexusFileWriter handler = mDataset.getHandler();

        // Try to list all nodes matching the path
        try {
            // Transform path into a NexusNode array
            NexusNode[] nodes = PathNexus.splitStringToNode(path);

            // Open path from root
            handler.open();

            // Call recursive method
            int level = 0;
            list = findAllContainer(handler, nodes, level);

            // Close the file
            handler.close();
        } catch (NexusException ne) {
            list.clear();
            handler.close();
            throw new NoResultException("Requested path doesn't exist!");
        }
        return list;
    }

    private List<IContainer> findAllContainer(NexusFileWriter handler, NexusNode[] nodes, int level)
            throws NexusException {
        List<IContainer> list = new ArrayList<IContainer>();

        if (nodes.length > level) {
            // List current node children
            List<NexusNode> child = handler.listChildren();

            NexusNode current = nodes[level];
            IContainer item;
            for (NexusNode node : child) {
                if (node.matchesPartNode(current)) {
                    // Open the node
                    handler.openNode(node);

                    // Recursive call
                    if (level < nodes.length - 1) {
                        list.addAll(findAllContainer(handler, nodes, level + 1));
                    }
                    // Create IContainer and add it to result list
                    else {
                        if (handler.getCurrentPath().getCurrentNode().isGroup()) {
                            item = new NexusGroup(mFactory, this, handler.getCurrentPath().clone(), mDataset);
                        } else {
                            item = new NexusDataItem(mFactory, handler.readDataInfo(), this, mDataset);
                        }
                        list.add(item);
                    }

                    // Close node
                    handler.closeData();
                }
            }
        }
        return list;
    }

    @Deprecated
    @Override
    public IDictionary findDictionary() {
        return null;
    }

    @Override
    public IContainer findContainer(String shortName) {
        IKey key = Factory.getFactory(mFactory).createKey(shortName);
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
    public void updateDataItem(String key, IDataItem dataItem) throws SignalNotAvailableException {
        throw new NotImplementedException();
    }

    /**
     * Return the internal PathNexus of the group
     */
    public PathNexus getPathNexus() {
        return mN4TCurPath;
    }

    // ---------------------------------------------------------
    // / protected methods
    // ---------------------------------------------------------
    /**
     * Set the internal PathNexus of the group
     */
    protected void setPath(PathNexus path) {
        mN4TCurPath = path;
    }

    protected void setChild(IContainer node) {
        if (!mChild.contains(node)) {
            mChild.add(node);
        }
    }

    /**
     * Recursive method filling the 'items' of all descendant IDataItem found
     * under the given IGroup 'group'
     * 
     * @param items
     *            output list of found IDataItem
     * @param group
     *            IGroup to start search in
     */
    static protected List<IDataItem> getDescendentDataItem(List<IDataItem> items, IGroup group) {
        List<IDataItem> list = group.getDataItemList();
        List<IGroup> gpList = group.getGroupList();

        for (IDataItem item : list) {
            items.add(item);
        }

        for (IGroup grp : gpList) {
            NexusGroup.getDescendentDataItem(items, grp);
        }

        return items;
    }

    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private void initChildren() {
        if (System.currentTimeMillis() - mLastLoadChild > MINIMUM_REFRESH_TIME) {
            long lastMod = mDataset.getLastModificationDate();
            if (lastMod > mLastLoadChild) {
                mLastLoadChild = mDataset.getLastModificationDate();
                NexusNode[] nexusNodes = new NexusNode[] {};
                NexusFileWriter handler = mDataset.getHandler();
                try {
                    handler.open();
                    nexusNodes = handler.listChildren(mN4TCurPath);
                } catch (NexusException ne) {
                    try {
                        if (handler.isFileOpened()) {
                            handler.closeAll();
                        }
                    } catch (NexusException e) {
                    }
                    handler.close();
                    return;
                }

                IContainer item;
                DataItem dataInfo;
                PathNexus path;
                for (int i = 0; i < nexusNodes.length; i++) {
                    try {
                        path = mN4TCurPath.clone();
                        path.pushNode(nexusNodes[i]);
                        if (nexusNodes[i].isGroup()) {
                            item = new NexusGroup(mFactory, PathGroup.Convert(path), mDataset);
                        } else {
                            if (nexusNodes[i].getClassName().equals("NXtechnical_data")) {
                                dataInfo = handler.readData(PathData.Convert(path));
                            } else {
                                handler.openPath(path);
                                dataInfo = handler.readDataInfo();
                            }
                            item = new NexusDataItem(mFactory, dataInfo, this, mDataset);
                        }
                        if (!mChild.contains(item)) {
                            mChild.add(item);
                        }
                    } catch (NexusException e) {
                    }
                }
                handler.close();
            }
        }
    }

    private void createFamilyTree() {
        if (mDataset != null && mN4TCurPath != null) {
            NexusNode[] nodes = mN4TCurPath.getParentPath().getNodes();
            NexusGroup ancestor = (NexusGroup) mDataset.getRootGroup();
            PathNexus path = PathNexus.ROOT_PATH.clone();
            NexusGroup group;

            for (NexusNode node : nodes) {
                group = (NexusGroup) ancestor.getGroup(node.getNodeName());
                if (group == null) {
                    path.pushNode(node);
                    group = new NexusGroup(mFactory, ancestor, path.clone(), mDataset);
                }
                ancestor = group;
            }

            setParent(ancestor);
        }
    }

    @Override
    public String toString() {
        return mN4TCurPath.toString();
    }

    @Override
    public String getFactoryName() {
        return mFactory;
    }

    @Override
    public IContainer findObjectByPath(Path path) {
        IContainer result = null;

        try {
            result = findContainerByPath(path.getValue());
        } catch (NoResultException e) {
            Factory.getLogger().log(Level.WARNING, e.getMessage());
        }

        return result;
    }

    private void readAttributes() {
        NexusAttribute tmpAttr;
        NexusFileWriter handler = mDataset.getHandler();
        try {
            handler.open();
            handler.openPath(mN4TCurPath);
            Collection<Attribute> attributes = handler.listAttribute();

            for (Attribute attribute : attributes) {
                try {
                    tmpAttr = new NexusAttribute(mFactory, attribute.name, handler.readAttr(attribute.name, null));
                    mAttributes.add(tmpAttr);
                } catch (NexusException e) {
                    Factory.getLogger().log(Level.WARNING, e.getMessage());
                }
            }
            handler.close();
        } catch (NexusException ne) {
            handler.close();
        }
    }

    @Override
    public long getLastModificationDate() {
        return mDataset.getLastModificationDate();
    }
}
