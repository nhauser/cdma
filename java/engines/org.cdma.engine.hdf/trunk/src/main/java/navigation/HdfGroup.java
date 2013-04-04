package navigation;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.swing.tree.DefaultMutableTreeNode;

import ncsa.hdf.hdf5lib.H5;
import ncsa.hdf.object.Attribute;
import ncsa.hdf.object.FileFormat;
import ncsa.hdf.object.Group;
import ncsa.hdf.object.HObject;
import ncsa.hdf.object.h5.H5File;
import ncsa.hdf.object.h5.H5Group;
import ncsa.hdf.object.h5.H5ScalarDS;

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
import org.cdma.interfaces.INode;
import org.cdma.utils.Utilities.ModelType;

import utils.HdfNode;
import utils.HdfObjectUtils;
import utils.HdfPath;

public class HdfGroup implements IGroup, Cloneable {

    // private class ObjectContainer<T> {
    // SoftReference<T> softRef;
    // T hardRef;
    //
    // public ObjectContainer(SoftReference<T> container) {
    // softRef = container;
    // }
    //
    // public ObjectContainer(T container) {
    // hardRef = container;
    // }
    //
    // public T get() {
    // T result = null;
    // if (hardRef != null) {
    // result = hardRef;
    // }
    // else if (softRef != null) {
    // result = softRef.get();
    // }
    // return result;
    // }
    // }

    private final HdfGroup parent;
    private IGroup root = null;
    private final String factoryName; // Name of the factory plugin that instantiate
    private final H5Group h5Group;
    private final H5File h5File;
    private final Map<String, IGroup> groupMap = new HashMap<String, IGroup>();
    private final Map<String, IDataItem> itemMap = new HashMap<String, IDataItem>();

    public HdfGroup(String factoryName, H5Group hdfGroup, HdfGroup parent) {
        this.h5Group = hdfGroup;
        this.factoryName = factoryName;
        this.h5File = (H5File) hdfGroup.getFileFormat();
        this.parent = parent;
    }

    public HdfGroup(String factoryName, Group hdfGroup, HdfGroup parent) {
        this(factoryName, (H5Group) hdfGroup, parent);
    }

    @Override
    public HdfGroup clone() {
        return new HdfGroup(factoryName, h5Group, parent);
    }

    @Override
    public ModelType getModelType() {
        return ModelType.Group;
    }

    public H5Group getH5Group() {
        return this.h5Group;
    }

    public H5File getH5File() {
        return this.h5File;
    }

    @Override
    public void addStringAttribute(String name, String value) {
        HdfObjectUtils.addStringAttribute(h5Group, name, value);
    }

    @Override
    public void addOneAttribute(IAttribute attribute) {
        HdfObjectUtils.addOneAttribute(h5Group, attribute);
    }

    @Override
    public IAttribute getAttribute(String name) {
        Attribute attr = HdfObjectUtils.getAttribute(h5Group, name);
        IAttribute result = new HdfAttribute(factoryName, attr);
        return result;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        List<IAttribute> result = HdfObjectUtils.getAttributeList(factoryName, h5Group);
        return result;
    }

    @Override
    public String getLocation() {
        return getName();
    }

    @Override
    public String getName() {
        return h5Group.getFullName();
    }

    @Override
    public String getShortName() {
        return h5Group.getName();
    }

    @Override
    public boolean hasAttribute(String name, String value) {
        boolean result = HdfObjectUtils.hasAttribute(h5Group, name, value);
        return result;
    }

    @Override
    public boolean removeAttribute(IAttribute a) {
        return HdfObjectUtils.removeAttribute(h5Group, a);
    }

    @Override
    public void setName(String name) {
        addStringAttribute("long_name", name);
    }

    @Override
    public void setShortName(String name) {
        try {
            h5Group.setName(name);
        }
        catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public void setParent(IGroup group) {
        try {
            h5Group.setPath(group.getName());
        }
        catch (Exception e) {
            Factory.getLogger().severe(e.getMessage());
        }
    }

    @Override
    public long getLastModificationDate() {
        long result = 0;
        String fileName = h5Group.getFile();
        File currentFile = new File(fileName);
        if (currentFile != null && currentFile.exists()) {
            result = currentFile.lastModified();
        }
        return result;
    }

    @Override
    public String getFactoryName() {
        return factoryName;
    }

    @Override
    public void addDataItem(IDataItem item) {
        item.setParent(this);
        if (item instanceof HdfDataItem) {
            h5Group.addToMemberList(((HdfDataItem) item).getH5DataItem());
        }
    }

    @Override
    public Map<String, String> harvestMetadata(String mdStandard) throws IOException {
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
    public void addOneDimension(IDimension dimension) {
    }

    @Override
    public void addSubgroup(IGroup group) {
        group.setParent(this);
    }

    @Override
    public IDataItem getDataItem(String shortName) {
        IDataItem result = null;
        if (shortName != null) {
            result = itemMap.get(shortName);

            if (result == null) {
                H5ScalarDS scalarDS = getH5ScalarDS(shortName);
                if (scalarDS != null) {
                    result = new HdfDataItem(factoryName, h5File, this, scalarDS);
                    itemMap.put(shortName, result);
                }
            }
        }

        return result;
    }

    private H5ScalarDS getH5ScalarDS(String shortName) {
        H5ScalarDS result = null;
        if (shortName != null && !shortName.trim().isEmpty()) {
            List<HObject> members = h5Group.getMemberList();
            for (HObject hObject : members) {
                if (hObject instanceof H5ScalarDS) {
                    H5ScalarDS scalarDS = (H5ScalarDS) hObject;
                    if (shortName.equals(scalarDS.getName())) {
                        result = scalarDS;
                        break;
                    }
                }
            }
        }
        return result;
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
    public IDimension getDimension(String name) {
        IDimension result = null;
        return result;
    }

    @Override
    public IContainer getContainer(String shortName) {
        // TDGV Refactor reporter dans un DefaultGroup
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
    public IGroup getGroup(String shortName) {
        IGroup result = null;
        if (shortName != null) {

            result = groupMap.get(shortName);

            if (result == null) {
                List<HObject> groups = h5Group.getMemberList();
                for (HObject hObject : groups) {
                    if (hObject instanceof H5Group) {
                        H5Group group = (H5Group) hObject;
                        if (shortName.equals(group.getName())) {
                            result = new HdfGroup(factoryName, group, this);
                            groupMap.put(shortName, result);
                            break;
                        }
                    }
                }
            }
        }
        return result;
    }

    @Override
    public IGroup getGroupWithAttribute(String attributeName, String value) {
        // TDGV Refactor reporter dans un DefaultGroup
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
        List<IDataItem> result = new ArrayList<IDataItem>();
        List<HObject> members = h5Group.getMemberList();
        for (HObject hObject : members) {
            if (hObject instanceof H5ScalarDS) {
                IDataItem itemToAdd = getDataItem(hObject.getName());
                if (itemToAdd != null) {
                    result.add(itemToAdd);
                }
            }
        }
        return result;
    }

    @Override
    public IDataset getDataset() {
        HdfDataset dataSet = new HdfDataset(factoryName, h5File);
        return dataSet;
    }

    @Override
    public List<IDimension> getDimensionList() {
        List<IDimension> result = new ArrayList<IDimension>();
        return result;
    }

    @Override
    public IGroup findGroup(String shortName) {
        IGroup result = null;
        result = getGroup(shortName);
        return result;
    }

    @Override
    public List<IGroup> getGroupList() {
        List<IGroup> result = new ArrayList<IGroup>();

        List<HObject> members = h5Group.getMemberList();
        for (HObject hObject : members) {
            if (hObject instanceof H5Group) {
                IGroup groupToAdd = getGroup(hObject.getName());
                if (groupToAdd != null) {
                    result.add(groupToAdd);
                }
            }
        }

        return result;
    }

    private List<HdfNode> getNodes() {
        List<HdfNode> nodes = new ArrayList<HdfNode>();
        List<HObject> members = h5Group.getMemberList();
        for (HObject hObject : members) {
            nodes.add(new HdfNode(hObject));
        }
        return nodes;
    }

    @Override
    public IContainer findContainerByPath(String path) throws NoResultException {
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

    @Override
    public List<IContainer> findAllContainerByPath(String path) throws NoResultException {
        List<IContainer> list = new ArrayList<IContainer>();
        IGroup root = getRootGroup();

        // Try to list all nodes matching the path
        // Transform path into a NexusNode array
        INode[] nodes = HdfPath.splitStringToNode(path);

        // Call recursive method
        int level = 0;
        list = findAllContainer(root, nodes, level);

        return list;
    }

    private List<IContainer> findAllContainer(IContainer container, INode[] nodes, int level) {
        List<IContainer> result = new ArrayList<IContainer>();
        if (container != null) {
            if (container instanceof HdfGroup) {
                HdfGroup group = (HdfGroup) container;
                if (nodes.length > level) {
                    // List current node children
                    List<HdfNode> childs = group.getNodes();

                    INode current = nodes[level];

                    for (HdfNode node : childs) {
                        if (node.matchesPartNode(current)) {

                            if (level < nodes.length - 1) {
                                result.addAll(findAllContainer(group.getContainer(node.getName()),
                                        nodes, level + 1));
                            }
                            // Create IContainer and add it to result list
                            else {
                                result.add(group.getContainer(node.getName()));
                            }
                        }
                    }
                }
            }
            else {
                HdfDataItem dataItem = (HdfDataItem) container;
                result.add(dataItem);
            }
        }
        return result;
    }

    @Override
    public boolean removeDataItem(IDataItem item) {
        return removeDataItem(item.getShortName());

    }

    @Override
    public boolean removeDataItem(String varName) {
        boolean result = true;
        h5Group.removeFromMemberList(getH5ScalarDS(varName));
        return result;
    }

    @Override
    public boolean removeDimension(String name) {
        return false;
    }

    @Override
    public boolean removeDimension(IDimension dimension) {
        return false;
    }

    @Override
    public boolean removeGroup(IGroup group) {
        return removeGroup(group.getShortName());
    }

    @Override
    public boolean removeGroup(String name) {
        boolean result = false;
        if (name != null) {
            H5Group groupToRemove = null;

            List<HObject> members = h5Group.getMemberList();
            for (HObject hObject : members) {
                if (hObject instanceof H5Group) {
                    H5Group group = (H5Group) hObject;
                    if (name.equals(group.getName())) {
                        groupToRemove = group;
                    }
                }
            }
            result = h5Group.getMemberList().remove(groupToRemove);
        }

        return result;
    }

    @Override
    public void updateDataItem(String key, IDataItem dataItem) throws SignalNotAvailableException {
        throw new NotImplementedException();
    }

    @Override
    public boolean isRoot() {
        return h5Group.isRoot();
    }

    @Override
    public boolean isEntry() {
        boolean result = false;
        result = getParentGroup().isRoot();
        return result;
    }

    @Override
    @Deprecated
    public void setDictionary(IDictionary dictionary) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IDictionary findDictionary() {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public List<IContainer> findAllContainers(IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public List<IContainer> findAllOccurrences(IKey key) throws NoResultException {
        throw new NotImplementedException();
    }

    @Override
    public IContainer findObjectByPath(Path path) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IContainer findContainer(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IGroup findGroup(IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem findDataItemWithAttribute(IKey key, String name, String value)
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
    public IGroup findGroupWithAttribute(IKey key, String name, String value) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IDataItem findDataItem(String shortName) {
        throw new NotImplementedException();
    }

    @Override
    @Deprecated
    public IDataItem findDataItem(IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public String toString() {
        StringBuffer buffer = new StringBuffer();
        IGroup parent = getParentGroup();
        buffer.append("Group " + getShortName() + " with parent "
                + ((parent == null) ? "Nobody" : parent.getName()));
        return buffer.toString();
    }

    public void save(FileFormat fileToWrite, Group parent) throws Exception {
        Group newParent = null;

        if (!isRoot()) {

            newParent = fileToWrite.createGroup(getShortName(), parent);

            List<IAttribute> attribute = getAttributeList();
            for (IAttribute iAttribute : attribute) {
                HdfAttribute attr = (HdfAttribute) iAttribute;
                attr.save(newParent);
            }
        }
        else {
            DefaultMutableTreeNode theRoot = (DefaultMutableTreeNode) fileToWrite.getRootNode();
            H5Group rootObject = (H5Group) theRoot.getUserObject();
            newParent = rootObject;
        }

        List<IDataItem> dataItems = getDataItemList();
        for (IDataItem dataItem : dataItems) {
            HdfDataItem hdfDataItem = (HdfDataItem) dataItem;
            hdfDataItem.save(fileToWrite, newParent);
        }

        List<IGroup> groups = getGroupList();
        for (IGroup iGroup : groups) {
            HdfGroup hdfGroup = (HdfGroup) iGroup;
            hdfGroup.save(fileToWrite, newParent);
        }

    }

    public static void main(String[] args) {

        System.setProperty(H5.H5PATH_PROPERTY_KEY,
                "/home/viguier/LocalSoftware/hdf-java/lib/linux/libjhdf5.so");

        String fileName = "/home/viguier/NeXusFiles/ANTARES/CKedge_2010-12-10_21-51-59.nxs";
        String groupPath = "/Pattern_281_salsa.Microscopy.Scan2D_Scienta_Is_XIA_vs_PIXY_1/ANTARES/ScientaAtt0";

        try {
            H5File file = new H5File(fileName);
            file.open();
            HObject hobject = file.get(groupPath);

            H5Group h5Group = (H5Group) hobject;
            h5Group.open();

            System.out.println("h5Group is opened");

            HdfGroup group = new HdfGroup("HDF", h5Group, null);

            String pathToFind = "/Pattern_281_salsa.Microscopy.Scan2D_Scienta_Is_XIA_vs_PIXY_1/ANTARES";
            System.out.println("Test: group.findContainerByPath(" + pathToFind + ")");
            IContainer container = group.findContainerByPath(pathToFind);
            System.out.println((container == null) ? "not found" : "Found ! -> " + container);

            System.out.println("Test: group.findAllContainerByPath()");
            pathToFind = "/Pattern_281_salsa.Microscopy.Scan2D_Scienta_Is_XIA_vs_PIXY_1/ANTARES/ScientaAtt0";
            pathToFind = "/Pattern_281_salsa.Microscopy.Scan2D_Scienta_Is_XIA_vs_PIXY_1/scan_data/data_0*";
            List<IContainer> containers = group.findAllContainerByPath(pathToFind);
            System.out.println("Found " + containers.size() + " containers");
            for (IContainer iContainer : containers) {
                System.out.println(iContainer);
            }

        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
