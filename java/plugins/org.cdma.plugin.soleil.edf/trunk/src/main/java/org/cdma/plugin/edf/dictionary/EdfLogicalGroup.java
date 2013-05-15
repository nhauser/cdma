package org.cdma.plugin.edf.dictionary;

import java.util.ArrayList;
import java.util.List;

import org.cdma.dictionary.ExtendedDictionary;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.BackupException;
import org.cdma.exception.FileAccessException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.plugin.edf.navigation.EdfDataset;
import org.cdma.plugin.edf.navigation.EdfKey;
import org.cdma.utils.Utilities.ModelType;

public class EdfLogicalGroup extends LogicalGroup {
    public static final String KEY_PATH_SEPARATOR = ":";

    IKey m_key; // List of IKey that populated this items
    LogicalGroup m_parent; // PathNexus corresponding to the current path
    List<IContainer> m_child; // List of belonging items
    ExtendedDictionary m_dictionary; // Dictionary that belongs to this current LogicalGroup
    EdfDataset m_dataset; // File handler
    String m_shortName = "";

    public EdfLogicalGroup(LogicalGroup parent, IKey key, EdfDataset dataset) {
        if (key != null) {
            m_key = key.clone();
        }
        else {
            m_key = null;
        }
        m_parent = parent;
        m_child = new ArrayList<IContainer>();
        m_dataset = dataset;
    }

    @Override
    public LogicalGroup clone() {
        EdfLogicalGroup group = new EdfLogicalGroup(m_parent, m_key, m_dataset);
        for (IContainer object : m_child) {
            group.m_child.add(object.clone());
        }
        group.m_dictionary = m_dictionary;
        return group;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LogicalGroup;
    }

    @Override
    public String getName() {
        if (m_parent == null || m_shortName == null) {
            return "";
        }
        else {
            return m_parent.getName() + "/" + m_shortName;
        }
    }

    @Override
    public LogicalGroup getParentGroup() {
        return m_parent;
    }

    @Override
    public LogicalGroup getRootGroup() {
        if (m_parent == null) {
            return this;
        }
        else {
            return m_parent.getParentGroup();
        }
    }

    @Override
    public String getShortName() {
        if (m_key == null) {
            return "";
        }
        else {
            return m_shortName;
        }
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
    public String getLocation() {
        return m_parent.getName() + getName();
    }

    /**
     * Get the dictionary belonging to the root group.
     * 
     * @return IDictionary the dictionary currently applied to this group
     */
    @Override
    public ExtendedDictionary getDictionary() {
        return (m_dictionary == null ? findDictionary() : m_dictionary);
    }

    /**
     * Set a dictionary to the root group.
     * 
     * @param dictionary the dictionary to set
     */
    @Override
    public void setDictionary(ExtendedDictionary dictionary) {
        m_dictionary = dictionary;
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

        for (IContainer object : list) {
            if (object.getModelType().equals(ModelType.DataItem)) {
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
        LogicalGroup grp = this;
        IDataItem result = null;

        if (keys.length >= 1) {
            while (i < (keys.length - 1)) {
                grp = grp.getGroup(new EdfKey(keys[i]));
            }
            result = grp.getDataItem(keys[keys.length - 1]);
        }

        return result;
    }

    @Override
    public List<IDataItem> getDataItemList(IKey key) {
        List<IContainer> list = new ArrayList<IContainer>();
        List<IDataItem> result = new ArrayList<IDataItem>();
        list = getItemByKey(key);

        for (IContainer object : list) {
            if (object.getModelType().equals(ModelType.DataItem)) {
                result.add((IDataItem) object);
            }
        }

        return result;
    }

    @Override
    public List<IDataItem> getDataItemList(String keyPath) {
        String[] keys = keyPath.split(KEY_PATH_SEPARATOR);

        int i = 0;
        LogicalGroup grp = this;
        List<IDataItem> result = null;

        if (keys.length >= 1) {
            while (i < (keys.length - 1)) {
                grp = grp.getGroup(new EdfKey(keys[i]));
            }
            result = grp.getDataItemList(keys[keys.length - 1]);
        }

        return result;
    }

    @Override
    public LogicalGroup getGroup(IKey key) {
        LogicalGroup item = null;
        List<IContainer> list = new ArrayList<IContainer>();
        list = getItemByKey(key);

        for (IContainer object : list) {
            if (object.getModelType().equals(ModelType.LogicalGroup)) {
                item = (LogicalGroup) object;
                break;
            }
        }

        return item;
    }

    @Override
    public LogicalGroup getGroup(String keyPath) {
        String[] keys = keyPath.split(KEY_PATH_SEPARATOR);

        int i = 0;
        LogicalGroup grp = this;
        LogicalGroup result = null;

        if (keys.length >= 1) {
            while (i < (keys.length - 1)) {
                grp = grp.getGroup(new EdfKey(keys[i]));
            }
            result = grp.getGroup(keys[keys.length - 1]);
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

    // ---------------------------------------------------------
    // / protected methods
    // ---------------------------------------------------------
    protected void addObject(IContainer object) {
        m_child.add(object);
    }

    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    /**
     * Apply the given filter on the given node. It permits determining which node we really want to
     * open. Then it's opened.
     * 
     * @param filters is the filter to applied
     * @param node is the node pattern we should modify to open the correct one
     * 
     * @note "node" must be a direct child of the currently opened node
     */
    private List<IContainer> applyKeyFilter(IKey key, List<IContainer> nodes, String[] nodesStatic) {
        List<IContainer> remove_list = new ArrayList<IContainer>();
        List<IContainer> node_list = nodes;

        // node_list.addAll(node.getDataItemList());
        // node_list.addAll(node.getGroupList());
        /*
        // Only add nodes matching  given "node" (by name or by class)
        for( IContainer tmpNode : node_list ) {
            if( tmpNode.getName().contains(seekNode) ) {
            	remove_list.add(tmpNode);
            }
        }
        node_list.removeAll(remove_list);
        remove_list.clear();
         */

        // Start filtering object
        List<IKeyFilter> filters = key.getFilterList();

        for (IKeyFilter keyFilter : filters) {
            switch (keyFilter.getLabel()) {
                // Node name must equal "value"
                case NAME: {
                    String value = (String) keyFilter.getValue();
                    for (IContainer n : node_list) {
                        if (!value.equals(n.getName())) {
                            remove_list.add(n);
                        }
                    }
                    node_list.removeAll(remove_list);
                    remove_list.clear();
                    break;
                }
                // Node name must be equal "value" (not case sensitive)
                case NAME_NO_CASE: {
                    String value = (String) keyFilter.getValue();
                    for (IContainer n : node_list) {
                        if (!value.equalsIgnoreCase(n.getName())) {
                            remove_list.add(n);
                        }
                    }
                    node_list.removeAll(remove_list);
                    remove_list.clear();
                    break;
                }
                // Node must have an attribute called "value"
                case ATTRIBUTE_NAME: {
                    String value = (String) keyFilter.getValue();
                    for (IContainer n : node_list) {
                        if (n.getAttribute(value) == null) {
                            remove_list.add(n);
                        }
                    }
                    node_list.removeAll(remove_list);
                    remove_list.clear();
                    break;
                }
                // Node must have an attribute with "value" for value
                // The attribute name is determined by another KeyFilter (ATTRIBUTE_NAME)
                case ATTRIBUTE_VALUE: {
                    String value = (String) keyFilter.getValue();
                    for (IContainer n : node_list) {
                        for (IAttribute attr : n.getAttributeList()) {
                            if (!attr.getStringValue().equals(value)) {
                                remove_list.add(n);
                            }
                        }
                    }
                    node_list.removeAll(remove_list);
                    remove_list.clear();
                    break;
                }
                // Seek item number "value"
                case INDEX: {
                    Integer value = (Integer) keyFilter.getValue();
                    if (node_list.size() > value) {
                        IContainer n = node_list.get(value);
                        node_list.clear();
                        node_list.add(n);
                    }
                    else {
                        node_list.clear();
                    }
                    break;
                }
                // Node name must match "value" (whom is a regular expression)
                case NAME_REGEXP: {
                    String value = (String) keyFilter.getValue();
                    for (IContainer n : node_list) {
                        if (!value.matches(n.getName())) {
                            remove_list.add(n);
                        }
                    }
                    node_list.removeAll(remove_list);
                    remove_list.clear();
                    break;
                }
            }
        }
        if (key.getName().equals("mi"))
            System.out.println("dddd");
        List<IContainer> result = new ArrayList<IContainer>();
        for (IContainer obj : node_list) {
            if (obj.getModelType() == ModelType.Group) {
                for (String name : nodesStatic) {
                    if (!name.isEmpty()) {
                        obj = listChildWithPartName((IGroup) obj, name).get(0);
                    }
                }
            }
            if (obj != null && obj.getModelType() == ModelType.Group) {
                EdfLogicalGroup lg = new EdfLogicalGroup(this, key, (EdfDataset) getDataset());
                lg.m_shortName = obj.getName();
                obj = lg;
            }
            result.add(obj);
        }

        remove_list.clear();
        return result;
    }

    private List<IContainer> getItemByKey(IKey key) {
        List<IContainer> list = new ArrayList<IContainer>();
        Path pPath = getDictionary().getPath(key);
        String path = pPath.getValue();

        int i = 0;

        IGroup root = m_dataset.getRootGroup();

        String pathFinal = "";
        String pathVaria = "";
        if (path.indexOf("_[") >= 0) {
            pathVaria = path.substring(path.indexOf("_[") + 2, path.indexOf("]_"));
            pathFinal = path.substring(path.indexOf("]_") + 2);
        }
        else {
            pathVaria = "";
            pathFinal = path;
        }
        String[] nodesVariab = pathVaria.split("/");
        String[] nodesStatic = pathFinal.split("/");

        // open all nodes but the last variable one
        IGroup tmpObj = root, tmpObj2;
        for (String node : nodesVariab) {
            if (i++ < (nodesVariab.length - 1)) {
                tmpObj2 = tmpObj.getGroup(node);
                if (tmpObj2 == null) {
                    List<IGroup> l = listGroupWithPartName(tmpObj, node);
                    if (l.size() > 0) {
                        tmpObj = l.get(0);
                    }
                }
                else {
                    tmpObj = tmpObj2;
                }
            }
            else {
                if (node != null && !node.trim().isEmpty()) {
                    list.addAll(listChildWithPartName(tmpObj, node));
                }
                else {
                    list.add(getDataset().getRootGroup());
                }
            }
        }
        list = applyKeyFilter(key, list, nodesStatic);

        return list;
    }

    private List<IContainer> listChildWithPartName(IGroup group, String name) {
        List<IContainer> list = new ArrayList<IContainer>();
        if (name != null && !name.trim().isEmpty()) {
            list.addAll(listGroupWithPartName(group, name));
            list.addAll(listDataItemWithPartName(group, name));
        }
        return list;
    }

    private List<IGroup> listGroupWithPartName(IGroup group, String name) {
        List<IGroup> list = new ArrayList<IGroup>();
        List<IGroup> grp_list = group.getGroupList();
        for (IGroup grp : grp_list) {
            if (grp.getName().contains(name) || grp.getShortName().equals(name)) {
                list.add(grp);
            }
        }
        grp_list = null;
        return list;
    }

    private List<IDataItem> listDataItemWithPartName(IGroup group, String name) {
        List<IDataItem> list = new ArrayList<IDataItem>();
        List<IDataItem> itm_list = group.getDataItemList();
        for (IDataItem itm : itm_list) {
            if (itm.getName().contains(name)) {
                list.add(itm);
            }
        }
        itm_list = null;

        return list;
    }

    private ExtendedDictionary findDictionary() {
        if (m_dictionary == null) {
            m_dictionary = new ExtendedDictionary();
            try {
                m_dictionary.readEntries(System
                        .getProperty("DICO_PATH", System.getenv("DICO_PATH")));
            }
            catch (FileAccessException e) {
                e.printStackTrace();
            }
        }
        return m_dictionary;
    }
}
