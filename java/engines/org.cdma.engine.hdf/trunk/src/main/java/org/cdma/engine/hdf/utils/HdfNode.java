package org.cdma.engine.hdf.utils;

import ncsa.hdf.object.HObject;

import org.cdma.interfaces.INode;

public class HdfNode implements INode {

    private final String name;
    private boolean isGroup;

    public HdfNode(String name) {
        this.name = name;
    }

    public HdfNode(HObject hObject) {
        this.name = hObject.getName();
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public String getNodeName() {
        return getName();
    }

    /**
     * Return true when the given node (which is this) matches this node.
     * 
     * @param node NexusNode that is a pattern referent: it should have name XOR class name defined
     * @return true when this node fit the given pattern node
     */
    @Override
    public boolean matchesNode(INode node) {
        boolean nameMatch;

        nameMatch = "".equals(node.getNodeName())
                || this.getNodeName().equalsIgnoreCase(node.getNodeName());

        return nameMatch;
    }

    @Override
    public boolean matchesPartNode(INode node) {
        boolean nameMatch;

        nameMatch = "".equals(node.getNodeName())
                || this.getNodeName().toLowerCase().replace("*", ".*")
                .matches(node.getNodeName().toLowerCase().replace("*", ".*"));
        return nameMatch;
    }

    @Override
    public boolean isGroup() {
        return isGroup;
    }

    @Override
    public String toString() {
        return name;
    }
}