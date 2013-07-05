package org.cdma.utils;

import org.cdma.interfaces.INode;

public class DefaultNode implements INode {

    private final String name;
    private boolean isGroup;

    public DefaultNode(String name) {
        this.name = name;
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
