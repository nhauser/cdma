package org.cdma.engine.hdf.utils;

import ncsa.hdf.object.HObject;

import org.cdma.interfaces.INode;

public class HdfNode implements INode {
    // Private definitions
    private static final String CLASS_SEPARATOR_START = "<";
    private static final String CLASS_SEPARATOR_START2 = "{";
    private static final String CLASS_SEPARATOR_END2 = "}";

    private final String name;
    private boolean isGroup;
    private String attribute = "";

    public HdfNode(String name, String attribute) {
        this.name = name;
        this.attribute = attribute;
    }

    public HdfNode(String fullName) {
        this(extractName(fullName), extractClass(fullName));
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

    @Override
    public String getAttribute() {
        return attribute;
    }

    @Override
    public boolean matchesNode(INode node) {
        boolean classMatch, nameMatch;
        classMatch = "".equals(node.getAttribute()) || node.getAttribute().equalsIgnoreCase(this.getAttribute());
        nameMatch = "".equals(node.getNodeName()) || this.getNodeName().equalsIgnoreCase(node.getNodeName());

        return (classMatch && nameMatch);
    }

    @Override
    public boolean matchesPartNode(INode node) {
        boolean classMatch = false, nameMatch = false;
        if (node != null) {
            classMatch = "".equals(node.getAttribute()) || node.getAttribute().equalsIgnoreCase(this.getAttribute());
            nameMatch = "".equals(node.getNodeName())
                    || this.getNodeName().toLowerCase().replace("*", ".*")
                            .matches(node.getNodeName().toLowerCase().replace("*", ".*"));
        }
        return (classMatch && nameMatch);
    }

    @Override
    public boolean isGroup() {
        return isGroup;
    }

    @Override
    public String toString() {
        return name;
    }

    public static String extractName(String sNodeName) {
        int iPosClassSep;
        String tmpNodeName = "";
        iPosClassSep = sNodeName.indexOf(CLASS_SEPARATOR_START);
        if (iPosClassSep < 0)
            iPosClassSep = sNodeName.indexOf(CLASS_SEPARATOR_START2);
        iPosClassSep = iPosClassSep < 0 ? sNodeName.length() : iPosClassSep;
        tmpNodeName = sNodeName.substring(0, iPosClassSep);
        return tmpNodeName;
    }

    public static String extractClass(String sNodeName) {
        int iPosClassSep;
        String tmpClassName = "";
        iPosClassSep = sNodeName.indexOf(CLASS_SEPARATOR_START);
        if (iPosClassSep < 0)
            iPosClassSep = sNodeName.indexOf(CLASS_SEPARATOR_START2);
        iPosClassSep = iPosClassSep < 0 ? sNodeName.length() : iPosClassSep;
        tmpClassName = iPosClassSep < sNodeName.length() ? sNodeName
                .substring(iPosClassSep + 1, sNodeName.length() - 1) : "";
                return tmpClassName;
    }

}
