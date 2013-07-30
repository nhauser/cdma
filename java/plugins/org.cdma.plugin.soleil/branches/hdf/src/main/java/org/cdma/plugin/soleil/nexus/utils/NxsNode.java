package org.cdma.plugin.soleil.nexus.utils;


import org.cdma.engine.hdf.utils.HdfNode;

public class NxsNode extends HdfNode {

    private final String className;
    // Private definitions
    private static final String CLASS_SEPARATOR_START = "<";
    private static final String CLASS_SEPARATOR_START2 = "{";
    private static final String CLASS_SEPARATOR_END2 = "}";

    public NxsNode(String name, String className) {
        super(name);
        this.className = className;
    }

    public NxsNode(String fullName) {
        this(extractName(fullName), extractClass(fullName));
    }

    public String getClassName() {
        return className;
    }
    /**
     * Return true when the given node (which is this) matches this node.
     * 
     * @param node NexusNode that is a pattern referent: it should have name XOR class name defined
     * @return true when this node fit the given pattern node
     */
    public boolean matchesNode(NxsNode node) {
        boolean classMatch, nameMatch;

        classMatch = "".equals(node.getClassName())
                || node.getClassName().equalsIgnoreCase(this.getClassName());
        nameMatch = super.matchesNode(node);

        return (classMatch && nameMatch);
    }

    public boolean matchesPartNode(NxsNode node) {
        boolean classMatch = false, nameMatch = false;
        if (node != null) {
            classMatch = "".equals(node.getClassName())
                    || node.getClassName().equalsIgnoreCase(this.getClassName());

            nameMatch = super.matchesPartNode(node);
        }
        return (classMatch && nameMatch);
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
        tmpClassName = iPosClassSep < sNodeName.length() ? sNodeName.substring(iPosClassSep + 1,
                sNodeName.length() - 1) : "";
        return tmpClassName;
    }

    @Override
    public String toString() {
        String sName = getNodeName();
        if (!getClassName().trim().equals(""))
            sName += CLASS_SEPARATOR_START2 + getClassName() + CLASS_SEPARATOR_END2;
        return sName;
    }
}
