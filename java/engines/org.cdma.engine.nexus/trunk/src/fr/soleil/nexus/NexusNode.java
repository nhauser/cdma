//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package fr.soleil.nexus;

import java.text.Collator;
import java.util.Comparator;


public class NexusNode implements Cloneable {
    // Private definitions
    private static final String CLASS_SEPARATOR_START  = "<";
    private static final String CLASS_SEPARATOR_START2 = "{";
    private static final String CLASS_SEPARATOR_END2   = "}";

    private String   m_sNodeName;
    private String   m_sClassName;
    private boolean m_bIsGroup;

    public NexusNode() {
    	m_sNodeName  = "";
        m_sClassName = "";
        m_bIsGroup   = false;
    }

    public NexusNode(String sNodeName, String sClassName) {
        m_sNodeName = sNodeName;
        m_sClassName = sClassName;
        m_bIsGroup = (!"SDS".equals(sClassName) || "".equals(sNodeName) ) && !"NXtechnical_data".equals(sClassName);
    }

    public NexusNode(String sNodeName, String sClassName, boolean bIsGroup) {
        m_sNodeName = sNodeName;
        m_sClassName = sClassName;
        m_bIsGroup = bIsGroup;
    }

    public void setNodeName(String sNodeName) {
        m_sNodeName = sNodeName;
    }

    public void setClassName(String sClassName) {
        m_sClassName = sClassName;
    }

    public void setIsGroup(boolean bIsGroup) {
        m_bIsGroup = bIsGroup;
    }

    public String getNodeName() {
        return m_sNodeName;
    }

    public String getClassName() {
        return m_sClassName;
    }

    public boolean isGroup() {
        return m_bIsGroup;
    }

    public boolean isRealGroup() {
        return (m_sClassName != null && !m_sClassName.isEmpty() && !m_sClassName.equals("SDS") );
    }

    protected NexusNode clone() {
        NexusNode nNewNode = new NexusNode();
        nNewNode.m_sNodeName = m_sNodeName;
        nNewNode.m_sClassName = m_sClassName;
        nNewNode.m_bIsGroup = m_bIsGroup;

        return nNewNode;
    }
    
    @Override
    public boolean equals(Object node) {
        if (node == this) {
            return true;
        }
        if (node == null || node.getClass() != this.getClass()) {
            return false;
        }

        NexusNode n = (NexusNode) node;
        return (m_sNodeName.equals(n.m_sNodeName) && m_sClassName.equals(n.m_sClassName) && m_bIsGroup == n.m_bIsGroup);
    }

    @Override
    public int hashCode() {
        return m_sNodeName.hashCode() + m_sClassName.hashCode();
    }

    public String toString() {
        String sName = getNodeName();
        if (!getClassName().trim().equals("") && isRealGroup())
            sName += CLASS_SEPARATOR_START2 + getClassName() + CLASS_SEPARATOR_END2;
        return sName;
    }

    public static NexusNode getNexusNode(String sNodeFullName, boolean bIsGroup) {
        NexusNode node = null;
        String tmpNodeName = NexusNode.extractName(sNodeFullName);
        String tmpClassName = NexusNode.extractClass(sNodeFullName);

        if (!"".equals(tmpNodeName) || !"".equals(tmpClassName))
            node = new NexusNode(tmpNodeName, tmpClassName, bIsGroup);

        return node;
    }

    public static String getNodeFullName(String sNodeName, String sNodeClass) {
        return sNodeName + (sNodeClass.equals("SDS") ? "" : (CLASS_SEPARATOR_START2 + sNodeClass + CLASS_SEPARATOR_END2));
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
        tmpClassName = iPosClassSep < sNodeName.length() ? sNodeName.substring(iPosClassSep + 1, sNodeName.length() - 1) : "";
        return tmpClassName;
    }

    /**
     * Return true when the given node (which is this) matches this node.
     * 
     * @param node
     *            NexusNode that is a pattern referent: it should have name XOR
     *            class name defined
     * @return true when this node fit the given pattern node
     */
    public boolean matchesNode(NexusNode node) {
        boolean classMatch, nameMatch;

        classMatch = "".equals(node.getClassName()) || node.getClassName().equalsIgnoreCase(this.getClassName());
        nameMatch = "".equals(node.getNodeName()) || this.getNodeName().equalsIgnoreCase(node.getNodeName());

        return (classMatch && nameMatch);
    }

    public boolean matchesPartNode(NexusNode node) {
        boolean classMatch, nameMatch;
        classMatch = "".equals(node.getClassName()) || node.getClassName().equalsIgnoreCase(this.getClassName());
        nameMatch = "".equals(node.getNodeName()) || this.getNodeName().toLowerCase().matches(node.getNodeName().toLowerCase().replace("*", ".*"));
        return (classMatch && nameMatch);
    }
    
    static public class NodeCollator implements Comparator<NexusNode> {

		@Override
		public int compare(NexusNode arg0, NexusNode arg1) {
			int result;
			// if arg0 is null
			if( arg0 == null ) {
				// equals if arg1 is null else negative
				result = arg1 == null ? 0 : -1;
			}
			// Check they are equals
			else if( arg0.equals(arg1) ) {
				result = 0;
			}
			// Lesser or greater test
			else {
				// If one is group and other not: group is greater
				if( arg0.isGroup() != arg1.isGroup() ) {
					result = arg0.isGroup() ? 1 : -1;
				}
				else {
					String class0 = arg0.getClassName();
					String class1 = arg1.getClassName();
					if( class0.equals( class1 ) ) {
						String name0 = arg0.getNodeName();
						String name1 = arg1.getNodeName();
						result = new NameCollator().compare( name0, name1 );
					}
					else {
						result = Collator.getInstance().compare( class0, class1 );
					}
				}
			}
			return result;
		}
    	
    }
    
    static public class NameCollator implements Comparator<String> {
    	
        @Override
        public int compare(final String arg0, final String arg1) {
            int iCmp;
            if (arg0.matches(".*[0-9].*") && arg1.matches(".*[0-9].*")) {
                // Prepare string by marking up every digit
                String argA, argB;
                argA = arg0.replaceAll("(\\d+)", "#$1#");
                argB = arg1.replaceAll("(\\d+)", "#$1#");

                // Separate characters and digit
                String[] arg0Parts, arg1Parts;
                arg0Parts = argA.split("#");
                arg1Parts = argB.split("#");

                // Compare strings until one is lesser than the other
                iCmp = 0;
                int index = 0;
                while (iCmp == 0) {
                    // If remains string in both parts
                    if (index < arg0Parts.length && index < arg1Parts.length) {
                        // If digits
                        if (arg0Parts[index].matches("[0-9]+") && arg1Parts[index].matches("[0-9]+")) {
                            int iArg0 = Integer.parseInt(arg0Parts[index]);
                            int iArg1 = Integer.parseInt(arg1Parts[index]);

                            if (iArg0 > iArg1)
                                iCmp = 1;
                            else if (iArg0 < iArg1)
                                iCmp = -1;
                            else
                                iCmp = 0;
                        }
                        // If characters
                        else {
                            iCmp = Collator.getInstance().compare(arg0Parts[index], arg1Parts[index]);
                        }
                    }
                    // One of the part is empty
                    else {
                        if (arg0Parts.length == arg1Parts.length)
                            iCmp = 0;
                        else
                            iCmp = (arg0Parts.length > arg1Parts.length) ? 1 : -1;
                        break;
                    }
                    index++;
                }
            } else {
                iCmp = Collator.getInstance().compare(arg0, arg1);
            }
            return iCmp;
        }
    }
}
