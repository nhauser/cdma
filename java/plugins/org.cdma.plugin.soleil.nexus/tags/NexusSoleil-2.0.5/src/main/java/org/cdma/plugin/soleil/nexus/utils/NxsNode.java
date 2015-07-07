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
package org.cdma.plugin.soleil.nexus.utils;

import org.cdma.engine.hdf.utils.HdfNode;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.INode;

public class NxsNode extends HdfNode {

    private static final String SELECTED_ATTRIBUTE_TO_LOAD = "NX_class";
    private String clazz = "";

    public NxsNode(final String name, final String className) {
        super(name);
        this.clazz = className;
    }

    public NxsNode(final String fullName) {
        this(extractName(fullName), extractClass(fullName));
    }

    public NxsNode(final IContainer container) {
        super(container);
        IAttribute attribute = container.getAttribute(SELECTED_ATTRIBUTE_TO_LOAD);
        if (attribute != null) {
            this.clazz = attribute.getStringValue();
        }
    }

    public String getClassName() {
        return clazz;
    }

    @Override
    public String toString() {
        return getName() + "{" + this.clazz + "}";
    }

    public static String extractClass(final String sNodeName) {
        int iPosClassSep;
        String tmpClassName = "";
        iPosClassSep = sNodeName.indexOf(ATTRIBUTE_SEPARATOR_START);
        if (iPosClassSep < 0)
            iPosClassSep = sNodeName.indexOf(ATTRIBUTE_SEPARATOR_START2);
        iPosClassSep = iPosClassSep < 0 ? sNodeName.length() : iPosClassSep;
        tmpClassName = iPosClassSep < sNodeName.length() ? sNodeName
                .substring(iPosClassSep + 1, sNodeName.length() - 1) : "";
        return tmpClassName;
    }

    @Override
    public boolean matchesNode(final INode node) {
        boolean classMatch, nameMatch;
        NxsNode nxsNode = (NxsNode) node;
        classMatch = "".equals(nxsNode.getClassName()) || nxsNode.getClassName().equalsIgnoreCase(this.clazz);
        nameMatch = "".equals(node.getNodeName()) || this.getNodeName().equalsIgnoreCase(node.getNodeName());

        return (classMatch && nameMatch);
    }

    @Override
    public boolean matchesPartNode(final INode node) {
        boolean classMatch = false, nameMatch = false;
        NxsNode nxsNode = (NxsNode) node;

        if (node != null) {
            classMatch = "".equals(nxsNode.getClassName()) || nxsNode.getClassName().equalsIgnoreCase(this.clazz);
            nameMatch = "".equals(node.getNodeName())
                    || this.getNodeName().toLowerCase().replace("*", ".*")
                            .matches(node.getNodeName().toLowerCase().replace("*", ".*"));
        }
        return (classMatch && nameMatch);
    }
}
