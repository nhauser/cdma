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
package org.cdma.engine.hdf.utils;

import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.INode;

public class HdfNode implements INode {
    // Private definitions
    public static final String ATTRIBUTE_SEPARATOR_START = "<";
    public static final String ATTRIBUTE_SEPARATOR_START2 = "{";

    protected final String name;
    private boolean isGroup;

    public HdfNode(final String name) {
        this.name = name;
    }

    public HdfNode(IContainer container) {
        this.name = container.getShortName();
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
    public boolean matchesNode(final INode node) {
        boolean nameMatch;
        nameMatch = "".equals(node.getNodeName()) || this.getNodeName().equalsIgnoreCase(node.getNodeName());

        return (nameMatch);
    }

    @Override
    public boolean matchesPartNode(final INode node) {
        boolean nameMatch = false;
        if (node != null) {
            nameMatch = "".equals(node.getNodeName())
                    || this.getNodeName().toLowerCase().replace("*", ".*")
                            .matches(node.getNodeName().toLowerCase().replace("*", ".*"));
        }
        return (nameMatch);
    }

    @Override
    public boolean isGroup() {
        return isGroup;
    }

    @Override
    public String toString() {
        return name;
    }

    public static String extractName(final String sNodeName) {
        int iPosClassSep;
        String tmpNodeName = "";
        iPosClassSep = sNodeName.indexOf(ATTRIBUTE_SEPARATOR_START);
        if (iPosClassSep < 0)
            iPosClassSep = sNodeName.indexOf(ATTRIBUTE_SEPARATOR_START2);
        iPosClassSep = iPosClassSep < 0 ? sNodeName.length() : iPosClassSep;
        tmpNodeName = sNodeName.substring(0, iPosClassSep);
        return tmpNodeName;
    }

}
