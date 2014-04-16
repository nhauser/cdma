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
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.utils;

import org.cdma.interfaces.INode;

public class DefaultNode implements INode {

    private final String name;
    private final String attribute;
    private boolean isGroup;

    public DefaultNode(String name, String attribute) {
        this.name = name;
        this.attribute = attribute;
    }

    public DefaultNode(String name) {
        this.name = name;
        this.attribute = "";
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

    @Override
    public String getAttribute() {
        return attribute;
    }
}
