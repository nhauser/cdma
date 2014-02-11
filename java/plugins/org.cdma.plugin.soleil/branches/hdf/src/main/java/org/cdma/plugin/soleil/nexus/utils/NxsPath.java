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
package org.cdma.plugin.soleil.nexus.utils;

import org.cdma.engine.hdf.utils.HdfPath;
import org.cdma.interfaces.INode;


public class NxsPath extends HdfPath {

    public static final String PARENT_NODE = "..";

    public NxsPath(NxsNode[] nodes) {
        super(nodes);
    }

    public NxsPath(INode[] nodes) {
        super(nodes);
    }

    public NxsPath(INode node) {
        super(node);
    }


    @Override
    public NxsNode[] getNodes() {
        return nodes.toArray(new NxsNode[nodes.size()]);
    }

    /**
     * getValue return the path value in a Nexus file as a String
     */
    public String getValue() {
        StringBuffer buf = new StringBuffer();

        if (!isRelative())
            buf.append(HdfPath.PATH_SEPARATOR);

        INode nnNode;
        for (int i = 0; i < nodes.size(); i++) {
            nnNode = nodes.get(i);
            if (!"".equals(nnNode.toString().trim())) {
                buf.append(nnNode.toString());
                if (nnNode.isGroup())
                    buf.append(HdfPath.PATH_SEPARATOR);
            }
        }
        String result = buf.toString();
        return result;
    }

    /**
     * isRelative Scan all nodes of the current path and check if it has a back value
     */
    public boolean isRelative() {
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).getNodeName().equals(PARENT_NODE))
                return true;
        }
        return false;
    }

    /**
     * getCurrentNode returns the node that the path is aiming to or null if it contains no node
     */
    public INode getCurrentNode() {

        if (nodes.size() > 0)
            return nodes.get(nodes.size() - 1);
        else
            return null;
    }

    public void addNode(INode nodeToAdd) {
        nodes.add(nodeToAdd);
    }
    public static NxsNode[] splitStringToNode(String sPath) {
        String[] names = splitStringPath(sPath);
        NxsNode[] nodes = null;

        int nbNodes = 0;
        for (String name : names) {
            if (!name.isEmpty()) {
                nbNodes++;
            }
        }

        if (nbNodes > 0) {
            nodes = new NxsNode[nbNodes];
            int i = 0;
            for (String name : names) {
                if (!name.isEmpty()) {
                    nodes[i] = new NxsNode(NxsNode.extractName(name), NxsNode.extractClass(name));
                    i++;
                }
            }
        }
        else {
            nodes = new NxsNode[0];
        }
        return nodes;
    }

}
