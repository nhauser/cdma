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
package org.cdma.engine.hdf.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.cdma.interfaces.INode;

public class HdfPath {

    public static final String PATH_SEPARATOR = "/";
    public static final String PARENT_NODE = "..";
    protected List<INode> nodes = new ArrayList<INode>();

    public HdfPath(final INode[] nodes) {
        this.nodes = Arrays.asList(nodes);
    }

    public HdfPath(final INode node) {
        this.nodes.add(node);
    }

    public HdfPath() {
        // Needed empty
    }

    /**
     * Split a string representing a NexusPath to extract each node name
     * 
     * @param path
     */
    public static String[] splitStringPath(final String path) {
        if (path.startsWith(HdfPath.PATH_SEPARATOR)) {
            return path.substring(1).split(HdfPath.PATH_SEPARATOR);
        }
        else {
            return path.split(HdfPath.PATH_SEPARATOR);
        }
    }

    public static INode[] splitStringToNode(final String sPath) {
        HdfNode[] result = null;
        if (sPath != null) {
            String[] names = splitStringPath(sPath);

            int nbNodes = 0;
            for (String name : names) {
                if (!name.isEmpty()) {
                    nbNodes++;
                }
            }

            if (nbNodes > 0) {
                result = new HdfNode[nbNodes];
                int i = 0;
                for (String name : names) {
                    if (!name.isEmpty()) {
                        result[i] = new HdfNode(name);
                        i++;
                    }
                }
            } else {
                result = new HdfNode[0];
            }
        }
        return result;
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
    public INode[] getNodes() {
        return nodes.toArray(new INode[nodes.size()]);
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


    @Override
    public String toString() {
        StringBuffer result = new StringBuffer();
        for (INode node : nodes) {
            result.append(node.toString());
            result.append(PATH_SEPARATOR);
        }
        return result.toString();
    }


}
