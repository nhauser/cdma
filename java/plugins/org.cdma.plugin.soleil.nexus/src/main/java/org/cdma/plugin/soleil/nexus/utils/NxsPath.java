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



    public NxsPath(final NxsNode[] nodes) {
        super(nodes);
    }

    public NxsPath(final INode[] nodes) {
        super(nodes);
    }

    public NxsPath(final INode node) {
        super(node);
    }

    public NxsPath(final HdfPath hdfPath) {
        super();

        INode[] hdfNodes = hdfPath.getNodes();

        nodes.clear();
        for (int i = 0; i < hdfNodes.length; i++) {
            INode hdfNode = hdfNodes[i];
            nodes.add(new NxsNode(hdfNode.getName(), hdfNode.getAttribute()));
        }

    }

    @Override
    public NxsNode[] getNodes() {
        return nodes.toArray(new NxsNode[nodes.size()]);
    }

    public static NxsNode[] splitStringToNode(final String sPath) {
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
