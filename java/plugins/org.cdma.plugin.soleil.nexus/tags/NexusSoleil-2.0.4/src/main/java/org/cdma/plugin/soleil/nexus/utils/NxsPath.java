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

import org.cdma.engine.hdf.utils.HdfPath;
import org.cdma.interfaces.IContainer;
import org.cdma.plugin.soleil.nexus.navigation.NxsGroup;

public class NxsPath extends HdfPath {

    public NxsPath(IContainer container) {
        IContainer parent = container.getParentGroup();
        if (parent != null) {
            NxsGroup nxsParent = (NxsGroup) parent;
            nodes.addAll(nxsParent.getNxsPath().nodes);
        }
        nodes.add(new NxsNode(container));
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
        } else {
            nodes = new NxsNode[0];
        }
        return nodes;
    }

}
