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
package fr.soleil.nexus;

import org.nexusformat.NexusException;

public class PathRelative extends PathNexus {

    public PathRelative(String[] groups) {
        super(groups, null);
    }

    public PathRelative(PathNexus pnPath) {
        super(new NexusNode[pnPath.getDepth()]);
        PathNexus pnBuf = pnPath.clone();
        setPath(pnBuf.getNodes());
        if (pnBuf.getFilePath() != null)
            setFile(pnBuf.getFilePath());
    }

    public PathRelative(String[] groups, String dataName) {
        super(groups, dataName);
    }

    @Override
    public boolean isRelative() {
        return true;
    }

    public PathNexus generateAbsolutePath(PathNexus pnStartingPath) throws NexusException {
        NexusNode nnNode;
        PathNexus pnInBuf = clone();
        PathNexus pnOutBuf = pnStartingPath.clone();
        int iDepth = pnInBuf.getDepth();

        for (int i = 0; i < iDepth; i++) {
            nnNode = pnInBuf.getNode(i);
            if (nnNode.getNodeName().equals(PathNexus.PARENT_NODE))
                pnOutBuf.popNode();
            else
                pnOutBuf.pushNode(nnNode);
        }

        return pnOutBuf;
    }
}
