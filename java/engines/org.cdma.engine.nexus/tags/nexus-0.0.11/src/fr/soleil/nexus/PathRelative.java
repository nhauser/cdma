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

import org.nexusformat.NexusException;

public class PathRelative extends PathNexus {

    public PathRelative(String[] groups) {
	super(groups, null);
    }

    public PathRelative(PathNexus pnPath) {
	super(new NexusNode[pnPath.getDepth()]);
	PathNexus pnBuf = pnPath.clone();
	setPath(pnBuf.getNodes());
	if( pnBuf.getFilePath() != null )
	    setFile(pnBuf.getFilePath());
    }

    public PathRelative(String[] groups, String dataName) {
	super(groups, dataName);
    }

    public boolean isRelative() { return true; }

    public PathNexus generateAbsolutePath(PathNexus pnStartingPath) throws NexusException
    {
	NexusNode nnNode;
	PathNexus pnInBuf  = clone();
	PathNexus pnOutBuf = pnStartingPath.clone();
	int iDepth = pnInBuf.getDepth();

	for( int i = 0; i < iDepth; i++ )
	{
	    nnNode = pnInBuf.getNode(i);
	    if( nnNode.getNodeName().equals(PathNexus.PARENT_NODE) )
		pnOutBuf.popNode();
	    else
		pnOutBuf.pushNode(nnNode);
	}

	return pnOutBuf;
    }
}

