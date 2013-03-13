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

import java.util.HashMap;
import java.util.Map;

import org.cdma.utilities.performance.Buffer;

/**
 * @brief The NexusBufferManager centralize navigation buffers to their file handler.
 * 
 * The NexusBufferManager associate a BufferNode (that is the navigation buffer) to its
 * corresponding a NexusFileBrowser.
 * <p>
 * It permits to share in navigation buffers of a same file between several NexusFileBrowser
 * that handle it.
 * 
 * @author rodriguez
 *
 */

public class NexusBufferManager {
	private final Map<String, Buffer<PathNexus, Attribute>> mAttributes;
    private final Map<String, Buffer<PathNexus, NexusNode>> mNodes;
    private final static NexusBufferManager mManager;

    static {
    	synchronized( NexusBufferManager.class ) {
    		mManager = new NexusBufferManager();
    	}
    }

    /**
     * Return the buffer of nodes corresponding to the given file
     * 
     * @param file browser from which we want the buffer
     */
    public static Buffer<PathNexus, NexusNode> getBufferNode(NexusFileBrowser file) {
    	Buffer<PathNexus, NexusNode> result;
    	synchronized (mManager.mNodes) {
    		result = mManager.mNodes.get(file.getFilePath());
    		if( result == null ) {
    			result = new Buffer<PathNexus, NexusNode>(100, new NexusNode.NodeCollator() );
    			mManager.mNodes.put(file.getFilePath(), result);
    		}
    	}
        return result;
    }
    
    /**
     * Return the buffer of attributes corresponding to the given file
     * 
     * @param file browser from which we want the buffer
     */
    public static Buffer<PathNexus, Attribute> getBufferAttribute(NexusFileBrowser file) {
    	Buffer<PathNexus, Attribute> result;
    	synchronized (mManager.mAttributes) {
	        result = mManager.mAttributes.get(file.getFilePath());
	        if( result == null ) {
	            result = new Buffer<PathNexus, Attribute>(200);
	            mManager.mAttributes.put(file.getFilePath(), result);
	        }
    	}
        return result;
    }
    
    private NexusBufferManager() {
        mNodes = new HashMap<String, Buffer<PathNexus, NexusNode> >();
        mAttributes = new HashMap<String, Buffer<PathNexus,Attribute>>();
    }
}

