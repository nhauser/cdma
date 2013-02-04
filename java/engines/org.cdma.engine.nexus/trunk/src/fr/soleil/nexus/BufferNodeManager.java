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

/**
 * @brief The BufferNodeManager centralize navigation buffers to their file handler.
 * 
 * The BufferNodeManager associate a BufferNode (that is the navigation buffer) to its
 * corresponding a NexusFileBrowser.
 * <p>
 * It permits to share in navigation buffers of a same file between several NexusFileBrowser
 * that handle it.
 * 
 * @author rodriguez
 *
 */

public class BufferNodeManager {
    private final Map<String, BufferNode> mBuffers;
    private static BufferNodeManager mManager;


    /**
     * Return the buffer corresponding to the given file
     * 
     * @param file browser from which we want the buffer
     */
    public static BufferNode getBuffer(NexusFileBrowser file) {
        if( mManager == null ) {
            mManager = new BufferNodeManager();
        }

        BufferNode result = mManager.mBuffers.get(file.getFilePath());
        if( result == null ) {
            result = new BufferNode(100);
            mManager.mBuffers.put(file.getFilePath(), result);
        }
        return result;
    }

    private BufferNodeManager() {
        mBuffers = new HashMap<String, BufferNode>();
    }
}

