package fr.soleil.nexus;

import java.util.HashMap;
import java.util.Map;

public class BufferNodeManager {
    private Map<String, BufferNode> mBuffers;
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
