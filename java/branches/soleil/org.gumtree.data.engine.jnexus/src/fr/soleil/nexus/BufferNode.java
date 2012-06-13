package fr.soleil.nexus;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;

import org.gumtree.data.util.performance.Benchmarker;

import fr.soleil.nexus.NexusNode.NameCollator;

public class BufferNode {
   
    // Maximum size of the buffer (number of available slots)
    private int m_iBufferSize; 
    
    // TreeMap of all node belonging to last listed group's children (node name => node class)
 //   private Map<String, String> m_tNodeTab;
    
    // TreeMap containing all node for a specific path (path in file => [node name => node class])
    private final Map<String, TreeMap<String, NexusNode>> m_tNodeInPath;
    
    // TreeMap containing path usage count (used to remove less used path when cleaning buffer)
    private final Map<String, Integer> m_hPathUsageWeigth;
    
    public BufferNode(int size) {
        m_iBufferSize      = size;
        m_tNodeInPath      = new HashMap<String, TreeMap<String, NexusNode> >();
        m_hPathUsageWeigth = new HashMap<String, Integer>();
    }

    /**
     * Returns the current maximum size of the node buffer (in number of slots)
     */
    public int getBufferSize() {
        return m_iBufferSize;
    }
    
    /**
     * Set the current maximum size of the node buffer (in number of slots)
     * 
     * @param iSize new number of available slots in the node buffer
     */
    public void setBufferSize(int iSize) {
        if (iSize > 10)
            m_iBufferSize = iSize;
    }
    
    protected void pushNodeInPath(PathNexus path, NexusNode node) {
        pushNodeInPath(path, node, 1);
    }

    private void pushNodeInPath(PathNexus path, NexusNode node, int iTimeToAccessNode) {
        TreeMap<String, NexusNode> tmpSet = m_tNodeInPath.get(path.toString());
        if (tmpSet == null) {
            tmpSet = new TreeMap<String, NexusNode> (new NameCollator());
            tmpSet.put( node.getNodeName(), node );
            putNodesInPath(path, tmpSet, iTimeToAccessNode);
        }
        else {
            tmpSet.put( node.getNodeName(), node );
        }
    }
    
    protected void putNodesInPath(PathNexus path, ArrayList<NexusNode> nodes, int iSpentTime) {
        TreeMap<String, NexusNode> tmpSet = m_tNodeInPath.get(path.toString());
        if (tmpSet == null) {
            tmpSet = new TreeMap<String, NexusNode> (new NameCollator());
        }
        for( NexusNode node : nodes ) {
            tmpSet.put(node.getNodeName(), node);
        }
        putNodesInPath(path, tmpSet, iSpentTime);
    }
    
    

    private void putNodesInPath(PathNexus path, TreeMap<String, NexusNode> tmNodes, int iTimeToAccessNode) {
        freeBufferSpace();

        Integer value = m_hPathUsageWeigth.get(path.toString());
        if (value == null)
            value = iTimeToAccessNode;
        else
            value += iTimeToAccessNode + 1;
        m_hPathUsageWeigth.put(path.toString(), value);

        m_tNodeInPath.put(path.toString(), tmNodes);
    }

    /**
     * getNodeInPath Return the buffered map of the node's names and node's
     * class for the current path
     */
    protected Collection<NexusNode> getNodeInPath(PathNexus path) {
        Integer value = m_hPathUsageWeigth.get(path.toString());
        if (value == null) {
            value = 1;
        }
        else {
            value++;
        }
        m_hPathUsageWeigth.put(path.toString(), value);

        
        if( m_tNodeInPath.containsKey(path.toString()) ) {
            Collection<NexusNode> result = m_tNodeInPath.get(path.toString()).values();
            return result;
        }
        else {
            return null;
        }
    }

    private void freeBufferSpace() {
        if (m_tNodeInPath.size() > m_iBufferSize) {
            int iNumToRemove = (m_iBufferSize / 2), iRemovedItem = 0, iInfLimit;
            Object[] frequency = m_hPathUsageWeigth.values().toArray();
            java.util.Arrays.sort(frequency);
            iInfLimit = (Integer) frequency[frequency.length / 2];
            Iterator<String> keys_iter = m_hPathUsageWeigth.keySet().iterator();
            int freq;
            String key;
            while (keys_iter.hasNext() && iRemovedItem < iNumToRemove) {
                key = keys_iter.next();
                freq = m_hPathUsageWeigth.get(key);

                if (freq <= iInfLimit) {
                    keys_iter.remove();
                    m_tNodeInPath.remove(key);
                    iRemovedItem++;
                }
            }
        }
    }
}
