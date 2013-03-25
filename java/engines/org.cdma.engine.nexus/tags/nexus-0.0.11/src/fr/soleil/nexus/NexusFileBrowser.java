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

// Tools lib
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Hashtable;
import java.util.List;
import java.util.Map.Entry;

import ncsa.hdf.hdflib.HDFNativeData;

import org.cdma.utilities.performance.Buffer;
import org.nexusformat.AttributeEntry;
import org.nexusformat.NXlink;
import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

public class NexusFileBrowser extends NexusFileInstance {

    // Member attributes
    // members concerning current path in file
    private PathNexus m_pVirtualPath; // Current path into the Nexus File (visible one by the user)
    private PathNexus m_pRealPath;    // Current path into the Nexus File (full real path)
    private boolean m_bFileChanged; // True if the file has been changed since last open

    // members concerning nodes' buffer
    private Collection<NexusNode> m_tNodeTab; // TreeMap of all node belonging to last listed group's children (node name => node class)
    
    private boolean m_autoOpen;

    // Constructors
    public NexusFileBrowser() {
        super();
        m_pVirtualPath = new PathNexus();
        m_pRealPath = new PathNexus();
        m_tNodeTab = new ArrayList<NexusNode>();
        m_bFileChanged = true;
        m_autoOpen = false;
    }

    public NexusFileBrowser(String sFilePath) {
        super(sFilePath);
        initPath(sFilePath);
        m_tNodeTab = new ArrayList<NexusNode>();
        m_bFileChanged = false;
        m_autoOpen = false;
    }

    /**
     * Reset all information stored into that buffer 
     */
    public void resetBuffer() {
        getBufferNode().resetBuffer();
    }
    
    /**
     * Set the current maximum size of the node buffer (in number of slots)
     * 
     * @param iSize new number of available slots in the node buffer
     */
    public void setBufferSize(int iSize) {
        getBufferNode().setBufferSize(iSize);
    }
    
    /**
     * Get the current maximum size of the node buffer (in number of slots)
     */
    public int getBufferSize() {
        return getBufferNode().getBufferSize();
    }

    /**
     * getCurrentPath return the current visible path inside the opened Nexus
     * file
     */
    public PathNexus getCurrentPath() {
        return m_pVirtualPath;
    }

    /**
     * openFile Set specified file as the current one and open it with
     * appropriate access mode
     */
    @Override
    protected void openFile(String sFilePath, int iAccessMode) throws NexusException {
        if (m_bFileChanged) {
            initPath(sFilePath);
        }
        super.openFile(sFilePath, iAccessMode);
    }

    /**
     * setFile Set specified file as the current one
     */
    @Override
    public void setFile(String sFilePath) {
        m_bFileChanged = true;
        m_pRealPath.clearNodes();
        m_pVirtualPath.clearNodes();
        initPath(sFilePath);
        super.setFile(sFilePath);
    }

    /**
     * finalize Free resources
     */
    @Override
    protected void finalize() throws Throwable {
        m_pVirtualPath = null;
        super.finalize();
    }

    // ---------------------------------------------------------
    // / Navigation through nodes
    // ---------------------------------------------------------
    /**
     * openPath opens groups and DataItems according to the path (PathNexus) in
     * the opened file given. All objects of the path must exist.
     * 
     * @param paPath The path string
     */
    public void openPath(PathNexus pnPath) throws NexusException {
        if (!pnPath.isRelative()) {
            closeAll();
        }

        // for each node of the path
        boolean opened;
        for (NexusNode pattern : pnPath.getNodes()) {
            opened = false;

            // for each child
            ArrayList<NexusNode> child = listChildren();
            for (NexusNode node : child) {
                // node comparison
                if (node.matchesPartNode(pattern)) {
                    opened = true;
                    openNode(node);
                    break;
                }
            }

            // No node opened throw exception
            if (!opened) {
                throw new NexusException("Path invalid path in file: " + pnPath.toString() + "\nfailed at: " + m_pVirtualPath.toString());
            }
        }
    }

    /**
     * openGroup opens the group name with class nxclass. The group must exist,
     * otherwise an exception is thrown. opengroup is similar to a cd name in a
     * filesystem.
     * 
     * @param name the name of the group to open.
     * @param nodeClass the classname of the group to open.
     * @param immediate try to open it now
     * @note this method is case insensitive for the node's name
     */
    protected void openGroup(String nodeName, String nodeClass) throws NexusException {
    	openGroup(nodeName, nodeClass, false);
    }
    
    protected void openGroup(String nodeName, String nodeClass, boolean immediate) throws NexusException {
    	String name = nodeName;
    	if( name == null ) {
    		name = "";
    	}
        if (name.equals(PathNexus.PARENT_NODE)) {
        	m_pVirtualPath.popNode();
        }
        else {
	        String sItemName = name;
	        String sItemClass = nodeClass;
	
	        // Ensure a class has been set
	        if (sItemClass.trim().equals("")) {
	            sItemClass = getNodeClassName(sItemName);
	        }
	
	        // Try to open requested node as it was given
	    	boolean bIsGroup = !(sItemClass.equals("SDS") || sItemClass.equals("NXtechnical_data"));
	        m_pVirtualPath.pushNode(new NexusNode(sItemName, sItemClass, bIsGroup));
        }
    }

    /**
     * openData opens the DataItem name with class SDS. The DataItem must exist,
     * otherwise an exception is thrown. opendata is similar to a 'cd' command
     * in a file system.
     * 
     * @param sNodeName the name of the group to open.
     * @note the pattern ".." means close current node
     */
    protected void openData(String sNodeName) throws NexusException {
    	if( sNodeName != null && ! sNodeName.isEmpty() ) {
	        if (sNodeName.equals(PathNexus.PARENT_NODE)) {
	        	m_pVirtualPath.popNode();
	        }
	        else {
	        	m_pVirtualPath.pushNode(new NexusNode(sNodeName, "", false));
	        }
    	}
    }

    /**
     * openNode Open the requested node of the currently opened file.
     * 
     * @param nnNode node to be opened
     */
    public void openNode(final NexusNode nnNode) throws NexusException {
        if (nnNode == null) {
            throw new NexusException("Invalid node to open: can't open a null node!");
        }
        String sNodeName = nnNode.getNodeName();
        String sNodeClass = nnNode.getClassName();
        
        // If the node is PARENT_NODE
        if (PathNexus.PARENT_NODE.equals(sNodeName)) {
        	// Pop the last node
        	m_pVirtualPath.popNode();
        }
        else {
	        // If the node is a group
	        if (nnNode.isGroup() || nnNode.isRealGroup() ) {
	        	// Check node name isn't empty
	            if (!"".equals(sNodeName)) {
	            	// Check class isn't empty
	                if ("".equals(sNodeClass)) {
	                    sNodeClass = getNodeClassName(sNodeName);
	                }
	            }
	            // Search the first group having the same class
	            else {
	                boolean found = false;
	                for( NexusNode node : listChildren() ) {
	                    if( node.getClassName().equals(sNodeClass) ) {
	                        found = true;
	                        sNodeName = node.getNodeName();
	                        break;
	                    }
	                }
	                if( ! found ) {
	                    throw new NexusException("Failed to open node: " + nnNode.toString());
	                }
	            }
	        }
	        // Create a new fully defined corresponding group
	        NexusNode node = new NexusNode(sNodeName, sNodeClass);
	        
	        
	        boolean found = false;
            for( NexusNode tmp : listChildren() ) {
                if( tmp.matchesPartNode(node) ) {
                    found = true;
                    break;
                }
            }
            if( ! found ) {
                throw new NexusException("Failed to open node: " + nnNode.toString());
            }
        	m_pVirtualPath.pushNode( node );
        }
    }

    /**
     * closeGroup Close currently opened group, and return to parent node
     * 
     * @throws NexusException
     */
    public void closeGroup() throws NexusException {
    	m_pVirtualPath.popNode();
    }

    /**
     * closeData Close currently opened DataItem, and return to parent node
     * 
     * @throws NexusException
     */
    public void closeData() throws NexusException {
        m_pVirtualPath.popNode();
    }

    /**
     * closeAll Close every opened DataItem and/or groups to step back until the
     * Nexus file root is reached
     * 
     * @note the NeXus file is kept opened
     */
    public void closeAll() throws NexusException {
        // Clearing current path
        m_pVirtualPath.setPath(new NexusNode[0]);
    }

    // ---------------------------------------------------------
    // / Nodes informations
    // ---------------------------------------------------------
    /**
     * getNodeClassName Return the class name of a specified node in currently
     * opened group
     * 
     * @param sNodeName name of the node from which we want to know the class name
     * @throws NexusException if no corresponding node was found
     */
    protected String getNodeClassName(String sNodeName) throws NexusException {
        // Parse children
        String sItemName = sNodeName.toUpperCase();
        listGroupChild();

        for( NexusNode node : listChildren() ) {
            // Check if names are equals
            if (sItemName.equals(node.getNodeName().toUpperCase())) {
                return node.getClassName();
            }
        }
        throw new NexusException("NexusNode not found: " + sNodeName);
    }

    /**
     * getNode Create a new NexusNode by scanning child of the current path
     * according to given node name
     */
    public NexusNode getNode(String sNodeName) throws NexusException {
        if (sNodeName == null || sNodeName.trim().equals(""))
            return null;

        // Parse children
        String sItemName = NexusNode.extractName(sNodeName).toUpperCase();
        String sItemClass = NexusNode.extractClass(sNodeName).toUpperCase();
        String sCurName, sCurFullName, sCurClass;
        listGroupChild();

        for( NexusNode node : listChildren() ) {
            // Check if names are equals
            sCurName = node.getNodeName();
            sCurClass = node.getClassName();
            sCurFullName = NexusNode.getNodeFullName(sCurName, sCurClass).toUpperCase();
            if (sItemName.equals(sCurFullName) || sItemName.equals(sCurName.toUpperCase())
                    || (sItemClass.equals(sCurClass.toUpperCase()) && sItemName.equals(""))) {
                return node;
            }
        }
        throw new NexusException("NexusNode not found: " + sNodeName);
    }

    /**
     * isOpenedDataItem Return true if the opened item is a DataItem
     * 
     * @param sNodeName name of the node from which we want to know the class name
     */
    public boolean isOpenedDataItem() {
        return m_pRealPath.getDataItemName() != null;
    }

    /**
     * tryGuessNodeName Try to find a node corresponding to given name. The
     * method will try name, then replace '/' by '__', then will try to add '__'
     * in end of name, or try adding '__#1'... The method will return first
     * matching pattern
     * 
     * @param sNodeName approximative node name
     * @return a string array containing the write name of a node as first
     *         element and classname as second element
     */
    protected String[] tryGuessNodeName(String sNodeName) throws NexusException {
        String[] sFoundNode = { "", "" };
        String sTmpName = sNodeName;
        try { // Try to find requested name
            sFoundNode[1] = getNodeClassName(sTmpName);
        } catch (NexusException ne1) {
            try { // don't succeed, so we try replacing "/" by "__"
                sTmpName = sTmpName.replace("/", "__");
                sFoundNode[1] = getNodeClassName(sTmpName);
            } catch (NexusException ne2) {
                try { // don't succeed: trying adding "#1" at end of name
                    sTmpName += "#1";
                    sFoundNode[1] = getNodeClassName(sTmpName);
                } catch (NexusException ne3) {
                    try { // don't succeed: trying with "__#1" at end of name
                        sTmpName = sTmpName.substring(0, sTmpName.length() - 2) + "__#1";
                        sFoundNode[1] = getNodeClassName(sTmpName);
                    } catch (NexusException ne4) {
                        // don't succeed, no more good idea... we send an
                        // Exception
                        throw new NexusException("NexusNode name not found: " + sNodeName + "!");
                    }
                }
            }
        }
        sFoundNode[0] = sTmpName;
        return sFoundNode;
    }

    protected NXlink getNXlink() throws NexusException {
        NXlink nlLink = null;
        openFile();
        if (isOpenedDataItem()) {
            nlLink = getNexusFile().getdataID();
        }
        else {
            nlLink = getNexusFile().getgroupID();
        }
        closeFile();
        return nlLink;
    }

    // ---------------------------------------------------------
    // / Browsing nodes
    // ---------------------------------------------------------
    /**
     * listChildren List all direct descendants of the node ending the given
     * path and returns it as a list of NexusNode
     * 
     * @throws NexusException
     */
    public NexusNode[] listChildren(PathNexus pnPath) throws NexusException {
        // Open the requested node
        openPath(pnPath);

        // Get all its descendants
        ArrayList<NexusNode> list = listChildren();
        NexusNode[] nodes = list.toArray(new NexusNode[list.size()]);

        // Return to document root
        closeAll();

        return nodes;
    }

    /**
     * listChildren List all direct descendants of an opened node and returns it
     * as a list of NexusNode
     * 
     * @throws NexusException
     */
    public ArrayList<NexusNode> listChildren() throws NexusException {
        ArrayList<NexusNode> nodes = new ArrayList<NexusNode>();
        nodes.addAll(listGroupChild());

        return nodes;
    }

    // ---------------------------------------------------------
    // / Link nodes
    // ---------------------------------------------------------
    /**
     * listAttribute List all attributes of the currently opened node and store
     * it into a hashtable member variable.
     * 
     * @note A Hashtable which will hold the names of the attributes as keys.
     *       For each key there is an AttributeEntry class as value.
     */
    @SuppressWarnings("unchecked")
    public Collection<Attribute> listAttribute() throws NexusException {
    	// Check we have to update the list
    	Collection<Attribute> attributes = getBufferAttribute().get(m_pVirtualPath);
    	if( attributes == null ) {
            // List attribute name and their properties
            openFile();
            Hashtable<String, AttributeEntry> attrTab = getNexusFile().attrdir();
            closeFile();
            
            // Create as many attribute
            Attribute attribute;
            attributes = new ArrayList<Attribute>();
            for( Entry<String, AttributeEntry> entry : attrTab.entrySet() ) {
            	attribute = new Attribute(
            						entry.getKey(), 
            						entry.getValue().length,
            						entry.getValue().type,
            						null
            						);
            	attributes.add( attribute );
            }
            // Add them to the buffer
            getBufferAttribute().push(m_pVirtualPath.clone(), attributes, attributes.size());
            attributes = getBufferAttribute().get(m_pVirtualPath);
        }
        return attributes;
    }

    public static String getStringValue(Byte[] reference) {
        byte[] toTransform = null;
        if (reference == null) {
            return "";
        } else {
            toTransform = new byte[reference.length];
        }
        for (int i = 0; i < reference.length; i++) {
            if (reference[i] == null) {
                toTransform[i] = (byte) 0;
            } else {
                toTransform[i] = reference[i].byteValue();
            }
        }

        return new String(toTransform);
    }

    protected NexusFileHandler getNexusFile() throws NexusException {
    	NexusFileHandler handler = super.getNexusFile();
    	
    	if( handler != null ) {
    		List<NexusNode> nodes = m_pRealPath.getShortestWayTo(m_pVirtualPath);
    		for( NexusNode node : nodes ) {
    			if( node != null ) {
    				physicallyOpenNode(node);
    			}
    		}
    	}
    	return handler;
    }
    
	// ---------------------------------------------------------
    // ---------------------------------------------------------
    // / Private methods
    // ---------------------------------------------------------
    // ---------------------------------------------------------
    /**
     * initPath Split the given string to initialize the member NexusPath
     * 
     * sFilePath file path to init
     */
    private void initPath(String sFilePath) {
        sFilePath = sFilePath.replace(File.separator, NexusFileInstance.PATH_SEPARATOR);
        m_pVirtualPath = new PathNexus();
        m_pVirtualPath.setFile(sFilePath);

        m_pRealPath = new PathNexus();
        m_pRealPath.setFile(sFilePath);

        m_bFileChanged = false;

    }

    protected Object getAttribute(String sAttrName) throws NexusException {
        int[] iAttrInf = { 0, 0 };

        // Get the list of attributes
        Collection<Attribute> attributes = listAttribute();
        
        // Seek the given one
        Attribute attribute = null;
        for( Attribute tmp : attributes ) {
        	if( tmp.name.equals(sAttrName) ) {
        		attribute = tmp;
        		break;
        	}
        }
        
        // If not found
        if (attribute == null ) {
            throw new NexusException("No corresponding attribute found: " + sAttrName + "!");
        }
        
        // Get its value
        Object result = attribute.value;
        
        // If not already loaded
        if( result == null ) {
            // Get infos on attribut
	        iAttrInf[0] = attribute.length;
	        iAttrInf[1] = attribute.type;
	
	        // Initialize an array of proper type with enough space to store
	        // attribute value
	        result = HDFNativeData.defineDataObject(iAttrInf[1], iAttrInf[0] + (iAttrInf[1] == NexusFile.NX_CHAR ? 1 : 0));
	
	        // Get attribute value
	        openFile();
	        getNexusFile().getattr(sAttrName, result, iAttrInf);
	        closeFile();
	        // Convert bytes array (representing chars) to string
	        if (iAttrInf[1] == NexusFile.NX_CHAR) {
	        	result = new String((byte[]) result);
	        	result = ((String) result).substring(0, ((String) result).length() - 1).toCharArray();
	        }
	        attribute.value = result;
        }
        return result;
    }

    /**
     * listGroupChild Stores in a buffer all children of the currently opened
     * group
     * 
     * @note this method is here to avoid JVM crash due to Nexus API (in version 4.2.0)
     * @throws NexusException
     */
    @SuppressWarnings("unchecked")
    private Collection<NexusNode> listGroupChild() throws NexusException {
        if (!isListGroupChildUpToDate()) {
        	m_tNodeTab = new ArrayList<NexusNode>();
        	try {
    		// Case we are in a group
            if (m_pVirtualPath.getGroupsName() != null) {
                Long time = System.currentTimeMillis();
                m_tNodeTab = new ArrayList<NexusNode>();
                openFile();
                Hashtable<String, String> map = getNexusFile().groupdir();
                closeFile();
                boolean bIsGroup;
                for( Entry<String, String> entry : map.entrySet() ) {
                    bIsGroup = !(entry.getValue().equals("SDS") || entry.getValue().equals("NXtechnical_data"));
                    m_tNodeTab.add( new NexusNode( entry.getKey(), entry.getValue(), bIsGroup ) );
                }
                time = System.currentTimeMillis() - time;
                getBufferNode().push(m_pVirtualPath.clone(), m_tNodeTab, time.intValue());
                m_tNodeTab = getBufferNode().get(m_pVirtualPath);
            }
        	}catch( NexusException e ) {
        		try {
        			closeFile();
        		} catch( NexusException ce ) {}
        		throw e;
        	}
        } else {
            m_tNodeTab = new ArrayList<NexusNode>(getBufferNode().get(m_pVirtualPath));
        }
        
        return m_tNodeTab;
    }

    /**
     * isListGroupChildUpToDate return true if the children buffer node list has
     * to be updated
     */
    private boolean isListGroupChildUpToDate() {
        return getBufferNode().get(m_pVirtualPath) != null;
    }
    
    protected void pushNodeInBuffer(String sCurName, String sCurClass) {
        boolean bIsGroup = !(sCurClass.equals("SDS") || sCurClass.equals("NXtechnical_data"));
        NexusNode node = new NexusNode( sCurName, sCurClass, bIsGroup );
        m_tNodeTab.add( node );
        getBufferNode().push(m_pVirtualPath.clone(), node, 1);
    }
    
    private Buffer<PathNexus, NexusNode> getBufferNode() {
        return NexusBufferManager.getBufferNode(this);
    }
    
    private Buffer<PathNexus, Attribute> getBufferAttribute() {
        return NexusBufferManager.getBufferAttribute(this);
    }

    // ------------------------------------------------------------------------
    // new methods
    // ------------------------------------------------------------------------
    private void physicallyCloseNode() throws NexusException {
        NexusNode node = m_pRealPath.popNode();
        if (node != null && !node.getClassName().equals("NXtechnical_data") && !node.isGroup()) {
            super.getNexusFile().closedata();
        } else if (node != null) {
        	super.getNexusFile().closegroup();
        }
    }
    
    private void physicallyOpenNode(NexusNode node) throws NexusException {
    	if (node == null) {
            throw new NexusException("Invalid node to open: can't open a null node!");
        }
    	
        String sNodeName = node.getNodeName();
        String sNodeClass = node.getClassName();
    	
    	if( sNodeName.equals(PathNexus.PARENT_NODE ) ) {
    		physicallyCloseNode();
    	}
    	else {
	        // Open the requested node
	        if ( node.isRealGroup() ) {
	        	super.getNexusFile().opengroup(sNodeName, sNodeClass);
	        }
	        // Open the DataItem
	        else {
	        	super.getNexusFile().opendata(sNodeName);
	        }
	        m_pRealPath.pushNode(node);
    	}
	}
    
    @Override
    public void open() {
    	super.open();
    	try {
			closeAll();
		} catch (NexusException e) {
		}
    	m_autoOpen = true;
    }
    
    @Override
	public void close() {
		m_autoOpen = false;
		try {
			closeFile();
		} catch (Exception e) {
		}
		super.close();
	}
    
    @Override
    protected void closeFile() throws NexusException {
		if (isFileOpened() && !m_autoOpen ) {
			try {
				while (m_pRealPath.getCurrentNode() != null) {
					physicallyCloseNode();
				}
			} catch (NexusException e) {
			}
			super.closeFile();
		}
    }
    
    @Override
    protected void openFile() throws NexusException {
    	if( ! m_autoOpen ) {
    		throw new NexusException("No file opened!");
    	}
    	else if( ! isFileOpened() ) {
    		super.openFile();
    	}
    }
}