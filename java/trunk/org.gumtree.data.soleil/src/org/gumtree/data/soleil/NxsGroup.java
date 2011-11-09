package org.gumtree.data.soleil;

// Standard import
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.impl.Key;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.exception.SignalNotAvailableException;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IDimension;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.utils.Utilities.ModelType;
import org.nexusformat.AttributeEntry;
import org.nexusformat.NexusException;

import fr.soleil.nexus4tango.DataSet;
import fr.soleil.nexus4tango.NexusFileWriter;
import fr.soleil.nexus4tango.NexusNode;
import fr.soleil.nexus4tango.PathData;
import fr.soleil.nexus4tango.PathGroup;
import fr.soleil.nexus4tango.PathNexus;

public class NxsGroup implements IGroup {
    /// Members
    // API CDM tree need
    NxsDataSet          m_dataset;        // File handler
    IGroup              m_parent = null;  // Parent group
    List<IContainer>    m_child;          // Children nodes (group, dataitem...)

    // Internal members
    PathNexus       	m_n4tCurPath;	  // Current path
    IDictionary         m_dictionary;     // Group dictionary
    List<IAttribute>    m_attributes;     // Attributes belonging to this
    List<IDimension>    m_dimensions;     // Dimensions direct child of this

    

    /// Constructors
    public NxsGroup(IGroup parent, PathNexus from, NxsDataSet dataset)
    {
        m_dictionary    = null;
        m_n4tCurPath     = from;
        m_dataset        = dataset;
        m_child          = new ArrayList<IContainer>();
        m_attributes     = new ArrayList<IAttribute>();
        m_dimensions     = new ArrayList<IDimension>();
        setParent(parent);
    }

    public NxsGroup(PathNexus from, NxsDataSet dataset)
    {
        m_dictionary    = null;
        m_n4tCurPath     = from;
        m_dataset        = dataset;
        m_child          = new ArrayList<IContainer>();
        m_attributes     = new ArrayList<IAttribute>();
        m_dimensions     = new ArrayList<IDimension>();
        if( from != null && dataset != null ) {
            createFamilyTree();
        }
    }

    public NxsGroup(NxsGroup group)
    {
        m_n4tCurPath	 = group.m_n4tCurPath.clone();
        m_dataset	     = group.m_dataset.clone();
        m_parent         = group.m_parent;
        m_child          = new ArrayList<IContainer>(group.m_child);
        m_attributes     = new ArrayList<IAttribute>(group.m_attributes);
        m_dimensions     = new ArrayList<IDimension>(group.m_dimensions);
        try {
            m_dictionary   = (IDictionary) group.m_dictionary.clone();
        } catch( CloneNotSupportedException e ) {
            m_dictionary = null;
        }
    }

    /**
     * Return a clone of this IGroup object.
     * @return new IGroup
     * Created on 18/09/2008
     */
    @Override
    public NxsGroup clone()
    {
        return new NxsGroup(this);
    }

	@Override
	public ModelType getModelType() {
		return ModelType.Group;
	}
    
    @Override
    public boolean isEntry() {
        return ( m_parent.getParentGroup().getParentGroup() == null );
    }
    
    @Override
    public void addDataItem(IDataItem v) {
        v.setParent(this);
        setChild(v);
    }

    @Override
    public void addOneAttribute(IAttribute attribute) {
        m_attributes.add(attribute);
    }

    @Override
    public void addOneDimension(IDimension dimension) {
        m_dimensions.add(dimension);
    }

    @Override
    public void addStringAttribute(String name, String value) {
        IAttribute attr = new NxsAttribute(name, value);
        m_attributes.add(attr);
    }

    @Override
    public void addSubgroup(IGroup g) {
        g.setParent(this);
    }

    @Override
    public IAttribute getAttribute(String name) {
        for( IAttribute attr : m_attributes ) {
            if( attr.getName().equals(name) )
                return attr;
        }

        return null;
    }


    @Override
    public IDataItem findDataItem(String shortName) {
        IDataItem resItem = null;

    	List<IDataItem> groups = getDataItemList();
    	for( Iterator<?> iter = groups.iterator(); iter.hasNext(); )
    	{
    		resItem = (IDataItem) iter.next();
    		if( resItem.getShortName().equals(shortName) )
    		{
    			groups.clear();
    			return resItem;
    		}
    	}

    	return null;
    }

    
    @Override
    public IDimension getDimension(String name) {
        for( IDimension dim : m_dimensions ) {
            if( dim.getName().equals(name) )
                return dim;
        }

        return null;
    }

    @Override
    public IGroup findGroup(String shortName) {
        NxsGroup resGroup = null;
        NxsGroup tmpGroup = null;

        if( shortName != null && shortName.trim().equals("") )
            return this;

        List<IGroup> groups = getGroupList();
        for( Iterator<?> iter = groups.iterator(); iter.hasNext(); )
        {
            tmpGroup = (NxsGroup) iter.next();
            if( tmpGroup.m_n4tCurPath.getCurrentNode().toString().equals(shortName) )
            {
                resGroup = tmpGroup;
                groups.clear();
                return resGroup;
            }
        }

        return resGroup;
    }

    @Override
    public IGroup getGroupWithAttribute(String attributeName, String value) {
        List<IGroup> groups = getGroupList();
        IAttribute attr;
        for( IGroup group : groups ) {
            attr = group.getAttribute(attributeName);
            if( attr.getStringValue().equals(value) )
                return group;
        }
        
        return null;
    }

    @Override
    public IGroup findGroupWithAttribute(IKey key, String name, String value) {
    	List<IContainer> found = findAllOccurrences(key);
    	IGroup result = null;
    	for( IContainer item : found ) {
    		if( 
    			item.getModelType().equals(ModelType.Group) && 
    			item.hasAttribute( name, value) 
    		) {
    			result = (IGroup) item;
    			break;
    		}
    		
    	}
        return result;
    }

    @Override
    public List<IAttribute> getAttributeList() {
        Hashtable<String, AttributeEntry> inList;
        ArrayList<IAttribute> outList = new ArrayList<IAttribute>();
        NxsAttribute tmpAttr;
        String sAttrName;


        try
        {
            m_dataset.getHandler().openPath(m_n4tCurPath);
            inList = m_dataset.getHandler().listAttribute();

            Iterator<String> iter = inList.keySet().iterator();
            while( iter.hasNext() )
            {
                sAttrName = iter.next();
                try
                {
                    tmpAttr = new NxsAttribute(sAttrName, m_dataset.getHandler().readAttr(sAttrName, null));
                    outList.add(tmpAttr);
                }
                catch (NexusException e)
                {
                    e.printStackTrace();
                }
            }
        }
        catch(NexusException ne)
        {
            try
            {
                m_dataset.getHandler().closeAll();
            }
            catch (NexusException e) {}
            return outList;
        }

        return outList;
    }
    
    @Override
    public IDataItem getDataItemWithAttribute(String name, String value) {
        IDataItem resItem = null;

    	List<IDataItem> groups = getDataItemList();
    	for( Iterator<?> iter = groups.iterator(); iter.hasNext(); )
    	{
    		resItem = (IDataItem) iter.next();
    		if( resItem.hasAttribute(name, value) )
    		{
    			groups.clear();
    			return resItem;
    		}
    	}

    	return null;
    }

    @Override
    public IDataItem findDataItemWithAttribute(IKey key, String name, String value) throws Exception {
    	List<IContainer> found = findAllOccurrences(key);
    	IDataItem result = null;
    	for( IContainer item : found ) {
    		if( 
    			item.getModelType().equals(ModelType.DataItem) && 
    			item.hasAttribute( name, value) 
    		) {
    			result = (IDataItem) item;
    			break;
    		}
    		
    	}
        return result;
    }
    
    @Override
    public IDataItem getDataItem(String shortName) {
        List<IDataItem> items = getDataItemList();
        for( IDataItem item : items ) {
            if( item.getShortName().equals(shortName) )
                return item;
        }
        
        return null;
    }
    
    @Override
    public IDataItem findDataItem(IKey key) {
        IDataItem item = null;
        List<IContainer> list = new ArrayList<IContainer>();
        list = findAllOccurrences(key);
        
        for( IContainer object : list ) {
        	if( object.getModelType().equals(ModelType.DataItem) ) {
        		item = (IDataItem) object;
        		break;
        	}
        }
        
        return item;
    }
    
    @Override
    public List<IDataItem> getDataItemList() {
    	List<IContainer> listItem = getGroupNodes(false);
		if (listItem == null) {
			return null;
		}

		List<IDataItem> dataItemList = new ArrayList<IDataItem>();
		for (IContainer variable : listItem) {
			dataItemList.add((IDataItem) variable);
		}
        
        for (IContainer variable : m_child) {
        	if( variable.getModelType().equals(ModelType.DataItem) ) {
        		dataItemList.add((IDataItem) variable);
        	}
        }
		return dataItemList;
    }

    @Override
    public IDataset getDataset() {
        return (IDataset) m_dataset;
    }

    @Override
    public List<IDimension> getDimensionList() {
        return m_dimensions;
    }
  
    @Override
    public IGroup findGroup(IKey key) {
        IGroup group = null;
        List<IContainer> list = new ArrayList<IContainer>();
        list = findAllOccurrences(key);
        
        for( IContainer object : list ) {
        	if( object.getModelType().equals(ModelType.Group) ) {
        		group = (IGroup) object;
        		break;
        	}
        }

        return group;
    }
    
    @Override
    public IGroup getGroup(String shortName) {
    	NexusNode node = PathNexus.splitStringToNode(shortName)[0];
        List<IGroup> groups = getGroupList();
        for( IGroup group : groups ) {
            if( group.getShortName().equals(shortName) || 
            	(
            		node.getNodeName().equals("") && 
            		node.getClassName().equals( ((NxsGroup) group).getClassName() ) 
            	)
            ) {
                return group;
            }
        }
        
        return null;
    }

    @Override
    public List<IGroup> getGroupList() {
    	List<IContainer> listItem = getGroupNodes(true);
		if (listItem == null) {
			return null;
		}

        List<IGroup> dataItemList = new ArrayList<IGroup>();
		for (IContainer variable : listItem) {
            if( ! m_child.contains(variable) ) {
                m_child.add((IGroup) variable);
            }
            dataItemList.add((IGroup) variable);
		}
		return dataItemList;
    }

    @Override
    public String getLocation() {
        return m_dataset.getCurrentPath().getValue();
    }


    @Override
    public String getName() {
        return m_n4tCurPath.getValue();
    }

    @Override
    public IContainer getContainer(String shortName) {
		if (shortName != null && shortName.equals("")) {
			return this;
		}
		IGroup resultGroupItem = getGroup(shortName);
		if (resultGroupItem != null) {
			return resultGroupItem;
		}
		IDataItem resultVariableItem = getDataItem(shortName);
		if (resultVariableItem != null) {
			return resultVariableItem;
		}

		return null;
    }

    @Override
    public IContainer findContainerByPath(String path) {
    	IContainer foundItem = null;
    	NexusNode nnNode;
    	String[] sNodes;
    	NexusFileWriter nfwFile = m_dataset.getHandler();
    	try
    	{
    		sNodes = PathNexus.splitStringPath(path);
    		nfwFile.openPath(PathNexus.ROOT_PATH);
    		for( int i = 0; i < sNodes.length; i++ )
    		{
    			nnNode = nfwFile.getNode(sNodes[i]);
    			if( nnNode != null )
    				nfwFile.openNode(nnNode);
    		}

    		if( nfwFile.getCurrentPath().getCurrentNode().isGroup() )
    		{
    			return new NxsGroup(nfwFile.getCurrentPath(), m_dataset);
    		}
    		else
    		{
    			DataSet dataInfo;

    			if( nfwFile.getCurrentPath().getCurrentNode().getClassName().equals("NXtechnical_data") )
    			{
    				dataInfo = nfwFile.readData(PathData.Convert(nfwFile.getCurrentPath()));
    			}
    			else
    			{
    				dataInfo = nfwFile.readDataInfo();
    			}
    			foundItem = new NxsDataItem( dataInfo, m_dataset );
    			((IDataItem) foundItem).setDimensions("*");
    		}
    	}
    	catch(NexusException ne) {}
    	return foundItem;
    }

    @Override
    public IGroup getParentGroup() {
        if( m_parent == null )
        {
        	PathNexus parentPath = m_n4tCurPath.getParentPath();
        	if( parentPath != null )
            {
                m_parent = new NxsGroup(parentPath, m_dataset); 
        		return m_parent;
            }
        	else
        		return null;
        }
        else
            return m_parent;
    }

    @Override
    public IGroup getRootGroup() {
        return m_dataset.getRootGroup();
    }

    @Override
    public String getShortName()
    {
    	NexusNode nnNode = m_n4tCurPath.getCurrentNode();
    	if( nnNode != null )
    		return nnNode.getNodeName();
    	else
    		return "";
    }
    
    private String getClassName()
    {
    	NexusNode nnNode = m_n4tCurPath.getCurrentNode();
    	if( nnNode != null )
    		return nnNode.getClassName();
    	else
    		return "";
    }

    @Override
    public Map<String, String> harvestMetadata(String md_standard)
            throws IOException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public boolean hasAttribute(String name, String value) {
    	IAttribute attr;
    	List<IAttribute> listAttr = getAttributeList();

        Iterator<IAttribute> iter = listAttr.iterator();
        while( iter.hasNext() )
        {
        	attr = iter.next();
        	if( attr.getStringValue().equals(value) )
        		return true;
		}
        return false;
    }

    @Override
    public boolean isRoot() {
        return (m_n4tCurPath.toString().equals(PathNexus.ROOT_PATH.toString()));
    }

    @Override
    public boolean removeAttribute(IAttribute attribute) {
        return m_attributes.remove(attribute);
    }

    @Override
    public boolean removeDataItem(IDataItem item) {
        return m_child.remove(item);
    }

    @Override
    public boolean removeDataItem(String varName) {
        IDataItem item = getDataItem(varName);
        if( item == null ) {
            return false;
        }
        
        return m_child.remove(item);
    }

    @Override
    public boolean removeDimension(String dimName) {
        IDimension dimension = getDimension(dimName);
        if( dimension == null ) {
            return false;
        }
        
        return m_child.remove(dimension);
    }
    
    @Override
    public boolean removeDimension(IDimension dimension) {
        return m_child.remove(dimension);
    }

    @Override
    public boolean removeGroup(IGroup group) {
        return m_child.remove(group);
    }

    @Override
    public boolean removeGroup(String shortName) {
        IGroup group = getGroup(shortName);
        if( group == null ) {
            return false;
        }
        
        return m_child.remove(group);
    }

    @Override
    public void setName(String name) {
        addStringAttribute("long_name", name);
    }

    @Override
    public void setShortName(String name) {
        NexusNode nnNode = m_n4tCurPath.popNode();
        if( nnNode != null )
        {
        	nnNode.setNodeName(name);
        	m_n4tCurPath.pushNode(nnNode);
        }
    }
    
    @Override
    public void setDictionary(IDictionary dictionary) {
        m_dictionary = dictionary;
    }

    @Override
    public void setParent(IGroup group) {
        m_parent = group;
        if( group != null ) {
            ((NxsGroup) group).setChild(this);
        }
    }

    @Override
	public List<IContainer> findAllContainers(IKey key) {
        return findAllOccurrences(new Key(key));
	}

    @Override
	public List<IContainer> findAllOccurrences(IKey key) {
    	String path = m_dictionary.getPath(key).toString();
		return findAllContainerByPath(path);
	}
    
    @Override
    public IContainer findObjectByPath(IPath path) {
    	List<IContainer> containers = findAllContainerByPath(path.toString());
    	if (containers != null && containers.size() > 0) {
    		return containers.get(0);
    	}
    	return null;
    }
    
    @Override
/*    public List<IContainer> findAllObjectByPath(String path) {
        List<IContainer> list = new ArrayList<IContainer>();
        
        // Try to list all nodes matching the path
        try
        {
        	// Transform path into a NexusNode array
            NexusNode[] nodes = PathNexus.splitStringToNode(path);
            
            // Open path from root
            m_dataset.getHandler().closeAll();
            
            list = listAllNodes(m_dataset.getHandler(), nodes, 0 );
        }
        catch(NexusException ne)
        {
            try
            {
            	list.clear();
        	    m_dataset.getHandler().closeAll();
            } catch(NexusException e) {}
        }
        return list;
    }
    */
    public List<IContainer> findAllContainerByPath(String path) {
        List<IContainer> list = new ArrayList<IContainer>();
        
        // Try to list all nodes matching the path
        try
        {
        	// Transform path into a NexusNode array
            NexusNode[] nodes = PathNexus.splitStringToNode(path);
            
            // Open path from root
            NexusFileWriter handler = m_dataset.getHandler();
            handler.closeAll();
            
            for( int i = 0; i < nodes.length - 1; i++ ) {
            	handler.openNode(nodes[i]);
            }
            
            // List child of the penultimate node in path
    		IContainer item;
    		NexusNode pattern = nodes[nodes.length - 1];
            List<NexusNode> child = handler.listChildren();
    		// Do nodes match
    		for( NexusNode node : child ) {
    			if( node.matchesPartNode(pattern) ) {
    				// Create IContainer
					handler.openNode(node);
					
					if( handler.getCurrentPath().getCurrentNode().isGroup() ) {
						item = new NxsGroup(handler.getCurrentPath().clone(), m_dataset);
					}
					else {
						item = new NxsDataItem(handler.readDataInfo(), m_dataset);
					}
					handler.closeData();
    				list.add( item );
    			}
    		}
        }
        catch(NexusException ne)
        {
            try
            {
            	list.clear();
        	    m_dataset.getHandler().closeAll();
            } catch(NexusException e) {}
        }
        return list;
    }


    @Override
	public IDictionary findDictionary()
    {
    	if( m_dictionary == null )
    	{
   			m_dictionary = new NxsDictionary();
            try {
                m_dictionary.readEntries(System.getProperty("DICO_PATH", System.getenv("DICO_PATH")));
            } catch (FileAccessException e) {
                e.printStackTrace();
            }
    	}
    	return m_dictionary;
    }

    @Override
	public IContainer findContainer(String shortName) {
		// TODO Auto-generated method stub
		return null;
	}
    
    @Override
    public void updateDataItem(String key, IDataItem dataItem)
            throws SignalNotAvailableException {
        // TODO Auto-generated method stub

    }

    // ---------------------------------------------------------
    /// protected methods
    // ---------------------------------------------------------
    /**
     * Return the internal PathNexus of the group
     */
    protected PathNexus getPath() {
        return m_n4tCurPath;
    }
    
    /**
     * Set the internal PathNexus of the group
     */
    protected void setPath(PathNexus path) {
        m_n4tCurPath = path;
    }
    
    protected void setChild(IContainer node) {
        if( ! m_child.contains(node) )
        {
            m_child.add(node);
        }
    }
    
    /**
     * 
     * @param node
     */
    static protected List<IDataItem> getDescendentDataItem(List<IDataItem> items, IGroup group) {
        List<IDataItem> list = group.getDataItemList();
        List<IGroup> gpList  = group.getGroupList();
        
        for(IDataItem item : list ) {
            items.add(item);
        }
        
        for(IGroup grp : gpList) {
            NxsGroup.getDescendentDataItem(items, grp);
        }

        return items;
    }
    // ---------------------------------------------------------
    /// private methods
    // ---------------------------------------------------------
    private List<IContainer> getGroupNodes(boolean bGroup)
    {
        List<IContainer> listItem = new ArrayList<IContainer>();
        NexusNode[] nexusNodes;
        NexusFileWriter handler = m_dataset.getHandler();

        try
        {
            nexusNodes = handler.listChildren(m_n4tCurPath);
        }
        catch(NexusException ne)
        {
            try
            {
                handler.closeAll();
            } catch(NexusException e) {}
            return null;
        }

        IContainer item;
        DataSet dataInfo;
        PathNexus path;
        for( int i = 0; i < nexusNodes.length; i++ )
        {
            if( nexusNodes[i].isGroup() == bGroup )
            {
                try
                {
                    path = m_n4tCurPath.clone();
                    path.pushNode(nexusNodes[i]);
                    if( bGroup )
                    {
                        item = new NxsGroup(PathGroup.Convert(path), m_dataset);
                    }
                    else
                    {
                        if( nexusNodes[i].getClassName().equals("NXtechnical_data") )
                        {
                            dataInfo = handler.readData(PathData.Convert(path));
                        }
                        else
                        {
                            handler.openPath(path);
                            dataInfo = handler.readDataInfo();
                        }
                        item = new NxsDataItem(dataInfo, m_dataset);
                    }
                    listItem.add(item);
                } catch(NexusException e) {}
            }
        }
        try
        {
            handler.closeAll();
        } catch(NexusException e) {}

        return (List<IContainer>) listItem;
    }

    private void createFamilyTree() throws NullPointerException {
        if( m_dataset == null || m_n4tCurPath == null ) {
            throw new NullPointerException("Defined file handler and path are required!");
        }
        
        NexusNode[] nodes = m_n4tCurPath.getParentPath().getNodes();
        NxsGroup ancestor = (NxsGroup) m_dataset.getRootGroup();
        PathNexus origin = m_dataset.getHandler().getCurrentPath().clone();
        PathNexus path = PathNexus.ROOT_PATH.clone();
        NxsGroup group;
        
        for( NexusNode node : nodes ) {
            group = (NxsGroup) ancestor.getGroup(node.getNodeName());
            if( group == null ) {
                path.pushNode(node);
                group = new NxsGroup(ancestor, path.clone(), m_dataset);
            }
            ancestor = group;
        }

        setParent(ancestor);
        try {
			m_dataset.getHandler().openPath(origin);
		} catch (NexusException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    public String toString() {
    	return m_n4tCurPath.toString();
    }
    
	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}
	
	/*
    private List<IContainer> listAllNodes(NexusFileWriter handler, NexusNode[] nodes, int iDepthLvl) throws NexusException {
    	List<IContainer> result  = new ArrayList<IContainer>();
    	List<NexusNode> child;
    	
    	if( iDepthLvl < nodes.length && nodes[iDepthLvl] != null ) {
    		
    		// Get current referent node
    		NexusNode node = nodes[iDepthLvl];
    		++iDepthLvl;
    		
    		// List current level node
    		child = handler.listChildren();
    		
    		// If not the last depth level recursive call will be made
	    	if( iDepthLvl < nodes.length ) {
	    		

    			// Do nodes matches
	    		for( NexusNode current : child ) {
	    			if( current.matchesNode(node) ) {
	    				// Recursive call
	    				handler.openNode(current);
	    				result.addAll( listAllNodes(handler, nodes, iDepthLvl) );
	    				handler.closeData();
	    			}
	    		}
	    	}
	    	// If last depth level directly add matching node
	    	else {
	    		IContainer item;
	    		// Do nodes matches
	    		for( NexusNode current : child ) {
	    			if( current.matchesNode(node) ) {
	    				// Create IContainer
						handler.openNode(current);
						
						if( handler.getCurrentPath().getCurrentNode().isGroup() ) {
							item = new NxsGroup(handler.getCurrentPath().clone(), m_dataset);
						}
						else {
							item = new NxsDataItem(handler.readDataInfo(), m_dataset);
						}
						handler.closeData();
	    				result.add( item );
	    			}
	    		}
	    	}
    	}
    	return result;
    }
    */
}
