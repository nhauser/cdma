package org.gumtree.data.soleil.navigation;

// Standard import
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.impl.Key;
import org.gumtree.data.engine.jnexus.navigation.NexusDataItem;
import org.gumtree.data.engine.jnexus.navigation.NexusGroup;
import org.gumtree.data.exception.NoResultException;
import org.gumtree.data.exception.SignalNotAvailableException;
import org.gumtree.data.interfaces.IAttribute;
import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IDataItem;
import org.gumtree.data.interfaces.IDataset;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IDimension;
import org.gumtree.data.interfaces.IGroup;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.soleil.NxsFactory;
import org.gumtree.data.utils.Utilities.ModelType;

import fr.soleil.nexus4tango.PathNexus;

public class NxsGroup implements IGroup {
	// ****************************************************
	// Members
	// ****************************************************
	private NxsDataset       m_dataset;     // Dataset to which this group belongs to
	private IGroup[]         m_groups;      // Groups having a similar path from different files
	private IGroup           m_parent;      // Parent group folder (mandatory)
	private List<IContainer> m_children;    // All containers that are below (physically) this one
	private boolean          m_childUpdate; // is the children list up to date
	private boolean          m_multigroup;  // is this group managing aggregation of group
	
	// ****************************************************
	// Constructors
	// ****************************************************
	private NxsGroup() {
		m_groups   = null; 
		m_parent   = null;
		m_dataset  = null;
		m_children = null;
		m_childUpdate = false;
	}
	
	public NxsGroup(IGroup[] groups, IGroup parent, NxsDataset dataset) {
		m_groups   = groups; 
		m_parent   = parent;
		m_dataset  = dataset;
		m_children = null;
		m_childUpdate = false;
	}
	
	public NxsGroup( NxsGroup original ) {
		m_groups = new IGroup[original.m_groups.length];
		int i = 0;
		for( IGroup group : original.m_groups ) {
			m_groups[i++] = group;
		}
		m_parent      = original.m_parent;
		m_dataset     = original.m_dataset;
		m_children    = null;
		m_childUpdate = false;
		m_multigroup  = m_groups.length > 1;
	}
	
	public NxsGroup(IGroup parent, PathNexus path, NxsDataset dataset) {
		try {
			m_groups  = dataset.getRootGroup().findAllContainerByPath(path.getValue()).toArray(new IGroup[0]);
		} catch (NoResultException e) {
		}
		m_parent      = parent;
		m_dataset     = dataset;
		m_children    = null;
		m_childUpdate = false;
	}

	// ****************************************************
	// Methods from interfaces
	// ****************************************************
    /**
     * Return a clone of this IGroup object.
     * @return new IGroup
     * Created on 18/09/2008
     */
    @Override
    public NxsGroup clone()
    {
    	NxsGroup clone = new NxsGroup();
		clone.m_groups = new IGroup[m_groups.length];
		int i = 0;
		for( IGroup group : m_groups ) {
			m_groups[i++] = group.clone();
		}
		clone.m_parent = m_parent.clone();
		clone.m_dataset = m_dataset;
		clone.m_childUpdate = false;
        return clone;
    }
	
	@Override
	public ModelType getModelType() {
		return ModelType.Group;
	}

	@Override
	public IAttribute getAttribute(String name) {
		IAttribute attr = null;
		for( IGroup group : m_groups ) {
			attr = group.getAttribute(name);
			if( attr != null ) {
				break;
			}
		}
		return attr;
	}

	@Override
	public List<IAttribute> getAttributeList() {
		List<IAttribute> result = new ArrayList<IAttribute>();
		for( IGroup group : m_groups ) {
			result.addAll( group.getAttributeList() );
		}
		return result;
	}

	@Override
	public String getLocation() {
		return m_parent.getLocation() + "/" + getShortName();
	}

	@Override
	public String getName() {
    	String name = "";
    	if( m_groups.length > 0 ) {
   			name = m_groups[0].getName();
    	}
    	return name;
	}

    @Override
    public String getShortName()
    {
    	String name = "";
    	if( m_groups.length > 0 ) {
    		name = m_groups[0].getShortName();
    	}
    	return name;
    }

	@Override
	public boolean hasAttribute(String name, String value) {
		for( IGroup group : m_groups ) {
			if( group.hasAttribute(name, value) ) {
				return true;
			}
		}
		return false;
	}

	@Override
	public void setName(String name) {
		for( IGroup group : m_groups ) {
			group.setName(name);
		}
	}

	@Override
	public void setShortName(String name) {
		for( IGroup group : m_groups ) {
			group.setShortName(name);
		}
	}

	@Override
	public void setParent(IGroup group) {
		m_parent = group;
	}

	@Override
	public String getFactoryName() {
		return NxsFactory.NAME;
	}

	@Override
	public Map<String, String> harvestMetadata(String mdStandard)
			throws IOException {
		return null;
	}

	@Override
	public IGroup getParentGroup() {
        if( m_parent == null )
        {
        	// TODO do not reconstruct the physical hierarchy: keep what has been done in construct
        	IGroup[] groups = new IGroup[m_groups.length];
        	int i = 0;
        	for( IGroup item : m_groups ) {
        		groups[i++] = item.getParentGroup();
        	}
        	
        	m_parent = new NxsGroup(groups, null, m_dataset);
        	((NxsGroup) m_parent).setChild(this);
        	
        }
		return m_parent;
		
		
		//return m_parent;
	}

	@Override
	public IGroup getRootGroup() {
		return m_dataset.getRootGroup();
	}

	@Override
	public IDataItem getDataItem(String shortName) {
		List<IDataItem> list = getDataItemList();
		IDataItem result = null;
		
		for( IDataItem item : list ) {
			if( item.getShortName() == shortName ) {
				result = item;
				break;
			}
		}
		
		return result;
	}

	@Override
    public IDataItem findDataItem(IKey key) {
        IDataItem item = null;
        List<IContainer> list = new ArrayList<IContainer>();
        try {
			list = findAllOccurrences(key);
		} catch (NoResultException e) {	}
        
        for( IContainer object : list ) {
        	if( object.getModelType().equals(ModelType.DataItem) ) {
        		item = (IDataItem) object;
        		break;
        	}
        }
        
        return item;
    }
    
    @Override
    public IDataItem findDataItem(String keyName) {
    	IKey key = NxsFactory.getInstance().createKey(keyName);
        
		return findDataItem(key);
    }

	@Override
	public IDataItem getDataItemWithAttribute(String name, String value) {
		List<IDataItem> list = getDataItemList();
		IDataItem result = null;
		for( IDataItem item : list ) {
			if( item.hasAttribute(name, value) ) {
				result = item;
				break;
			}
		}
		
		return result;
	}

	@Override
	public IDataItem findDataItemWithAttribute(IKey key, String name,
			String attribute) throws Exception {
		List<IContainer> list = findAllContainers(key);
		IDataItem result = null;
		for( IContainer item : list ) {
			if( item.getModelType() == ModelType.DataItem ) {
				if( item.hasAttribute(name, attribute) ) {
					result = (IDataItem) item;
					break;
				}
			}
		}
		
		return result;
	}

	@Override
	public IGroup findGroupWithAttribute(IKey key, String name, String value) {
		List<IContainer> list;
		try {
			list = findAllContainers(key);
		}
		catch(NoResultException e) {
			list = new ArrayList<IContainer>();
		}
		IGroup result = null;
		for( IContainer item : list ) {
			if( item.getModelType() == ModelType.Group ) {
				if( item.hasAttribute(name, value) ) {
					result = (IGroup) item;
					break;
				}
			}
		}
		
		return result;
	}

	@Override
	public IContainer getContainer(String shortName) {
		List<IContainer> list = listChildren();
		IContainer result = null;
		
		for( IContainer container : list ) {
			if( container.getShortName() == shortName ) {
				result = container;
				break;
			}
		}
		
		return result;
	}

	@Override
	public IGroup getGroup(String shortName) {
		List<IGroup> list = getGroupList();
		IGroup result = null;
		
		for( IGroup group : list ) {
			if( group.getShortName() == shortName ) {
				result = group;
				break;
			}
		}
		
		return result;
	}

	@Override
	public IGroup getGroupWithAttribute(String attributeName, String attributeValue) {
		List<IGroup> list = getGroupList();
		IGroup result = null;
		for( IGroup item : list ) {
			if( item.hasAttribute(attributeName, attributeValue) ) {
				result = item;
				break;
			}
		}
		
		return result;
	}

	@Override
	public List<IDataItem> getDataItemList() {
		listChildren();
		
		ArrayList<IDataItem> list = new ArrayList<IDataItem>();
		for( IContainer container : m_children ) {
			if( container.getModelType() == ModelType.DataItem ) {
				list.add( (IDataItem) container);
			}
		}
		
		return list;
	}

	@Override
	public IDataset getDataset() {
		return m_dataset;
	}

	@Override
    public IGroup findGroup(IKey key) {
		IGroup item = null;
        List<IContainer> list = new ArrayList<IContainer>();
        try {
			list = findAllOccurrences(key);
		} catch (NoResultException e) {	}
        
        for( IContainer object : list ) {
        	if( object.getModelType().equals(ModelType.Group) ) {
        		item = (IGroup) object;
        		break;
        	}
        }
        
        return item;
    }
    
    @Override
    public IGroup findGroup(String keyName) {
    	IKey key = NxsFactory.getInstance().createKey(keyName);
        
		return findGroup(key);
    }

	@Override
	public List<IGroup> getGroupList() {
		listChildren();
		
		ArrayList<IGroup> list = new ArrayList<IGroup>();
		for( IContainer container : m_children ) {
			if( container.getModelType() == ModelType.Group ) {
				list.add( (IGroup) container);
			}
		}
		
		return list;
	}

    @Override
	public IContainer findContainer(String shortName) {
    	IKey key = NxsFactory.getInstance().createKey(shortName);
    	IContainer result;
    	try {
			List<IContainer> list = findAllOccurrences(key);
			if( list.size() > 0) {
				result = list.get(0);
			}
			else {
				result = null;
			}
		} catch (NoResultException e) {
			result = null;
		}
    	
    	
		return result;
	}

	@Override
	public IContainer findContainerByPath(String path) throws NoResultException {
		List<IContainer> containers = findAllContainerByPath(path);
		IContainer result = null;

		if( containers.size() > 0 ) {
			result = containers.get(0);
		}
		
		return result;
	}

	@Override
	public List<IContainer> findAllContainerByPath(String path)	throws NoResultException {
		List<IContainer> list = new ArrayList<IContainer>();
		List<IContainer> tmp = null;
		String tmpName;

		// Store in a map all different containers from all m_groups 
		Map< String, ArrayList<IContainer> > items = new HashMap<String, ArrayList<IContainer> >();
		for( IGroup group : m_groups ) {
			try {
				tmp = group.findAllContainerByPath(path);
				for( IContainer item : tmp ) {
					tmpName = item.getShortName();
					if( items.containsKey( tmpName ) ) {
						items.get( tmpName ).add( item );
					}
					else {
						ArrayList<IContainer> tmpList = new ArrayList<IContainer>();
						tmpList.add( item );
						items.put( tmpName, tmpList );
					}
				}
			}
			catch(NoResultException e) { 
				// Nothing to do 
			}
		}
		
		// Construct that were found
		for( String entry : items.keySet() ) {
			tmp = items.get(entry);
			// If a Group list then construct a new Group folder
			if( tmp.get(0).getModelType() == ModelType.Group ) {
				list.add(
					new NxsGroup(
						tmp.toArray( new IGroup[0] ),
						this,
						m_dataset
					)
				);
			}
			// If a DataItem list then construct a new compound NxsDataItem 
			else {
				ArrayList<NexusDataItem> dataItems = new ArrayList<NexusDataItem>();
				for( IContainer item : tmp ) {
					if( item.getModelType() == ModelType.DataItem ) {
						dataItems.add( (NexusDataItem) item );
					}
				}
				NexusDataItem[] array = new NexusDataItem[dataItems.size()];
				dataItems.toArray(array);
				list.add(
					new NxsDataItem(
						array,
						this,
						m_dataset
					)
				);
			}
		}
		
		return list;
	}

	@Override
	public boolean removeDataItem(IDataItem item) {
		return removeDataItem( item.getShortName() );
	}

	@Override
	public boolean removeDataItem(String varName) {
		boolean succeed = false;
		for( IGroup group : m_groups ) {
			if( group.removeDataItem( varName ) ) {
				succeed = true;
			}
		}
		return succeed;
	}

	@Override
	public boolean removeGroup(IGroup group) {
		return removeGroup( group.getShortName() );
	}

	@Override
	public boolean removeGroup(String shortName) {
		boolean succeed = false;
		for( IGroup group : m_groups ) {
			if( group.removeGroup( shortName ) ) {
				succeed = true;
			}
		}
		return succeed;
	}

	@Override
	public void setDictionary(IDictionary dictionary) {
		if( m_groups.length > 0 ) {
			m_groups[0].setDictionary( dictionary );
		}
	}

	@Override
	public IDictionary findDictionary() {
		IDictionary dictionary = null;
		if( m_groups.length > 0 ) {
			dictionary = m_groups[0].findDictionary();
		}
		return dictionary;
	}

	@Override
	public boolean isRoot() {
		return (m_groups.length > 0 && m_groups[0].isRoot());
	}

	@Override
	public boolean isEntry() {
		return ( m_parent.getParentGroup().getParentGroup() == null );
	}

    @Override
	public List<IContainer> findAllContainers(IKey key) throws NoResultException {
        return findAllOccurrences(new Key(key));
	}

    @Override
	public List<IContainer> findAllOccurrences(IKey key) throws NoResultException {
    	String path = findDictionary().getPath(key).toString();
		return findAllContainerByPath(path);
	}
    
	@Override
	public IContainer findObjectByPath(IPath path) {
		IContainer result = null;
		
		try {
			result = findContainerByPath(path.getValue());
		} catch (NoResultException e) {
			e.printStackTrace();
		}
		
		return result;
	}
    
	@Override
	public void addOneAttribute(IAttribute attribute) {
		// TODO Auto-generated method stub

	}

	@Override
	public void addStringAttribute(String name, String value) {
		// TODO Auto-generated method stub

	}
	
	@Override
	public boolean removeDimension(IDimension dimension) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean removeDimension(String dimName) {
		// TODO Auto-generated method stub
		return false;
	}
	
	@Override
	public List<IDimension> getDimensionList() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public IDimension getDimension(String name) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void addDataItem(IDataItem v) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean removeAttribute(IAttribute attribute) {
		// TODO Auto-generated method stub
		return false;
	}
	
	@Override
	public void addOneDimension(IDimension dimension) {
		// TODO Auto-generated method stub

	}

	@Override
	public void addSubgroup(IGroup group) {
		// TODO Auto-generated method stub

	}
	
	@Override
	public void updateDataItem(String key, IDataItem dataItem)
			throws SignalNotAvailableException {
		// TODO Auto-generated method stub

	}
	
    // ------------------------------------------------------------------------
    /// Protected methods
    // ------------------------------------------------------------------------
	protected void setChild(IContainer node) {
		if( ! m_children.contains(node) ) {
			m_children.add(node);
		}
	}
	// ****************************************************
	// private methods
	// ****************************************************
	private List<IContainer> listChildren() {
		List<IContainer> result;
		if( m_multigroup ) {
			result = listChildrenMultiGroup();
		}
		else {
			result = listChildrenMonoGroup();
		}
		return result;
	}
	
	private List<IContainer> listChildrenMultiGroup() {
		if( ! m_childUpdate ) 
		{
			List<IContainer> tmp = null;
			m_children = new ArrayList<IContainer>();
			String tmpName;
	
			// Store in a map all different containers from all m_groups 
			Map< String, ArrayList<IContainer> > items = new HashMap<String, ArrayList<IContainer> >();
			for( IGroup group : m_groups ) {
				tmp = new ArrayList<IContainer>();
				tmp.addAll( group.getDataItemList() );
				tmp.addAll( group.getGroupList() );
				for( IContainer item : tmp ) {
					tmpName = item.getShortName();
					if( items.containsKey( tmpName ) ) {
						items.get( tmpName ).add( item );
					}
					else {
						ArrayList<IContainer> tmpList = new ArrayList<IContainer>();
						tmpList.add( item );
						items.put( tmpName, tmpList );
					}
				}
			}
			
			// Construct what were found
			for( String entry : items.keySet() ) {
				tmp = items.get(entry);
				// If a Group list then construct a new Group folder
				if( tmp.get(0).getModelType() == ModelType.Group ) {
					m_children.add(
						new NxsGroup(
							tmp.toArray( new IGroup[0] ),
							this, 
							m_dataset
						)
					);
				}
				// If a DataItem list then construct a new compound NxsDataItem 
				else {
					ArrayList<NexusDataItem> nxsDataItems = new ArrayList<NexusDataItem>();
					for( IContainer item : tmp ) {
						if( item.getModelType() == ModelType.DataItem ) {
							nxsDataItems.add( (NexusDataItem) item );
						}
					}
					NexusDataItem[] array = new NexusDataItem[nxsDataItems.size()];
					nxsDataItems.toArray(array);
					m_children.add(
						new NxsDataItem(
							array,
							this,
							m_dataset
						)
					);
				}
			}
			m_childUpdate = true;
		}
		return m_children;	
	}
	
	private List<IContainer> listChildrenMonoGroup() {
		if( ! m_childUpdate )
		{
			m_children = new ArrayList<IContainer>();
			
			// Store in a list all different containers from all m_groups 
			for( IDataItem item : m_groups[0].getDataItemList() ) {
				m_children.add( new NxsDataItem( (NexusDataItem) item, m_dataset ) );
			}
			
			for( IGroup group : m_groups[0].getGroupList() ) {
				m_children.add( new NxsGroup( new IGroup[] {group}, this, m_dataset) );
			}
			m_childUpdate = true;
		}
		return m_children;
	}

	// ****************************************************
 	// Specific methods
	// ****************************************************
	public PathNexus getPathNexus() {
		PathNexus path = ((NexusGroup) m_groups[0]).getPathNexus();
		return path;
	}
}
