package org.cdma.engine.archiving.navigation;

import java.io.IOException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.Path;
import org.cdma.engine.archiving.internal.Constants;
import org.cdma.engine.archiving.internal.GroupUtils;
import org.cdma.engine.archiving.internal.attribute.Attribute;
import org.cdma.engine.archiving.internal.attribute.AttributePath;
import org.cdma.engine.archiving.internal.attribute.AttributeProperties;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.SignalNotAvailableException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;
import org.cdma.utils.Utilities.ModelType;

public class ArchivingGroup implements IGroup {
    private Map<String, IAttribute> mAttributes; // map of attributes 
    private List<IContainer> mChildren;   // list of container both items and groups 
    private List<IDimension> mDimensions; // list of dimension associated to this group
    private boolean  mInitialized;       // has this group been initialized
    private ArchivingDataset mDataset;    // handler on the dataset
    private ArchivingGroup   mParent;     // parent group of this
    private Attribute mDbAttr;      // Archiving attribute object (path, properties and connection)
    private String    mFactory;     // Plug-in factory name
    
    private Timestamp mDateStart;   // Last starting date used for extraction
    private Timestamp mDateEnd;     // Last ending date used for extraction
    private String    mDateFormat;  // Last date format used for extraction
    
    protected ArchivingGroup(String factory, ArchivingDataset dataset) {
		this( factory, dataset, null, null );
	}
    
	public ArchivingGroup(String factory, ArchivingDataset dataset, ArchivingGroup parent, String name) {
		mFactory    = factory;
		mAttributes = new HashMap<String, IAttribute>();
		mChildren   = new ArrayList<IContainer>();
		mDimensions = new ArrayList<IDimension>();
		mDataset    = dataset;
		mParent     = parent;

		if( parent != null ) {
			mDbAttr = parent.mDbAttr.clone();
			AttributePath path = mDbAttr.getPath();
			if( path != null ) {
				path.setNextDbField(name);
			}
		}
		else {
			mDbAttr = new Attribute( dataset.getSqldataset(), dataset.getSchema() );
		}
		mInitialized = false;
		mDateStart   = new Timestamp(0);   // Last starting date used for extraction
	    mDateEnd     = new Timestamp(0);   // Last ending date used for extraction
	    mDateFormat  = "";
		// Initialize archived attribute properties
	    initProperties();
	}

	@Override
	public ArchivingDataset getDataset() {
		return mDataset;
	}
	
    @Override
	public IDataItem getDataItem(String shortName) {
    	initChildren();
    	IDataItem result = null;
		if( shortName != null ) {
			for( IContainer container : mChildren ) {
				if( 
					container.getModelType().equals( ModelType.DataItem ) &&
					shortName.equals( container.getShortName() ) 
				) {
					result = (IDataItem) container;
				}
			}
		}
		return result;
	}

	@Override
	public IDataItem getDataItemWithAttribute(String name, String value) {
		initChildren();
    	IDataItem result = null;
		if( name != null ) {
			for( IContainer container : mChildren ) {
				if( 
					container.getModelType().equals( ModelType.DataItem ) &&
					container.hasAttribute(name, value)
				) {
					result = (IDataItem) container;
				}
			}
		}
		return result;
	}

	@Override
	public List<IDataItem> getDataItemList() {
		initChildren();
		List<IDataItem> result = new ArrayList<IDataItem>();
		for( IContainer container : mChildren ) {
			if( container.getModelType().equals( ModelType.DataItem ) ) {
				result.add( (IDataItem) container );
			}
		}
		return result;
	}

	@Override
	public IContainer getContainer(String shortName) {
		initChildren();
		IContainer result = null;
		if( shortName != null ) {
			for( IContainer container : mChildren ) {
				if( shortName.equals( container.getShortName() ) ) {
					result = container;
					break;
				}
			}
		}
		return result;
	}
	
	@Override
	public IGroup clone() {
		ArchivingGroup result = new ArchivingGroup(mFactory, mDataset);
		result.mParent = this.mParent;
		result.mDbAttr = this.mDbAttr.clone();
		result.mAttributes = new HashMap<String, IAttribute>(this.mAttributes);
		result.mChildren = new ArrayList<IContainer>(this.mChildren);
		result.mDimensions = new ArrayList<IDimension>(this.mDimensions);
		result.mDbAttr = this.mDbAttr.clone();
		return result;
	}
	
	public final Attribute getArchivedAttribute() {
		return mDbAttr;
	}
	
	@Override
	public ModelType getModelType() {
		return ModelType.Group;
	}

	@Override
	public void addOneAttribute(IAttribute attribute) {
		if( attribute != null ) {
			mAttributes.put(attribute.getName(), attribute);
		}
	}

	@Override
	public void addStringAttribute(String name, String value) {
		addOneAttribute( new ArchivingAttribute( mFactory, name, value ) );
	}

	@Override
	public IAttribute getAttribute(String name) {
		IAttribute result = mAttributes.get(name);
		return result;
	}

	@Override
	public List<IAttribute> getAttributeList() {
		return new ArrayList<IAttribute>(mAttributes.values());
	}

	@Override
	public String getLocation() {
		return mDataset.getLocation();
	}

	@Override
	public String getName() {
		return getArchivedAttribute().getPath().getName();
	}

	@Override
	public String getShortName() {
		return getArchivedAttribute().getPath().getShortName();
	}

	@Override
	public boolean hasAttribute(String name, String value) {
		boolean result = false;
		IAttribute attr = getAttribute(name);
		if( attr != null && attr.isString() && attr.getStringValue().equalsIgnoreCase( value) ) {
			result = true;
		}
		return result;
	}

	@Override
	public boolean removeAttribute(IAttribute attribute) {
		boolean removed = false;
		if( attribute != null ) {
			IAttribute attr = mAttributes.remove(attribute.getName());
			removed = (attr != null);
		}
		return removed;
	}

	@Override
	public void setName(String name) {
		AttributePath path = new AttributePath(name);
		mDbAttr.setPath(path);
	}

	@Override
	public void setShortName(String name) {
		getArchivedAttribute().getPath().setShortName(name);
	}

	@Override
	public void setParent(IGroup group) {
		if( group instanceof ArchivingGroup ) {
			mParent = (ArchivingGroup) group;
		}
	}

	@Override
	public long getLastModificationDate() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String getFactoryName() {
		return mFactory;
	}

	@Override
	public void addDataItem(IDataItem item) {
		mChildren.add( item );
	}

	@Override
	public Map<String, String> harvestMetadata(String mdStandard) throws IOException {
		return new HashMap<String, String>();
	}

	@Override
	public IGroup getParentGroup() {
		return mParent;
	}

	@Override
	public IGroup getRootGroup() {
		return mDataset.getRootGroup();
	}

	@Override
	public void addOneDimension(IDimension dimension) {
		mDimensions.add(dimension);
	}

	@Override
	public void addSubgroup(IGroup group) {
		mChildren.add( group );
		
	}

	@Override
	public IDataItem findDataItem(IKey key) {
		throw new NotImplementedException();
	}

	@Override
	public IDataItem findDataItemWithAttribute(IKey key, String name, String attribute) throws NoResultException {
		throw new NotImplementedException();
	}

	@Override
	public IGroup findGroupWithAttribute(IKey key, String name, String value) {
		throw new NotImplementedException();
	}

	@Override
	public IDimension getDimension(String name) {
		IDimension result = null;
		if( name != null && ! name.isEmpty() ) {
			for( IDimension dim : mDimensions ) {
				if( dim.getName().equalsIgnoreCase( name ) ) {
					result = dim;
				}
			}
		}
		return result;
	}

	@Override
	public IGroup getGroup(String shortName) {
		initChildren();
		IGroup result = null;
		if( shortName != null ) {
			for( IContainer container : mChildren ) {
				if( 
					container.getModelType().equals( ModelType.Group ) &&
					container.getShortName().equals(shortName)
				) {
					result = (IGroup) container;
				}
			}
		}
		return result;
	}

	@Override
	public IGroup getGroupWithAttribute(String name, String value) {
		initChildren();
		IGroup result = null;
		if( name != null ) {
			for( IContainer container : mChildren ) {
				if( 
					container.getModelType().equals( ModelType.Group ) &&
					container.hasAttribute(name, value)
				) {
					result = (IGroup) container;
				}
			}
		}
		return result;
	}

	@Override
	public IDataItem findDataItem(String shortName) {
		throw new NotImplementedException();
	}

	@Override
	public List<IDimension> getDimensionList() {
		initChildren();
		return mDimensions;
	}

	@Override
	public IGroup findGroup(String shortName) {
		throw new NotImplementedException();
	}

	@Override
	public IGroup findGroup(IKey key) {
		throw new NotImplementedException();
	}

	@Override
	public List<IGroup> getGroupList() {
		initChildren();
		List<IGroup> result = new ArrayList<IGroup>();
		for( IContainer container : mChildren ) {
			if( container.getModelType().equals( ModelType.Group ) ) {
				result.add( (IGroup) container );
			}
		}
		return result;
	}

	@Override
	public IContainer findContainer(String shortName) {
		throw new NotImplementedException();
	}

	@Override
	public IContainer findContainerByPath(String path) throws NoResultException {
		IContainer result = null;

		// Split the path into nodes
		String[] nodes = path.split(AttributePath.SEPARATOR);
		IGroup group = getRootGroup();
		
		// For each one try to get its child
		for( String node : nodes ) {
			if( group != null && node != null && !node.isEmpty() ) {
				result = group.getContainer(node);
				if( result != null && result.getModelType().equals(ModelType.Group) ) {
					group = (IGroup) result;
				}
				else {
					break;
				}
			}
		}
		
		return result;
	}

	@Override
	public List<IContainer> findAllContainerByPath(String path) throws NoResultException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean removeDataItem(IDataItem item) {
		boolean result = false;
		if( item != null ) {
			String name = item.getShortName();
			result = removeDataItem(name);
		}
		return result;
	}

	@Override
	public boolean removeDataItem(String shortName) {
		return removeContainer(shortName);
	}

	@Override
	public boolean removeDimension(String name) {
		boolean result = false;
		if( name != null ) {
			for( IDimension dim : mDimensions ) {
				if( name.equals( dim.getName() ) ) {
					result = mDimensions.remove(dim);
					break;
				}
			}
		}
		return result;
	}

	@Override
	public boolean removeGroup(IGroup group) {
		boolean result = false;
		if( group != null ) {
			String name = group.getShortName();
			result = removeDataItem(name);
		}
		return result;
	}

	@Override
	public boolean removeGroup(String shortName) {
		return removeContainer(shortName);
	}

	@Override
	public boolean removeDimension(IDimension dimension) {
		boolean result = false;
		if( dimension != null ) {
			String name = dimension.getName();
			result = removeDimension(name);
		}
		return result;
	}

	@Override
	public void updateDataItem(String key, IDataItem dataItem)
			throws SignalNotAvailableException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setDictionary(IDictionary dictionary) {
		throw new NotImplementedException();
	}

	@Override
	public IDictionary findDictionary() {
		throw new NotImplementedException();
	}

	@Override
	public boolean isRoot() {
		return (mParent == null && getArchivedAttribute().getPath().isEmpty());
	}

	@Override
	public boolean isEntry() {
		return (mParent != null && mParent.isRoot() );
	}

	@Override
	public List<IContainer> findAllContainers(IKey key)
			throws NoResultException {
		throw new NotImplementedException();
	}

	@Override
	public List<IContainer> findAllOccurrences(IKey key)
			throws NoResultException {
		throw new NotImplementedException();
	}

	@Override
	public IContainer findObjectByPath(Path path) {
		IContainer result = null;

		// If path is empty return the root group
		AttributePath attrPath = new AttributePath(path.getValue());
		if( attrPath.isEmpty() ) {
			result = mDataset.getRootGroup();
		}
		else {
			// Get nodes of this attribute
			String[] nodes = attrPath.getNodes();
			IGroup group = mDataset.getRootGroup();
			
			// Get corresponding groups
			for( String node : nodes ) {
				if( group != null ) {
					group = group.getGroup( node );
				}
			}
			
			// Check if the path targets an item
			String itemName = attrPath.getField();
			if( itemName != null && !itemName.isEmpty() ) {
				result = group.getDataItem(itemName);
			}
			// Else it targets a group
			else {
				result = group;
			}
		}
		return result;
	}

	@Override
	public String toString() {
		StringBuffer result = new StringBuffer();
		result.append( getName() );
		result.append("\nattrib: \n" );
		for( IAttribute attr : getAttributeList() ) {
			result.append("  - " + attr.toString() + "\n" );
		}
		return result.toString();
	}
	
	// ------------------------------------------------------------------------
	// Private methods
	// ------------------------------------------------------------------------
	private boolean removeContainer(String shortName) {
		boolean result = false;
		if( shortName != null ) {
			for( IContainer container : mChildren ) {
				if( shortName.equals( container.getShortName() ) ) {
					result = mChildren.remove(container);
					break;
				}
			}
		}
		return result;
	}

	private void initChildren() {
		if( ! isInitialized() ) {
			mInitialized = true;
			
			// remove children and dimensions
			mChildren.clear();
			mDimensions.clear();
			
			// Initialize child groups if any
			GroupUtils.initGroupList(this);
			
			// Initialize child items if any
			GroupUtils.initItemList(this);
		}
	}
	
	private boolean isInitialized() {
		boolean result = mInitialized;
		
		if( result ) {
			// In case of fully defined attribute path
			AttributePath path = mDbAttr.getPath();
			if( path != null && path.isFullyQualified() ) {
				
				// Check if start date has changed
				Timestamp time = GroupUtils.getStartDate(this);
				if( !mDateStart.equals(time) ) {
					mDateStart = time;
					result = false;
				}
				
				// Check if end date has changed
				time = GroupUtils.getEndDate(this);
				if( !mDateEnd.equals(time) ) {
					mDateEnd = time;
					result = false;
				}

				// Check if date format has changed
				String format = GroupUtils.getDateFormat(this);
				if( !mDateFormat.equals(format) ) {
					mDateFormat = format;
					result = false; 
				}
			}
		}
		return result;
	}
	
	/**
	 * Initialize archived attribute's properties
	 */
	private void initProperties() {
		// Get the archived attribute's properties
		if( getArchivedAttribute().getPath().isFullyQualified() && ! mInitialized ) {
			mInitialized = true;
			try {
				String name = getArchivedAttribute().getPath().getName();
				String dbName = mDataset.getSchema();
				SqlDataset dataset = mDataset.getSqldataset();
				AttributeProperties properties = new AttributeProperties(name, dataset, dbName);
				mDbAttr.setProperties( properties );
				
				ArchivingAttribute startArchiving = new ArchivingAttribute(mFactory, Constants.ORIGIN_DATE, properties.getOrigin() );
				addOneAttribute(startArchiving);
				
			} catch (IOException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize the attribute properties!", e);
			}
		}
	}
}
