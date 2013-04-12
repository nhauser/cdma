//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.mambo.navigation;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.archiving.internal.GroupUtils;
import org.cdma.engine.archiving.internal.attribute.AttributePath;
import org.cdma.engine.archiving.navigation.ArchivingAttribute;
import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.engine.archiving.navigation.ArchivingGroup;
import org.cdma.exception.BackupException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.mambo.SoleilMamboFactory;
import org.cdma.plugin.mambo.internal.MamboConstants;
import org.cdma.plugin.xml.navigation.XmlGroup;
import org.cdma.utilities.navigation.AbstractGroup;
import org.cdma.engine.sql.utils.DateFormat;

public class SoleilMamboGroup extends AbstractGroup {
	private static final String PATH_SEPARATOR = "/";
	private XmlGroup       mXmlGroup;
	private ArchivingGroup mArcGroup;
	
	public SoleilMamboGroup(SoleilMamboDataset dataset, SoleilMamboGroup parent, String name) throws BackupException {
		super( SoleilMamboFactory.NAME, dataset, parent, name );
		mXmlGroup = null;
		mArcGroup = null;
	}
	
	public SoleilMamboGroup(SoleilMamboDataset dataset, SoleilMamboGroup parent, XmlGroup group) throws BackupException {
		super( SoleilMamboFactory.NAME, dataset, parent, group.getShortName() );
		mXmlGroup = group;
		mArcGroup = null;
	}
	
	public SoleilMamboGroup(SoleilMamboDataset dataset, SoleilMamboGroup parent, XmlGroup xmlGroup, ArchivingGroup arcGroup) throws BackupException {
		super( SoleilMamboFactory.NAME, dataset, parent, arcGroup.getShortName() );
		mXmlGroup = xmlGroup;
		mArcGroup = arcGroup;
	}
	
	@Override
	public long getLastModificationDate() {
		long result = 0;
		
		IDataset dataset = getDataset();
		if( dataset != null ) {
			result = dataset.getLastModificationDate();
		}
		
		return result;
	}

	@Override
	public Map<String, String> harvestMetadata(String standard) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public IGroup clone() {
		SoleilMamboGroup result = null;
		try {
			result = new SoleilMamboGroup(
										(SoleilMamboDataset) getDataset(), 
										(SoleilMamboGroup) getParentGroup(), 
										getShortName()
									);
			result.mXmlGroup = this.mXmlGroup;
			result.mArcGroup = this.mArcGroup;
		} catch (BackupException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to create a clone!", e);
		}

		return result;
	}
	
	@Override
	public void addOneAttribute(IAttribute attribute) {
		if( attribute != null ) {
			IAttribute attr = prepareAttributeForAdd(attribute);
			super.addOneAttribute( attr );

			if( mArcGroup != null ) {
				String[] attributes = ArchivingDataset.getDrivingAttributes();
				String name = attribute.getName();
				for( String tmp : attributes ) {
					if( tmp.equals( name ) ) {
						mArcGroup.addOneAttribute(attr);
						break;
					}
				}
			}
		}
	}
	
	/**
	 * Prepare an a given IAttribute for being added to this group. Mainly,
	 * it will convert a long timestamp into the properly formated String date
	 * representation
	 *  
	 * @param attribute to be converted
	 * @return attribute of the right format
	 */
	private IAttribute prepareAttributeForAdd(IAttribute attribute) {
		IAttribute result = attribute;
		
		// If attribute is a number
		if( attribute != null && !attribute.isString() ) {
			String name = attribute.getName();
			
			// If attribute is matching a dated expected attribute
			for( String dateAttr : ArchivingDataset.getDatedAttributes() ) {
				if( dateAttr.equals(name) ) {
					String dateFormat = GroupUtils.getDateFormat(this);
					if( dateFormat != null ) {
						try {
							String date = DateFormat.convertDate( attribute.getNumericValue().longValue(), dateFormat );
							result = new ArchivingAttribute(SoleilMamboFactory.NAME, name, date);
						} catch (ParseException e) {
							// Let the attribute as it is and return it
						}
					}
					break;
				}
			}
			
		}
		return result;
	}

	// ------------------------------------------------------------------------
	// protected methods
	// ------------------------------------------------------------------------
	@Override
	protected String getPathSeparator() {
		return PATH_SEPARATOR;
	}

	@Override
	protected void initAttributes() {
		List<IGroup> groups = new ArrayList<IGroup>();
		if( mArcGroup != null ) {
			groups.add(mArcGroup);
		}
		
		if( mXmlGroup != null ) {
			groups.add(mXmlGroup);
		}
		
		List<IAttribute> attributes;
		for( IGroup group : groups ) {
			attributes = group.getAttributeList();
			for( IAttribute attribute : attributes ) {
				addOneAttribute(attribute);
			}
		}
		
	}
	
	@Override
	protected void initChildren() {
		initGroupList();
		initItemList();
	}
	
	// ------------------------------------------------------------------------
	// private methods
	// ------------------------------------------------------------------------
	private void initItemList() {
		List<IGroup> groups = new ArrayList<IGroup>();
		if( mXmlGroup != null ) {
			groups.add(mXmlGroup);
		}
		if( mArcGroup != null ) {
			groups.add(mArcGroup);
		}

		List<IDataItem> items;
		List<IAttribute> attributes;
		IDataItem child;
		for( IGroup group : groups ) {
			SoleilMamboDataset dataset = (SoleilMamboDataset) getDataset();
			items = group.getDataItemList();
			try {
				for( IDataItem item : items ) {
					// Create a child data item with loaded data
					IArray data = item.getData();
					String name = item.getShortName();
					child = new SoleilMamboDataItem( dataset, this, name, data);
					
					// Set its attributes
					attributes = item.getAttributeList();
					for( IAttribute attribute : attributes ) {
						child.addOneAttribute(attribute);
					}
					
					addDataItem( child );
					
				}
			} catch (BackupException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize item child", e);
			} catch (IOException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize item child", e);
			}
		}
	}
	
	private void initGroupList() {
		SoleilMamboDataset dataset = (SoleilMamboDataset) getDataset();
		if( mXmlGroup != null ) {
			String name;
			SoleilMamboGroup group;
			List<IGroup> xmlGroups = mXmlGroup.getGroupList();
			try {
				for( IGroup xmlGroup : xmlGroups ) {
					// If group is an 'attribute' create the archiving group
					name = xmlGroup.getShortName();
					if( name.matches(MamboConstants.ATTRIBUTE_MARKUP_REGEXP) ) {
						IAttribute attr = xmlGroup.getAttribute(MamboConstants.ATTRIBUTE_NAME);
						if( attr != null ) {
							String path = attr.getStringValue();
							group = initGroupChild(dataset, path, (XmlGroup) xmlGroup);
						}
						else {
							// Create a corresponding group
							group = new SoleilMamboGroup( dataset, this, (XmlGroup) xmlGroup );
						}
					}
					else {
						// Create a corresponding group
						group = new SoleilMamboGroup( dataset, this, (XmlGroup) xmlGroup );
					}
					
					addSubgroup( group );
					
				}
			} catch (BackupException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize group child", e);
			} catch (NoResultException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to initialize group child", e);
			}
		}
	}
	
	private SoleilMamboGroup initGroupChild(SoleilMamboDataset dataset, String path, XmlGroup xmlGroup) throws NoResultException, BackupException {
		SoleilMamboGroup result = null;
		SoleilMamboGroup parent = this;
		SoleilMamboGroup group = null;
		
		// Get the archiving root group
		ArchivingDataset arcDataset = dataset.getArchivingDataset();
		IGroup arcGroup = arcDataset.getRootGroup();
		
		// Split the path into nodes
		String[] nodes = path.split(AttributePath.SEPARATOR);
		// For each node construct a SoleilMamboGroup
		for( String node : nodes ) {
			if( node != null ) {
				group = (SoleilMamboGroup) parent.getGroup(node);
				arcGroup = arcGroup.getGroup(node);
				if( group == null ) {
					if( arcGroup == null ) {
						throw new NoResultException("Path is invalid: " + path + "\nNode wasn't found: " + node );
					}
					
					// Create the mambo group
					group = new SoleilMamboGroup(dataset, parent, arcGroup.getShortName() );
					
					// Return the first created group (and only it!) as the current children
					if( parent == this ) {
						result = group;
					}
					// Descendants are added to the first created group
					else {
						parent.addSubgroup(group);
					}
				}	
				parent = group;
			}
		}
		
		// Continue the initializing normally at last group of the archiving attribute
		group.mXmlGroup = xmlGroup;
		group.mArcGroup = (ArchivingGroup) arcGroup;

		// Add attributes from VC files
		if( result != null ) {
			String[] attributes = ArchivingDataset.getDrivingAttributes();
			for( String name : attributes ) {
				IAttribute attribute = this.getAttribute(name);
				if( attribute != null ) {
					mArcGroup.addOneAttribute(attribute);
				}
			}
		}
		
		// Add dimension to that last group
		List<IDimension> dimensions = arcGroup.getDimensionList();
		for( IDimension dimension : dimensions ) {
			group.addOneDimension(dimension);
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
	
}
