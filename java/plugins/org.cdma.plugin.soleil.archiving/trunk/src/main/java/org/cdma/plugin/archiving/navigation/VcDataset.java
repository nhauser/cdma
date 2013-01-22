package org.cdma.plugin.archiving.navigation;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.List;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.archiving.VcFactory;
import org.cdma.plugin.archiving.internal.VcXmlConstants;
import org.cdma.plugin.xml.navigation.XmlDataset;

public class VcDataset implements IDataset {

	public static final int ATTRIBUTE_ENTITIES_LENGTH = 4;

	private File mCurrentFile;
	private boolean isOpen;
	private String mTitle;

	private VcGroup mRootGroup;
	private boolean mDbInitialized;
	private SqlDataset mHdbDataset;
	private SqlDataset mTdbDataset;
	private boolean mIsUSFormat;

	public VcDataset(File target) {
		mCurrentFile   = target;
		isOpen         = false;
		mDbInitialized = false;
		mIsUSFormat    = false;
	}
	
	protected SqlDataset getHdbdataset() {
		synchronized ( this ) {
			if( ! mDbInitialized ) {
				initDb();
			}	
		}
		return mHdbDataset;
	}
	
	protected SqlDataset getTdbdataset() {
		synchronized ( this ) {
			if( ! mDbInitialized ) {
				initDb();
			}	
		}
		return mTdbDataset;
	}
	
	private void initDb() {
		String driver = System.getProperty("HDB_DRIVER",   System.getenv("HDB_DRIVER"));
		String host   = System.getProperty("HDB_HOST",     System.getenv("HDB_HOST"));
		String port   = System.getProperty("HDB_PORT",     System.getenv("HDB_PORT"));
		String name   = System.getProperty("HDB_NAME",     System.getenv("HDB_NAME"));
		String schema = System.getProperty("HDB_SCHEMA",   System.getenv("HDB_SCHEMA"));
		//String rac    = System.getProperty("HDB_RAC",      System.getenv("HDB_RAC"));
		String user   = System.getProperty("HDB_USER",     System.getenv("HDB_USER"));
		String pwd    = System.getProperty("HDB_PASSWORD", System.getenv("HDB_PASSWORD"));
		String url    = driver + ":@" + host +
						( "".equals(port) ? "" : (":" + port) ) + 
						( "".equals(name) ? "" : (":" + name) ) +
						( "".equals(schema) ? "" : (":" + schema) );
		mHdbDataset = new SqlDataset( VcFactory.NAME, url, user, pwd);
				
		driver = System.getProperty("TDB_DRIVER",   System.getenv("TDB_DRIVER"));
		host   = System.getProperty("TDB_HOST",     System.getenv("TDB_HOST"));
		port   = System.getProperty("TDB_PORT",     System.getenv("TDB_PORT"));
		name   = System.getProperty("TDB_NAME",     System.getenv("TDB_NAME"));
		schema = System.getProperty("TDB_SCHEMA",   System.getenv("TDB_SCHEMA"));
		//rac    = System.getProperty("TDB_RAC",      System.getenv("TDB_RAC"));
		user   = System.getProperty("TDB_USER",     System.getenv("TDB_USER"));
		pwd    = System.getProperty("TDB_PASSWORD", System.getenv("TDB_PASSWORD"));
		url    = driver + ":@" + host +
				( "".equals(port) ? "" : (":" + port) ) + 
				( "".equals(name) ? "" : (":" + name) ) +
				( "".equals(schema) ? "" : (":" + schema) );
		mTdbDataset = new SqlDataset( VcFactory.NAME, url, user, pwd);
		
		mDbInitialized = true;
	}

	public VcDataset(URI target) {
		this(new File(target));
	}

	private void initDatasetFromXml(IDataset mXmlDataset) {
		IGroup xmlRoot = mXmlDataset.getRootGroup();
		mRootGroup = new VcGroup(xmlRoot.getName(), this, null);
		for (IAttribute attribute : xmlRoot.getAttributeList()) {
			mRootGroup.addOneAttribute(attribute);
		}
		for (IContainer container : xmlRoot.getGroupList()) {

			if (container instanceof IGroup) {
				VcGroup tempGroup = null;
				String containerName = container.getName();
				if (VcXmlConstants.ATTRIBUTE_XML_TAG
						.equals(containerName)) {

					// Retrieve attribute data
					IAttribute dataAttribute = container
							.getAttribute(VcXmlConstants.ATTRIBUTE_COMPLETE_NAME_PROPERTY_XML_TAG);
					if (dataAttribute != null) {

						tempGroup = initGroupFromAttribute(dataAttribute
								.getStringValue());

						// Attribute
						tempGroup
								.addOneAttribute(container
										.getAttribute(VcXmlConstants.ATTRIBUTE_FACTOR_PROPERTY_XML_TAG));

						// Plot properties group initialization
						List<IGroup> childGroup = ((IGroup) container)
								.getGroupList();
						if (childGroup != null && childGroup.size() == 1) {
							VcGroup plotPropertiesGroup = copyGroup(childGroup
									.get(0));
							tempGroup.addSubgroup(plotPropertiesGroup);
						}
					}
				} else if (VcXmlConstants.VC_GENERIC_PLOT_PARAMETERS_XML_TAG
						.equals(containerName)) {
					tempGroup = copyGroup(container);
				}
			}
		}
	}

	private VcGroup initGroupFromAttribute(String attributeName) {
		VcGroup result = null;
		// Retrieve all entities of the tango attribute
		String[] attributeEntities = attributeName.split("/");
		if (attributeEntities != null
				&& attributeEntities.length == ATTRIBUTE_ENTITIES_LENGTH) {
			boolean isIterationUsefull = true;
			VcGroup referentGroup = mRootGroup;
			// Iteration in existing groups to find if the current entity
			// already exist
			for (int index = 0; index < ATTRIBUTE_ENTITIES_LENGTH; ++index) {
				IContainer currentContainer = null;
				String currentEntity = attributeEntities[index];
				if (isIterationUsefull) {
					for (IGroup group : referentGroup.getGroupList()) {
						if (currentEntity.equalsIgnoreCase(group.getName())) {
							currentContainer = group;
							break;
						}
					}
				}
				// If no entity already exists, we create it and add it to the
				// current referent group
				if (currentContainer == null) {
					currentContainer = new VcGroup(currentEntity, this,
							referentGroup);
					referentGroup.addSubgroup((IGroup) currentContainer);
					// We don't search anymore into existing group (if this
					// group doesn't exist, the next one either)
					isIterationUsefull = false;
				}

				if (currentContainer instanceof VcGroup
						&& currentContainer != null) {
					referentGroup = (VcGroup) currentContainer;
				}
			}
			// Set the last group as attribute container group
			if (referentGroup != null) {
				result = referentGroup;
				referentGroup.setHiddenAttribute(attributeName);
			}
		}
		return result;
	}

	private VcGroup copyGroup(IContainer container) {
		VcGroup result = null;
		if (container instanceof VcGroup) {
			result = new VcGroup((VcGroup) container);
		}
		return result;
	}

	@Override
	public String getFactoryName() {
		return VcFactory.NAME;
	}

	@Override
	public void close() throws IOException {
	}

	@Override
	public IGroup getRootGroup() {
		return mRootGroup;
	}

	@Override
	public LogicalGroup getLogicalRoot() {
		return new LogicalGroup(null, this);
	}

	@Override
	public String getLocation() {
		return mCurrentFile.getPath();
	}

	@Override
	public String getTitle() {
		return mTitle;
	}

	@Override
	public void setLocation(String location) {
		if (location != null) {
			File file = new File(location);
			if (file.exists() && file.isFile()) {
				mCurrentFile = file;
			}
		}
	}

	@Override
	public void setTitle(String title) {
		mTitle = title;
	}

	@Override
	public boolean sync() throws IOException {
		return false;
	}

	@Override
	public void open() {
		File originalVC = mCurrentFile;
		IDataset xmlDataset = new XmlDataset(getFactoryName(), originalVC);
		if (xmlDataset.isOpen()) {
			if (!isOpen) {
				initDatasetFromXml(xmlDataset);
				isOpen = true;
			}
		}
		xmlDataset = null;
	}

	@Override
	public void save() throws WriterException {
		throw new NotImplementedException();
	}

	@Override
	public void saveTo(String location) throws WriterException {
		throw new NotImplementedException();
	}

	@Override
	public void save(IContainer container) throws WriterException {
		throw new NotImplementedException();
	}

	@Override
	public void save(String parentPath, IAttribute attribute) throws WriterException {
		throw new NotImplementedException();
	}

	@Override
	public boolean isOpen() {
		return isOpen;
	}

	@Override
	public long getLastModificationDate() {
		long result = -1;
		IAttribute lastUpdate = mRootGroup
				.getAttribute(VcXmlConstants.VC_LAST_UPDATE_DATE_PROPERTY_XML_TAG);
		if (lastUpdate != null) {
			try {
				result = Long.parseLong(lastUpdate.getStringValue());
			} catch (NumberFormatException e) {
			}
		}
		return result;
	}

	public boolean getUSDateFormat() {
		return mIsUSFormat;
	}
	
	public void setUSdateFormat(boolean isUS ) {
		mIsUSFormat = isUS;
	}
}
