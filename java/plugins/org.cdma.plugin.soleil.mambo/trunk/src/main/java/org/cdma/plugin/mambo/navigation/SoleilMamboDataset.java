package org.cdma.plugin.mambo.navigation;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLEncoder;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.engine.archiving.navigation.ArchivingDataset.ArchivingMode;
import org.cdma.engine.archiving.navigation.ArchivingGroup;
import org.cdma.exception.BackupException;
import org.cdma.exception.NoResultException;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.mambo.SoleilMamboFactory;
import org.cdma.plugin.mambo.internal.MamboConstants;
import org.cdma.plugin.xml.navigation.XmlDataset;
import org.cdma.plugin.xml.navigation.XmlGroup;
import org.cdma.utilities.configuration.ConfigDataset;
import org.cdma.utilities.configuration.ConfigManager;

public class SoleilMamboDataset implements IDataset {
	private ConfigDataset    mConfig;
	private ArchivingDataset mArcDataset;
	private XmlDataset       mXmlDataset;
	private ArchivingMode    mArchivingMode;
	private SoleilMamboGroup mPhyRoot;
	private LogicalGroup     mLogRoot;
	
	public SoleilMamboDataset( File file) {
		mXmlDataset = new XmlDataset(SoleilMamboFactory.NAME, file);
		mArcDataset = null;
		mArchivingMode = ArchivingMode.HDB;
	}

	@Override
	public String getFactoryName() {
		return SoleilMamboFactory.NAME;
	}

	@Override
	public void close() throws IOException {
		if( mArcDataset != null ) {
			mArcDataset.close();
			mArcDataset = null;
		}
		mXmlDataset.close();
	}

	@Override
	public IGroup getRootGroup() {
		if( mPhyRoot == null ) {
			try {
				XmlGroup xmlRoot = (XmlGroup) mXmlDataset.getRootGroup();
				ArchivingGroup arcRoot = (ArchivingGroup) mArcDataset.getRootGroup();
				mPhyRoot = new SoleilMamboGroup( this, null, xmlRoot, arcRoot );
				
				/*
				// Get the XML root group and initialize child
				XmlGroup group = (XmlGroup) mXmlDataset.getRootGroup();
				mPhyRoot = new SoleilMamboGroup( this, null, group );
				
				// Get archiving attributes
				IGroup arcRoot = mArcDataset.getRootGroup();
				
				// Parameterize archiving dataset according xml dataset
				for( IAttribute attribute : mPhyRoot.getAttributeList() ) {
					arcRoot.addOneAttribute(attribute);
				}
				
				// Set attributes from archiving root group
				List<IAttribute> arcAttr = arcRoot.getAttributeList();
				for( IAttribute attr : arcAttr ) {
					mPhyRoot.addOneAttribute(attr);
				}
				*/
			} catch (BackupException e) {
				e.printStackTrace();
			}
		}
		return mPhyRoot;
	}
	
	@Override
	public LogicalGroup getLogicalRoot() {
		if( mLogRoot == null ) {
			mLogRoot = new LogicalGroup(null, this);
		}
		return mLogRoot;
	}

	@Override
	public String getLocation() {
		return mXmlDataset.getLocation();
	}

	@Override
	public String getTitle() {
		return mXmlDataset.getTitle();
	}

	@Override
	public void setLocation(String location) {
		mXmlDataset.setLocation(location);
	}

	@Override
	public void setTitle(String title) {
		mXmlDataset.setTitle(title);
	}

	@Override
	public boolean sync() throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public void open() throws IOException {
		initialize();
		
		if( mArcDataset == null ) {
			String user = getUser(mArchivingMode);
			String pwrd = getPassword(mArchivingMode);
			URI uri = null;
			try {
				uri = getDbConnectionURI();
				mArcDataset = new ArchivingDataset(SoleilMamboFactory.NAME, uri);
				mArcDataset.setArchivingMode(mArchivingMode);
				mArcDataset.setUser(user);
				mArcDataset.setPassword(pwrd);
				mArcDataset.setNumericalDate(true);
				
			} catch (URISyntaxException e) {
				mArcDataset = null;
				throw new IOException("Unable to connect to: " + uri, e);
			}
		}
		
		mArcDataset.open();
	}

	/**
	 * Open the VC file and parameterize the archiving dataset.
	 */
	private void initialize() {
		if( mXmlDataset != null ) {
			// Get the XML root group and initialize child
			XmlGroup group = (XmlGroup) mXmlDataset.getRootGroup();
			
			if( group != null ) {
				// Parameterize archiving mode according VC file
				IAttribute attr = group.getAttribute(MamboConstants.ATTRIBUTE_HISTORIC);
				if( attr != null && attr.isString() ) {
					boolean historic = Boolean.parseBoolean(attr.getStringValue());
					mArchivingMode = historic ? ArchivingMode.HDB : ArchivingMode.TDB;
				}
			}
		}
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
		return mArcDataset != null && mArcDataset.isOpen();
	}

	@Override
	public long getLastModificationDate() {
		return mXmlDataset.getLastModificationDate();
	}
	
	public XmlDataset getXmlDataset() {
		return mXmlDataset;
	}
	
	public ArchivingDataset getArchivingDataset() {
		return mArcDataset;
	}

	
	// ------------------------------------------------------------------------
	// Private methods
	// ------------------------------------------------------------------------
	private URI getDbConnectionURI() throws URISyntaxException {
		URI uri = null;
		String driver = getDriver();
		String schema = getSchema();
		String dbName = getDbName();
		String host   = getHost();
		String port   = getPort();
		String rac    = getRac();
		StringBuffer bufUri = new StringBuffer();
		if( driver != null && !driver.isEmpty() ) {
			bufUri.append(driver);
			if( rac != null && !rac.isEmpty() ) {
				bufUri.append(":");
				bufUri.append(rac);
			}
			else {
				if( host != null && !host.isEmpty() ) {
					bufUri.append(":@");
					bufUri.append(host);
				}
				if( port != null && !port.isEmpty() ) {
					bufUri.append(":");
					bufUri.append(port);
				}
				if( schema != null && !schema.isEmpty() ) {
					bufUri.append(":");
					bufUri.append(schema);
				}
				if( dbName != null && !dbName.isEmpty() ) {
					bufUri.append(":");
					bufUri.append(dbName);
				}
			}
		}
		String value;
		try {
			value = URLEncoder.encode(bufUri.toString(), "UTF-8");
			uri = new URI(value);
		} catch (UnsupportedEncodingException e) {
			uri = new URI(bufUri.toString());
		}
		
		return uri;
	}
	
	private String getDriver() {
		return getParam("_DRIVER", mArchivingMode, false);
	}
	
	private String getHost() {
		return getParam("_HOST", mArchivingMode, false);
	}
	
	private String getPort() {
		return getParam("_PORT", mArchivingMode, false);
	}
	
	private String getRac() {
		return getParam("_RAC", mArchivingMode, false);
	}
	
	private String getSchema() {
		return getParam("_SCHEMA", mArchivingMode, false);
	}
	
	private String getDbName() {
		return getParam("_NAME", mArchivingMode, false);
	}
	
	private String getPassword(ArchivingMode mode) {
		return getParam("_PASSWORD", mode, true);
	}

	private String getUser(ArchivingMode mode) {
		return getParam("_USER", mode, true);
	}
	
	/**
	 * Search for the value of the given parameter. The order of search is the following:<br/>
	 * 1 - system properties,<br/>
	 * 2 - system environment,<br/>
	 * 3 - cdma config files
	 * 
	 * @param name to search
	 * @param archiving mode
	 * @param forceConfig cdma config override system prop  
	 * @return the value of the given parameter
	 */
	private String getParam(String name, ArchivingMode mode, boolean forceConfig) {
		String result = null;
		
		if( mode != null ) {
			// Check in env. 
			String fullName = mode.getName() + name;
			result = System.getProperty(fullName, System.getenv(fullName));
			
			// Check in config file
			if( result == null || forceConfig ) {
				ConfigDataset conf = getConfiguration();
				if( conf != null && conf.hasParameter(fullName) ) {
					result = conf.getParameter(fullName);
				}
			}
		}
		return result;
	}
	
	/**
	 * Get the configuration matching this dataset.
	 * @return
	 */
    public ConfigDataset getConfiguration() {
		if (mConfig == null) {
            try {
				mConfig = ConfigManager.getInstance(SoleilMamboFactory.getInstance(), SoleilMamboFactory.CONFIG_FILE).getConfig(this);
			} catch (NoResultException e) {
				Factory.getLogger().log(Level.SEVERE, "Unable to get configuration!", e);
			}
        }
        return mConfig;
    }
}
