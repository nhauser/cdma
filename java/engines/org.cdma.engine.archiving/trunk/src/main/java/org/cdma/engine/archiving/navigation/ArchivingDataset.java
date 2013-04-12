package org.cdma.engine.archiving.navigation;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLDecoder;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.dictionary.LogicalGroup;
import org.cdma.engine.archiving.internal.Constants;
import org.cdma.engine.archiving.internal.attribute.Attribute;
import org.cdma.engine.archiving.internal.sql.ArchivingQueries;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.exception.NotImplementedException;
import org.cdma.exception.WriterException;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;

public class ArchivingDataset implements IDataset {
	
	public enum ArchivingMode {
		HDB  ("HDB"),
		TDB  ("TDB"),
		ADB  ("ADB"),
		SNAP ("SNAP");
		
        private String mName;

        private ArchivingMode(String type) { mName = type; }
        public  String getName()           { return mName; }
	}
	
	private URI            mURI;
	private String         mUser;
	private String         mPassword;
	private String         mTitle;
	private ArchivingGroup mRootGroup;
	private ArchivingMode  mArchivingMode;
	private SqlDataset     mSqlDataset;
	private boolean       mIsOpen;
	private String         mSchemaName;
	private String         mFactory;
	private boolean       mNumDate;
	
	/**
	 * At opening, the given uri will be decoded using UTF-8 URLDecoder.
	 * @param factory name of the plug-in
	 * @param uri of the targeted archiving database
	 */
	public ArchivingDataset(String factory, URI uri) {
		mRootGroup     = null;
		mURI           = uri;
		mIsOpen        = false;
		mTitle         = "";
		mFactory       = factory;
		mArchivingMode = ArchivingMode.HDB;
		mSqlDataset    = null;
	}
	
	@Override
	public String getFactoryName() {
		return mFactory;
	}

	@Override
	public void close() throws IOException {
		synchronized( this ) {
			mIsOpen = false;
			mSqlDataset.close();
		}
	}

	@Override
	public IGroup getRootGroup() {
		if( mRootGroup == null && isOpen() ) {
			ArchivingGroup group = new ArchivingGroup(mFactory, this);
			Attribute attribute = group.getArchivedAttribute();
			if( ArchivingQueries.checkDatabaseConformity(attribute) ) {
				mRootGroup = group;
				mRootGroup.addOneAttribute( new ArchivingAttribute( mFactory, Constants.DATE_FORMAT, Constants.ISO_DATE_PATTERN) );
			}
		}
		return mRootGroup;
	}

	@Override
	public LogicalGroup getLogicalRoot() {
		// TODO Auto-generated method stub
		throw new NotImplementedException();
	}

	@Override
	public String getLocation() {
		return mURI.toString();
	}

	@Override
	public String getTitle() {
		return mTitle;
	}

	@Override
	public void setLocation(String location) {
		try {
			URI uri = new URI( location );
			mURI = uri;
		} catch (URISyntaxException e) {
			mURI = null;
			try {
				close();
			} catch (IOException e1) {
			}
			Factory.getLogger().log(Level.SEVERE, "Unable to set new location!", e);
		}
		
	}

	@Override
	public void setTitle(String title) {
		mTitle = title;
	}

	@Override
	public boolean sync() throws IOException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void open() throws IOException {
		synchronized( this ) {
			mIsOpen = true;
			
			String value;
			try {
				value = URLDecoder.decode(mURI.toString(), "UTF-8");
			} catch (UnsupportedEncodingException e) {
				value = mURI.toString();
			}
			
			// Check that the SqlDataset is set
			if( mSqlDataset == null ) {
				mSqlDataset = new SqlDataset(mFactory, value, getUser(), getPassword() );
			}
			// Check it is the same
			else if( getLocation().equals( mSqlDataset.getLocation() ) ) {
				// Close connection
				try {
					mSqlDataset.close();
				} catch( IOException e) {
				}
				// Create a new SqlDataset
				mSqlDataset = new SqlDataset(mFactory, value, getUser(), getPassword() );
				mSqlDataset.setNumericalDate(mNumDate);
			}
			
			// open the connection
			mSqlDataset.open();
		}
	}
	
	@Override
	public void save() throws WriterException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void saveTo(String location) throws WriterException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void save(IContainer container) throws WriterException {
		// TODO
		throw new NotImplementedException();
	}

	@Override
	public void save(String parentPath, IAttribute attribute) throws WriterException {
		// TODO 
		throw new NotImplementedException();
		
	}

	@Override
	public boolean isOpen() {
		boolean result;
		synchronized( this ) {
			result = mIsOpen;
		}
		return result;
	}

	@Override
	public long getLastModificationDate() {
		long result = -1;
		if( isOpen() ) {
			mSqlDataset.getLastModificationDate();
		}
		return result;
	}
	
	public void setArchivingMode( ArchivingMode mode ) {
		mArchivingMode = mode;
	}
	
	public ArchivingMode getArchivingMode() {
		return mArchivingMode;
	}
	
	public void setUser( String user ) {
		mUser = user;
	}
	
	public void setPassword( String password ) {
		mPassword = password;
	}
	
	/**
	 * Are dates expressed as numerical timestamp or as String
	 * 
	 * @param is true if dates are expressed as timestamps
	 */
	public void setNumericalDate(boolean numerical) {
		mNumDate = numerical;
		if( mSqlDataset != null ) {
			mSqlDataset.setNumericalDate(mNumDate);
		}
	}
	
	/**
	 * Are dates expressed as numerical timestamp or as String
	 * 
	 * @return true if dates are returned as timestamps
	 */
	public boolean getNumericalDate() {
		return mNumDate;
	}
	
	/**
	 * Return the name of all IAttributes that have meaning for the extraction of data.
	 * 
	 * @return a String array of attributes' names
	 */
	static public String[] getDrivingAttributes() {
		return Constants.DRIVING_ATTRIBUTE;
	}
	
	/**
	 * Return the name of all IAttributes that have a date representation
	 * 
	 * @return a String array of attributes' names
	 */
	static public String[] getDatedAttributes() {
		return Constants.DATE_ATTRIBUTE;
	}
	
    // ---------------------------------------------------------
    // protected methods
    // ---------------------------------------------------------
	public SqlDataset getSqldataset() {
		return mSqlDataset;
	}
	
	public String getSchema() {
		return mSchemaName;
	}
	
	public void setSchema(String name) {
		mSchemaName = name;
	}
	
    // ---------------------------------------------------------
    // private methods
    // ---------------------------------------------------------
	private String getUser() {
		String user = mUser;
		
		if( user == null && mArchivingMode != null ) {
			String name = mArchivingMode.getName() + "_USER";
			user = System.getProperty(name, System.getenv(name));
		}
		
		return user;
	}
	
	private String getPassword() {
		String password = mPassword;
		
		if( password == null && mArchivingMode != null ) {
			String name = mArchivingMode.getName() + "_PASSWORD";
			password    = System.getProperty(name, System.getenv(name));
		}
		return password;
	}
}
