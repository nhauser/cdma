package org.cdma.engine.archiving.internal.attribute;

import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.archiving.internal.SqlFieldConstants;
import org.cdma.engine.archiving.internal.sql.ArchivingQueries;
import org.cdma.engine.sql.navigation.SqlCdmaCursor;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.engine.sql.navigation.SqlGroup;
import org.cdma.interfaces.IDataItem;

public class AttributeProperties implements Cloneable {
	private int mId; 
	private int mType;
	private int mFormat;
	private int mWritable;
	private String mName;
	private Class<?> mClass;
	private long mOrigin;
	
	public AttributeProperties( int ID, int type, int format, int writable, String name, Class<?> clazz) {
		mId       = ID;
		mType     = type;
		mFormat   = format;
		mWritable = writable;
		mName     = name;
		mClass    = clazz;
	}
	
	public AttributeProperties( String attrName, SqlDataset dbDataset, String dbName ) throws IOException {
		mName     = attrName;
		mId       = -1;
		mType     = -1;
		mFormat   = -1;
		mWritable = -1;
		initialize(dbDataset, dbName);
	}
	
	/**
	 * Name of the considered attribute
	 */
	public String getName() {
		return mName;
	}

	/**
	 * Return the write property of this attribute
	 * @return int value: 0=read_only, 1=read_with_write, 2=write, 3=read_write
	 */
	public int getWritable() {
		return mWritable;
	}
	
	/**
	 * Return the canonical rank of element considered by this attribute.
	 * It means the interpretation rank: scalar, spectrum, image
	 * 
	 * @return rank of a single element of this attribute
	 */
	public int getFormat() {
		return mFormat;
	}
	
	/**
	 * return the interpretation type of data for this attribute
	 * 
	 * @return int value: 1=bool; 2=short; 3=long; 4=float; 5=double; 6=ushort; 7=ulong; 8=string
	 */
	public int getType() {
		return mType;
	}
	
	/**
	 * return the starting date of the archiving process for this attribute
	 * 
	 * @return long representing a time value in milliseconds
	 */
	public long getOrigin() {
		return mOrigin;
	}
	
	/**
	 * return the interpretation class for this attribute's data
	 */
	public Class<?> getTypeClass() {
		if( mClass == null ) {
			switch( mType ) {
				case 1:
					mClass = Boolean.TYPE;
					break;
				case 2:
				case 6:
					mClass = Short.TYPE;
					break;
				case 3:
				case 7:
					mClass = Long.TYPE;
					break;
				case 4:
					mClass = Float.TYPE;
					break;
				case 5:
					mClass = Double.TYPE;
					break;
				case 8:
				default:
					mClass = String.class;
					break;
			}
		}
		return mClass;
	}
	
	
	/**
	 * Return the table name where all attribute's values are stored
	 * @return
	 */
	public String getDbTable() {
		return SqlFieldConstants.ATT_TABLE_PREFIX + mId;
	}
	
	/**
	 * Return an array of the available fields in database for that attribute
	 * @return
	 */
	public String[] getDbFields() {
		List<String> result = new ArrayList<String>();
		
		switch( mWritable ) {
			case 0:
			case 2:
				result.add( SqlFieldConstants.ATT_FIELD_VALUE );
				break;
			case 1:
			case 3:
				result.add( SqlFieldConstants.ATT_FIELD_READ );
				result.add( SqlFieldConstants.ATT_FIELD_WRITE );
				break;
			default:
				break;
		}
		
		switch( mFormat ) {
		case 0:
			break;
		case 1:
			result.add( SqlFieldConstants.ATT_FIELD_DIMX );
			break;
		case 2:
			result.add( SqlFieldConstants.ATT_FIELD_DIMX );
			result.add( SqlFieldConstants.ATT_FIELD_DIMY );
			break;
		default:
			break;
		}
		
		return result.toArray(new String[] {});
	}
	
	public AttributeProperties clone() {
		return new AttributeProperties(mId, mType, mFormat, mWritable, mName, mClass );
	}
	
	// ------------------------------------------------------------------------
	// Private methods
	// ------------------------------------------------------------------------
	/**
	 * Initialize the properties according database content
	 * 
	 * @param dbDataset SQL dataset handling connection to database
	 * @param dbName name of database to search in
	 * @throws IOException
	 */
	private void initialize( SqlDataset dbDataset, String dbName ) throws IOException {
		if( mName != null && dbDataset != null ) {
			// Get the query to extract attribute's properties
			String query_attr_aptitude = ArchivingQueries.queryAttributeProperties( dbName, mName );

			// Execute the query
		    SqlCdmaCursor cursor = dbDataset.executeQuery( query_attr_aptitude.toString() );
		    
	    	SqlGroup group;
	    	IDataItem item;
		    
		    try {
			    if( cursor.next() ) {
			    	group = cursor.getGroup();
			    	// Read attribute table ID
			    	item = group.getDataItem( SqlFieldConstants.ADT_FIELDS_ID );
			    	mId = item.readScalarInt();
			    	
			    	// Read attribute data type
			    	item = group.getDataItem( SqlFieldConstants.ADT_FIELDS_TYPE );
			    	mType = item.readScalarInt();
			    	
			    	// Read attribute data format
			    	item = group.getDataItem( SqlFieldConstants.ADT_FIELDS_FORMAT );
			    	mFormat = item.readScalarInt();
			    	
			    	// Read attribute data write capability
			    	item = group.getDataItem( SqlFieldConstants.ADT_FIELDS_WRITABLE );
			    	mWritable = item.readScalarInt();
			    	
			    	// Read attribute starting time
			    	item = group.getDataItem( SqlFieldConstants.ADT_FIELDS_ORIGIN );
		    		mOrigin = item.readScalarLong();
			    }
		    } catch( SQLException e ) {
		    	throw new IOException("Unable to initialize the attribute's properties!", e);
		    }		    
		    try {
				cursor.close();
			} catch (SQLException e) {
				Factory.getLogger().log(Level.WARNING, e.getMessage(), e);
			}
		}
		else {
			throw new IOException( "Invalid parameters: no null values are allowed!" );
		}
	}
}
