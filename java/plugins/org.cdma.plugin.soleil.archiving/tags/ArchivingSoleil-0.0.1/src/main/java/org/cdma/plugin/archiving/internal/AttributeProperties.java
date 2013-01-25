package org.cdma.plugin.archiving.internal;

import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.sql.navigation.SqlCdmaCursor;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.engine.sql.navigation.SqlGroup;
import org.cdma.interfaces.IDataItem;

public class AttributeProperties {
	private int mId; 
	private int mType;
	private int mFormat;
	private int mWritable;
	private String mName;
	private Class<?> mClass;
	
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
		return VcSqlConstants.ATT_TABLE_PREFIX + mId;
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
//				result.add( VcSqlConstants.ATT_FIELD_TIME );
				result.add( VcSqlConstants.ATT_FIELD_VALUE );
				break;
			case 1:
			case 3:
//				result.add( VcSqlConstants.ATT_FIELD_TIME );
				result.add( VcSqlConstants.ATT_FIELD_READ );
				result.add( VcSqlConstants.ATT_FIELD_WRITE );
				break;
			default:
				break;
		}
		
		switch( mFormat ) {
		case 0:
			break;
		case 1:
			result.add( VcSqlConstants.ATT_FIELD_DIMX );
			break;
		case 2:
			result.add( VcSqlConstants.ATT_FIELD_DIMX );
			result.add( VcSqlConstants.ATT_FIELD_DIMY );
			break;
		default:
			break;
		}
		
		return result.toArray(new String[] {});
	}
	
	/**
	 * Returns true if the given name is the name of dimension for that attribute 
	 */
	public boolean isDimension(String name) {
		boolean result = false;

		if ((name != null && !name.isEmpty())
				&& (name.equalsIgnoreCase(VcSqlConstants.ATT_FIELD_DIMX) || 
					name.equalsIgnoreCase(VcSqlConstants.ATT_FIELD_DIMY) ||
					name.equalsIgnoreCase(VcSqlConstants.ATT_FIELD_TIME))) {
			result = true;
		}

		return result;
	}
	
	/**
	 * Initialize the properties according database content
	 * 
	 * @param dbDataset SQL dataset handling connection to database
	 * @param dbName name of database to search in
	 * @throws IOException
	 */
	private void initialize( SqlDataset dbDataset, String dbName ) throws IOException {
		if( mName != null && dbDataset != null && dbName != null ) {
		    StringBuffer query_attr_aptitude = new StringBuffer();
		    query_attr_aptitude.append( "SELECT " );
		    
		    // Construct the fields for the query
		    for( int i = 0; i < VcSqlConstants.ADT_FIELDS.length; i++ ) {
		    	query_attr_aptitude.append( VcSqlConstants.ADT_TABLE_NAME + "." + VcSqlConstants.ADT_FIELDS[i] );
		    	if( i < VcSqlConstants.ADT_FIELDS.length - 1 ) {
		    		query_attr_aptitude.append( ", " );
		    	}
		    }
		    
		    // Add the from section to query
		    query_attr_aptitude.append( " FROM " + dbName + "." + VcSqlConstants.ADT_TABLE_NAME );
		    
		    // Add the clause section to the query
		    query_attr_aptitude.append( " WHERE " + VcSqlConstants.ADT_TABLE_NAME + "." + VcSqlConstants.ADT_FIELDS_NAME + " = '" + mName + "'");

		    // Execute the query
		    SqlCdmaCursor cursor = dbDataset.execute_query( query_attr_aptitude.toString() );
		    
	    	SqlGroup group;
	    	IDataItem item;
		    
		    try {
			    if( cursor.next() ) {
			    	group = cursor.getGroup();
			    	// Read attribute table ID
			    	item = group.getDataItem( VcSqlConstants.ADT_FIELDS_ID );
			    	mId = item.readScalarInt();
			    	
			    	// Read attribute data type
			    	item = group.getDataItem( VcSqlConstants.ADT_FIELDS_TYPE );
			    	mType = item.readScalarInt();
			    	
			    	// Read attribute data format
			    	item = group.getDataItem( VcSqlConstants.ADT_FIELDS_FORMAT );
			    	mFormat = item.readScalarInt();
			    	
			    	// Read attribute data write capability
			    	item = group.getDataItem( VcSqlConstants.ADT_FIELDS_WRITABLE );
			    	mWritable = item.readScalarInt();
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
