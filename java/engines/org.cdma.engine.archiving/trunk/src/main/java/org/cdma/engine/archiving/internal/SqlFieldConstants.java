package org.cdma.engine.archiving.internal;

/**
 * Constant fields for the HDB and TDB schemas. It regroup all tables's and fields' names.
 * 
 * @author rodriguez
 */
public class SqlFieldConstants {
	/**
	 * All descriptive tables of an archiving database
	 */
	// Data base table names
	public static final String AMT_TABLE_NAME = "APT";
	public static final String APT_TABLE_NAME = "AMT";
	public static final String ADT_TABLE_NAME = "ADT";
	public static final String[] ARC_TABLES = new String[] { ADT_TABLE_NAME, AMT_TABLE_NAME, APT_TABLE_NAME };
	
	// Data base attributes' properties table (ADT)
	public static final String ADT_FIELDS_ID        = "id";
	public static final String ADT_FIELDS_TYPE      = "data_type";
	public static final String ADT_FIELDS_FORMAT    = "data_format";
	public static final String ADT_FIELDS_WRITABLE  = "writable";
	public static final String ADT_FIELDS_FULL_NAME = "full_name";
	public static final String ADT_FIELDS_DOMAIN    = "domain";
	public static final String ADT_FIELDS_FAMILY    = "family";
	public static final String ADT_FIELDS_MEMBER    = "member";
	public static final String ADT_FIELDS_ATT_NAME  = "att_name";
	public static final String ADT_FIELDS_ORIGIN    = "time";
	
	/**
	 * Available fields of the ADT table (descriptive table of each archived attribute)
	 */
	public static final String[] ADT_FIELDS = new String[] {ADT_FIELDS_ID, ADT_FIELDS_TYPE, ADT_FIELDS_FORMAT, ADT_FIELDS_WRITABLE, ADT_FIELDS_FULL_NAME, ADT_FIELDS_ORIGIN};
	
	// Data base attributes' values table (ATT_***)
	public static final String ATT_TABLE_PREFIX = "ATT_";
	public static final String ATT_FIELD_DIMX   = "DIM_X";
	public static final String ATT_FIELD_DIMY   = "DIM_Y";
	public static final String ATT_FIELD_TIME   = "TIME";
	public static final String ATT_FIELD_VALUE  = "VALUE";
	public static final String ATT_FIELD_READ   = "READ_VALUE";
	public static final String ATT_FIELD_WRITE  = "WRITE_VALUE";
	public static final String CELL_SEPARATOR   = ", ";
	
	
}
