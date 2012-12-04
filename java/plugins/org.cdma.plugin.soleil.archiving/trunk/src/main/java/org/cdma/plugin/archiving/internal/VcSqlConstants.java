package org.cdma.plugin.archiving.internal;


public class VcSqlConstants {
	// Data bases' names
	public static final String HDB_NAME = "HDB";
	public static final String TDB_NAME = "TDB";

	// Data base attributes' properties table
	public static final String ADT_TABLE_NAME      = "ADT";
	public static final String ADT_FIELDS_ID       = "id";
	public static final String ADT_FIELDS_TYPE     = "data_type";
	public static final String ADT_FIELDS_FORMAT   = "data_format";
	public static final String ADT_FIELDS_WRITABLE = "writable";
	public static final String ADT_FIELDS_NAME     = "full_name";
	public static final String[] ADT_FIELDS        = new String[] {ADT_FIELDS_ID, ADT_FIELDS_TYPE, ADT_FIELDS_FORMAT, ADT_FIELDS_WRITABLE, ADT_FIELDS_NAME};

	// Data base attributes' values table
	public static final String ATT_TABLE_PREFIX = "ATT_";
	public static final String ATT_FIELD_DIMX   = "DIM_X";
	public static final String ATT_FIELD_DIMY   = "DIM_Y";
	public static final String ATT_FIELD_TIME   = "TIME";
	public static final String ATT_FIELD_VALUE  = "VALUE";
	public static final String ATT_FIELD_READ   = "READ_VALUE";
	public static final String ATT_FIELD_WRITE  = "WRITE_VALUE";
	//public static final String[] ATT_FIELDS     = new String[] { ATT_FIELD_TIME, ATT_FIELD_READ, ATT_FIELD_WRITE };
	public static final String CELL_SEPARATOR = ", ";
}
