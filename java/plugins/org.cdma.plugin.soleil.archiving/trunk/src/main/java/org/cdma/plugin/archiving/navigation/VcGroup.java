package org.cdma.plugin.archiving.navigation;

import java.io.IOException;
import java.text.ParseException;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.arrays.DefaultArray;
import org.cdma.engine.sql.array.SqlArray;
import org.cdma.engine.sql.navigation.SqlCdmaCursor;
import org.cdma.engine.sql.navigation.SqlDataItem;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IContainer;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDimension;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.archiving.VcFactory;
import org.cdma.plugin.archiving.internal.AttributeProperties;
import org.cdma.plugin.archiving.internal.VcSqlConstants;
import org.cdma.plugin.archiving.internal.VcXmlConstants;
import org.cdma.plugin.archiving.internal.sql.DateFormat;
import org.cdma.plugin.archiving.internal.sql.DbUtils;
import org.cdma.plugin.archiving.internal.sql.DbUtils.BaseType;
import org.cdma.plugin.xml.navigation.XmlGroup;
import org.cdma.utils.ArrayTools;

public class VcGroup extends XmlGroup {

    public static final int INVALID_INDEX = -1;
    public static final String ATTRIBUTE_READ_DATA_LABEL = "read_data";
    public static final String ATTRIBUTE_WRITE_DATA_LABEL = "write_data";
    public static final String ATTRIBUTE_SINGLE_DATA_LABEL = "data";
    public static final int READ_ATTRIBUTE_DB_INDEX = 0; 
    public static final int WRITE_ATTRIBUTE_DB_INDEX = 1;

	private String mHiddenAttribute;
	private boolean mHiddenAttributeLoaded;

	public VcGroup(String name, VcDataset dataset, IGroup parent) {
		super(VcFactory.NAME, name, INVALID_INDEX, dataset, parent);
		mHiddenAttributeLoaded = false;
		mHiddenAttribute = null;
	}

	public VcGroup(VcGroup group) {
		super(group);
	}

	public VcDataset getDataset() {
		return (VcDataset) super.getDataset();
	}
	
	public void setHiddenAttribute(String waitingAttribute) {
		this.mHiddenAttribute = waitingAttribute;
	}

	public boolean isHiddenAttributeLoaded() {
		return mHiddenAttributeLoaded;
	}

	public String getHiddenAttribute() {
		return mHiddenAttribute;
	}

    @Override
	public IDataItem getDataItem(String shortName) {
    	initDataItems();
		return super.getDataItem(shortName);
	}

	@Override
	public IDataItem getDataItemWithAttribute(String name, String value) {
		initDataItems();
		return super.getDataItemWithAttribute(name, value);
	}

	@Override
	public List<IDataItem> getDataItemList() {
		initDataItems();
		return super.getDataItemList();
	}

	@Override
	public IContainer getContainer(String shortName) {
		initDataItems();
		return super.getContainer(shortName);
	}

	@Override
	public void printHierarchy(int level) {

		String tabFormatting = "";
		for (int i = 0; i < level; ++i) {
			tabFormatting += "\t";
		}
		// Tag beginning
		System.out.print(tabFormatting);
		System.out.print("<");
		System.out.print(getShortName());

		// Attributes of this group
		for (IAttribute attr : getAttributeList()) {
			System.out.print(" ");
			System.out.print(attr.getName());
			System.out.print("=\"");
			System.out.print(attr.getStringValue());
			System.out.print("\"");
		}
		System.out.print(">\n");

		// Subgroup description
		for (IGroup group : getGroupList()) {
			if (group instanceof VcGroup) {
				((VcGroup) group).printHierarchy(level + 1);
			}
		}

		// Items description
		for (IDataItem item : getDataItemList()) {
			if (item instanceof VcDataItem) {
				((VcDataItem) item).printHierarchy(level + 1);
			}
		}

		// Tag Ending
		System.out.print(tabFormatting);
		System.out.print("</");
		System.out.print(getShortName());
		System.out.print(">\n");
	}
	
	
	///////////////////////////////////////////////////////////////////////////////
	private void initDataItems() {
		if( mHiddenAttribute != null ) {
			if( ! mHiddenAttributeLoaded ) {
			    // Database request parameters
			    IGroup rootGroup = getRootGroup();
			    IAttribute historicProperty = rootGroup.getAttribute(VcXmlConstants.VC_HISTORIC_PROPERTY_XML_TAG);
			    boolean historic = false;
			    if( historicProperty != null ) {
			    	historic = Boolean.parseBoolean( historicProperty.getStringValue() );
			    }
			    
			    // Determines on which DB to execute query: HDB / TDB
			    VcDataset dataset = getDataset();
			    SqlDataset dbDataset;
			    
			    String     dbName;
			    if( historic ) {
			    	dbDataset = dataset.getHdbdataset();
			    	dbName = VcSqlConstants.HDB_NAME;
			    }
			    else {
			    	dbDataset = dataset.getTdbdataset();
			    	dbName = VcSqlConstants.TDB_NAME;
			    }
			    
				try {
				    // Get the tango-attribute's properties
					AttributeProperties attribute = new AttributeProperties(mHiddenAttribute, dbDataset, dbName);
					
					// Construct the dataitem
					constructDataItem(attribute, dbDataset, dbName );
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		mHiddenAttributeLoaded = true;
	}
	
	private void constructDataItem(AttributeProperties attribute, SqlDataset dbDataset, String dbName) {
	    try {
		    // Prepare the query (between dates)
	    	BaseType dbType = DbUtils.detectDb( dbDataset );
	    	String query = prepareQueryDataItem(dbType, dbName, attribute);

		    if( query != null && ! query.isEmpty() ) {
			    // Execute the query
			    SqlCdmaCursor cursor = dbDataset.execute_query(query);
			    
			    // Prepare items for each found column
		    	List<SqlDataItem> sql_items = cursor.getDataItemList();
		    	IDimension dimension;
		    	IDataItem child;
		    	for( SqlDataItem item : sql_items ) {
		    		if( attribute.isDimension( item.getShortName() ) ) {
		    			IArray data = constructVcArray( attribute, (SqlArray) item.getData() );
						dimension = new VcDimension( data, item.getShortName());
						addOneDimension(dimension);
		    		}
		    		else {
		    			IArray data = constructVcArray( attribute, (SqlArray) item.getData() );
		    			child = new VcDataItem(item.getShortName(), this, data);
		    			addDataItem(child);
		    		}
		    	}
		    }
		} catch (ParseException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to interprete dates in query", e);
		} catch (IOException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to initialize data items!", e);
		}
	}
	
	private IArray constructVcArray(AttributeProperties attribute, SqlArray array) {
		IArray result = null;
		if( array != null ) {
			int[] shape = array.getShape();
			Class<?> clazz = array.getElementType();
			Object storage = array.getStorage(); 
			if( java.io.Reader.class.isAssignableFrom( clazz ) ) {
				storage = DbUtils.extractDataFromReader(attribute, (java.io.Reader[]) storage, VcSqlConstants.CELL_SEPARATOR );
				shape = ArrayTools.detectShape(storage);
			}
			try {
				result = DefaultArray.instantiateDefaultArray( VcFactory.NAME , storage, shape);
			} catch (InvalidArrayTypeException e) {
				Factory.getLogger().log( Level.SEVERE, "Unable to create array!", e );
			}
		}
		
		return result;
	}

	private String prepareQueryDataItem( BaseType dbType, String dbName, AttributeProperties attribute ) throws ParseException {
		StringBuffer query = new StringBuffer();
		
		if( attribute != null ) {
			String tableName = attribute.getDbTable();
			VcDataset dataset = getDataset();
			boolean frFormat = ! dataset.getUSDateFormat();
			String[] fields = attribute.getDbFields();

			// Compute starting and ending dates (if null then '01/01/1970' and 'now')
			String start  = prepareStartDate( dbType, frFormat );
			String end    = prepareEndDate( dbType, frFormat );
			String select = "SELECT " + DateFormat.dateToSqlString( VcSqlConstants.ATT_FIELD_TIME, dbType, frFormat) + " as " + VcSqlConstants.ATT_FIELD_TIME;
			String from   = " FROM " + dbName + "." + tableName;
			String where  = " WHERE (time BETWEEN '" + start + "' AND '" + end + "')";
			String order  = " ORDER BY time";
			
			for( String field : fields ) {
				select += ", " + field;
			}
			
			query.append( select );
			query.append( from );
			query.append( where);
			query.append( order );
		}
		return query.toString();
	}
	
	private String prepareStartDate(BaseType dbType, boolean frFormat) throws ParseException {
		// Get the starting and ending dates of the archiving
		IGroup rootGroup = getRootGroup();
		
		// Check if this has 'start' date attribute
		IAttribute startTime = getAttribute(VcXmlConstants.VC_START_DATE_PROPERTY_XML_TAG);
		
		if( startTime == null ) {
			startTime = rootGroup.getAttribute(VcXmlConstants.VC_START_DATE_PROPERTY_XML_TAG);
		}
		
		// Compute starting date (if null then 01/01/1970)
		String start;
		if( startTime != null ) {
			Class<?> clazz = startTime.getType();
			if( Number.class.isAssignableFrom(clazz) ) {
				start = DateFormat.convertDate((Long) startTime.getNumericValue(), dbType, frFormat );
			}
			else {
				start = DateFormat.convertDate(startTime.getStringValue(), dbType, frFormat );
			}
		}
		else {
			start = DateFormat.convertDate(0, dbType, frFormat);
		}
		
		return start;
	}
	
	private String prepareEndDate(BaseType dbType, boolean frFormat) throws ParseException {
		// Get the starting and ending dates of the archiving
		IGroup rootGroup = getRootGroup();

		// Check if this has 'end' date attribute
		IAttribute endTime = getAttribute(VcXmlConstants.VC_END_DATE_PROPERTY_XML_TAG);
		if( endTime == null ) {
			endTime = rootGroup.getAttribute(VcXmlConstants.VC_END_DATE_PROPERTY_XML_TAG);
		}
		
		// Compute starting date (if null then 01/01/1970)
		String end;
		if( endTime != null ) {
			Class<?> clazz = endTime.getType();
			if( Number.class.isAssignableFrom(clazz) ) {
				end = DateFormat.convertDate((Long) endTime.getNumericValue(), dbType, frFormat );
			}
			else {
				end = DateFormat.convertDate(endTime.getStringValue(), dbType, frFormat );
			}
		}
		else {
			end = DateFormat.convertDate(0, dbType, frFormat);
		}
		
		return end;
	}
}
