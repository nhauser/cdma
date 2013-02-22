package org.cdma.engine.archiving.internal.sql;

import java.text.ParseException;
import java.util.Map.Entry;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.archiving.internal.SqlFieldConstants;
import org.cdma.engine.archiving.internal.attribute.Attribute;
import org.cdma.engine.archiving.internal.attribute.AttributeConnector;
import org.cdma.engine.archiving.internal.attribute.AttributePath;
import org.cdma.engine.archiving.internal.attribute.AttributeProperties;
import org.cdma.engine.sql.utils.DateFormat;
import org.cdma.engine.sql.utils.DbUtils.BaseType;

/**
 * This helper class contains all executed queries in the Soleil's Archiving plug-in.
 * @author rodriguez
 *
 */
public class ArchivingQueries {

	/**
	 * Prepare a query that permits extracting an attribute's properties.
	 * The returned fields' properties are the following in the same order:
	 * {@link SqlFieldConstants.ADT_FIELDS}
	 * 
	 * @param dbName name of the archiving data base (HDB, TDB, ...)
	 * @param attribute full name of the attribute
	 * @return a SQL query string
	 */
	static public String queryAttributeProperties(String dbName, String attribute) {
		StringBuffer query_attr_aptitude = new StringBuffer();

		// Construct the fields for the query
		query_attr_aptitude.append( "SELECT " );
	    for( int i = 0; i < SqlFieldConstants.ADT_FIELDS.length; i++ ) {
	    	query_attr_aptitude.append( SqlFieldConstants.ADT_TABLE_NAME + "." + SqlFieldConstants.ADT_FIELDS[i] );
	    	if( i < SqlFieldConstants.ADT_FIELDS.length - 1 ) {
	    		query_attr_aptitude.append( ", " );
	    	}
	    }

		// Check the db name is specified
		if( dbName != null && !dbName.isEmpty() ) {
			dbName += ".";
		}
		else {
			dbName = "";
		}
	    
	    // Add the from section to query
	    query_attr_aptitude.append( " FROM " + dbName + SqlFieldConstants.ADT_TABLE_NAME );
	    
	    // Add the clause section to the query
	    query_attr_aptitude.append( " WHERE " + SqlFieldConstants.ADT_TABLE_NAME + "." + SqlFieldConstants.ADT_FIELDS_FULL_NAME + " = '" + attribute + "'");
	    
	    return query_attr_aptitude.toString();
	}
	
	/**
	 * Prepare a query that permits to list children group of the given attribute.
	 * 
	 * @param dbName name of the archiving data base (HDB, TDB, ...)
	 * @param attribute Attribute object that define the archived attribute group from which children should be listed
	 * @return null if this is a leaf group or a query to find children groups' names
	 */
	static public String queryChildGroups( Attribute attribute ) {
		String query = null;
		String dbName = attribute.getDbConnector().getDbName();
		AttributePath path = attribute.getPath();
		String field = path.getNextDbFieldName();
		
		// Check the db name is specified
		if( dbName != null && !dbName.isEmpty() ) {
			dbName += ".";
		}
		else {
			dbName = "";
		}
		
		if( field != null && path != null && dbName != null ) {
			String select = "SELECT DISTINCT(" + SqlFieldConstants.ADT_TABLE_NAME + "." + field + ")";
			String from = " FROM " + dbName + SqlFieldConstants.ADT_TABLE_NAME;
	
			// construct the 'where' clause
			StringBuffer where = new StringBuffer();
			int i = 0;
			for( Entry<String, String> entry : path.getSqlCriteria().entrySet() ) {
				if( i != 0 ) {
					where.append( " AND ");
				}
				else {
					where.append( " WHERE ");
				}
				where.append( entry.getKey() );
				where.append( "='");
				where.append(entry.getValue());
				where.append("'");
				i++;
			}
			query = select + from + where;
		}
		return query;
	}
	
	/**
	 * Prepare a query that permits to list children group of the given attribute.
	 * 
	 * @param attribute Attribute object describing the archived attribute
	 * @param datePattern format expected in the query (i.e 'yyyy-MM-dd HH:mm:ss.SSS') 
	 * @return SQL string for a preparedStatement
	 * @throws ParseException
	 * @note 2 parameters are expected (1st start date, 2nd end date) and not set
	 */
	
	static 	public String queryChildItems( Attribute attribute, String datePattern ) {
		StringBuffer query = new StringBuffer();
		
		if( attribute != null ) {
			AttributeConnector dbCon = attribute.getDbConnector();
			AttributeProperties prop = attribute.getProperties();
			if( dbCon != null && prop != null ) {
				String tableName = prop.getDbTable();
				String[] fields  = prop.getDbFields();
				String dbName    = dbCon.getDbName();
				BaseType dbType  = dbCon.getDbType();
				
				// Check the db name is specified
				if( dbName != null && !dbName.isEmpty() ) {
					dbName += ".";
				}
				else {
					dbName = "";
				}
				
				// Compute starting and ending dates (if null then '01/01/1970' and 'now')
				try {
					// Prepare each part of the query (SELECT, FROM, WHERE, ORDER)
					String select = "SELECT " 
									+ DateFormat.dateToSqlString( SqlFieldConstants.ATT_FIELD_TIME, dbType, datePattern) 
									+ " as " + SqlFieldConstants.ATT_FIELD_TIME + " ";
					String from   = " FROM " + dbName + tableName;
					String where = " WHERE (time BETWEEN ? AND ?)";

					String order  = " ORDER BY time";
					
					// Populate the SELECT section
					for( String field : fields ) {
						select += ", " + field;
					}
					
					// Assembly the  query
					query.append( select );
					query.append( from );
					query.append( where);
					query.append( order );
				} catch (ParseException e) {
					Factory.getLogger().log(Level.SEVERE, "Unable to prepare query to get item list!", e );
				}
			}
		}
		
		return query.toString();
	}
	/*
	static private String prepareStartDate(IAttribute startTime, BaseType dbType, boolean frFormat) throws ParseException {
		// Compute starting date (if null then 01/01/1970)

		String start;
		if( startTime != null ) {
			Class<?> clazz = startTime.getType();
			if( Number.class.isAssignableFrom(clazz) ) {
				//Long time = startTime.getNumericValue();
				start = DateFormat.convertDate((Long) startTime.getNumericValue(), dbType, frFormat );
			}
			else {
				start = DateFormat.convertDate(startTime.getStringValue(), dbType, frFormat );
			}
		}
		else {
			start = DateFormat.convertDate(System.currentTimeMillis() - 3600*1000, dbType, frFormat);
		}
		
		return start;
	}
	
	static private String prepareEndDate(IAttribute endTime, BaseType dbType, boolean frFormat) throws ParseException {
		// Compute starting date (if null then now)
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
			end = DateFormat.convertDate(System.currentTimeMillis(), dbType, frFormat);
		}
		
		return end;
	}
	*/
	
}
