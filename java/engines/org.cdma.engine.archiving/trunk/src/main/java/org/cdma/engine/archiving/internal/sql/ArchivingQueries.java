package org.cdma.engine.archiving.internal.sql;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.text.ParseException;
import java.util.List;
import java.util.Map.Entry;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.archiving.internal.SqlFieldConstants;
import org.cdma.engine.archiving.internal.attribute.Attribute;
import org.cdma.engine.archiving.internal.attribute.AttributeConnector;
import org.cdma.engine.archiving.internal.attribute.AttributePath;
import org.cdma.engine.archiving.internal.attribute.AttributeProperties;
import org.cdma.engine.sql.internal.SqlConnector;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.engine.sql.utils.DateFormat;
import org.cdma.engine.sql.utils.DbUtils;
import org.cdma.engine.sql.utils.DbUtils.BaseType;
import org.cdma.engine.sql.utils.SamplingType;
import org.cdma.engine.sql.utils.SamplingType.SamplingPeriod;
import org.cdma.engine.sql.utils.SamplingType.SamplingPolicy;

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
				String dbName    = dbCon.getDbName();
				BaseType dbType  = dbCon.getDbType();
				
				// Get the sampling type
				SamplingPeriod sampling = prop.getSampling();
				SamplingType samplingType = DbUtils.getSqlSamplingType(sampling, dbType);
				
				// Check the db name is specified
				if( dbName != null && !dbName.isEmpty() ) {
					dbName += ".";
				}
				else {
					dbName = "";
				}
				
				// Compute starting and ending dates (if null then '01/01/1970' and 'now')
				try {
					// Check if date are expected as numerical
					String group = "";
					String time;
					// Case of sampling
					if( sampling != SamplingPeriod.ALL ) {
						// Get pattern of sampling
						String pattern = samplingType.getPattern(sampling);
						
						// Convert dates to string (truncated dates according sampling)
						String tmp = DateFormat.dateToSqlString( SqlFieldConstants.ATT_FIELD_TIME, dbType, pattern);
						
						// Reconstruct timestamp using truncated dates
						time = DateFormat.stringToSqlDate(tmp, dbType, pattern);
						
						// Sort by truncated dates
						group = " GROUP BY " + tmp;
					}
					else {
						time = SqlFieldConstants.ATT_FIELD_TIME ;
					}
					
					if( datePattern != null) {
						time = DateFormat.dateToSqlString( time, dbType, datePattern);
					}
					
					if( datePattern != null || sampling != SamplingPeriod.ALL ) {
						time += " as " + SqlFieldConstants.ATT_FIELD_TIME + " ";
					}
					
					
					// Prepare each part of the query (SELECT, FROM, WHERE, ORDER)
					String select = "SELECT " + time;
					String from   = " FROM " + dbName + tableName;
					String where  = " WHERE (" + SqlFieldConstants.ATT_FIELD_TIME + " BETWEEN ? AND ?)";
					String order  = " ORDER BY " + SqlFieldConstants.ATT_FIELD_TIME;
					
					
					// Populate the SELECT section according sampling
					select = populateSelectClause(prop, sampling, select, samplingType);
					
					// Assembly the  query
					query.append( select );
					query.append( from );
					query.append( where);
					if( sampling != SamplingPeriod.ALL ) {
						query.append( group );						
					}
					query.append( order );
				} catch (ParseException e) {
					Factory.getLogger().log(Level.SEVERE, "Unable to prepare query to get item list!", e );
				}
			}
		}
		return query.toString();
	}

	/**
	 * Check the database contains the right tables for an archiving database
	 * 
	 * @param attribute
	 * @return
	 */
	static public boolean checkDatabaseConformity(Attribute attribute) {
		boolean result = true;
		
		if( attribute != null ) {
			AttributeConnector dbCon = attribute.getDbConnector();
			SqlDataset dataset   = dbCon.getSqlDataset();
			SqlConnector connector = dataset.getSqlConnector();
			try {
				Connection connection  = connector.getConnection();
				for( String table : SqlFieldConstants.ARC_TABLES ) {
					if( !ArchivingQueries.existe(connection, table) ) {
						result = false;
						break;
					}
				}
			} catch( SQLException e ) {
				result = false;
			} catch (IOException e) {
				result = false;
			}
			
			
		}
		return result;
	}
	
	private static boolean existe(Connection connection, String nomTable) throws SQLException{
	   boolean existe;
	   DatabaseMetaData dmd = connection.getMetaData();
	   ResultSet tables = dmd.getTables(connection.getCatalog(),null,nomTable,null);
	   existe = tables.next();
	   tables.close();
	   return existe;
	}
	
	/**
	 * Will populate the SELECT SQL clause according to the db type, the sampling policy...
	 * @param prop
	 * @param sampling
	 * @param select
	 * @param samplingType
	 * @return
	 */
	private static String populateSelectClause(AttributeProperties prop, SamplingPeriod sampling, String select, SamplingType samplingType) {
		String result = select;
		
		List<String> fields = prop.getDbFields();
		
		if( sampling == SamplingPeriod.ALL ) {
			for( String field : fields ) {
				result += ", " + field;
			}
		}
		else {
			// Manages CLOB fields
			String stringField;
			fields = prop.getDbClobFields();
			for( String field : fields ) {
				stringField = samplingType.getFieldAsStringSelector(field);
				result += ", " + samplingType.getSamplingSelector(stringField, SamplingPolicy.MIN, field);
			}
			
			// Manages numerical fields
			fields = prop.getDbNumericalFields();
			for( String field : fields ) {
				result += ", " + samplingType.getSamplingSelector(field, SamplingPolicy.AVERAGE, field);
			}
		}
		return result;
	}

}
