package org.cdma.plugin.archiving.internal.sql;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.sql.navigation.SqlDataset;
import org.cdma.plugin.archiving.internal.AttributeProperties;
import org.cdma.plugin.archiving.internal.sql.SamplingType.SamplingPeriod;
import org.cdma.utilities.conversion.StringArrayToPrimitveArray;

public class DbUtils {
	public static class Convert2DArray {
		Object mArray;
		
		public Convert2DArray( Object array ) {
			mArray = array;
		}
		
		public void set( int x, int y, String cell ) {
			((double[][]) mArray)[x][y] = Double.valueOf(cell);
		}
	}
	
	
	public enum BaseType {
		ORACLE ("oracle"),
		MYSQL  ("mysql");
		
	    private String name;
		
		private BaseType(String type) {
			name = type;
		}
		
		public String getName() {
			return name;
		}
	}
	
	/**
	 * Return the Data Base type according the given URL (must contains the protocol) 
	 */
	public static BaseType detectDb( SqlDataset dataset ) {
		String host = dataset.getSqlConnector().getHost();
		BaseType type;
		if( host.matches( "[^:]*:oracle.*" ) ) {
			type = BaseType.ORACLE;
		}
		else {
			type = BaseType.MYSQL;
		}
		return type;
	}
	
	/**
	 * Returns the sampling date format according the given database and the given sampling identifier. 
	 * @param samplingType
	 * @param type
	 * @return
	 */
	public static SamplingType getSqlSamplingType(final short samplingType, final BaseType type) {
		SamplingType sampler;
		SamplingPeriod period = SamplingPeriod.instantiate(samplingType);
		switch( type ) {
			case MYSQL:
				sampler = Enum.valueOf( SamplingTypeMySQL.class, period.name().trim() );
				break;
			case ORACLE:
			default:
				sampler = Enum.valueOf( SamplingTypeOracle.class, period.name().trim() );
				break;
		}
		return sampler;
	}
	
	public static Object extractDataFromReader(AttributeProperties attribute, Reader[] storage, String cellSeparator) {
		Object result = null;
		
		BufferedReader reader;
		StringBuffer content;
		String line;
		
		List<String> wholeData = new ArrayList<String>();
		try {
			//For each reader
			for( Reader current : storage ) {
				content = new StringBuffer();
				
				// Read all lines from the reader
				reader = new BufferedReader(current);
				line = reader.readLine();
				while( line != null ) {
					content.append(line);
					content.append(cellSeparator);
					line = reader.readLine();
				}
				
				// Add the content to list
				wholeData.add(content.toString());
			}
		} catch (IOException e) {
			Factory.getLogger().log(Level.SEVERE, "Unable to read the whole matrix of data!", e);
		}
		
		result = convertClobData( attribute, wholeData, cellSeparator );
		
		return result;
	}
	
	static public Object convertClobData(AttributeProperties attribute, List<String> wholeData, String cellSeparator ) {
		Object result = null;
		
		// Convert the string list into string array
		String[] array = new String[ wholeData.size() ];
		wholeData.toArray(array);
		
		// If expected type is String return it as it is
		Class<?> clazz = attribute.getTypeClass();
		if( clazz.equals(String.class) ) {
			result = array;
		}
		// Else try to convert each line into cells
		else {
			List< String[] > lines = new ArrayList< String[] >();
			int maxSize = 0;

			// For each line
			for( String buffer : array ) {
				// Split it according the cell separator
				String[] line = buffer.split(cellSeparator);
				lines.add(line);
				
				// Update max size (to avoid raged arrays)
				if( line.length > maxSize ) {
					maxSize = line.length;
				}
			}
			
			// Instantiate a well defined array
			int[] shape = new int[] { wholeData.size(), maxSize };
			StringArrayToPrimitveArray converter = new StringArrayToPrimitveArray( lines.toArray(), shape, clazz );
			result = converter.convert();
		}
		
		return result;
	}
	

}
