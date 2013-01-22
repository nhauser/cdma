package org.cdma.plugin.archiving.internal.sql;

import java.text.ParseException;
import java.text.SimpleDateFormat;

import org.cdma.plugin.archiving.internal.sql.DbUtils.BaseType;

public class DateFormat {
	public static final String SQL_TIME_FR = "'DD-MM-YYYY HH24:MI:SS.FF'";
	public static final String SQL_TIME_US = "'MM-DD-YYYY HH24:MI:SS.FF'";

	
    public static final String FR_DATE_PATTERN  = "dd-MM-yyyy HH:mm:ss.SSS";
    public static final String US_DATE_PATTERN  = "MM-dd-yyyy HH:mm:ss.SSS";
    public static final String ISO_DATE_PATTERN = "yyyy-MM-dd HH:mm:ss.SSS";
    
    public static final java.util.GregorianCalendar CALENDAR = new java.util.GregorianCalendar();
    public static final java.text.SimpleDateFormat FR_FORMAT = new java.text.SimpleDateFormat(FR_DATE_PATTERN);
    public static final java.text.SimpleDateFormat US_FORMAT = new java.text.SimpleDateFormat(US_DATE_PATTERN);
    public static final java.text.SimpleDateFormat ISO_FORMAT = new java.text.SimpleDateFormat(ISO_DATE_PATTERN);
    
	static public final String getSqlDatePattern( boolean toFrSqlFormat ) {
		String result;
		if( toFrSqlFormat ) {
			result = SQL_TIME_US;
		}
		else {
			result = SQL_TIME_FR;
		}
		return result;
	}
	/**
	 * Insert a given string into the   
	 * @param type
	 * @param date
	 * @param toFrSqlFormat
	 * @return
	 * @throws ParseException
	 */
	static public String dateToSqlString( final String date, BaseType type, boolean toFrSqlFormat ) throws ParseException {
		String result;
		synchronized( CALENDAR ) {
			// Select localized time formater
			SimpleDateFormat formater = US_FORMAT;
			
			if( toFrSqlFormat ) {
				formater = FR_FORMAT;
			}
			
			// Get the DBMS (generic) time pattern
			SamplingType sampler = DbUtils.getSqlSamplingType((short) -1, type);
			String pattern = sampler.getSQLRepresentation(formater);
			// Insert formated date into SQL query
			switch( type ) {
				case MYSQL:
					result = "DATE_FORMAT(" + date + " , '" + pattern + "')";
					break;
				case ORACLE:
				default:
					result = "to_char(" + date + " , '" + pattern + "')";
					break;
			}
		}
		return result;
	}
	
	static public String convertDate( String date, BaseType type, boolean toFrSqlFormat ) throws ParseException {
			String result;
			synchronized( CALENDAR ) {
				// Concert string to time value
				long time = stringToMilli(date);
				
				result = convertDate(time, type, toFrSqlFormat);
			}
			return result;
	}
	
	static public String convertDate(final long timeInMillis, BaseType type, boolean toFrSqlFormat ) throws ParseException {
		String result;
		synchronized( CALENDAR ) {
			// Select localized time formater
			SimpleDateFormat format = US_FORMAT;
			
			if( toFrSqlFormat ) {
				format = FR_FORMAT;
			}
			// Formate the date
			result = formatDate(timeInMillis, format);
		}
		return result;
}
	
	/**
     * Cast a string format date (dd-MM-yyyy HH:mm:ss or yyyy-MM-dd HH:mm:ss)
     * into long (number of milliseconds since January 1, 1970)
     * 
     * @param date
     * @return
     */
	protected static synchronized long stringToMilli(String date) throws ParseException {

		final boolean isFr = date.indexOf("-") != 4;
		final int currentLength = date.length();
		final String toDate     = "yyyy-MM-dd";
		final String toDateTime = "yyyy-MM-dd 00:00:00";

		int delta = toDateTime.length() - currentLength;
		
		if( delta > 0 && currentLength >= toDate.length() ) {
			date += toDateTime.substring( currentLength );
		}

		if (date.indexOf(".") == -1) {
			date = date + ".000";
		}

		long result;
		synchronized( CALENDAR ) {
			if (isFr) {
				FR_FORMAT.parse(date);
				result = FR_FORMAT.getCalendar().getTimeInMillis();
			} else {
				ISO_FORMAT.parse(date);
				result = ISO_FORMAT.getCalendar().getTimeInMillis();
			}
		}
		
		return result;
    }
	
	protected static String formatDate(final long timeInMillis, final SimpleDateFormat format) {
		String date = null;
		if (format != null) {
			synchronized (CALENDAR) {
				CALENDAR.setTimeInMillis(timeInMillis);
				date = format.format(CALENDAR.getTime());
			}
		}
		return date;
	}
	
}
