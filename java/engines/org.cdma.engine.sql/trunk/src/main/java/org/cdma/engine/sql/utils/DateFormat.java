//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.sql.utils;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.GregorianCalendar;
import java.util.concurrent.ConcurrentHashMap;

import org.cdma.engine.sql.utils.DbUtils.BaseType;
import org.cdma.engine.sql.utils.SamplingType.SamplingPeriod;

public class DateFormat {

	static private final ConcurrentHashMap<String, SimpleDateFormat> DATE_FORMATS;
	
    /**
	 * Insert a given string into the   
	 * @param field name in the database
	 * @param type database type
	 * @param pattern of date
	 * @param sampling period see {@link SamplingPeriod}
	 * @return
	 * @throws ParseException
	 */
	static public String dateToSqlString( final String field, BaseType type, String pattern ) throws ParseException {
		return dateToSqlString( field, type, pattern, SamplingPeriod.instantiate((short) -1) );
	}

	static public String dateToSqlString( final String field, BaseType type, String pattern, SamplingPeriod period ) throws ParseException {
		String result;
		
		SimpleDateFormat formater = getDateFormater(pattern);
		
		// Get the DBMS (generic) time pattern
		SamplingType sampler = DbUtils.getSqlSamplingType(period, type);
		String datePattern = sampler.getSQLRepresentation(formater);
		
		// Insert formated date into SQL query
		switch( type ) {
			case MYSQL:
				result = "DATE_FORMAT(" + field + " , '" + datePattern + "')";
				break;
			case ORACLE:
			default:
				result = "to_char(" + field + " , '" + datePattern + "')";
				break;
		}
		return result;
	}
	
	static public String convertDate( String date, String pattern ) throws ParseException {
			// Concert string to time value
			long time = stringToMilli(date, pattern);
			
			String result = convertDate(time, pattern);
			return result;
	}
	
	static public String convertDate(final long timeInMillis, String pattern ) throws ParseException {
		String result;

		// Select localized time formater
		SimpleDateFormat format = getDateFormater(pattern);
			
		// Formate the date
		result = formatDate(timeInMillis, format);
		return result;
	}
	
	/**
     * Cast a string format date (dd-MM-yyyy HH:mm:ss or yyyy-MM-dd HH:mm:ss)
     * into long (number of milliseconds since January 1, 1970)
     * 
     * @param date
     * @return
     */
	static public synchronized long stringToMilli(String date, String pattern) throws ParseException {
		long result = 0;
		if( pattern != null ) {
			SimpleDateFormat format = getDateFormater(pattern);
			format.parse(date);
			result = format.getCalendar().getTimeInMillis();
		}
		
		return result;
    }
	
	// ------------------------------------------------------------------------
	// protected methods
	// ------------------------------------------------------------------------
	static protected String formatDate(final long timeInMillis, final SimpleDateFormat format) {
		String date = null;
		if (format != null) {
			GregorianCalendar calendar = new java.util.GregorianCalendar();

			calendar.setTimeInMillis(timeInMillis);
			date = format.format(calendar.getTime());
		}
		return date;
	}
	
	static protected SimpleDateFormat getDateFormater(String pattern) {
		// Select localized time formater
		SimpleDateFormat formater = null;
		if( pattern != null ) {
			if( DATE_FORMATS.containsKey( pattern ) ) {
				formater = DATE_FORMATS.get(pattern);
			}
			else {
				formater = new SimpleDateFormat(pattern);
				DATE_FORMATS.putIfAbsent( pattern, formater );
			}
		}
		return formater;
	}
	
	// ------------------------------------------------------------------------
	// static block
	// ------------------------------------------------------------------------
	static {
		DATE_FORMATS = new ConcurrentHashMap<String, SimpleDateFormat>();
	}
	
}
