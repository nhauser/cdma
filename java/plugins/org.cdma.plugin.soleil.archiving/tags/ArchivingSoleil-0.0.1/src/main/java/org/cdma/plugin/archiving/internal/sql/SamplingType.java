package org.cdma.plugin.archiving.internal.sql;

import java.text.SimpleDateFormat;


public interface SamplingType {
	/**
	 * The DBMS string representation for the date transformation
	 */
	public String getSQLRepresentation();
	
	/**
	 * The DBMS string representation for the date transformation
	 * according the given format.
	 * @note only following keys are authorized: [ yyyy, MM, dd, HH, mm, ss, SSS ]
	 */
	public String getSQLRepresentation(SimpleDateFormat format);
	
	/**
	 * For the given SamplingPeriod it will return the corresponding sampling format 
	 * @param sample
	 */
	public SamplingType getType(SamplingPeriod sample);
	
	
	
	public enum SamplingPeriod {
	    MONTH   (5),
	    DAY     (4),
	    HOUR    (3),
	    MINUTE  (2),
	    SECOND  (1),
	    ALL     (-1);
	    
	    private int mSampling;
	    
	    private SamplingPeriod( int sampling ) {
	    	mSampling = sampling;
	    }
	    
	    public int value() {
	    	return mSampling;
	    }
	    
	    public static SamplingPeriod instantiate(int period) {
	    	SamplingPeriod result;
	        switch(period) {
	        case 5:
	            result = MONTH;
	            break;
	        case 4:
	        	result = DAY;
	            break;
	        case 3:
	        	result = HOUR;
	            break;
	        case 2:
	        	result = MINUTE;
	            break;
	        case 1:
	        	result = SECOND;
	            break;
	        case -1:
    		default:
	        	result = ALL;
	            break;
	        }
	        return result;
	    }
	}
	

	
}
