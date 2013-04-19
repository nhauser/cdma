//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.engine.sql.utils.sampling;

import java.text.SimpleDateFormat;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

import org.cdma.engine.sql.utils.SamplingType;


public enum SamplingTypeOracle implements SamplingType {
    MONTH      ("YYYY-"),
    DAY        ("YYYY-MM-"),
    HOUR       ("YYYY-MM-DD "),
    MINUTE     ("YYYY-MM-DD HH24:"),
    SECOND     ("YYYY-MM-DD HH24:MI:"),
    FRACTIONAL ("YYYY-MM-DD HH24:MI:SS."),
    ALL        ("YYYY-MM-DD HH24:MI:SS.FF"),
    NONE       ("YYYY-MM-DD HH24:MI:SS.FF");
    
    private String mSampling;
    static private LinkedHashMap<String, String> mCorrespondance;
    
    static {
    	synchronized( SamplingTypeMySQL.class ) {
    		mCorrespondance = new LinkedHashMap<String, String>();
    		mCorrespondance.put("yyyy", "YYYY");
    		mCorrespondance.put("MM", "MM");
    		mCorrespondance.put("dd", "DD");
    		mCorrespondance.put("HH", "HH24");
    		mCorrespondance.put("mm", "MI");
    		mCorrespondance.put("ss", "SS");
    		mCorrespondance.put("SSS", "FF");
    	}
    	
    }
    
    private SamplingTypeOracle( String sampling ) {
    	mSampling = sampling;
    }
    
    @Override
    public String getPattern(SamplingPeriod period) {
    	String result = SamplingTypeOracle.valueOf(period.name()).mSampling;
    	
    	for( Entry<String, String> entry : mCorrespondance.entrySet() ) {
    		result = result.replace(entry.getValue(), entry.getKey());
    	}
    	
    	return result;
    }
    
    @Override
    public String getSQLRepresentation() {
    	return mSampling;
    }
    
    @Override
    public String getSQLRepresentation(SimpleDateFormat format) {
    	String result = format.toPattern();
    	
    	for( Entry<String, String> entry : mCorrespondance.entrySet() ) {
    		result = result.replace(entry.getKey(), entry.getValue());
    	}
    	
    	return result;
    }
    
    @Override
    public SamplingType getType(SamplingPeriod time) {
    	SamplingType result = SamplingTypeOracle.valueOf(time.name());
    	return result;
    }
    
    @Override
    public String getSamplingSelectClause(String field, SamplingPolicy policy, String name) {
    	String result = field;
    	switch( policy ) {
    		case NONE:
    			break;
    		case AVERAGE:
    			result = "AVG(" + field + ") as " + name;
    			break;
    		case MAX:
    			result = "MAX(" + field + ") as " + name;
    			break;
    		case MIN:
    			result = "MIN(" + field + ") as " + name;
    			break;
    		default:
    			break;
    	}
    	
    	return result;
    }
    
    @Override
    public String getFieldAsStringSelector( String field ) {
    	return "to_char(" + field + ")";
    }
    
    @Override
    public String getDateSampling(String field, SamplingPeriod period, int factor) {
    	String result = "";
    	String periodPattern = null;
    	
    	periodPattern = getSamplingPeriodUnit(period);
    	
    	if( periodPattern != null ) {
    		if( period.equals(SamplingPeriod.MONTH) || period.equals(SamplingPeriod.DAY) ) {
    			result = "decode(";
    		}
    		result += "FLOOR(to_number(to_char(" + field + " , '" + periodPattern + "')) / " + factor + ") * " + factor;
    		if( period.equals(SamplingPeriod.MONTH) || period.equals(SamplingPeriod.DAY) ) {
    			result += ",0,1,";
    			result += "FLOOR(to_number(to_char(" + field + " , '" + periodPattern + "')) / " + factor + ") * " + factor;
    			result += ")";
    		}
    	}
    	
		return result;
    	
    }

    @Override
    public String getPatternPeriodUnit(SamplingPeriod period) {
    	String result = getSamplingPeriodUnit(period);
    	
    	for( Entry<String, String> entry : mCorrespondance.entrySet() ) {
    		result = result.replace(entry.getValue(), entry.getKey());
    	}
    	
    	return result;
    }
    
	public String getSamplingPeriodUnit(SamplingPeriod period) {
		String result = null;
		switch (period) {
		case FRACTION:
			result = "FF";
			break;
		case SECOND:
			result = "SS";
			break;
    	case MINUTE:
    		result = "MI";
			break;
    	case HOUR:
    		result = "HH24";
			break;
		case DAY:
			result = "DD";
			break;
		case MONTH:
			result = "MM";
			break;
		}
		return result;
	}
}
