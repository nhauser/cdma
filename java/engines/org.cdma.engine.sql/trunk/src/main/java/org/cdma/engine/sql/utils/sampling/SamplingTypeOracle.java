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
import java.util.HashMap;
import java.util.Map.Entry;

import org.cdma.engine.sql.utils.SamplingType;
import org.cdma.engine.sql.utils.SamplingType.SamplingPeriod;
import org.cdma.engine.sql.utils.SamplingType.SamplingPolicy;


public enum SamplingTypeOracle implements SamplingType {
    MONTH      ("YYYY-MM"),
    DAY        ("YYYY-MM-DD"),
    HOUR       ("YYYY-MM-DD HH24"),
    MINUTE     ("YYYY-MM-DD HH24:MI"),
    SECOND     ("YYYY-MM-DD HH24:MI:SS"),
    FRACTIONAL ("YYYY-MM-DD HH24:MI:SS.FF"),
    ALL        ("YYYY-MM-DD HH24:MI:SS.FF"),
    NONE       ("YYYY-MM-DD HH24:MI:SS.FF");
    
    private String mSampling;
    static private HashMap<String, String> mCorrespondance;
    
    static {
    	synchronized( SamplingTypeMySQL.class ) {
    		mCorrespondance = new HashMap<String, String>();
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
    	
    	switch (period) {
		case SECOND:
			periodPattern = "SS";
			break;
    	case MINUTE:
    		periodPattern = "MI";
			break;
    	case HOUR:
    		periodPattern = "HH24";
			break;
		case DAY:
			periodPattern = "DD";
			break;
		case MONTH:
			periodPattern = "MM";
			break;
		}
    	
    	if( periodPattern != null ) {
    		result = "MOD(to_number(to_char(" + field + " , '" + periodPattern + "')), " + factor + ") = 0";
    	}
    	
		return result;
    	
    }
}
