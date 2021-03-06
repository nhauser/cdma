package org.cdma.plugin.archiving.internal.sql;

import java.text.SimpleDateFormat;
import java.util.HashMap;
import java.util.Map.Entry;


public enum SamplingTypeOracle implements SamplingType {
    MONTH      ("YYYY-MM"),
    DAY        ("YYYY-MM-DD"),
    HOUR       ("YYYY-MM-DD HH24"),
    MINUTE     ("YYYY-MM-DD HH24:MI"),
    SECOND     ("YYYY-MM-DD HH24:MI:SS"),
    FRACTIONAL ("YYYY-MM-DD HH24:MI:SS.FF"),
    ALL        ("YYYY-MM-DD HH24:MI:SS.FF");
    
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
    
    public String getSQLRepresentation() {
    	return mSampling;
    }
    
    public String getSQLRepresentation(SimpleDateFormat format) {
    	String result = format.toPattern();
    	
    	for( Entry<String, String> entry : mCorrespondance.entrySet() ) {
    		result = result.replace(entry.getKey(), entry.getValue());
    	}
    	
    	return result;
    }
    
    public SamplingType getType(SamplingPeriod time) {
    	SamplingType result = SamplingTypeOracle.valueOf(time.name());
    	/*
    	switch( time ) {
    	case MONTH:
    		result = SamplingTypeOracle.MONTH;
    		break;
    	case DAY:
    		result = SamplingTypeOracle.DAY;
    		break;
    	case HOUR:
    		result = SamplingTypeOracle.HOUR;
    		break;
    	case MINUTE:
    		result = SamplingTypeOracle.MINUTE;
    		break;
    	case SECOND:
    		result = SamplingTypeOracle.SECOND;
    		break;
    	case ALL:
    	default:
    		result = SamplingTypeOracle.ALL;
    		break;
    	}*/
    	return result;
    }
}
