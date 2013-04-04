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


public enum SamplingTypeMySQL implements SamplingType {
    MONTH      ("%Y-%m"),
    DAY        ("%Y-%m-%d"),
    HOUR       ("%Y-%m-%d %H"),
    MINUTE     ("%Y-%m-%d %H:%i"),
    SECOND     ("%Y-%m-%d %H:%i:%s"),
    FRACTIONAL ("%Y-%m-%d %H:%i:%s.%f"),
    ALL        ("%Y-%m-%d %H:%i:%s.%f");
    
    private String mSampling;
    static private HashMap<String, String> mCorrespondance;
    
    static {
    	synchronized( SamplingTypeMySQL.class ) {
    		mCorrespondance = new HashMap<String, String>();
    		mCorrespondance.put("yyyy", "%Y");
    		mCorrespondance.put("MM", "%m");
    		mCorrespondance.put("dd", "%d");
    		mCorrespondance.put("HH", "%H");
    		mCorrespondance.put("mm", "%i");
    		mCorrespondance.put("ss", "%s");
    		mCorrespondance.put("SSS", "%f");
    	}
    	
    }
    
    private SamplingTypeMySQL( String sampling ) {
    	mSampling = sampling;
    }
    
    public String getSQLRepresentation() {
    	return mSampling;
    }
    
    public SamplingType getType(SamplingPeriod time) {
    	SamplingType result = SamplingTypeMySQL.valueOf( time.name() );
    	return result;
    }
    
    public String getSQLRepresentation(SimpleDateFormat format) {
    	String result = format.toPattern();
    	
    	for( Entry<String, String> entry : mCorrespondance.entrySet() ) {
    		result = result.replace(entry.getKey(), entry.getValue());
    	}
    	
    	return result;
    }
}
