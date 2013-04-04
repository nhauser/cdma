//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.mambo.internal;

import java.io.File;
import java.net.URI;

import org.cdma.plugin.mambo.SoleilMamboDataSource.ValidURIFilter;

public class DetectedSource {
    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private URI mURI;

    public DetectedSource(URI uri) {
        mURI = uri;
        init(uri);
    }

    public URI getURI() {
        return mURI;
    }

    public boolean isExperiment() {
        return mIsExperiment;
    }

    public boolean isBrowsable() {
        return mIsBrowsable;
    }

    public boolean isProducer() {
        return mIsProducer;
    }

    public boolean isReadable() {
        return mIsReadable;
    }

    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private void init(URI uri) {
    	mIsExperiment = false;
		mIsBrowsable = false;
		mIsProducer = false;
		mIsReadable = false;

    	if (uri != null) {
        	String path = uri.getPath();
        	File file = new File(path);
        	if( file.exists() ) {
        		mIsBrowsable = file.isDirectory();
        		
        		if( ! mIsBrowsable ) {
        			ValidURIFilter filter = new ValidURIFilter();
        			if( filter.accept( file ) ) {
        				mIsReadable = true;
        				mIsProducer = true;
        				mIsExperiment = true;
        			}
        		}
        	}
        	
    		
    		
    		/*
        	if( scheme != null && scheme.equals("jdbc") && uri.getSchemeSpecificPart() != null ) {
        		try {
					Driver driver = DriverManager.getDriver(uri.toString());
					if( driver != null ) {
			    		ArchivingDataset dataset = new ArchivingDataset(SoleilMamboFactory.NAME, uri);
			    		for( ArchivingMode mode : ArchivingMode.values() ) {
			    			try {
			    				dataset.setArchivingMode(mode);
			    				dataset.open();
			    				if( dataset.getRootGroup() != null ) {
			    					mIsBrowsable = true;
						    		mIsExperiment = true;
						    		mIsProducer = true;
						    		mIsReadable = true;
						    		dataset.close();
						    		break;
			    				}
			    				else {
			    					dataset.close();
			    				}
			    			}
			    			catch( IOException e ) {
			    				// Nothing to do
			    			}
			    		}
					}
				} catch (SQLException e) {
					// Nothing to do: no suitable driver for the given URI 
				}
        	}
        	*/
        }
    }
}
