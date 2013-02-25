//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.archiving.internal;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.net.URI;

import org.cdma.engine.archiving.internal.Constants;
import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.archiving.SoleilArcFactory;

public class DetectedSource {
	public static final class ViewConfigurationFilter implements FilenameFilter {

		private final String EXTENSION = ".vc";

		@Override
		public boolean accept(File dir, String name) {
			return (name.endsWith(EXTENSION));
		}
		
		private ViewConfigurationFilter() {}
	}
	
	private static ViewConfigurationFilter mFilter;
    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private URI mURI;

    public DetectedSource(URI uri) {
    	if( mFilter == null ) {
    		synchronized( DetectedSource.class ) {
    			if( mFilter == null ) {
    				mFilter = new ViewConfigurationFilter();
    			}
    		}
    	}
        mURI = uri;
        init(uri);
    }

    public FilenameFilter getFilenameFilter() {
    	return mFilter;
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
        if (uri != null) {
            // Check if the URI is considered as browsable
            mIsBrowsable = initBrowsable(uri);
        	
        	// Check it is a Vc file
            mIsReadable = initReadable(uri);

            // Check if we are producer of the source
            mIsProducer = initProducer(uri);

            // Check if the uri corresponds to dataset experiment
            mIsExperiment = initExperiment(uri);
        }
    }

    private boolean initReadable(URI target) {
		boolean result = false;
		if ( ! mIsBrowsable) {
			try {
				File file = new File(target);
				String name = file.getName();
			
				// Check if the URI is a ViewConfiguration file
				if ( mFilter.accept(file, name)) {
					result = true;
				}
			}
			catch( IllegalArgumentException e ) {
				// nothing to do
			}
		}
		return result;
	}

	private boolean initProducer(URI target) {
		boolean result = false;
		if ( mIsReadable ) {
			IDataset dataset = new ArchivingDataset(SoleilArcFactory.NAME, target);
			try {
				dataset.open();
				if (dataset.isOpen()) {
					IGroup rootGroup = dataset.getRootGroup();
					if (rootGroup != null) {
						IAttribute startDate = rootGroup.getAttribute(Constants.START_DATE);
						IAttribute endDate = rootGroup.getAttribute(Constants.END_DATE);
						result = ((startDate != null) && (endDate != null));
					}
				}
			} catch (IOException e) {
				// nothing to do
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return result;
	}

	private boolean initBrowsable(URI target) {
		boolean result = false;
		try {
			File file = new File(target);
			if (file.exists() && file.isDirectory()) {
				result = true;
			}
		}
		catch( IllegalArgumentException e ) {
			// nothing to do
		}
		return result;
	}

	private boolean initExperiment(URI target) {
		boolean result = mIsProducer;
		return result;
	}
    
}
