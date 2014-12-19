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

import java.io.IOException;
import java.net.URI;

import org.cdma.engine.archiving.navigation.ArchivingDataset;
import org.cdma.engine.archiving.navigation.ArchivingDataset.ArchivingMode;
import org.cdma.interfaces.IDataset;
import org.cdma.plugin.archiving.SoleilArcFactory;

public class DetectedSource {
    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private final URI mURI;

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
        if (uri != null) {
            String scheme = uri.getScheme();

            mIsBrowsable = false;
            mIsExperiment = false;
            mIsProducer = false;
            mIsReadable = false;

            if( (scheme != null) && scheme.equals("jdbc") && (uri.getSchemeSpecificPart() != null) ) {
                try {
                    IDataset dataset = new SoleilArcFactory().createDatasetInstance(uri);
                    if(dataset instanceof ArchivingDataset) {
                        ArchivingDataset archivingdataset = (ArchivingDataset)dataset;
                        for( ArchivingMode mode : ArchivingMode.values() ) {
                            try {
                                archivingdataset.setArchivingMode(mode);
                                archivingdataset.open();
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
                } catch (Exception e) {
                    // Nothing to do: no suitable driver for the given URI
                }

            }
        }
    }
}
