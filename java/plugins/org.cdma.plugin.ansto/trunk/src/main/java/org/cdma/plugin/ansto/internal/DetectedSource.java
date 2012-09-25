package org.cdma.plugin.ansto.internal;

import java.io.File;
import java.io.FilenameFilter;
import java.net.URI;

public class DetectedSource {
    private static final String EXTENSION[] = new String[] {".nxs", ".nx.hdf" };

    public static final class NetCDFFilter implements FilenameFilter {

        public boolean accept(File dir, String name) {
            boolean result = false;
            
            for( String ext : EXTENSION ) {
                if ( name.endsWith(ext) ) {
                    result = true;
                    break;
                }
            }
            
            return result;
        }
    }

    private boolean mIsExperiment;
    private boolean mIsBrowsable;
    private boolean mIsProducer;
    private boolean mIsReadable;
    private URI mURI;
    private boolean mIsFolder;

    public DetectedSource(URI uri, boolean browsable, boolean readable, boolean producer, boolean experiment, boolean datasetFolder) {
        mIsReadable = readable;
        mIsProducer = producer;
        mIsBrowsable = browsable;
        mIsExperiment = experiment;
        mURI = uri;
    }

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

    public boolean isFolder() {
        return mIsFolder;
    }
    
    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private void init(URI uri) {
        if (uri != null) {
            // Check if the uri is a folder
            File file = new File(uri.getPath());
            if ( ! file.isDirectory() ) {
                mIsFolder = true;
            }
            else {
                mIsFolder = false;
            }

            // Check it is a NeXus file
            mIsReadable = initReadable(uri);

            // Check if we are producer of the source
            mIsProducer = initProducer(uri);

            // Check if the uri corresponds to dataset experiment
            mIsExperiment = initExperiment(uri);

            // Check if the URI is considered as browsable
            mIsBrowsable = initBrowsable(uri);
        }

    }

    private boolean initReadable(URI uri) {

        boolean result = false;
        File file = new File(uri.getPath());
        String name = file.getName();

        if( file.exists() && !file.isDirectory() && file.length() != 0L ) {
            // Check if the URI is a NetCDF file
            for( String ext : EXTENSION ) {
                if( name.endsWith(ext)) {
                    result = true;
                    break;
                }
            }
        }

        return result;
    }

    private boolean initProducer(URI uri) {
        boolean result = false;
        
        // Check if the URI targets an ANSTO file
        if ( mIsReadable ) {
            // TODO [SOLEIL][clement] make a real test case
            return true;
        }
        return result;
    }

    private boolean initExperiment(URI uri) {
        boolean result = false;

        // Check if the URI targets an ANSTO experiment dataset
        if ( mIsProducer ) {
            // TODO [SOLEIL][clement] make a real test case
            return true;
        }
        return result;
    }

    private boolean initBrowsable(URI uri) {
        boolean result = false;

        // If experiment not browsable
        if ( mIsFolder || (!mIsExperiment && mIsFolder) ) {
            return true;
        }
        return result;
    }
}
