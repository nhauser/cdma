package org.cdma.plugin.soleil.edf.internal;

import java.io.File;
import java.net.URI;

import org.cdma.plugin.soleil.edf.EdfDatasource;
import org.cdma.plugin.soleil.edf.EdfDatasource.ValidURIFilter;

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
        mIsExperiment = false;
        mIsBrowsable = false;
        mIsProducer = false;
        mIsReadable = false;

        if (uri != null) {
            String path = uri.getPath();
            if (path != null) {
                File file = new File(path);
                if (file.exists()) {
                    mIsBrowsable = file.isDirectory();
                    
//                    if (mIsBrowsable){
//                        if (EdfDatasource.findEDFFiles(file)){
//                            mIsReadable = true;
//                        }
//                    }
                    ValidURIFilter filter = new ValidURIFilter();
                    if (filter.accept(file)) {
                        mIsBrowsable = true;
                        mIsReadable = true;
                        mIsProducer = true;
                        mIsExperiment = true;
                    }
                }
            }
        }
    }
}
