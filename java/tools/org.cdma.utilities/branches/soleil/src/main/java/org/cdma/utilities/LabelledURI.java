package org.cdma.utilities;

import java.net.URI;

import org.cdma.interfaces.IDatasource;

public class LabelledURI {
    private String mLabel;
    private URI    mURI;
    private IDatasource mSource;

    public LabelledURI( String label, URI uri, IDatasource datasource ) {
        mLabel = label;
        mURI = uri;
        mSource = datasource;
    }
    
    public URI getURI() {
        return mURI;
    }
    
    public String getLabel() {
        return mLabel;
    }
    
    public IDatasource getDatasource() {
        return mSource;
    }
    
    @Override
    public String toString() {
        return mURI.toString();
    }
}
