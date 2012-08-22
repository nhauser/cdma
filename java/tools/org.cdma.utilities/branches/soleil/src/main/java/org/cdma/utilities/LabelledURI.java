package org.cdma.utilities;

import java.net.URI;

public class LabelledURI {
    private String mLabel;
    private URI    mURI;

    public LabelledURI( String label, URI uri ) {
        mLabel = label;
        mURI = uri;
    }
    
    public URI getURI() {
        return mURI;
    }
    
    public String getLabel() {
        return mLabel;
    }
}
