package org.cdma.utilities;

import java.net.URI;

import org.cdma.interfaces.IDatasource;

public class LabelledURI {
    private String mLabel;
    private URI    mURI;
    private IDatasource mSource;

    public LabelledURI( String label, URI uri, IDatasource datasource ) {
        mLabel  = label;
        mURI    = uri;
        mSource = datasource;
    }

    public LabelledURI( URI uri, IDatasource datasource ) {
        this(extractLabel(uri, datasource), uri, datasource);
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
    
    protected static String extractLabel(URI uri, IDatasource datasource) {
        String label;
        if( datasource != null ) {
            String[] parts = datasource.getURIParts(uri);
            if( parts != null && parts.length > 0 ) {
                label = parts[parts.length - 1];
            }
            else {
                label = String.valueOf(uri);
            }
        }
        else {
            label = String.valueOf(uri);
        }
        return label;
    }

    @Override
    public String toString() {
        return mLabel.toString();
    }
}
