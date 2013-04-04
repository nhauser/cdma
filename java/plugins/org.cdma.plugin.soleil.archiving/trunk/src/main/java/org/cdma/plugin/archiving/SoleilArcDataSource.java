package org.cdma.plugin.archiving;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.cdma.interfaces.IDatasource;
import org.cdma.plugin.archiving.internal.DetectedSource;

public class SoleilArcDataSource implements IDatasource {
	private static final int MAX_SOURCE_BUFFER_SIZE = 200;
	
    private static HashMap<String, DetectedSource> detectedSources; // map of analyzed URIs
    private static SoleilArcDataSource datasource;

    public static SoleilArcDataSource getInstance() {
        synchronized (SoleilArcDataSource.class ) {
            if( datasource == null ) {
                datasource  = new SoleilArcDataSource();
                detectedSources = new HashMap<String, DetectedSource>();
            }
        }
        return datasource;
    }
    
	@Override
	public String getFactoryName() {
		return SoleilArcFactory.NAME;
	}
	
	@Override
	public boolean isReadable(URI target) {
		return getSource(target).isReadable();
	}

	@Override
	public boolean isProducer(URI target) {
		return getSource(target).isProducer();
	}

	@Override
	public boolean isBrowsable(URI target) {
		return getSource(target).isBrowsable();
	}

	@Override
	public boolean isExperiment(URI target) {
		return getSource(target).isExperiment();
	}

	@Override
	public List<URI> getValidURI(URI target) {
		return new ArrayList<URI>();
	}

	@Override
	public String[] getURIParts(URI target) {
		// No parts for URI
		return new String[] {target.toString()};
	}

	@Override
	public long getLastModificationDate(URI target) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public URI getParentURI(URI target) {
		// No parent URI are available
		return null;
	}
	
    private DetectedSource getSource(URI uri) {
        DetectedSource source = null;
        synchronized (detectedSources) {
            source = detectedSources.get(uri.toString());
            if (source == null) {
                if (detectedSources.size() > MAX_SOURCE_BUFFER_SIZE) {
                    int i = MAX_SOURCE_BUFFER_SIZE / 2;
                    List<String> remove = new ArrayList<String>();
                    for (String key : detectedSources.keySet()) {
                        remove.add(key);
                        if (i-- < 0) {
                            break;
                        }
                    }
                    for (String key : remove) {
                        detectedSources.remove(key);
                    }
                }

                source = new DetectedSource(uri);
                detectedSources.put(uri.toString(), source);
            }
        }
        return source;
    }

    private static final String URI_DESC = "URI must target an Archiving database";
    
	@Override
	public String getURITypeDescription() {
		return URI_DESC;
	}
}
