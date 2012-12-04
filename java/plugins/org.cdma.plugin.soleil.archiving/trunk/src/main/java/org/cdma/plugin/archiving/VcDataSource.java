package org.cdma.plugin.archiving;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.cdma.interfaces.IDatasource;
import org.cdma.plugin.archiving.internal.DetectedSource;

public class VcDataSource implements IDatasource {
	private static final int MAX_SOURCE_BUFFER_SIZE = 200;
    private static HashMap<String, DetectedSource> detectedSources; // map of analyzed URIs
    private static VcDataSource datasource;

    public static VcDataSource getInstance() {
        synchronized (VcDataSource.class ) {
            if( datasource == null ) {
                datasource  = new VcDataSource();
                detectedSources = new HashMap<String, DetectedSource>();
            }
        }
        return datasource;
    }
    
	@Override
	public String getFactoryName() {
		return VcFactory.NAME;
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
		List<URI> result = new ArrayList<URI>();
		DetectedSource source = getSource(target);
		
		if ( source.isReadable() ) {
			result.add(target);
		}
		else if ( source.isBrowsable() ) {
			File file = new File(target);
			File[] files = file.listFiles( source.getFilenameFilter() );
			for (File subFile : files) {
				URI fileURI = subFile.toURI();
				if (isReadable(fileURI)) {
					result.add(fileURI);
				}
			}
		}
		System.out.println("getValidURI: " + result.size() );		
		return result;
	}

	@Override
	public String[] getURIParts(URI target) {
		return target.getPath().split("/");
	}

	@Override
	public long getLastModificationDate(URI target) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public URI getParentURI(URI target) {
		String[] elements = target.getPath().split("/");
		String stringResult = "";
		URI result = null;
		for (int i = 0; i < elements.length - 1; ++i) {
			stringResult += elements[i];
		}

		try {
			result = new URI(stringResult);
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
		return result;
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
}
