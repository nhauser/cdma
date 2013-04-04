package org.cdma.plugin.mambo;

import java.io.File;
import java.io.FileFilter;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.mambo.internal.DetectedSource;

public class SoleilMamboDataSource implements IDatasource {
	private static final int MAX_SOURCE_BUFFER_SIZE = 200;
    private static final String EXTENSION = ".vc";
    
    private static HashMap<String, DetectedSource> detectedSources; // map of analyzed URIs
    private static SoleilMamboDataSource datasource;

    public static SoleilMamboDataSource getInstance() {
        synchronized (SoleilMamboDataSource.class ) {
            if( datasource == null ) {
                datasource  = new SoleilMamboDataSource();
                detectedSources = new HashMap<String, DetectedSource>();
            }
        }
        return datasource;
    }
    
    public static class ValidURIFilter implements FileFilter {

        @Override
        public boolean accept(File path) {
            boolean result = path.isDirectory();
            if( ! result ) {
            	String fileName = path.getPath();
            	int length = fileName.length();
            	return (length > EXTENSION.length() && fileName.substring(length - EXTENSION.length()).equalsIgnoreCase(EXTENSION));
            }
            return result;
        }
    }
    
	@Override
	public String getFactoryName() {
		return SoleilMamboFactory.NAME;
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
        if (source != null) {
            File folder = new File(target.getPath());
            if( folder.isDirectory() ) {
                File[] files = folder.listFiles( (FileFilter) new ValidURIFilter() );
                if( files != null ) {
                    for (File file : files) {
                        result.add(file.toURI());
                    }
                }
            }
        }
        return result;
	}

	@Override
	public String[] getURIParts(URI target) {
		// No parts for URI
		return new String[] {target.toString()};
	}

	@Override
	public long getLastModificationDate(URI target) {
        long last = 0;
        if (isReadable(target) || isBrowsable(target)) {
            File file = new File(target.getPath());
            if (file.exists()) {
                last = file.lastModified();
            }
        }
        return last;
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
