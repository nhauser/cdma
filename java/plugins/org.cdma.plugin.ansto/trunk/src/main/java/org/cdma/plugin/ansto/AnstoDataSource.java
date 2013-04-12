package org.cdma.plugin.ansto;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.cdma.interfaces.IDatasource;
import org.cdma.plugin.ansto.internal.DetectedSource;
import org.cdma.plugin.ansto.internal.DetectedSource.NetCDFFilter;

public final class AnstoDataSource implements IDatasource {
    private static final int MAX_SOURCE_BUFFER_SIZE = 200;
	private static final String FILE_SEPARATOR = "/";
    private static HashMap<String, DetectedSource> mDetectedSources; // map of analyzed URIs
    private static AnstoDataSource datasource;

    static {
        synchronized (AnstoDataSource.class ) {
            if( datasource == null ) {
                datasource  = new AnstoDataSource();
            }
            if (mDetectedSources == null) {
                mDetectedSources = new HashMap<String, DetectedSource>();
            }
        }
    }
    
    public static AnstoDataSource getInstance() {
        return datasource;
    }
    
    public String getFactoryName() {
        return AnstoFactory.NAME;
    }

    @Override
    public boolean isReadable(URI target) {
        boolean result = false;
        DetectedSource source = getSource(target);
        if (source != null) {
            result = source.isReadable();
        }
        return result;
    }

    @Override
    public boolean isProducer(URI target) {
        boolean result = false;
        DetectedSource source = getSource(target);
        if (source != null) {
            result = source.isProducer();
        }
        return result;
    }

    @Override
    public boolean isExperiment(URI target) {
        boolean result = false;
        DetectedSource source = getSource(target);
        if (source != null) {
            result = source.isExperiment();
        }
        return result;
    }

    @Override
    public boolean isBrowsable(URI target) {
        boolean result = false;
        DetectedSource source = getSource(target);
        if (source != null) {
            result = source.isBrowsable();
        }
        return result;
    }

    @Override
    public List<URI> getValidURI(URI target) {
        List<URI> result = new ArrayList<URI>();

        DetectedSource source = getSource(target);
        if (source != null) {
            if ( source.isFolder() ) {
                File folder = new File(target.getPath());
                NetCDFFilter filter = new NetCDFFilter(true);
                File[] files = folder.listFiles(filter);
                if( files != null ) {
                    for ( File file : files ) {
                        result.add(file.toURI());
                    }
                }
            }
        }
        return result;
    }
    
    @Override
    public URI getParentURI( URI target ) {
        URI result = null;
        
        if ( isBrowsable(target) || isReadable(target) ) {

            File current = new File(target.getPath());
            String filePath = "";
            current = current.getParentFile();
            filePath = current.toURI().toString();
            
            result = URI.create(filePath);
        }
        
        return result;
    }

    @Override
    public String[] getURIParts(URI target) {
        List<String> parts = new ArrayList<String>();
        if( target != null ) {
	        String path = target.getPath();
	        File file = new File(target);
	        if( path != null && file.exists() ) {
		        for( String part : path.split( FILE_SEPARATOR ) ) {
		            if( part != null && ! part.isEmpty() ) {
		                parts.add(part);
		            }
		        }
	        }
        }
        return parts.toArray(new String[] {});
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
    
    private static final String URI_DESC = "File system: folders and NetCDF files";
    
	@Override
	public String getURITypeDescription() {
		return URI_DESC;
	}

    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private DetectedSource getSource(URI uri) {
        DetectedSource source = null;
        synchronized (mDetectedSources) {
            source = mDetectedSources.get(uri.toString());
            if (source == null) {
                if (mDetectedSources.size() > MAX_SOURCE_BUFFER_SIZE) {
                    int i = MAX_SOURCE_BUFFER_SIZE / 2;
                    List<String> remove = new ArrayList<String>();
                    for (String key : mDetectedSources.keySet()) {
                        remove.add(key);
                        if (i-- < 0) {
                            break;
                        }
                    }
                    for (String key : remove) {
                        mDetectedSources.remove(key);
                    }
                }

                source = new DetectedSource(uri);
                if( source.isStable() ) {
                	mDetectedSources.put(uri.toString(), source);
                }
            }
        }
        return source;
    }
    
    private AnstoDataSource() {
    }
}
