package org.cdma.plugin.ansto;

import java.io.File;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.engine.netcdf.navigation.Constants;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.ansto.internal.DetectedSource;
import org.cdma.plugin.ansto.internal.DetectedSource.NetCDFFilter;

public class AnstoDataSource implements IDatasource {
    private static final int MAX_SOURCE_BUFFER_SIZE = 200;
    private static HashMap<String, DetectedSource> mDetectedSources; // map of analyzed URIs
    private static AnstoDataSource datasource;

    public static AnstoDataSource getInstance() {
        synchronized (AnstoDataSource.class ) {
            if( datasource == null ) {
                datasource  = new AnstoDataSource();
            }
        }
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
            else {
                if (source.isReadable() && source.isBrowsable()) {
                    try {
                        // Extract the path internal of file from the given target
                        String uri = target.toString();
                        String sep = target.getFragment() == null ? "#" : "";

                        // List all sub-groups if the it is browsable
                        AnstoDataset dataset = AnstoDataset.instantiate(target);
                        IGroup group = dataset.getRootGroup();
                        for (IGroup node : group.getGroupList()) {
                            result.add(URI.create(uri + sep
                                    + URLEncoder.encode(Constants.PATH_SEPARATOR + node.getShortName(), "UTF-8")));
                        }

                    }
                    catch (UnsupportedEncodingException e) {
                        Factory.getLogger().log( Level.WARNING, e.getMessage());
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
            String fragment = target.getFragment();
            String filePath = "";
            
            if( fragment == null ) {
                current = current.getParentFile();
                filePath = current.toURI().toString();
                fragment = "";
            }
            else {
                filePath = current.toURI().toString();
                
                try {
                    fragment = URLDecoder.decode(fragment, "UTF-8");
                    String[] nodes = fragment.split( Constants.PATH_SEPARATOR );
                    fragment = "";
                    for ( int i = 0; i < nodes.length - 1; i++ ) {
                        fragment += Constants.PATH_SEPARATOR + nodes[i];
                    }

                    if( ! fragment.isEmpty() ) {
                        fragment = "#" + URLEncoder.encode(fragment, "UTF-8");
                    }
                } catch (UnsupportedEncodingException e) {
                    Factory.getLogger().log( Level.WARNING, e.getMessage());
                }
                
            }
            
            result = URI.create(filePath + fragment);
        }
        
        return result;
    }

    @Override
    public String[] getURIParts(URI target) {
        List<String> parts = new ArrayList<String>();
        if ( isBrowsable(target) || isReadable(target) ) {
            String path = target.getPath();
            String fragment = target.getFragment();
            for( String part : path.split( Constants.PATH_SEPARATOR ) ) {
                if( part != null && ! part.isEmpty() ) {
                    parts.add(part);
                }
            }

            if (fragment != null) {
                try {
                    fragment = URLDecoder.decode(fragment, "UTF-8");
                    String[] nodes = fragment.split( Constants.PATH_SEPARATOR );
                    for (String node : nodes ) {
                        parts.add( node );
                    }
                }
                catch (UnsupportedEncodingException e) {
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
	public String getTypeDescription() {
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
                mDetectedSources.put(uri.toString(), source);
            }
        }
        return source;
    }
    
    private AnstoDataSource() {
        if (mDetectedSources == null) {
            synchronized (AnstoDataSource.class) {
                if (mDetectedSources == null) {
                    mDetectedSources = new HashMap<String, DetectedSource>();
                }
            }
        }
    }
}
