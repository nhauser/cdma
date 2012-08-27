package org.cdma.plugin.soleil;

import java.io.File;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IGroup;
import org.cdma.plugin.soleil.internal.DetectedSource;
import org.cdma.plugin.soleil.navigation.NxsDataset;

public final class NxsDatasource implements IDatasource {
    private static final int MAX_SOURCE_BUFFER_SIZE = 200;
    private static HashMap<String, DetectedSource> mDetectedSources; // map of analyzed URIs
    
    public NxsDatasource() {
        if( mDetectedSources == null ) {
            synchronized( NxsDatasource.class ) {
                if( mDetectedSources == null ) {
                    mDetectedSources = new HashMap<String, DetectedSource>();
                }
            }
        }
    }
    
    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }

    @Override
    public boolean isReadable(URI target) {
        DetectedSource source = getSource(target);
        return source.isReadable();
    }

    @Override
    public boolean isProducer(URI target) {
        DetectedSource source = getSource(target);
        return source.isProducer();
    }

    @Override
    public boolean isExperiment(URI target) {
        DetectedSource source = getSource(target);
        return source.isExperiment();
    }

    @Override
    public boolean isBrowsable(URI target) {
        DetectedSource source = getSource(target);
        return source.isBrowsable();
    }

    @Override
    public List<URI> getValidURI(URI target) {
        List<URI> result = new ArrayList<URI>();

        DetectedSource source = getSource( target );
        if (source.isFolder() && !source.isDatasetFolder()) {
            File folder = new File(target.getPath());
            for (File file : folder.listFiles()) {
                result.add(file.toURI());
            }
        } else {
            if ( source.isReadable() && source.isBrowsable()) {
                try {
                    String uri = target.toString();
                    String sep = target.getFragment() == null ? "#" : "";

                    NxsDataset dataset = NxsDataset.instanciate(target);
                    IGroup group = dataset.getRootGroup();
                    for (IGroup node : group.getGroupList()) {
                        result.add(URI.create(uri + sep + URLEncoder.encode("/" + node.getShortName(), "UTF-8")));
                    }

                } catch (NoResultException e) {
                    e.printStackTrace();
                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                }
            }
        }
        return result;
    }

    @Override
    public String[] getURIParts(URI target) {
        List<String> parts = new ArrayList<String>();
        DetectedSource source = getSource(target);
        if ( source.isProducer() ) {
            String name = target.getPath();
            String fragment = target.getFragment();

            parts.add(name);

            if (fragment != null) {
                try {
                    fragment = URLDecoder.decode(fragment, "UTF-8");

                    for (String node : fragment.split("/")) {
                        parts.add(node);
                    }
                } catch (UnsupportedEncodingException e) {
                }
            }
        }

        return parts.toArray(new String[] {});
    }
    
    @Override
    public long getLastModificationDate(URI target) {
        long last = 0;
        if( isReadable( target ) || isBrowsable( target ) ) {
            File file = new File(target.getPath());
            if( file.exists() ) {
                last = file.lastModified();
            }
        }
        return last;
    }
    
    // ---------------------------------------------------------
    // / private methods
    // ---------------------------------------------------------
    private DetectedSource getSource(URI uri) {
        DetectedSource source = null;
        synchronized( mDetectedSources ) {
            source = mDetectedSources.get( uri.toString() );
            if( source == null ) {
                if( mDetectedSources.size() > MAX_SOURCE_BUFFER_SIZE ) {
                    int i = MAX_SOURCE_BUFFER_SIZE / 2;
                    List<String> remove = new ArrayList<String>();
                    for( String key : mDetectedSources.keySet() ) {
                        remove.add(key);
                        if( i-- < 0 ) {
                            break;
                        }
                    }
                    for( String key : remove ) {
                        mDetectedSources.remove( key );
                    }
                }
                
                source = new DetectedSource( uri );
                mDetectedSources.put( uri.toString(), source );
            }
        }
        return source;
    }
}
