/*******************************************************************************
 * Copyright (c) 2008 - ANSTO/Synchrotron SOLEIL
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors:
 * 	Norman Xiong (nxi@Bragg Institute) - initial API and implementation
 * 	Tony Lam (nxi@Bragg Institute) - initial API and implementation
 *        Majid Ounsy (SOLEIL Synchrotron) - API v2 design and conception
 *        Stéphane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.mambo;

import java.io.File;
import java.io.FileFilter;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.cdma.interfaces.IDatasource;
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
        List<String> parts = new ArrayList<String>();
        if( target != null ) {
	        String path = target.getPath();
	        if( path != null ) {
		        for( String part : path.split( File.pathSeparator ) ) {
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

	@Override
	public URI getParentURI(URI target) {
        URI result = null;

        if (isReadable(target) || isBrowsable(target)) {
            File current = new File(target.getPath());
            if( current != null ) {
                if( current != null ) {
                	current = current.getParentFile();
                }
            }
            result = current.toURI();
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

    private static final String URI_DESC = "URI must target a Mambo's view configuration file";
    
	@Override
	public String getURITypeDescription() {
		return URI_DESC;
	}
}
