// ******************************************************************************
//Copyright (c) 2011 Synchrotron Soleil.
//The CDMA library is free software; you can redistribute it and/or modify it
//under the terms of the GNU General Public License as published by the Free
//Software Foundation; either version 2 of the License, or (at your option)
//any later version.
//Contributors :
//See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.edf;

import java.io.File;
import java.io.FileFilter;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import javax.swing.filechooser.FileSystemView;

import org.cdma.interfaces.IDatasource;
import org.cdma.plugin.soleil.edf.internal.DetectedSource;
import org.cdma.plugin.soleil.edf.utils.FileComparator;

public final class EdfDatasource implements IDatasource {
    private static final int MAX_SOURCE_BUFFER_SIZE = 200;
    private static final String EXTENSION = ".edf";

    private static HashMap<String, DetectedSource> detectedSources; // map of analyzed URIs
    private static EdfDatasource datasource;

    public static EdfDatasource getInstance() {
        synchronized (EdfDatasource.class) {
            if (datasource == null) {
                datasource = new EdfDatasource();
                detectedSources = new HashMap<String, DetectedSource>();
            }
        }
        return datasource;
    }

    public static class ValidURIFilter implements FileFilter {

        @Override
        public boolean accept(File path) {
            boolean result = false;
            boolean isDir = path.isDirectory();
            if (!isDir) {
                String fileName = path.getPath();
                int length = fileName.length();
                result = (length > EXTENSION.length() && fileName.substring(length - EXTENSION.length())
                        .equalsIgnoreCase(EXTENSION));
            } else {
                result = findEDFFiles(path);
            }
            return result;
        }
    }

    private static boolean findEDFFiles(File path) {
        boolean hasEdfFile = false;
        boolean hasDirectory = false;
        File[] files = FileSystemView.getFileSystemView().getFiles(path, false);
        if (files != null) {
            ArrayList<File> fileList = new ArrayList<File>();
            fileList.addAll(Arrays.asList(files));
            Collections.sort(fileList, new FileComparator());
            for (File file : fileList) {
                String fPath = file.getAbsolutePath();
                int pointIndex = fPath.lastIndexOf('.');
                if ((pointIndex > -1) && "edf".equalsIgnoreCase(fPath.substring(pointIndex + 1))) {
                    hasEdfFile = true;
                }
                if (file.isDirectory()) {
                    hasDirectory = true;
                }
            }
        }
        return hasEdfFile && !hasDirectory;
    }

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
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
            if (folder.isDirectory()) {

                File[] files = folder.listFiles(new ValidURIFilter());
                if (files != null) {
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
        if (target != null) {
            String path = target.getPath();
            if (path != null) {
                for (String part : path.split("/")) {
                    if (part != null && !part.isEmpty()) {
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
            if (current != null) {
                if (current != null) {
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

    private static final String URI_DESC = "URI must target an EDF Directory";

    @Override
    public String getURITypeDescription() {
        return URI_DESC;
    }
}
