//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package org.cdma.plugin.soleil.nexus;

import java.io.File;
import java.io.FileFilter;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;

import org.cdma.Factory;
import org.cdma.exception.NoResultException;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.INode;
import org.cdma.plugin.soleil.nexus.internal.DetectedSource;
import org.cdma.plugin.soleil.nexus.navigation.NxsDataset;
import org.cdma.plugin.soleil.nexus.utils.NxsPath;

public final class NxsDatasource implements IDatasource {
    private static final int MAX_SOURCE_BUFFER_SIZE = 200;
    private static final HashMap<String, DetectedSource> mDetectedSources; // map of analyzed URIs
    private static NxsDatasource datasource;

    static {
        synchronized (NxsDatasource.class) {
            mDetectedSources = new HashMap<String, DetectedSource>();
        }
    }

    public static NxsDatasource getInstance() {
        synchronized (NxsDatasource.class) {
            if (datasource == null) {
                datasource = new NxsDatasource();
            }
        }
        return datasource;
    }

    private static class ValidURIFilter extends DetectedSource.NeXusFilter implements FileFilter {

        @Override
        public boolean accept(File pathname) {
            boolean result = pathname.isDirectory();
            if (!result) {
                result = accept(pathname.getParentFile(), pathname.getName());
            }
            return result;
        }
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
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
            if (source.isFolder() && !source.isDatasetFolder()) {
                File folder = new File(target.getPath());
                File[] files = folder.listFiles((FileFilter) new ValidURIFilter());
                Arrays.sort(files, new Comparator<File>() {
                    @Override
                    public int compare(File f1, File f2) {
                        return Long.valueOf(f2.lastModified()).compareTo(f1.lastModified());
                    }
                });
                if (files != null) {
                    for (File file : files) {
                        result.add(file.toURI());
                    }
                }
            }
            else {
                if (source.isReadable() && source.isBrowsable()) {
                    try {
                        String uri = target.toString();
                        String sep = target.getFragment() == null ? "#" : "";

                        NxsDataset dataset = NxsDataset.instanciate(target);
                        IGroup group = dataset.getRootGroup();
                        if (group != null) {
                            for (IGroup node : group.getGroupList()) {
                                result.add(URI.create(uri + sep
                                        + URLEncoder.encode("/" + node.getShortName(), "UTF-8")));
                            }
                        }

                    }
                    catch (NoResultException e) {
                        Factory.getLogger().log(Level.WARNING, e.getMessage());
                    }
                    catch (UnsupportedEncodingException e) {
                        Factory.getLogger().log(Level.WARNING, e.getMessage());
                    }
                }
            }
        }
        return result;
    }

    @Override
    public URI getParentURI(URI target) {
        URI result = null;

        if (isReadable(target) || isBrowsable(target)) {

            File current = new File(target.getPath());
            String fragment = target.getFragment();
            String filePath = "";

            if (fragment == null) {
                current = current.getParentFile();
                filePath = current.toURI().toString();
                fragment = "";
            }
            else {
                filePath = current.toURI().toString();

                try {
                    fragment = URLDecoder.decode(fragment, "UTF-8");
                    INode[] nodes = NxsPath.splitStringToNode(fragment);
                    fragment = "";
                    for (int i = 0; i < nodes.length - 1; i++) {
                        fragment += "/" + nodes[i].getNodeName();
                    }

                    if (!fragment.isEmpty()) {
                        fragment = "#" + URLEncoder.encode(fragment, "UTF-8");
                    }
                }
                catch (UnsupportedEncodingException e) {
                    Factory.getLogger().log(Level.WARNING, e.getMessage());
                }

            }

            result = URI.create(filePath + fragment);
        }

        return result;
    }

    @Override
    public String[] getURIParts(URI target) {
        List<String> parts = new ArrayList<String>();
        String path = target.getPath();
        String fragment = target.getFragment();
        if (path != null) {
            for (String part : path.split("/")) {
                if (part != null && !part.isEmpty()) {
                    parts.add(part);
                }
            }
        }
        if (fragment != null) {
            try {
                fragment = URLDecoder.decode(fragment, "UTF-8");
                INode[] nodes = NxsPath.splitStringToNode(fragment);
                for (INode node : nodes) {
                    parts.add(node.getNodeName());
                }
            }
            catch (UnsupportedEncodingException e) {
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

    private static final String URI_DESC = "File system: folders and NeXus files";

    @Override
    public String getURITypeDescription() {
        return URI_DESC;
    }

    // ---------------------------------------------------------
    // Plug-in specific methods
    // ---------------------------------------------------------
    public DetectedSource getSource(URI uri) {
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
                if (source.isStable()) {
                    mDetectedSources.put(uri.toString(), source);
                }
            }
        }
        return source;
    }

    private NxsDatasource() {
    }
}
