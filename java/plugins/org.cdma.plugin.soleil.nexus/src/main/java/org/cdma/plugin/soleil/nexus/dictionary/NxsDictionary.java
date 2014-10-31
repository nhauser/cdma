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
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
 * 	Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
 * 	Gregory VIGUIER (SOLEIL Synchrotron) - API evolution
 ******************************************************************************/
package org.cdma.plugin.soleil.nexus.dictionary;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cdma.IFactory;
import org.cdma.dictionary.Key;
import org.cdma.dictionary.Path;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IKey;
import org.cdma.plugin.soleil.nexus.NxsFactory;
import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.jdom2.input.sax.XMLReaders;

/**
 * @note This class is just a test and is not representative of how the real implementation should
 *       work. Behaviors and algorithms of this class do not apply to the CDMA dictionary's
 *       behaviour!
 * @author rodriguez
 * 
 */
@Deprecated
public final class NxsDictionary implements IDictionary, Cloneable {
    private String mPath; // Path of the XML file carrying the dictionary
    private Map<IKey, Path> mItemMap; // Map associating keys from view file to path from mapping
                                      // file

    public NxsDictionary() {
        mItemMap = new HashMap<IKey, Path>();
    }

    @Override
    public void addEntry(String keyName, String path) {
        IFactory factory = NxsFactory.getInstance();
        mItemMap.put(new Key(factory, keyName), new Path(factory, path));
    }

    @Override
    public boolean containsKey(String keyName) {
        return mItemMap.containsKey(keyName);
    }

    @Override
    public List<IKey> getAllKeys() {
        return new ArrayList<IKey>(mItemMap.keySet());
    }

    @Override
    public List<Path> getAllPaths(IKey keyName) {
        return new ArrayList<Path>(mItemMap.values());
    }

    @Override
    public Path getPath(IKey keyName) {
        if (mItemMap.containsKey(keyName)) {
            return mItemMap.get(keyName);
        } else {
            return null;
        }
    }

    @Override
    public void readEntries(URI uri) throws FileAccessException {
        File dicFile = new File(uri);
        if (!dicFile.exists()) {
            throw new FileAccessException("the target dictionary file does not exist");
        }
        try {
            BufferedReader br = new BufferedReader(new FileReader(dicFile));
            while (br.ready()) {
                String line = br.readLine();
                if (line != null) {
                    String[] temp = line.split("=");
                    if (0 < (temp[0].length())) {
                        addEntry(temp[0], temp[1]);
                    }
                }
            }
            br.close();
        } catch (IOException ex) {
            throw new FileAccessException("failed to open the dictionary file\n", ex);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void readEntries(String filePath) throws FileAccessException {
        File dicFile = new File(filePath);
        if (!dicFile.exists()) {
            throw new FileAccessException("the target dictionary file does not exist");
        }

        // Parse the XML dictionary
        SAXBuilder xmlFile = new SAXBuilder(XMLReaders.DTDVALIDATING);
        Element root;
        Document dictionary;
        try {
            dictionary = xmlFile.build(dicFile);
        } catch (JDOMException e1) {
            throw new FileAccessException("error while to parsing the dictionary!\n", e1);
        } catch (IOException e1) {
            throw new FileAccessException("an I/O error prevent parsing dictionary!\n", e1);
        }

        mPath = dicFile.getAbsolutePath();
        root = dictionary.getRootElement();

        List<?> nodes = root.getChildren("item");

        String key;
        String path = "";
        for (Element node : (List<Element>) nodes) {
            key = node.getAttributeValue("key");
            path = node.getChildText("path");
            if (key != null && !key.isEmpty() && path != null && !path.isEmpty()) {
                addEntry(key, path);
            }
        }
    }

    @Override
    public void removeEntry(String keyName, String path) {
        mItemMap.remove(keyName);
    }

    @Override
    public void removeEntry(String keyName) {
        mItemMap.remove(keyName);
    }

    @SuppressWarnings("unchecked")
    @Override
    public IDictionary clone() throws CloneNotSupportedException {
        NxsDictionary dict = new NxsDictionary();
        dict.mItemMap = (HashMap<IKey, Path>) ((HashMap<IKey, Path>) mItemMap).clone();
        return dict;
    }

    /**
     * @return path of the dictionary file
     */
    public String getPath() {
        return mPath;
    }

    @Override
    public void addEntry(String key, Path path) {
        throw new NotImplementedException();
    }

    @Override
    public String getFactoryName() {
        return NxsFactory.NAME;
    }
}
