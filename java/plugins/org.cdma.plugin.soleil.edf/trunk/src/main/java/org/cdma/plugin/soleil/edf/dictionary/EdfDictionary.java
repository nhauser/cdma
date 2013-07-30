package org.cdma.plugin.soleil.edf.dictionary;

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
import org.cdma.plugin.soleil.edf.EdfFactory;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.JDOMException;
import org.jdom.input.SAXBuilder;

/**
 * @note This class is just a test and is not representative of how the real implementation should
 *       work. Behaviours and algorithms of this class do not apply to the CDM dictionary's
 *       behaviour!
 * @author rodriguez
 * 
 */
public class EdfDictionary implements IDictionary {
    // private String m_path; // Path of the XML file carrying the dictionary
    private Map<IKey, Path> m_itemMap = new HashMap<IKey, Path>();

    @Override
    public void addEntry(String keyName, String path) {
        IFactory factory = EdfFactory.getInstance();
        m_itemMap.put(new Key(factory, keyName), new Path(factory, path));
    }

    @Override
    public boolean containsKey(String keyName) {
        return m_itemMap.containsKey(keyName);
    }

    @Override
    public List<IKey> getAllKeys() {
        return new ArrayList<IKey>(m_itemMap.keySet());
    }

    @Override
    public List<Path> getAllPaths(IKey key) {
        return new ArrayList<Path>(m_itemMap.values());
    }

    @Override
    public Path getPath(IKey key) {
        if (m_itemMap.containsKey(key)) {
            return m_itemMap.get(key);
        }
        else {
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
                String[] temp = br.readLine().split("=");
                if (0 < (temp[0].length())) {
                    addEntry(temp[0], temp[1]);
                }
            }
        }
        catch (Exception ex) {
            throw new FileAccessException("failed to open the dictionary file", ex);
        }
    }

    @Override
    public void readEntries(String filePath) throws FileAccessException {
        File dicFile = new File(filePath);
        if (!dicFile.exists()) {
            throw new FileAccessException("the target dictionary file does not exist");
        }

        // Parse the XML dictionary
        SAXBuilder xmlFile = new SAXBuilder();
        Element root;
        Document dictionary;
        try {
            dictionary = xmlFile.build(dicFile);
        }
        catch (JDOMException e1) {
            throw new FileAccessException("error while to parsing the dictionary!\n"
                    + e1.getMessage());
        }
        catch (IOException e1) {
            throw new FileAccessException("an I/O error prevent parsing dictionary!\n"
                    + e1.getMessage());
        }

        // m_path = dicFile.getAbsolutePath();
        root = dictionary.getRootElement();

        List<?> nodes = root.getChildren("entry"), tmpList;

        Element result;
        String key;
        String xml_name;
        for (Object obj : nodes) {
            Element node = (Element) obj;
            key = node.getChildText("key");
            result = node.getChild("return");
            xml_name = result.getChildText("data-item");
            tmpList = result.getChildren("data");
            for (Object tmpObj : tmpList) {
                Element tmpNode = (Element) tmpObj;
                if (tmpNode.getAttributeValue("name").equals(xml_name)) {
                    String path = "";
                    for (Object pathObj : tmpNode.getChild("path").getChildren("node")) {
                        Element pathNode = (Element) pathObj;
                        String attr = pathNode.getAttributeValue("filter");
                        if (!"false".equals(attr)) {
                            path += "_[" + pathNode.getText() + "]_";
                        }
                        else {
                            path += pathNode.getText();
                        }

                    }
                    addEntry(key, path);
                }
            }
        }
    }

    @Override
    public void removeEntry(String keyName, String path) {
        m_itemMap.remove(keyName);
    }

    @Override
    public void removeEntry(String keyName) {
        m_itemMap.remove(keyName);
    }

    @SuppressWarnings("unchecked")
    @Override
    public IDictionary clone() throws CloneNotSupportedException {
        EdfDictionary dict = new EdfDictionary();
        dict.m_itemMap = (HashMap<IKey, Path>) ((HashMap<IKey, Path>) m_itemMap).clone();
        return dict;
    }

    @Override
    public String getFactoryName() {
        return EdfFactory.NAME;
    }

    @Override
    public void addEntry(String key, Path path) {
        throw new NotImplementedException();
    }

}
