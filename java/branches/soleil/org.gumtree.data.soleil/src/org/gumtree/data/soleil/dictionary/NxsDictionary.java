package org.gumtree.data.soleil.dictionary;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.gumtree.data.IFactory;
import org.gumtree.data.dictionary.IPath;
import org.gumtree.data.dictionary.impl.Key;
import org.gumtree.data.dictionary.impl.Path;
import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IKey;
import org.gumtree.data.soleil.NxsFactory;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.JDOMException;
import org.jdom.input.SAXBuilder;


/**
 * @note This class is just a test and is not representative of how the real implementation should work.
 * Behaviors and algorithms of this class do not apply to the CDMA dictionary's behaviour!
 * @author rodriguez
 *
 */
public final class NxsDictionary implements IDictionary, Cloneable {
    private String mPath;              // Path of the XML file carrying the dictionary 
  private Map<IKey, IPath> mItemMap; // Map associating keys from view file to path from mapping file
  
  public NxsDictionary() {
    mItemMap = new HashMap<IKey, IPath>();
  }
  
    @Override
  public void addEntry(String keyName, String path)
  {
      IFactory factory = NxsFactory.getInstance();
    mItemMap.put(new Key(factory, keyName), new Path(factory, path));
  }

    @Override
  public boolean containsKey(String keyName)
  {
    return mItemMap.containsKey(keyName);
  }

  @Override
  public List<IKey> getAllKeys()
  {
    return new ArrayList<IKey>(mItemMap.keySet());
  }

  @Override
  public List<IPath> getAllPaths(IKey keyName)
  {
    return new ArrayList<IPath>(mItemMap.values());
  }

  @Override
  public IPath getPath(IKey keyName)
  {
    if( mItemMap.containsKey(keyName) )
    {
      return mItemMap.get(keyName);
    }
    else
    {
      return null;
    }
  }

  @Override
  public void readEntries(URI uri) throws FileAccessException
  {
    File dicFile = new File(uri);
    if (!dicFile.exists()) 
    {
      throw new FileAccessException("the target dictionary file does not exist");
    }
    try 
    {
      BufferedReader br = new BufferedReader(new FileReader(dicFile));
      while (br.ready())
      {
        String line = br.readLine();
        if( line != null ) {
          String[] temp = line.split("=");
          if (0 < (temp[0].length())) 
          {
            addEntry(temp[0], temp[1]);
          }
        }
      }
      br.close();
    } 
    catch (IOException ex) 
    {
      throw new FileAccessException("failed to open the dictionary file\n", ex);
    }
  }

  @SuppressWarnings("unchecked")
  @Override
    public void readEntries(String filePath) throws FileAccessException
    {
        File dicFile = new File(filePath);
        if (!dicFile.exists()) 
        {
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
            throw new FileAccessException("error while to parsing the dictionary!\n", e1);
        }
        catch (IOException e1) {
            throw new FileAccessException("an I/O error prevent parsing dictionary!\n", e1);
        }
        
        mPath = dicFile.getAbsolutePath();
        root = dictionary.getRootElement();
        
        List<?> nodes = root.getChildren("entry"), tmpList;
        
        Element result;
        String key;
        String xml_name;
        for( Element node : (List<Element>) nodes ) {
            key = node.getChildText("key");
            result = node.getChild("return");
            xml_name = result.getChildText("data-item");
            tmpList = result.getChildren("data");
            for( Element tmpNode : (List<Element>) tmpList ) {
                if( tmpNode.getAttributeValue("name").equals(xml_name) ) {
                    String path = "";
                    for( Element pathNode : (List<Element>) tmpNode.getChild("path").getChildren("node") ) {
                        String filter = pathNode.getAttributeValue("filter");
                        
                        if( filter == null || !"false".equals(filter) ) {
                          // If "filter" == null then any filters can be applied
                          if( filter == null ) {
                            path += pathNode.getText();
                          }
                          // If "filter" == "name" then only filter "name" can be applied
                          else {
                            String type = pathNode.getAttributeValue("type");
                            String operand = pathNode.getAttributeValue("value");
                            path += "_[" +
                                ( type != null ? (type) : "" ) +
                                ( operand != null ? ("=" + operand) : "" ) +
                                "]_" +  pathNode.getText();
                          }
                            
                        }
                        // else "filter" == false then NO filter can be applied
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
    mItemMap.remove(keyName);
  }

  @Override
  public void removeEntry(String keyName) {
    mItemMap.remove(keyName);
  }
  
    
    @SuppressWarnings("unchecked")
  @Override
  public IDictionary clone() throws CloneNotSupportedException
  {
    NxsDictionary dict = new NxsDictionary();
    dict.mItemMap = (HashMap<IKey, IPath>) ((HashMap<IKey, IPath>) mItemMap).clone();
    return dict;
  }

  /**
   * @return path of the dictionary file
   */
  public String getPath() {
    return mPath;
  }
  
  @Override
  public void addEntry(String key, IPath path) {
    // TODO Auto-generated method stub
    
  }

  @Override
  public String getFactoryName() {
    return NxsFactory.NAME;
  }
}
