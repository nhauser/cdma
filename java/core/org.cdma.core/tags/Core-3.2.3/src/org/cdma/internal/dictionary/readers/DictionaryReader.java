package org.cdma.internal.dictionary.readers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.cdma.Factory;
import org.cdma.IFactory;
import org.cdma.dictionary.Concept;
import org.cdma.exception.FileAccessException;
import org.cdma.internal.dictionary.classloader.PluginMethodManager;
import org.cdma.internal.dictionary.solvers.ItemSolver;
import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import org.jdom2.input.sax.XMLReaders;

public class DictionaryReader {
	
	private static volatile PluginMethodManager mMethodMgr;
	private static volatile DataManager mDataManager;
	private String   mViewSrc;      // View file to read
	private String   mConceptSrc;   // Concept file to read
	private boolean mLoadSynonyms; // Have synonyms to be reloaded

	public DictionaryReader( String viewFile ) {
		mMethodMgr    = PluginMethodManager.instantiate();
		mDataManager  = DataManager.instantiate();
		mViewSrc      = viewFile;
		mConceptSrc   = null;
		mLoadSynonyms = false;
	}
	
	public DataView getView() throws FileAccessException {
		DataView result = mDataManager.getView(mViewSrc);
		if( result == null ) {
			init();
			result = mDataManager.getView(mViewSrc);
		}
		return result;
	}
	
	public DataMapping getMapping(IFactory factory, String mappingFile) throws FileAccessException {
		DataMapping result = mDataManager.getMapping(mappingFile);
		if( result == null ) {
			readMappingFile(factory, mappingFile);
			result = mDataManager.getMapping(mappingFile);
		}
		return result;
	}
	
	public DataConcepts getConcepts() throws FileAccessException {
		DataConcepts result = mDataManager.getConcept(mViewSrc);
		if( result == null ) {
			init();
			result = mDataManager.getConcept(mViewSrc);
		}
		return result;
	}
	
	public void init( ) throws FileAccessException {
		readConceptsFile();
		readViewFile();
		readSynonyms();
	}
	
	protected void readConceptsFile() throws FileAccessException {
		// Check the need of reading concepts file
		if( mDataManager.getConcept( mViewSrc ) == null ) {
			mLoadSynonyms = true;
			try {
				// Get the specific concept file's path if any
				Element startNode = saxBuildFile(mViewSrc);
				if( startNode != null ) {
					String concept = startNode.getAttributeValue("concept");
					if (concept != null) {
						mConceptSrc = Factory.getPathConceptDictionaryFolder() + concept;
					}
				}
				
				// Read concept files
				List<Concept> list = readConceptsFiles();
				mDataManager.registerConcept( mViewSrc, new DataConcepts( list ) );
			} catch (FileAccessException e) {
				throw new FileAccessException( "Error while reading the concept file: " + e.getMessage(), e );
			}
		}
	}
	
    protected void readMappingFile( IFactory factory, String file ) throws FileAccessException {
    	DataMapping mapping;
    	try {
	        Element root = saxBuildFile(file);
	        if( root == null ) {
	        	mapping = new DataMapping("");
	        }
	        else {
		        String version = root.getAttributeValue("version");
		
		        mapping = new DataMapping(version);
		        
		        List<Element> nodes = root.getChildren("item");
		        String keyName;
		
		        // Updating the KeyID / Path map
		        for (Element elem : nodes) {
		            // Get the key name
		            keyName = elem.getAttributeValue("key");
		
		            // Add the corresponding mapping
		            mapping.addSolver( keyName, new ItemSolver(factory, mMethodMgr, elem) );
		        }
	        }
			mDataManager.registerMapping(file, mapping);
		} catch (FileAccessException e) {
			throw new FileAccessException( "Error while reading the mapping file: " + e.getMessage(), e );
		}
    }
    
	protected void readViewFile() throws FileAccessException {
		if( mDataManager.getView( mViewSrc ) == null ) {
			mLoadSynonyms = true;
			try {
				// Read view dictionary
		        Element root = saxBuildFile(mViewSrc);
		        DataView view;
		        if( root != null ) {
		            view = readViewFile(root);
		            view.setName( root.getAttributeValue("name") );
		            
				}
				else {
					view = new DataView();
				}
		        mDataManager.registerView(mViewSrc, view);
			} catch (FileAccessException e) {
				throw new FileAccessException( "Error while reading the view file: " + e.getMessage(), e );
			}
		}
	}
	
	protected void readSynonyms() throws FileAccessException {
		if( mLoadSynonyms ) {
			mLoadSynonyms = false;
			try {
	            // Read view dictionary
	            Element root = saxBuildFile(mViewSrc);
	            if( root != null ) {
		            String synonymFile = root.getAttributeValue( "synonym" );
		            if( synonymFile != null && ! synonymFile.trim().isEmpty() ) {
		            	File file = new File(mViewSrc);
		            	String fileName = file.getParentFile().getAbsolutePath();
		            	readSynonyms(fileName + "/" + synonymFile);
					}
	            }
			} catch (FileAccessException e) {
				throw new FileAccessException( "Error while reading the synonyms file: " + e.getMessage(), e );
			}
		}
	}
	
	// ---------------------------------------------------------------
    // PRIVATE : Reading methods
    // ---------------------------------------------------------------
    private List<Concept> readConceptsFiles() throws FileAccessException {
        List<Concept> concepts = new ArrayList<Concept>();
        
        String commonFile = Factory.getPathCommonConceptDictionary();
        // Read common concept file
        if( commonFile != null ) {
            concepts.addAll( readConceptsFile(commonFile) );
        }
        
        // Read specific concept file
        concepts.addAll( readConceptsFile( mConceptSrc ) );
        
        return concepts;
    }
    
    @SuppressWarnings("unchecked")
    private List<Concept> readConceptsFile( String filePath ) throws FileAccessException {
		// Check file exists
		List<Concept> concepts = new ArrayList<Concept>();
        Element elem = saxBuildFile(filePath);
        if( elem != null ) {
	        Concept concept;
	        List<?> nodes = elem.getChildren("concept");
	        for (Object child : (List<Element>) nodes) {
	            elem = (Element) child;
	            concept = Concept.instantiate(elem);
	            if( concept != null ) {
	            	concepts.add(concept);
	            }
	        }
        }

        return concepts;
    }
    
    private DataView readViewFile(Element startNode) throws FileAccessException {
    	DataView view = new DataView();
    	
    	// List all child nodes
        List<Element> nodes = startNode.getChildren();
        String keyName;
        
        // For each element of the view (item or group)
        for( Element current : nodes ) {
            // Get the key name
            keyName = current.getAttributeValue("key");
            if( keyName != null && !keyName.isEmpty() ) {
                // If the element is an item
                if( current.getName().equals("item") ) {
                	view.addKey(keyName);
                }
                // If the element is a group of keys
                else if( current.getName().equals("group") ){
                	// Read the sub view
                	DataView subView = readViewFile( current );
                	
                    // Add it to current view
                	view.addView( keyName, subView );
                }
            }
        }
        return view;
    }

    private void readSynonyms(String synonymFile) throws FileAccessException {
    	Element root = saxBuildFile(synonymFile);
    	if( root != null ) {
	    	String plugID, key, mapping;
	    	for( Element plugin : root.getChildren( "plugin" ) ) {
	    		plugID = plugin.getAttributeValue("name");
	    		
	    		for( Element synonym : plugin.getChildren("synonym") ) {
	    			// Get synonym values
	    			key     = synonym.getAttributeValue( "key" );
	    			mapping = synonym.getAttributeValue( "mapping" );

	    			// Add the synonym
	    			addSynonym(key, mapping, plugID);
	    		}
	    	}
    	}
	}
    
    private void addSynonym(String key, String mapping, String plugID) throws FileAccessException {
    	DataConcepts concepts = getConcepts();
    	Concept concept = concepts.getConcept( key );
		if( concept != null ) {
			if (! concept.getSynonymList().contains( mapping ) ) {
				concept.addSynonym(mapping, plugID);
			}
		}
		else {
			concept = new Concept(key);
			concept.addSynonym(mapping, plugID);
			concepts.addConcept(concept);
		}
	}

	/**
     * Check the given file exists and open the root node
     * 
     * @param filePath XML file to be opened
     * @return Sax element that is the root of the XML file
     * @throws FileAccessException
     */
    private Element saxBuildFile(String filePath) throws FileAccessException {
    	Element result;
    	// Check the file path isn't empty
        if( filePath == null || filePath.isEmpty() ) {
        	result = null;	
        }
        else {
	        // Check the file path is a valid XML file 
	        File dicFile = new File(filePath);
	        if ( !dicFile.exists() ) {
	            throw new FileAccessException("the target file does not exist:\n" + filePath);
	        }
	        else if( dicFile.isDirectory() ) {
	            throw new FileAccessException("the target is a folder not a XML file:\n" + filePath);
	        }
	
	        // Open the XML file to get its root element
	        SAXBuilder xmlFile = new SAXBuilder(XMLReaders.DTDVALIDATING);
	        Document document;
	        try {
	            document = xmlFile.build(dicFile);
	        }
	        catch (JDOMException e) {
	            throw new FileAccessException("error while to parsing file:\n" + filePath + "\n" + e.getMessage(), e);
	        }
	        catch (IOException e) {
	            throw new FileAccessException("an I/O error prevent parsing file:\n" + filePath + "\n" +e.getMessage(), e);
	        }
	        result = document.getRootElement();
        }
        return result;
    }
}
