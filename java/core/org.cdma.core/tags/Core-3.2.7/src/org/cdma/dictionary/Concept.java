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
// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    Tony Lam (nxi@Bragg Institute) - initial API and implementation
//    Clement Rodriguez (ALTEN for SOLEIL Synchrotron) - API evolution
// ****************************************************************************
package org.cdma.dictionary;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jdom2.Element;

/**
 * @brief The Concept class is used to define a physical concept
 * 
 * It also tells how data are expected when using the Extended Dictionary mechanism (EDM).
 * <p>
 * This class summaries what should compound a specific measurement:
 * <br/> - definition of the measurement (energy of a beam)
 * <br/> - units if any are specified
 * <br/> - some mandatory attributes (unit, instrument the measurement belongs to...) 
 * <br/> - synonyms that could be used by the EDM to reference that concept.
 * <p>
 * Synonyms are used to permit a plug-in specific mapping to answer to that concept.
 * 
 * @author rodriguez
 *
 */
public class Concept {
    private String mLabel;
    private String mDescription;
    private String mUnit;
    private String mID;
    private Map<String, String> mAttributes;
    private Map<String, List<String>> mSynonyms;
    
    public Concept(String keyName) {
		mID = keyName;
		mLabel = keyName;
		mSynonyms = new HashMap<String, List<String>>();
		mDescription = "";
		mAttributes = new HashMap<String, String>();
	}
    
    /**
     * Instantiate the Concept corresponding to the given XML node  
     * @param element DOM named 'concept' respecting the concept_file.dtd
     * 
     * @return the loaded concept from XML file
     */
    public static Concept instantiate( Element elem ) {
    	Concept result = null;
    	if( elem.getName().equals("concept") ) {
    		result = new Concept(elem); 
    	}
    	
    	return result;
    }
    
    /**
     * Returns the available synonyms for that concept
     * 
     * @return list of String of any synonym
     */
    public List<String> getSynonymList() {
        return new ArrayList<String>( mSynonyms.keySet());
    }
    
    /**
     * Returns the expected unit for that concept.
     * 
     * @return String unit's representation
     */
    public String getUnit() {
        return mUnit;
    }
    
    /**
     * Returns the name of all mandatory attributes for that concept
     * 
     * @return list of String of attributes' names
     */
    public List<String> getAttributeList() {
        List<String> result = new ArrayList<String>( mAttributes.keySet() );
        return result;
    }
    
    /**
     * Returns the name description of the physical concept.
     * 
     * @return String description
     */
    public String getDescription() {
        return mDescription;
    }
    
    /**
     * Returns the label of the physical concept, its most usual name.
     * 
     * @return String name
     */
    public String getLabel() {
        return mLabel;
    }
    
    /**
     * Returns true if the given synonym is for that concept
     * 
     * @param synonym we want to test
     * @param plugin name for this synonym
     * @return boolean value that is true in case of matching 
     */
    public boolean matches( String synonym, String factoryName ) {
    	boolean result;
    	if( factoryName != null ) {
    		result = mSynonyms.containsKey( synonym ) && mSynonyms.get(synonym).contains(factoryName);
    	}
    	else {
    		result = mSynonyms.containsKey( synonym );
    	}
    	return result;
    }
    
    /**
     * Returns the concept ID by definition it is its first synonym.
     * 
     * @return the ID of the concept
     */
    public String getConceptID() {
        return mID;
    }
    
    public void addSynonym(String value, String factoryName ) {
    	List<String> synonyms = mSynonyms.get(value);
    	if( synonyms == null ) {
    		synonyms = new ArrayList<String>();
    		mSynonyms.put(value, synonyms);
    	}
    	
    	if( ! synonyms.contains(factoryName) ) {
    		synonyms.add(value);
    	}
    }
    
    private Concept(Element elem) {
    	mSynonyms = new HashMap<String, List<String>>();
    	
        mLabel = elem.getAttributeValue("label");
        mID    = elem.getAttributeValue("key");
        // Local variables
        Element dom;
        List<?> list;
        
        // Constructing the definition part
        dom = elem.getChild("definition");
        mDescription = dom.getChildText("description");
        dom = dom.getChild("unit");
        if( dom != null ) {
            mUnit = dom.getText();
        }
        
        // Constructing the attributes part
        mAttributes = new HashMap<String, String>();
        dom = elem.getChild("attributes");
        if( dom != null ) {
            list = dom.getChildren("attribute");
            
            for( Object synonym : list ) {
                dom = (Element) synonym;
                mAttributes.put( dom.getAttributeValue("name"), dom.getText() );
            }
        }            
    }
}
