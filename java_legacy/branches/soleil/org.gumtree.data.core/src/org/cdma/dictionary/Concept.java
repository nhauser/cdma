// ****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors
//    Clement Rodriguez - initial API and implementation
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
    private Map<String, String> mAttributes;
    private List<String> mSynonyms;
    
    public Concept(Element elem) {
        if( elem.getName().equals("concept") ) {
            mLabel = elem.getAttributeValue("label");
            
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
            
            // Constructing the synonyms part
            mSynonyms = new ArrayList<String>();
            dom = elem.getChild("synonyms");
            list = dom.getChildren("key");
            for( Object synonym : list ) {
                mSynonyms.add( ((Element) synonym).getText() );
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
    
    /**
     * Returns the available synonyms for that concept
     * 
     * @return list of String of any synonym
     */
    public List<String> getSynonymList() {
        return mSynonyms;
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
     * @return boolean value that is true in case of matching 
     */
    public boolean matches( String synonym ) {
        return mSynonyms.contains( synonym );
    }
    
    /**
     * Returns the concept ID by definition it is its first synonym.
     * 
     * @return the ID of the concept
     */
    public String getConceptID() {
        return mSynonyms.get(0);
    }
}
