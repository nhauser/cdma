// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - initial API
// ****************************************************************************
package org.cdma.internal.dictionary;

import java.util.List;
import org.cdma.dictionary.Concept;

/// @cond internal

/**
 * @brief The ConceptManager aims to retrieve the Concept that corresponds to an IKey
 * 
 * The ConceptManage stores all the concepts that can ask the end user when using the 
 * <i>Extended Dictionary Mechanism</i> for specific experiment.
 * <p>
 * The main goal is to retrieve the right Concept object for a given IKey. <b>Several IKey
 * can lead to the same Concept. But only one Concept can match a IKey</b>
 * 
 * @see Concept
 * @author rodriguez
 */
public class ConceptManager {
    private List<Concept> mConcepts; // List of all concepts currently loaded 
    
    public ConceptManager( List<Concept> concepts ) {
        mConcepts = concepts;
    }
    
    /**
     * Retrieve the concept corresponding the given Key
     * 
     * @param key which the matching concept is required 
     * @return Concept object 
     */
    public Concept getConcept(String keyName) {
        Concept result = null;
        
        for( Concept concept : mConcepts ) {
            if( concept.matches( keyName ) ) {
                result = concept;
                break;
            }
        }
        
        return result;
    }
    
    /**
     * Retrieve the concept's ID corresponding the given key name or null if not found.
     * 
     * @param key name which the concept's ID is required 
     * @return String representing the identifier of the corresponding concept
     */
    public String getConceptID(String keyName) {
        String result = null;
        for( Concept concept : mConcepts ) {
            if( concept.matches( keyName ) ) {
                result = concept.getConceptID();
                break;
            }
        }
        
        return result; 
    }
    
    /**
     * Add a given concept to this concept manager
     * 
     * @param concept
     */
    public void addConcept(Concept concept) {
        mConcepts.add(concept);
    }

}

/// @endcond internal