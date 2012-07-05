/****************************************************************************** 
 * Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0 
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 * 
 * Contributors: 
 * 	  Clement Rodriguez - initial API and implementation
 *    Norman Xiong
 ******************************************************************************/
package org.gumtree.data.dictionary;

import org.gumtree.data.exception.FileAccessException;
import org.gumtree.data.interfaces.IDictionary;
import org.gumtree.data.interfaces.IKey;

public interface IExtendedDictionary extends IDictionary {
	/**
	 * Get a sub part of this dictionary that corresponds to a key.
	 * @param IKey object
	 * @return IExtendedDictionary matching the key
	 */
	public IExtendedDictionary getDictionary(IKey key);
	
	/**
	 * Get the version number (in 3 digits default implementation) that is plug-in
	 * dependent. This version corresponds of the dictionary defining the path. It  
	 * permits to distinguish various generation of IDataset for a same institutes.
	 * Moreover it's required to select the right class when using a IClassLoader
	 * invocation. 
	 */
	public String getVersionNum();
	
	/**
	 * Get the plug-in implementation of a IClassLoader so invocations of external
	 * are made possible.
	 */
	public IClassLoader getClassLoader();
	
	/**
	 * Get the view name matching this dictionary
	 */
	public String getView();
	
	/**
	 * Read all keys stored in the XML dictionary file
	 */
	public void readEntries() throws FileAccessException;
	
	/**
	 * Return the path to reach the key dictionary file
	 */
	public String getKeyFilePath();
	
	/**
	 * Return the path to reach the mapping dictionary file
	 */
	public String getMappingFilePath();
}
