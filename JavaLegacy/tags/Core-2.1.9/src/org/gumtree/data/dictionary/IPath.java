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

import java.util.List;
import org.gumtree.data.interfaces.IModelObject;


/**
 * The IPath is an interface that describe how to reach a specific item in the
 * IDataset. The way that path are written is format plug-in dependent.
 * The more often it will be a <code>String</code> with a separator between nodes
 * that is relevant for the plug-in.
 * <p>
 * The String representation will only be interpreted by the plug-in. But it can be modified
 * using some parameters to make selective choice while browsing the nodes structure of the
 * IDataset.
 * <p>
 * In other cases (for the extended dictionary mechanism) it can also be (or be completed by)
 * a call on a plug-in method. It permits to describe from the dictionary how to have
 * access to the requested node.
 * 
 * @author rodriguez
 * @see org.gumtree.data.interfaces.IKey
 * @see org.gumtree.data.dictionary.IPathParameter
 * @see org.gumtree.data.dictionary.IPathMethod
 */
public interface IPath extends IModelObject, Cloneable {
	/**
	 * Set the value of the path in string
	 */
	void setValue(String path);
	
	/**
	 * Get the value of the path
	 * 
	 * @return string path value
	 */
	String getValue();
	
	/**
	 * Clone the path so it keep unmodified when updated while it
	 * has an occurrence in a dictionary
	 * @return
	 */
	Object clone();

	/**
	 * @return the String representation of the path
	 */
	String toString();
	
	/**
	 * Getter on methods that should be invoked to get data.
	 * The returned list is unmodifiable, key is the method and
	 * value is an array of Object arguments.
	 *  
	 * @return unmodifiable list of methods
	 */
	List<IPathMethod> getMethods();
	
	/**
	 * Set methods that should be invoked to get data.
	 * The list contains PathMethod having method and array of Object for
	 * arguments.
	 * 
	 * @param list that will be copied to keep it unmodifiable
	 * @return unmodifiable map of methods
	 */
	void setMethods(List<IPathMethod> methods);
	
	/**
	 * Will modify the path to make given parameters efficient.
	 * 
	 * @param parameters list of parameters to be inserted
	 */
	void applyParameters(List<IPathParameter> parameters);
	
	/**
	 * Will modify the path to remove all traces of parameters 
	 * that are not defined.
	 */
	void removeUnsetParameters();
	
	/**
	 * Will modify the path to unset all parameters that were defined
	 */
	void resetParameters();
	
	/**
	 * Analyze the path to reach the first undefined parameter.
	 * Return the path parameter to open the node. The parameter has 
	 * a wildcard for value (i.e: all matching nodes can be opened) 
	 * 
	 * @param param 
	 *           output path that will be updated with the appropriate node's
	 *           type and name until to reach the first path parameter
	 * 
	 * @return IPathParameter 
	 * 			 having the right type and name and a wildcard for value (empty)
	 */
	IPathParameter getFirstPathParameter(StringBuffer output);
}
