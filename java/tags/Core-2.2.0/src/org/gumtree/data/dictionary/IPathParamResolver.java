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

import org.gumtree.data.interfaces.IContainer;
import org.gumtree.data.interfaces.IModelObject;


/**
 * IPathParamResolver is used internally by the extended dictionary mechanism. 
 * Its aim is to determine according to a given path how to reach a specific IContainer
 * without any ambiguity by returning the IPathParameter that corresponds to the IContainer.
 * <p>
 * The IPathParaResolver is associated to a IPath that is stored in the dictionary.
 * It's the comparison between the associated path and the IContainer's physical real path, that
 * will define the IPathParameter.
 * 
 * @see org.gumtree.data.dictionary.IPathParameter
 * @see org.gumtree.data.dictionary.IPath
 * @author rodriguez
 */

public interface IPathParamResolver extends IModelObject {

	/**
	 * It will returns
	 * @param node 
	 * @return
	 */
	IPathParameter resolvePathParameter(IContainer node);
}
