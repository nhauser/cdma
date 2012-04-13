// ****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Rodriguez Clément (clement.rodriguez@synchrotron-soleil.fr) - initial API and implementation
// ****************************************************************************

/**
 * @bried The CDMA dictionary package permits to abstract the data source physical structure.
 *
 * The Extended Dictionary mechanism proposes to not rely on the physical structure of a data 
 * source, but to use a virtual structure that the plug-in will conform to.
 * <p/>
 * A physical browsing require to know where and how data are stored according a particular
 * institute's format. At the opposite the Extended Dictionary mechanism defines how data is
 * expected by the above application according an experiment. The dictionary associate keys
 * (that have a physical meaning to a path in the data source). Thus the plug-in should only
 * care of resolving the path to return the requested data. The API user only cares about 
 * data manipulation not any more on its access.
 * <p/>
 * The main objects that allows such an use are ILogicalGroup and IExtendedDictionary. The
 * ILogicalGroup contains a IExtendedDictionary which have all definitions and there logical
 * organization. The group gives access to several sub-group and and IDataItem.  
 */
package org.gumtree.data.dictionary;