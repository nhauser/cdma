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
/**
 * @brief The CDMA dictionary package describes the Extended Dictionary mechanism.
 *
 * The Extended Dictionary mechanism permits to abstract the data source physical structure.
 * It proposes to not rely on the physical structure of a data source, but to use a virtual 
 * structure that the plug-in will conform to.
 * <p>
 * A physical browsing require to know where and how data are stored according a particular
 * institute's format. At the opposite the Extended Dictionary mechanism defines how data is
 * expected by the above application according an experiment. The dictionary associate keys
 * (that have a physical meaning to a path in the data source: Key) to a path (defined 
 * by a plug-in mapping dictionary: Path). Thus the plug-in should only care of resolving 
 * the path to return the requested data. The API user only cares about data manipulation 
 * not any more on its access.
 * <p>
 * Main objects that allow such an use are Key, LogicalGroup and ExtendedDictionary. The
 * LogicalGroup contains an ExtendedDictionary which have all definitions and there logical
 * organization. The group gives access to several sub-group and and IDataItem.
 * <p>
 * All those object are available in the package 'org.gumtree.data.dictionary'.
 * The plug-in must only provide its own mapping file to activate the Extended Dictionary mechanism.
 */


package org.cdma.dictionary;
