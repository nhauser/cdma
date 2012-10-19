// ****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************

/**
 * @brief The CDMA internal dictionary reader package contains helpers for managing XML files.
 * 
 * This package isn't exposed, it is internally used by the <i>Extended Dictionary Mechanism</i>.
 * <p>
 * It contains classes that:
 * <br/> - a loader for all XML files relative to the dictionary
 * <br/> - manages View, Concept and Mapping
 * <br/> - a cache preventing multiple loadings
 * <br/> - ...
 * <p>
 * To activate it's API reference documentation use the '<i>internal</i>' condition flag.
 * 
 */

package org.cdma.internal.dictionary.readers;