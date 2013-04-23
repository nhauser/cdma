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
 * @brief The CDMA internal dictionary classloader package aims to dynamically load plug-in's code.
 * 
 * This package isn't exposed, it's internally used by the <i>Extended Dictionary Mechanism</i>.
 * <p>
 * It contains proposes following functionalities:
 * <br/> - a common interface for a seeker of plug-in dynamic code
 * <br/> - a manager to centralize all found code preventing multiple loading 
 * <br/> - various loaders: based on java class loader or OSGI service loader 
 * <br/> ...
 * <p>
 * To activate it's API reference documentation use the '<i>internal</i>' condition flag.
 * 
 */

package org.cdma.internal.dictionary.classloader;