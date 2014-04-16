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
//    ClÃ©ment Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************

/**
 * @brief The CDMA interfaces package contains all interfaces a plug-in must implement.
 * 
 * All interfaces, but the IContainer one, in this package should be implemented by a plug-in.
 * To be functional a plug-in should also implement the IFactory interface in the package
 * org.cdma so it can create CDMA objects.
 * 
 * To activate the Extended Dictionary the interface IPathParamResolver should be implemented.
 * Indeed it aims to resolve the paths while using that mechanism.
 */

package org.cdma.interfaces;
