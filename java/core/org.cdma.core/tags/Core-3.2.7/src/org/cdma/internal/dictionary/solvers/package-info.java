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
 *        St�phane Poirier (SOLEIL Synchrotron) - API v2 design and conception
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
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr) - API evolution
// ****************************************************************************

/**
 * @brief The CDMA internal dictionary readers package concerns the resolution of IKey.
 * 
 * This package isn't exposed, it is internally used by the <i>Extended Dictionary Mechanism</i>.
 * <p>
 * It contains classes that:
 * <br/> - know how to resolve a IDataItem or a LogicalGroup according a path, a method call or a key
 * <br/> - know how to resolve an IAttribute according a path, a method call or a key
 * <br/> - ...
 * <p>
 * To activate it's API reference documentation use the '<i>internal</i>' condition flag.
 * 
 */

package org.cdma.internal.dictionary.solvers;
