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
 * @brief The CDMA internal dictionary package contains sub-packages for the <i>Extended Dictionary Mechanism</i>.
 * 
 * Those package aren't exposed, they are internally used by the <i>Extended Dictionary Mechanism</i>.
 * <p>
 * It contains following packages:
 * <br/> - solvers: allow having same behavior to resolve an IKey 
 * <br/> - readers: managing XML relative to the dictionary mechanism (views, mapping, concept and synonyms)
 * <br/> - classloader: permit discovering all code that will be dynamically loaded and executed according mappings
 * <p>
 * To activate it's API reference documentation use the '<i>internal</i>' condition flag.
 * 
 */

package org.cdma.internal.dictionary;