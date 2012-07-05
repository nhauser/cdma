// ****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors
//    Clement Rodriguez - initial API and implementation
//    Norman Xiong
// ****************************************************************************

/**
 * @brief The CDMA dictionary filter package provides some filter that can be used on IKey.
 *
 * A filter is applied on an IKey and represents conditions that permits identifying a 
 * specific node using the <i>Extended Dictionary Mechanism</i>.
 * <p>
 * When a given IKey can return several IContainer, the IFilter applied on that key
 * will make possible to define which one is relevant.
 * <p>
 * As the end user (application programmer) does not know what institute's plug-in is 
 * in used, plug-in shouldn't define its own IFilter implementation. Indeed they won't
 * be applied by the application. 
 * <p>
 * Instead the plug-in developer should submit IFilter implementations to the community.
 * Each filter must only use CDMA's methods to guarantee they can be used by every plug-in.
 */


package org.cdma.dictionary.filter;
