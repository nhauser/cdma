//****************************************************************************
// Copyright (c) 2008 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors: 
//    Norman Xiong (nxi@Bragg Institute) - initial API and implementation
// ****************************************************************************

/**
 * @brief The utils package provides some tools that can be used to drive the Core or manipulate object obtained using the plug-in.
 * 
 * Two categories can be differentiated in that package:<br>
 * - Core behaviors utilities<br>
 * - Plug-in behaviors utilities<br>
 * 
 * In the former category there are tools to redefine the plug-ins manager and the way to 
 * discover them in the class path.
 * @see org.gumtree.data.utils.IFactoryManager
 * @see org.gumtree.data.utils.IFactoryResolver
 * 
 * In the plug-in category, tools to manipulate arrays like IArrayUtils to reshape, resize, revert, slice... matrices.
 * @see org.gumtree.data.utils.IArrayUtils
 */
package org.gumtree.data.utils;