// ****************************************************************************
// Copyright (c) 2010 Australian Nuclear Science and Technology Organisation.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the Eclipse Public License v1.0 
// which accompanies this distribution, and is available at
// http://www.eclipse.org/legal/epl-v10.html
// 
// Contributors
//    Clement Rodriguez - initial API and implementation
// ****************************************************************************
package org.cdma.dictionary;

import org.cdma.exception.CDMAException;
import org.cdma.internal.IModelObject;

/**
 * @brief IPluginMethod interface permits to post process data when needed.
 *
 * The IPluginMethod aims to provide a mechanism permitting to call
 * methods that are specified in the XML mapping document from the <i>Extended 
 * Dictionary Mechanism</i>.
 * This interface have to be implemented into the 'institute's plug-in'
 * for each such methods.
 * <p/> 
 * When using the Extended Dictionary Mechanism, plug-ins need sometimes to 
 * post processed to conform to what is expected {@link Concept}. The treatment 
 * will be done by the {@link execute} method's implementation.
 * <p>
 * For examples we can consider:
 * <br/> - the problem of transforming an energy into a wavelength.
 * <br/> - to gather data items into bigger one
 * <p>
 * The IPluginMethod has all required information to process item
 * using the {@link Context} object. The context is an in/out value,
 * each result of the executed method must be stored. They are accessible 
 * using getContainers and setContainers methods. Only the result of lastly 
 * executed method is stored, the institute is responsible to flush it.
 * 
 * @see org.cdma.dictionary.Context
 * 
 * @note <b/Important:</b> As those methods are loaded dynamically using OSGI service or basic 
 * loading service the plug-in must describes it:
 * <br/> - for OSGI: in OSGI-INF/ declare the service providers
 * <br/> - for basic service loader: in META-INF/services/org.cdma.dictionary.IPluginMethod declare
 * the class in its namespace.
 */
public interface IPluginMethod extends IModelObject {
  
    /**
     * Executes the method
     *
     * @param context input/output context
     * @throw  Exception in case of any trouble
     */
    void execute(Context context) throws CDMAException;
}
