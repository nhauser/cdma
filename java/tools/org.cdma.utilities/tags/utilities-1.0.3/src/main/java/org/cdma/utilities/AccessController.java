// ******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
//    Clement Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities;


/**
 * This class is used to manage access to CDMA in order not to have JVM crashes with not thread-safe
 * plugins.
 * 
 * @author GIRARDOT
 */
public class AccessController {

    public static final Object ACCESS_LOCK = new Object();

}
