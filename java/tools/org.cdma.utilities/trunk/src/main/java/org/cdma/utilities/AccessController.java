//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
//    Clément Rodriguez (clement.rodriguez@synchrotron-soleil.fr)
// See AUTHORS file
//******************************************************************************
package org.cdma.utilities;

import java.util.concurrent.Semaphore;

/**
 * This class is used to manage access to CDMA in order not to have JVM crashes with not thread-safe
 * plugins.
 * 
 * @author GIRARDOT
 */
public class AccessController {

    private final static Semaphore accessControl = new Semaphore(1, true);

    /**
     * Asks for access
     * 
     * @throws InterruptedException if the current thread is interrupted while waiting for permit
     */
    public static void takeAccess() throws InterruptedException {
        accessControl.acquire();
    }

    /**
     * Releases access
     */
    public static void releaseAccess() {
        accessControl.release();
    }

}
