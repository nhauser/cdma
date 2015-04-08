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
package org.cdma.plugin.soleil.edf.utils;

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class EdfFileReader {

    public EdfFileReader() {
    }

    public static ByteBuffer readAsBytes(int sizeToRead, boolean littleEndian, DataInputStream dis)
            throws IOException {

        byte[] flatByteImageValue = new byte[sizeToRead];
        ByteBuffer result = ByteBuffer.allocate(sizeToRead);

        if (littleEndian) {
            result = result.order(ByteOrder.LITTLE_ENDIAN);
        }

        dis.readFully(flatByteImageValue);
        result = result.put(flatByteImageValue);

        result.rewind();

        return result;
    }
}
