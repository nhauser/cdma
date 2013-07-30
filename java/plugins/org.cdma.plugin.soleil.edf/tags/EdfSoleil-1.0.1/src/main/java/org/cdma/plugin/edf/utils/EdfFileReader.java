package org.cdma.plugin.edf.utils;

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
            result.order(ByteOrder.LITTLE_ENDIAN);
        }

        dis.readFully(flatByteImageValue);
        result = result.put(flatByteImageValue);

        result.rewind();

        return result;
    }
}
