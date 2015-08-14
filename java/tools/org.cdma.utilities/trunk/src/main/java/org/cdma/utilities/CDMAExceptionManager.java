package org.cdma.utilities;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

import org.cdma.exception.CDMAException;

public class CDMAExceptionManager {

    private static final List<ICDMAExceptionHandler> listeners = new ArrayList<ICDMAExceptionHandler>();

    // private static final Logger LOGGER = LoggerFactory.getLogger(CDMAExceptionManager.class);

    public static void addCDMAExceptionHandler(ICDMAExceptionHandler handler) {
        if (!listeners.contains(handler)) {
            listeners.add(handler);
        }
    }

    public static void removeCDMAExceptionHandler(ICDMAExceptionHandler handler) {
        if (listeners.contains(handler)) {
            synchronized (listeners) {
                listeners.remove(handler);
            }
        }
    }

    public static void notifyHandler(Object source, CDMAException error) {
        // TODO Logging centralization LOGGER
        ListIterator<ICDMAExceptionHandler> iterator = listeners.listIterator();
        while (iterator.hasNext()) {
            iterator.next().handleCDMAException(source, error);
        }
    }

}
