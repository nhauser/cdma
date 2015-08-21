package org.cdma.utilities;
import org.cdma.exception.CDMAException;


public interface ICDMAExceptionHandler {

    public void handleCDMAException(Object source, CDMAException cdmaException);

}
