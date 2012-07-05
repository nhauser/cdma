package org.cdma.exception;

public class CDMAException extends Exception {
    private static final long serialVersionUID = 7442908563199393797L;
    
    public CDMAException() { }

    public CDMAException(String message) {
        super(message);
    }

    public CDMAException(Throwable cause) {
        super(cause);
    }

    public CDMAException(String message, Throwable cause) {
        super(message, cause);
    }
}
