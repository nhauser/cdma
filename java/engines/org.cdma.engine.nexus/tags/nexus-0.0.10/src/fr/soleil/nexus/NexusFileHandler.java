//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
package fr.soleil.nexus;

import org.nexusformat.NexusException;
import org.nexusformat.NexusFile;

public class NexusFileHandler extends NexusFile {
    public NexusFileHandler(String filename, int access) throws NexusException  {
        super(filename, access);
    }

    /**
     * getSubItemName Returns the name of the item having the given class name
     * 
     * @param iIndex index of the sub-item to open (class name dependent)
     * @param sNodeClass class name of the sub-item to open
     * @return item's name
     * @throws NexusException
     * @note the first item has index number 0
     */
    public String getSubItemName(int iIndex, String sNodeClass) throws NexusException {
    	if(handle < 0) throw new NexusException("NAPI-ERROR: File not open");
    	String names[] = new String[2];
    	int i = 0;
    	while(nextentry(handle,names) != -1)
    	{
    	    if( names[1].equals(sNodeClass) ) {
	    		if( i == iIndex ) {
	    		    return names[0];
	    		}
	    		i++;
    	    }
    	}
    	return null;
    }
    
    /**
     * Will make the loading of the NeXusAPI without crashing the whole system.
     * Send an error if the NeXus API can't be found physiqcally
     * Send an exception if the API has'nt been installed  
     */
    static public void loadAPI() {
    	// do nothing
    }
}
