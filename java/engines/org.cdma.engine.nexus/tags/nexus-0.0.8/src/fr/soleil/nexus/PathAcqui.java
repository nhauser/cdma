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

public class PathAcqui extends PathGroup {
    /**
     * PathAcqui
     * Create an object PathAcqui
     * 
     * @param sAcquiName name of the acquisition
     * @note group's class can be specified by adding "<" and ">" to a class name: i.e. "my_entry<NXentry>"
     * @note BE AWARE that it's better not to force the group's class. By default they are mapped by the API to apply Nexus format DTD
     */
    public PathAcqui(String sAcquiName) { super(new String[] {sAcquiName}); }

    protected PathAcqui(String[] sGroups)           { super(sGroups); }
}
