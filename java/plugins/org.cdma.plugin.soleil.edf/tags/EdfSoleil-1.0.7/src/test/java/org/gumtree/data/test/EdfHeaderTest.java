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
package org.gumtree.data.test;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.junit.Ignore;

@Ignore
public class EdfHeaderTest {

    /**
     * @param args
     */
    public static void main(String[] args) {
        for (String path : args) {
            int pointIndex = path.lastIndexOf('.');
            if ((pointIndex > -1) && "edf".equalsIgnoreCase(path.substring(pointIndex + 1))) {
                BufferedInputStream reader = null;
                try {
                    reader = new BufferedInputStream(new FileInputStream(path));
                    int character = -1;
                    int count = 0;
                    StringBuffer headerBuffer = new StringBuffer();
                    boolean newLine = false;
                    while (true) {
                        try {
                            count++;
                            character = reader.read();
                            if (character == -1) {
                                break;
                            }
                            headerBuffer.append((char) character);
                            if (character == '\n') {
                                newLine = true;
                            } else {
                                if ((character == '}') && newLine) {
                                    count++;
                                    character = reader.read();
                                    break;
                                }
                                newLine = false;
                            }
                        } catch (IOException e) {
                            character = -1;
                            break;
                        }
                    }
                    if (character != -1) {
                        System.out.print("=========== ");
                        System.out.print(path);
                        System.out.println(" ===========");
                        System.out.println("Header length: " + count);
                        System.out.println("Header value:");
                        System.out.println(headerBuffer.toString());
                        char toCheck = headerBuffer.charAt(0);
                        while (Character.isWhitespace(toCheck) || (toCheck == '{')) {
                            headerBuffer.delete(0, 1);
                            toCheck = headerBuffer.charAt(0);
                        }
                        toCheck = headerBuffer.charAt(headerBuffer.length() - 1);
                        while (Character.isWhitespace(toCheck) || (toCheck == '}')) {
                            headerBuffer.delete(headerBuffer.length() - 1, headerBuffer.length());
                            toCheck = headerBuffer.charAt(headerBuffer.length() - 1);
                        }
                        System.out.println("-------------------");
                        System.out.println("Transformed header value:");
                        System.out.println(headerBuffer.toString());
                    }
                } catch (FileNotFoundException e) {
                    // Just ignore this case;
                } finally {
                    if (reader != null) {
                        try {
                            reader.close();
                        } catch (Exception e) {
                            // Ignore this one
                        }
                    }
                }
            }
        }
        System.out.println("End of EDF analyze");
    }

}
