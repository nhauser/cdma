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
                try {
                    BufferedInputStream reader = new BufferedInputStream(new FileInputStream(path));
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
                            }
                            else {
                                if ((character == '}') && newLine) {
                                    count++;
                                    character = reader.read();
                                    break;
                                }
                                newLine = false;
                            }
                        }
                        catch (IOException e) {
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
                }
                catch (FileNotFoundException e) {
                    // Just ignore this case;
                }
            }
        }
        System.out.println("End of EDF analyze");
    }

}
