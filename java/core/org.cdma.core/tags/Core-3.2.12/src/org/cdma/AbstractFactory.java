package org.cdma;

import java.io.IOException;
import java.net.URI;

import org.cdma.dictionary.LogicalGroup;
import org.cdma.dictionary.Path;
import org.cdma.exception.CDMAException;
import org.cdma.exception.FileAccessException;
import org.cdma.exception.InvalidArrayTypeException;
import org.cdma.exception.NotImplementedException;
import org.cdma.interfaces.IArray;
import org.cdma.interfaces.IAttribute;
import org.cdma.interfaces.IDataItem;
import org.cdma.interfaces.IDataset;
import org.cdma.interfaces.IDatasource;
import org.cdma.interfaces.IDictionary;
import org.cdma.interfaces.IGroup;
import org.cdma.interfaces.IKey;

public class AbstractFactory implements IFactory {

    @Override
    public IDataset openDataset(URI uri) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IDictionary openDictionary(URI uri) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IDictionary openDictionary(String filepath) throws FileAccessException {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArray(Class<?> clazz, int[] shape, Object storage) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArray(Object javaArray) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createStringArray(String string) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createDoubleArray(double[] javaArray) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createDoubleArray(double[] javaArray, int[] shape) {
        throw new NotImplementedException();
    }

    @Override
    public IArray createArrayNoCopy(Object javaArray) {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem createDataItem(IGroup parent, String shortName, IArray array) throws InvalidArrayTypeException {
        throw new NotImplementedException();
    }

    @Override
    public IDataItem createDataItem(final IGroup parent, final String shortName, final Object value)
            throws CDMAException {
        throw new NotImplementedException();
    }

    @Override
    public IGroup createGroup(IGroup parent, String shortName) {
        throw new NotImplementedException();
    }

    @Override
    public IGroup createGroup(String shortName) throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public LogicalGroup createLogicalGroup(IDataset dataset, IKey key) {
        throw new NotImplementedException();
    }

    @Override
    public IAttribute createAttribute(String name, Object value) {
        throw new NotImplementedException();
    }

    @Override
    public IDataset createDatasetInstance(URI uri) throws CDMAException {
        throw new NotImplementedException();
    }

    @Override
    public IDataset createDatasetInstance(URI uri, boolean withWriteAccess) throws CDMAException {
        throw new NotImplementedException();
    }

    @Override
    public IDataset createEmptyDatasetInstance() throws IOException {
        throw new NotImplementedException();
    }

    @Override
    public IKey createKey(String name) {
        throw new NotImplementedException();
    }

    @Override
    public Path createPath(String path) {
        throw new NotImplementedException();
    }

    @Override
    public String getName() {
        throw new NotImplementedException();
    }

    @Override
    public String getPluginLabel() {
        throw new NotImplementedException();
    }

    @Override
    public String getPluginDescription() {
        throw new NotImplementedException();
    }

    @Override
    public IDatasource getPluginURIDetector() {
        throw new NotImplementedException();
    }

    @Override
    public IDictionary createDictionary() {
        throw new NotImplementedException();
    }

    @Override
    public String getPluginVersion() {
        throw new NotImplementedException();
    }

    @Override
    public String getCDMAVersion() {
        throw new NotImplementedException();
    }

    @Override
    public void processPostRecording() {
        throw new NotImplementedException();
    }

    @Override
    public boolean isLogicalModeAvailable() {
        throw new NotImplementedException();
    }
}
