package org.cdma.plugin.mambo.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.cdma.engine.sql.utils.SamplingType.SamplingPeriod;

import fr.soleil.lib.project.xmlhelpers.XMLLine;
import fr.soleil.lib.project.xmlhelpers.XMLUtils;

public class MamboVCGenerator {

    public static final String CRLF = System.getProperty("line.separator");

    public static final String VC_XML_TAG = "viewConfiguration";
    public static final String CREATION_DATE_PROPERTY_XML_TAG = "creationDate";
    public static final String DATE_RANGE_PROPERTY_XML_TAG = "dateRange";
    public static final String DYNAMIC_DATE_RANGE_PROPERTY_XML_TAG = "dynamicDateRange";
    public static final String END_DATE_PROPERTY_XML_TAG = "endDate";
    public static final String GENERIC_PLOT_PARAMETERS_XML_TAG = "genericPlotParameters";
    public static final String HISTORIC_PROPERTY_XML_TAG = "historic";
    public static final String IS_MODIFIED_PROPERTY_XML_TAG = "isModified";
    public static final String LAST_UPDATE_DATE_PROPERTY_XML_TAG = "lastUpdateDate";
    public static final String LONG_TERM_PROPERTY_XML_TAG = "longTerm";
    public static final String NAME_PROPERTY_XML_TAG = "name";
    public static final String PATH_PROPERTY_XML_TAG = "path";
    public static final String SAMPLING_FACTOR_PROPERTY_XML_TAG = "samplingFactor";
    public static final String SAMPLING_TYPE_PROPERTY_XML_TAG = "samplingType";
    public static final String START_DATE_PROPERTY_XML_TAG = "startDate";

    public static final String ATT_XML_TAG = "attribute";
    public static final String COMPLETE_NAME_PROPERTY_XML_TAG = "completeName";
    public static final String FACTOR_PROPERTY_XML_TAG = "factor";

    private File vcFile = null;
    private final String fileName;
    private final List<String> attributeList = new ArrayList<String>();
    private Date startDate = null;
    private Date endDate = null;
    private long creationTMS = 0;
    private long lastUpdateTMS = 0;
    private boolean longTerm = false;
    private boolean historic = true;
    private SamplingPeriod samplingPeriod = SamplingPeriod.NONE;
    private int samplingFactor = 1;
    private String userPath = null;

    public MamboVCGenerator(final String fileName) {
        this.fileName = fileName;
        creationTMS = System.currentTimeMillis();
    }

    public String getUserPath() {
        return userPath;
    }

    public void setUserPath(String userPath) {
        this.userPath = userPath;
    }

    public SamplingPeriod getSamplingPeriod() {
        return samplingPeriod;
    }

    public void setSamplingPeriod(SamplingPeriod samplingPeriod) {
        this.samplingPeriod = samplingPeriod;
        update();
    }

    public int getSamplingFactor() {
        return samplingFactor;
    }

    public void setSamplingFactor(int samplingFactor) {
        this.samplingFactor = samplingFactor;
        update();
    }


    public List<String> getAttributeList() {
        return attributeList;
    }

    public void setAttributeList(List<String> attributeList) {
        this.attributeList.clear();
        if (attributeList != null) {
            this.attributeList.addAll(attributeList);
        }
        update();
    }

    public Date getStartDate() {
        return startDate;
    }

    public void setStartDate(Date startDate) {
        this.startDate = startDate;
        update();
    }

    public Date getEndDate() {
        return endDate;
    }

    public void setEndDate(Date endDate) {
        this.endDate = endDate;
        update();
    }

    public boolean isLongTerm() {
        return longTerm;
    }

    public void setLongTerm(boolean longTerm) {
        this.longTerm = longTerm;
        update();
    }

    public boolean isHistoric() {
        return historic;
    }

    public void setHistoric(boolean historic) {
        this.historic = historic;
        update();
    }

    public File getVcFile() throws IOException {
        generateVCFile();
        return vcFile;
    }

    private void update() {
        lastUpdateTMS = System.currentTimeMillis();
    }

    private File generateVCFile() throws IOException {
        if ((userPath != null) && !userPath.trim().isEmpty()) {
            try {
                vcFile = new File(userPath + ".vc");
                if ((vcFile != null) && !vcFile.exists()) {
                    vcFile.createNewFile();
                }
            } catch (IOException e) {
                System.err.println("Cannot create file " + userPath + ".vc" + " " + e.getMessage());
                vcFile = null;
            }
        }

        if (vcFile == null) {
            vcFile = File.createTempFile(fileName, ".vc");
        }
        if (vcFile != null) {
            PrintWriter writer = new PrintWriter(new FileWriter(vcFile.getAbsolutePath(), false));
            writer.println(XMLUtils.XML_HEADER);
            writer.println(toString());
            writer.flush();
            writer.close();
        }
        if (((userPath == null) || (userPath.trim().isEmpty())) && (vcFile != null) && vcFile.exists()) {
            // Delete file if it is a temporary file
            vcFile.deleteOnExit();
        }
        return vcFile;
    }

    private boolean isModified() {
        return creationTMS < lastUpdateTMS;
    }

    @Override
    public String toString() {
        final StringBuilder ret = new StringBuilder();
        final XMLLine openingLine = new XMLLine(VC_XML_TAG, XMLLine.OPENING_TAG_CATEGORY);
        final boolean dynamicDateRange = false;
        final String dateRange = "Last hour";
        final String name = fileName;
        final String path = vcFile.getAbsolutePath();
        final Timestamp creationDate = new Timestamp(creationTMS);
        final Timestamp lastUpdateDate = new Timestamp(lastUpdateTMS);
        Timestamp startTMS = creationDate;
        if(startDate != null) {
            startTMS = new Timestamp(startDate.getTime());
        }
        Timestamp endTMS = lastUpdateDate;
        if(endDate != null) {
            endDate = new Timestamp(endDate.getTime());
        }

        openingLine.setAttribute(IS_MODIFIED_PROPERTY_XML_TAG, isModified() + "");
        if (creationDate != null) {
            openingLine.setAttribute(CREATION_DATE_PROPERTY_XML_TAG, creationDate.toString());
        }
        if (lastUpdateDate != null) {
            openingLine
            .setAttribute(LAST_UPDATE_DATE_PROPERTY_XML_TAG, lastUpdateDate.toString());
        }
        if (startDate != null) {
            openingLine.setAttribute(START_DATE_PROPERTY_XML_TAG, startTMS.toString());
        }
        if (endDate != null) {
            openingLine.setAttribute(END_DATE_PROPERTY_XML_TAG, endTMS.toString());
        }
        openingLine.setAttribute(HISTORIC_PROPERTY_XML_TAG, String.valueOf(historic));
        openingLine.setAttribute(LONG_TERM_PROPERTY_XML_TAG, String.valueOf(longTerm));

        openingLine.setAttribute(DYNAMIC_DATE_RANGE_PROPERTY_XML_TAG,
                String.valueOf(dynamicDateRange));
        openingLine.setAttribute(DATE_RANGE_PROPERTY_XML_TAG, dateRange);

        if (name != null) {
            openingLine.setAttribute(NAME_PROPERTY_XML_TAG, name);
        }
        if (path != null) {
            openingLine.setAttribute(PATH_PROPERTY_XML_TAG, path);
        }
        // System.out.println (
        // "---------------------------------------/samplingType/"+samplingType+"/"
        // );
        if (samplingPeriod != null) {
            openingLine.setAttribute(SAMPLING_TYPE_PROPERTY_XML_TAG, String.valueOf(samplingPeriod.value()));
            openingLine.setAttribute(SAMPLING_FACTOR_PROPERTY_XML_TAG, String.valueOf(samplingFactor));
        }

        final XMLLine closingLine = new XMLLine(VC_XML_TAG, XMLLine.CLOSING_TAG_CATEGORY);

        ret.append(openingLine.toString());
        ret.append(CRLF);

        // attribute list
        if (!attributeList.isEmpty()) {
            for (String attributeName : attributeList) {
                XMLLine att_openingLine = new XMLLine(ATT_XML_TAG, XMLLine.OPENING_TAG_CATEGORY);
                att_openingLine.setAttribute(COMPLETE_NAME_PROPERTY_XML_TAG, attributeName);
                att_openingLine.setAttribute(FACTOR_PROPERTY_XML_TAG, String.valueOf(samplingFactor) + "");
                XMLLine att_closingLine = new XMLLine(ATT_XML_TAG, XMLLine.CLOSING_TAG_CATEGORY);
                ret.append(att_openingLine.toString());
                ret.append(CRLF);
                ret.append(att_closingLine.toString());
            }
        }

        ret.append(closingLine.toString());
        ret.append(CRLF);
        return ret.toString();
    }

    public static void main(String[] args) {
        MamboVCGenerator mamboVCGenerator = new MamboVCGenerator("MonFichierVc");
        List<String> attributeList = new ArrayList<String>();
        attributeList.add("tango/tangotest/1/double_scalar");
        attributeList.add("tango/tangotest/1/double_spectrum_ro");
        attributeList.add("tango/tangotest/1/short_scalar");
        mamboVCGenerator.setAttributeList(attributeList);
        mamboVCGenerator.setStartDate(new Date());
        mamboVCGenerator.setEndDate(new Date());

        try {
            File vcFiletmp = mamboVCGenerator.getVcFile();
            System.out.println("vcFiletmp generated =" + vcFiletmp.getAbsolutePath());
            Thread.sleep(120000);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

}
