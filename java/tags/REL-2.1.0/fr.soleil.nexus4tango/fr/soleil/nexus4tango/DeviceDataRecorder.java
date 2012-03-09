package fr.soleil.nexus4tango;

import fr.esrf.Tango.DevFailed;
import fr.esrf.Tango.DevState;
import fr.esrf.TangoApi.DeviceAttribute;
import fr.esrf.TangoApi.DeviceData;
import fr.esrf.TangoApi.DeviceProxy;

/**
 * This class allows to group device datarecorder's actions 
 * for saving devices' data in a nexus file.
 * @author MARECHAL
 *
 */
public class DeviceDataRecorder 
{
	//deviceproxy which executes commands
	private DeviceProxy deviceDataRecorder = null;
	
	//datarecorder commands
	public final String START_RECORDING = "StartRecording";
	public final String END_RECORDING = "EndRecording";
	public final String WRITE_USER_DATA = "WriteUserData";
	public final String WRITE_EXPERIMENTAL_DATA = "WriteExperimentalData";
	public final String WRITE_TANGO_DEVICE_DATA = "WriteTangoDeviceData";
	public final String WRITE_TECHNICAL_DATA = "WriteTechnicalData";
	public final String IS_DEVICE_RECORDABLE = "IsTangoDeviceRecordable";
	public final String FILE_ROOT_NAME = "fileRootName";
	
	public final String ASYNCHRONOUS_MODE = "asynchronousWrite";
	
	private boolean isMoving = false; //return device datarecorder state (moving or not)

	/**
	 * Default constructor
	 *
	 */
	public DeviceDataRecorder()
	{
		// do nothing
	}
	
	//------------------------------------------------------------
	// Getter and setter for device
	//-----------------------------------------------------------
	
	public DeviceProxy getDevice()
	{
		return deviceDataRecorder;
	}

	public void setDevice(String device) 
	{
		try 
		{
			deviceDataRecorder = new DeviceProxy(device);
		} 
		catch (DevFailed e) 
		{
			e.printStackTrace();
		}
	}
	
	/**
	 * Save data from device (device_name) in a NeXus file
	 * @param String device_name
	 * @param boolean save_whole_context which notices if technical data is saved or not
	 * @throws Exception
	 */
	public void saveData(String device_name, boolean save_whole_context) throws Exception
	{
		//create a string array and put in device name
		String[] devices_list = {device_name};
		saveData(devices_list,save_whole_context);
	}
	
	/**
	 * Save data from each device in the list (devices_list) in a NeXus file
	 * @param String[] devices_list
	 * @param boolean save_whole_context which notices if technical data is saved or not
	 * @throws Exception
	 */
	public void saveData(String[] devices_list, boolean save_whole_context) throws Exception
	{
		if(devices_list != null && deviceDataRecorder != null)
		{
			try
			{
				//set datarecorder in asynchrone mode
				DeviceAttribute attr = new DeviceAttribute(ASYNCHRONOUS_MODE, true);
				deviceDataRecorder.write_attribute(attr);
				
				executeCommand(START_RECORDING, null);
				executeCommand(WRITE_USER_DATA, null);
				
				if(save_whole_context)
					executeCommand(WRITE_TECHNICAL_DATA, null);
				
				//iteration on the devices list, for each device do : WriteTangoDeviceData and WriteExperimentalData
				String device_name = null;
				for (int i=0 ; i<devices_list.length ; i++)
				{
					device_name = devices_list[i];
					writeDeviceData(device_name);
				}
			}
			catch(Exception e)
			{
				throw e;
			}
			finally
			{
				executeCommand(END_RECORDING, null);
				
				//set datarecorder in synchrone mode
				DeviceAttribute attr = new DeviceAttribute(ASYNCHRONOUS_MODE, false);
				deviceDataRecorder.write_attribute(attr);
			}
		}
	}
	
	/**
	 * @return int : notices if the device is recordable or not by device DataRecorder
	 */
	public int isDeviceRecordable(String device_name) throws Exception 
	{
		if(device_name != null)
		{
			DeviceData argin = new DeviceData();
       	  	argin.insert(device_name);
       	  	DeviceData argout = deviceDataRecorder.command_inout(IS_DEVICE_RECORDABLE, argin);
       	  	if(argout != null)
       	  		return argout.extractShort();
		}
		return -1;
	}
	
	/**
	 * Execute method WriteTangoDeviceData and WriteExperimentalData for device
	 * specified in parameter
	 * @param device_name
	 * @throws Exception
	 */
	private void writeDeviceData(String device_name) throws Exception
	{
		DeviceData in = null;
		try 
		{
			in = new DeviceData();
		} 
		catch (DevFailed e) 
		{
			throw new Exception("Can't create DeviceData");
		}
		in.insert(device_name);
		try 
		{
			executeCommand(WRITE_TANGO_DEVICE_DATA, in);
		} 
		catch (Exception e) 
		{
			throw new Exception("Can't execute command: WriteTangoDeviceData for device: " + device_name, e);
		}
		try 
		{
			executeCommand(WRITE_EXPERIMENTAL_DATA, in);
		} 
		catch (Exception e) 
		{
			throw new Exception("Can't execute command: WriteExperimentalData for device: " + device_name, e);
		}
	}
	
	
	
	/**
	 * Execute the specified command with the specified argin (or null)
	 * @param String command
	 * @param String argin
	 * @throws Exception 
	 */
	private void executeCommand(String command, DeviceData argin) throws Exception
	{
		if(deviceDataRecorder != null)
		{
			//while ds is running
			isMoving = deviceDataRecorder.state().equals(DevState.MOVING);
			while(isMoving)
			{
				 isMoving = deviceDataRecorder.state().equals(DevState.MOVING);
			}
			
			try
			{
				if(argin == null)
				{
					deviceDataRecorder.command_inout(command);
				}
				else
				{
					deviceDataRecorder.command_inout(command, argin);
				}
			}
			catch (Exception e) 
			{
				throw new Exception("Can't execute command: " + command, e);
			}
		}
	}

	//------------------------------------------------------------
	// Getter and setter for nexus file root name (String)
	//-----------------------------------------------------------
	
	public String getFileRootName() throws DevFailed 
	{
		//return attribute fileRootName value
		DeviceAttribute devAttr= deviceDataRecorder.read_attribute(FILE_ROOT_NAME);
		return devAttr.extractString();
	}

	
	public void setFileRootName(String fileRootName) throws DevFailed 
	{
		//write attribute fileRootName
		DeviceAttribute attr = new DeviceAttribute(FILE_ROOT_NAME, fileRootName);
		deviceDataRecorder.write_attribute(attr);
	}
	
	
}

