//*****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez Clément
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
//*****************************************************************************

#ifndef __CDMA_DICTIONARYDETECTOR_H__
#define __CDMA_DICTIONARYDETECTOR_H__

#include <yat/utils/String.h>
#include "nxfile.h"
#include <cdma/Common.h>
#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>
#include <internal/common.h>

/**
* DictionaryDetector aims to detect the correct dictionary mapping file name
* according to file structure. Each beamline defines its own structure model
* and there can be several ones for each beamline.
*
* @author nxi
*
*/
/*
class DictionaryMethod
{
  operator()
};
#define DECLARE_DICT_METHOD(name) 
class name:: public DictionaryMethod
*/

namespace cdma
{
  class DictionaryDetector
  {
    public:
      DictionaryDetector();
      DictionaryDetector(const NexusFilePtr& handle, const yat::String& uri);
      ~DictionaryDetector();
      yat::String getDictionaryName() throw ( cdma::Exception );

    protected:
      void detectBeamline();
      void detectDataModel();
      const yat::String& getBeamline() { return m_beamline; };
      void setBeamline(const yat::String& beam){ m_beamline = beam; };
      const yat::String& getModel() { return m_model; };
      void setModel(const yat::String& model){ m_model = model; };

    private:
      yat::String  m_beamline;     ///< beamline model
      yat::String  m_model;        ///< beamline's structure data model
      NexusFilePtr m_ptrNxFile;    ///< handle on file
      yat::String  m_uri;          ///< file location

    private:
      bool isFlyScan();
      bool isScanServer();
  };

} // namespace

#endif // __CDMA_DICTIONARYDETECTOR_H__
