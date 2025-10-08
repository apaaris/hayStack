// ======================================================================== //
// Copyright 2022-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SiloContent.h"
#include <fstream>
#include <umesh/UMesh.h>
#include <umesh/extractIsoSurface.h>
#include <miniScene/Scene.h>

#ifdef HS_HAVE_SILO
#include <silo.h>
//#include <pmpio.h>
#endif

namespace umesh {
  UMesh::SP tetrahedralize(UMesh::SP in,
                           int ownedTets,
                           int ownedPyrs,
                           int ownedWedges,
                           int ownedHexes);
}  
namespace hs {
  
  SiloContent::SiloContent(const std::string &fileName,
                                     int thisPartID,
                                     const box3i &cellRange,
                                     vec3i fullVolumeDims,
                                     const std::string &texelFormat,
                                     int numChannels,
                                     float isoValue,
                                     const std::string &variableName)
    : fileName(fileName),
      thisPartID(thisPartID),
      cellRange(cellRange),
      fullVolumeDims(fullVolumeDims),
      texelFormat(texelFormat),
      numChannels(numChannels),
      isoValue(isoValue),
      variableName(variableName)
  {}

  void siloSplitKDTree(std::vector<box3i> &regions,
                   box3i cellRange,
                   int numParts)
  {
    if (numParts == 1) {
      regions.push_back(cellRange);
      return;
    }
    
    vec3i size = cellRange.size();
    int dim = arg_max(size);
    if (size[dim] < 2) {
      regions.push_back(cellRange);
      return;
    }

    int nRight = numParts/2;
    int nLeft  = numParts - nRight;
    
    box3i lBox = cellRange, rBox = cellRange;
    lBox.upper[dim]
      = rBox.lower[dim]
      = cellRange.lower[dim] + (cellRange.size()[dim]*nLeft)/numParts;
    siloSplitKDTree(regions,lBox,nLeft);
    siloSplitKDTree(regions,rBox,nRight);
  }

  bool siloContains(const std::string &hayStack,
                const std::string &needle)
    { return hayStack.find(needle) != hayStack.npos; }
  
  bool siloContains(const ResourceSpecifier &hayStack,
                const std::string &needle)
    { return siloContains(hayStack.where,needle); }
  
  void SiloContent::create(DataLoader *loader,
                                const ResourceSpecifier &dataURL)
  {
    std::string type = dataURL.get("type",dataURL.get("format",""));
    std::string texelFormat;
    if (type == "") {
      std::cout << "#hs.silo: no type specified, trying to guess form '" << dataURL.where << "'..." << std::endl;
      if (siloContains(dataURL,"uint8"))
        texelFormat = "uint8_t";
      else if (siloContains(dataURL,"uint16"))
        texelFormat = "uint16_t";
      else if (siloContains(dataURL,"float64"))
        texelFormat = "double";
      else if (siloContains(dataURL,"float"))
        texelFormat = "float";
      else
        throw std::runtime_error("could not get silo volume file format");
    } else {
      if (type == "uint8" || type == "byte")
        texelFormat = "uint8_t"; //scalarType = StructuredVolume::UINT8;
      else if (type == "float" || type == "f")
        texelFormat = "float"; //scalarType = StructuredVolume::FLOAT;
      else if (type == "uint16")
        texelFormat = "uint16_t"; //scalarType = StructuredVolume::UINT16;
      else
        throw std::runtime_error("SiloContent: invalid type '"+type+"'");
    }
    
    int numChannels = dataURL.get_int("channels",1);
    
    std::string dimsString = dataURL.get("dims","");
    // if (dimsString.empty())
    //   throw std::runtime_error("RAWVolumeContent: 'dims' not specified");
    
    vec3i dims;
    if (dimsString == "") {
#ifdef HS_HAVE_SILO
      // Try to read dimensions directly from Silo file
      std::cout << "#hs.silo: no dims specified, reading from Silo file" << std::endl;
      DBfile *dbfile = DBOpen(dataURL.where.c_str(), DB_UNKNOWN, DB_READ);
      if (!dbfile)
        throw std::runtime_error("hs::SiloContent: could not open Silo file to read dimensions");
      
      DBtoc *toc = DBGetToc(dbfile);
      if (!toc || toc->nqvar == 0) {
        DBClose(dbfile);
        throw std::runtime_error("hs::SiloContent: no quad variables found in file");
      }
      
      DBquadvar *qvar = DBGetQuadvar(dbfile, toc->qvar_names[0]);
      if (!qvar) {
        DBClose(dbfile);
        throw std::runtime_error("hs::SiloContent: could not read variable to get dimensions");
      }
      
      dims.x = qvar->dims[0];
      dims.y = qvar->dims[1];
      dims.z = qvar->ndims > 2 ? qvar->dims[2] : 1;
      
      std::cout << "#hs.silo: read dimensions from file: " << dims << std::endl;
      
      DBFreeQuadvar(qvar);
      DBClose(dbfile);
#else
      // Fall back to parsing from filename
      std::cout << "#hs.silo: no dims specified and Silo not available, trying to guess from filename" << std::endl;
      const char *fileName = dataURL.where.c_str();
      const char *nextScanPos = fileName;
      PRINT(fileName);
      while (true) {
        const char *nextUS = strstr(nextScanPos,"_");
        PRINT(nextUS);
        if (!nextUS)
          throw std::runtime_error
            ("could not find '_<width>x<height>x<depth>_' in Silo file name");
        int n = sscanf(nextUS,"_%ix%ix%i_",&dims.x,&dims.y,&dims.z);
        if (n != 3) { nextScanPos = nextUS+1; continue; }

        std::cout << "guessing dims from " << (nextUS+1) << std::endl;;
        break;
      }
#endif
    } else {
      int n = sscanf(dimsString.c_str(),"%i,%i,%i",&dims.x,&dims.y,&dims.z);
      if (n != 3)
        throw std::runtime_error
          ("SiloContent:: could not parse dims from '"+dimsString+"'");
    }
    
    box3i initRegion = { vec3i(0), dims-1 };
    std::string extractString = dataURL.get("extract");
    if (!extractString.empty()) {
      vec3i lower, size;
      int n = sscanf(extractString.c_str(),"%i,%i,%i,%i,%i,%i",
                     &lower.x,&lower.y,&lower.z,
                     &size.x,&size.y,&size.z);
      if (n != 6)
        throw std::runtime_error("SiloContent:: could not parse 'extract' value from '"
                                 +extractString
                                 +"' (should be 'f,f,f,f,f,f' format)");
      initRegion.lower = lower;
      initRegion.upper = lower+size-1;
    }

    std::vector<box3i> regions;
    siloSplitKDTree(regions,initRegion,dataURL.numParts);
    // splitKDTree(regions,box3i(vec3i(0),dims-1),dataURL.numParts);
    if (regions.size() < dataURL.numParts)
      throw std::runtime_error("input data too small to split into indicated number of parts");

    if (loader->myRank() == 0) {
      std::cout << "Silo Volume: input data file of " << dims << " voxels will be read in the following bricks:" << std::endl;
      for (int i=0;i<regions.size();i++)
        std::cout << " #" << i << " : " << regions[i] << std::endl;
    }
    float isoValue = NAN;
    std::string isoString = dataURL.get("iso",dataURL.get("isoValue"));
    if (!isoString.empty())
      isoValue = std::stof(isoString);
    
    std::string variableName = dataURL.get("var", dataURL.get("variable", ""));
    
    for (int i=0;i<dataURL.numParts;i++) {
      loader->addContent(new SiloContent(dataURL.where,i,
                                              regions[i],
                                              dims,texelFormat,//scalarType,
                                              numChannels,
                                              isoValue,
                                              variableName));
    }
  }
  
size_t SiloContent::projectedSize()
  {
    vec3i numVoxels = cellRange.size()+1;
    return numVoxels.x*size_t(numVoxels.y)*numVoxels.z*numChannels*sizeOf(texelFormat);
  }
  
  void SiloContent::executeLoad(DataRank &dataGroup, bool verbose)
  {
#ifndef HS_HAVE_SILO
    throw std::runtime_error("hs::SiloContent: Silo library support not enabled at compile time");
#else
    vec3i numVoxels = (cellRange.size()+1);
    size_t numScalars = 
      size_t(numVoxels.x)*size_t(numVoxels.y)*size_t(numVoxels.z);
    std::vector<uint8_t> rawData(numScalars*sizeOf(texelFormat));
    
    // Open Silo file
    DBfile *dbfile = DBOpen(fileName.c_str(), DB_UNKNOWN, DB_READ);
    if (!dbfile)
      throw std::runtime_error("hs::SiloContent: could not open Silo file '"+fileName+"'");
    
    // Get the list of variables in the file
    DBtoc *toc = DBGetToc(dbfile);
    if (!toc || toc->nqvar == 0) {
      DBClose(dbfile);
      throw std::runtime_error("hs::SiloContent: no quad variables found in file '"+fileName+"'");
    }
    
    // Use the specified variable name, or the first quad variable by default
    const char *varname = nullptr;
    if (!variableName.empty()) {
      varname = variableName.c_str();
      if (verbose) {
        std::cout << "Loading Silo variable (specified): " << varname << std::endl;
      }
    } else {
      varname = toc->qvar_names[0];
      if (verbose) {
        std::cout << "Loading Silo variable (first in file): " << varname << std::endl;
      }
    }
    
    // Read the quad variable
    DBquadvar *qvar = DBGetQuadvar(dbfile, varname);
    if (!qvar) {
      DBClose(dbfile);
      throw std::runtime_error("hs::SiloContent: could not read variable '"+std::string(varname)+"'");
    }
    
    // Get mesh dimensions from the quad variable
    vec3i fileDims(qvar->dims[0], qvar->dims[1], qvar->ndims > 2 ? qvar->dims[2] : 1);
    
    std::cout << "  Silo file dimensions: " << fileDims << std::endl;
    std::cout << "  Expected dimensions: " << fullVolumeDims << std::endl;
    std::cout << "  Cell range: " << cellRange << std::endl;
    std::cout << "  Num voxels to extract: " << numVoxels << std::endl;
    std::cout << "  Data type: " << qvar->datatype << " (DB_FLOAT=" << DB_FLOAT     
            << ", DB_DOUBLE=" << DB_DOUBLE << ", DB_INT=" << DB_INT << ")" << std::endl;
    std::cout << "  Texel format: " << texelFormat << std::endl;
    std::cout << "  Buffer size: " << rawData.size() << " bytes" << std::endl;
    
    if (fileDims != fullVolumeDims) {
      DBFreeQuadvar(qvar);
      DBClose(dbfile);
      throw std::runtime_error("hs::SiloContent: dimension mismatch - expected " 
                               + std::to_string(fullVolumeDims.x) + "x" 
                               + std::to_string(fullVolumeDims.y) + "x" 
                               + std::to_string(fullVolumeDims.z) 
                               + " but got " + std::to_string(fileDims.x) + "x" 
                               + std::to_string(fileDims.y) + "x" 
                               + std::to_string(fileDims.z));
    }
    
    // Extract data from the specified cell range
    void *srcData = qvar->vals[0]; // First component
    int datatype = qvar->datatype;
    
    // Convert and copy data to our buffer based on data type
    char *dataPtr = (char *)rawData.data();
    size_t texelSize = sizeOf(texelFormat);
    
    for (int iz=cellRange.lower.z;iz<=cellRange.upper.z;iz++) {
      for (int iy=cellRange.lower.y;iy<=cellRange.upper.y;iy++) {
        for (int ix=cellRange.lower.x;ix<=cellRange.upper.x;ix++) {
          size_t srcIdx = ix + iy*size_t(fullVolumeDims.x) 
                            + iz*size_t(fullVolumeDims.x)*size_t(fullVolumeDims.y);
          
          // Convert based on source and destination types
          if (texelFormat == "float") {
            float value;
            if (datatype == DB_FLOAT) {
              value = ((float*)srcData)[srcIdx];
            } else if (datatype == DB_DOUBLE) {
              value = (float)((double*)srcData)[srcIdx];
            } else if (datatype == DB_INT) {
              value = (float)((int*)srcData)[srcIdx];
            } else {
              DBFreeQuadvar(qvar);
              DBClose(dbfile);
              throw std::runtime_error("hs::SiloContent: unsupported data type in Silo file");
            }
            ((float*)dataPtr)[0] = value;
            dataPtr += sizeof(float);
          } else if (texelFormat == "uint8_t") {
            uint8_t value;
            if (datatype == DB_FLOAT) {
              value = (uint8_t)(((float*)srcData)[srcIdx] * 255.0f);
            } else if (datatype == DB_DOUBLE) {
              value = (uint8_t)(((double*)srcData)[srcIdx] * 255.0);
            } else if (datatype == DB_INT) {
              value = (uint8_t)((int*)srcData)[srcIdx];
            } else {
              DBFreeQuadvar(qvar);
              DBClose(dbfile);
              throw std::runtime_error("hs::SiloContent: unsupported data type in Silo file");
            }
            ((uint8_t*)dataPtr)[0] = value;
            dataPtr += sizeof(uint8_t);
          } else if (texelFormat == "uint16_t") {
            uint16_t value;
            if (datatype == DB_FLOAT) {
              value = (uint16_t)(((float*)srcData)[srcIdx] * 65535.0f);
            } else if (datatype == DB_DOUBLE) {
              value = (uint16_t)(((double*)srcData)[srcIdx] * 65535.0);
            } else if (datatype == DB_INT) {
              value = (uint16_t)((int*)srcData)[srcIdx];
            } else {
              DBFreeQuadvar(qvar);
              DBClose(dbfile);
              throw std::runtime_error("hs::SiloContent: unsupported data type in Silo file");
            }
            ((uint16_t*)dataPtr)[0] = value;
            dataPtr += sizeof(uint16_t);
          }
        }
      }
    }
    
    // Clean up Silo resources
    DBFreeQuadvar(qvar);
    DBClose(dbfile);
    
    // Validate the data for debugging
    float *floatData = (float*)rawData.data();
    size_t numValues = rawData.size() / sizeof(float);
    float minVal = floatData[0], maxVal = floatData[0];
    int nanCount = 0, infCount = 0;
    for (size_t i = 0; i < numValues; i++) {
    if (std::isnan(floatData[i])) nanCount++;
    if (std::isinf(floatData[i])) infCount++;
    if (std::isfinite(floatData[i])) {
        minVal = std::min(minVal, floatData[i]);
        maxVal = std::max(maxVal, floatData[i]);
    }
    }
    std::cout << "  Data validation: " << numValues << " values, range=[" 
            << minVal << ":" << maxVal << "], NaNs=" << nanCount 
            << ", Infs=" << infCount << std::endl;
    
    std::vector<uint8_t> rawDataRGB; // Empty for now - multi-channel not implemented for Silo yet
#endif
    vec3f gridOrigin(cellRange.lower);
    vec3f gridSpacing(1.f);
    
    bool doIso = !isnan(isoValue);
    if (doIso) {
      umesh::UMesh::SP
        volume = std::make_shared<umesh::UMesh>();
      volume->perVertex = std::make_shared<umesh::Attribute>();
      
      for (int iz=0;iz<numVoxels.z;iz++)
        for (int iy=0;iy<numVoxels.y;iy++)
          for (int ix=0;ix<numVoxels.x;ix++) {
            volume->vertices.push_back(umesh::vec3f(umesh::vec3i(ix,iy,iz))*(const umesh::vec3f&)gridSpacing+(const umesh::vec3f&)gridOrigin);
            size_t idx = ix+size_t(numVoxels.x)*(iy+size_t(numVoxels.y)*iz);
            float scalar;
            if (texelFormat == "float") {
            // switch(texelFormat) {
            // case BN_FLOAT:
              scalar = ((const float*)rawData.data())[idx];
            } else if (texelFormat == "uint16_t") {
            //   break;
            // case BN_UFIXED16:
              scalar = ((const uint16_t*)rawData.data())[idx]*(1.f/((1<<16)-1));
              // break;
            // case BN_UFIXED8:
            } else if (texelFormat == "uint8_t") {
              scalar = ((const uint8_t*)rawData.data())[idx]*(1.f/((1<<8)-1));
            //   break;
            // default:
            } else {
              throw std::runtime_error("not implemented...");
            };
            
              // = ( == StructuredVolume::FLOAT)
              // ? ((const float*)rawData.data())[idx]
              // : (1.f/255.f*((const uint8_t*)rawData.data())[idx]);
              
            volume->perVertex->values.push_back(scalar);
          }
      if (size_t(numVoxels.x)*size_t(numVoxels.y)*size_t(numVoxels.z) > (1ull<<30))
        throw std::runtime_error("volume dims too large to extract iso-surface via umesh");
      volume->finalize();
      for (int iz=0;iz<numVoxels.z-1;iz++)
        for (int iy=0;iy<numVoxels.y-1;iy++)
          for (int ix=0;ix<numVoxels.x-1;ix++) {
            umesh::Hex hex;
            int i000 = (ix+0)+int(numVoxels.x)*((iy+0)+int(numVoxels.y)*(iz+0));
            int i001 = (ix+1)+int(numVoxels.x)*((iy+0)+int(numVoxels.y)*(iz+0));
            int i010 = (ix+0)+int(numVoxels.x)*((iy+1)+int(numVoxels.y)*(iz+0));
            int i011 = (ix+1)+int(numVoxels.x)*((iy+1)+int(numVoxels.y)*(iz+0));
            int i100 = (ix+0)+int(numVoxels.x)*((iy+0)+int(numVoxels.y)*(iz+1));
            int i101 = (ix+1)+int(numVoxels.x)*((iy+0)+int(numVoxels.y)*(iz+1));
            int i110 = (ix+0)+int(numVoxels.x)*((iy+1)+int(numVoxels.y)*(iz+1));
            int i111 = (ix+1)+int(numVoxels.x)*((iy+1)+int(numVoxels.y)*(iz+1));
            hex.base = { i000,i001,i011,i010 };
            hex.top  = { i100,i101,i111,i110 };
            volume->hexes.push_back(hex);
          }

      // volume = umesh::tetrahedralize(volume,0,0,0,volume->hexes.size());
      // PRINT(volume->toString());
      umesh::UMesh::SP surf = umesh::extractIsoSurface(volume,isoValue);
      surf->finalize();
      PRINT(surf->toString());

      mini::Mesh::SP mesh = mini::Mesh::create();
      for (auto vtx : surf->vertices)
        mesh->vertices.push_back((const vec3f&)vtx);
      for (auto idx : surf->triangles)
        mesh->indices.push_back((const vec3i&)idx);
      mini::Object::SP obj = mini::Object::create({mesh});
      mini::Instance::SP inst = mini::Instance::create(obj);
      mini::Scene::SP model = mini::Scene::create({inst});
      
      if (!mesh->indices.empty())
        dataGroup.minis.push_back(model);
    } else {
      dataGroup.structuredVolumes.push_back
        (std::make_shared<StructuredVolume>(numVoxels,texelFormat,rawData,rawDataRGB,
                                            gridOrigin,gridSpacing));
    }
  }
  
  std::string SiloContent::toString() 
  {
    std::stringstream ss;
    ss << "SiloContext{#" << thisPartID << ",fileName="<<fileName<<",cellRange="<<cellRange<< "}";
    return ss.str();
  }

  
}
