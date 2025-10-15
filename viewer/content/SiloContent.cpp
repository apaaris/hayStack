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
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <umesh/UMesh.h>
#include <umesh/extractIsoSurface.h>
#include <miniScene/Scene.h>

// Forward declarations for derived velocity field computations
namespace hs {
  namespace vorticity {
    void computeVelocityGradients(
      const std::vector<float>& vel1, const std::vector<float>& vel2, const std::vector<float>& vel3,
      int nx, int ny, int nz, float dx, float dy, const std::vector<float>& z,
      std::vector<float>& dux, std::vector<float>& duy, std::vector<float>& duz,
      std::vector<float>& dvx, std::vector<float>& dvy, std::vector<float>& dvz,
      std::vector<float>& dwx, std::vector<float>& dwy, std::vector<float>& dwz);
    
    void computeLambda2(const std::vector<float>& dux, const std::vector<float>& duy, const std::vector<float>& duz,
                        const std::vector<float>& dvx, const std::vector<float>& dvy, const std::vector<float>& dvz,
                        const std::vector<float>& dwx, const std::vector<float>& dwy, const std::vector<float>& dwz,
                        std::vector<float>& result);
    
    void computeQCriterion(const std::vector<float>& dux, const std::vector<float>& duy, const std::vector<float>& duz,
                           const std::vector<float>& dvx, const std::vector<float>& dvy, const std::vector<float>& dvz,
                           const std::vector<float>& dwx, const std::vector<float>& dwy, const std::vector<float>& dwz,
                           std::vector<float>& result);
    
    void computeVorticity(const std::vector<float>& duy, const std::vector<float>& duz,
                          const std::vector<float>& dvx, const std::vector<float>& dvz,
                          const std::vector<float>& dwx, const std::vector<float>& dwy,
                          std::vector<float>& result);
    
    void computeHelicity(const std::vector<float>& vel1, const std::vector<float>& vel2, const std::vector<float>& vel3,
                         const std::vector<float>& duy, const std::vector<float>& duz,
                         const std::vector<float>& dvx, const std::vector<float>& dvz,
                         const std::vector<float>& dwx, const std::vector<float>& dwy,
                         std::vector<float>& result);
  }
}

#ifdef HS_HAVE_SILO
#include <silo.h>
// pmpio.h requires MPI, but we're not using it yet for single-file Silo loading
// #include <pmpio.h>
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
                                     const std::string &variableName,
                                     const std::string &meshBlockName,
                                     bool isMultiMesh,
                                     const std::string &isoExtractPath,
                                     const std::string &mappedScalarField)
    : fileName(fileName),
      thisPartID(thisPartID),
      cellRange(cellRange),
      fullVolumeDims(fullVolumeDims),
      texelFormat(texelFormat),
      numChannels(numChannels),
      isoValue(isoValue),
      variableName(variableName),
      meshBlockName(meshBlockName),
      isMultiMesh(isMultiMesh),
      isoExtractPath(isoExtractPath),
      mappedScalarField(mappedScalarField)
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
#ifdef HS_HAVE_SILO
    // First, check if this is a multi-mesh file
    DBfile *dbfile = DBOpen(dataURL.where.c_str(), DB_UNKNOWN, DB_READ);
    if (!dbfile)
      throw std::runtime_error("hs::SiloContent: could not open Silo file '"+dataURL.where+"'");
    
    DBtoc *toc = DBGetToc(dbfile);
    bool isMultiMesh = (toc && toc->nmultivar > 0);
    std::vector<std::string> meshBlockNames;
    std::string variableName = dataURL.get("var", dataURL.get("variable", ""));
    
    if (isMultiMesh) {
      // Multi-mesh file: get list of blocks
      std::cout << "#hs.silo: detected multi-mesh file with " << toc->nmultivar << " multi-variables" << std::endl;
      
      // Check if this is a derived variable
      bool isDerivedVar = (variableName == "lambda2" || variableName == "qCriterion" || 
                          variableName == "vorticity" || variableName == "helicity" ||
                          variableName == "vel_mag");
      
      const char *multivarName;
      if (isDerivedVar) {
        // For derived variables, use vel1 to get the mesh structure
        std::cout << "#hs.silo: '" << variableName << "' is a derived variable, using vel1 for mesh structure" << std::endl;
        multivarName = "vel1";
      } else {
        multivarName = variableName.empty() ? toc->multivar_names[0] : variableName.c_str();
      }
      
      DBmultivar *mv = DBGetMultivar(dbfile, multivarName);
      if (!mv) {
        DBClose(dbfile);
        throw std::runtime_error("hs::SiloContent: could not read multi-variable '"+std::string(multivarName)+"'");
      }
      
      std::cout << "#hs.silo: multi-variable '" << multivarName << "' has " << mv->nvars << " blocks" << std::endl;
      
      for (int i = 0; i < mv->nvars; i++) {
        if (mv->varnames[i]) {
          meshBlockNames.push_back(std::string(mv->varnames[i]));
        }
      }
      
      DBFreeMultivar(mv);
      DBClose(dbfile);
      
      // Distribute blocks across ranks
      int numParts = dataURL.numParts;
      if (meshBlockNames.size() < numParts) {
        std::cout << "#hs.silo: WARNING: requested " << numParts << " ranks but only " 
                  << meshBlockNames.size() << " blocks available. Using " << meshBlockNames.size() << " ranks." << std::endl;
        numParts = meshBlockNames.size();
      }
      
      if (loader->myRank() == 0) {
        std::cout << "Silo Multi-Mesh: " << meshBlockNames.size() << " blocks will be distributed across " 
                  << numParts << " ranks:" << std::endl;
      }
      
      // Distribute blocks round-robin across ranks
      std::vector<int> blocksPerRank(numParts, 0);
      for (int i = 0; i < meshBlockNames.size(); i++) {
        int rankID = i % numParts;
        blocksPerRank[rankID]++;
        if (loader->myRank() == 0) {
          std::cout << " Block #" << i << " (" << meshBlockNames[i] << ") -> rank " << rankID << std::endl;
        }
      }
      
      if (loader->myRank() == 0) {
        std::cout << "\nBlock distribution summary:" << std::endl;
        for (int r = 0; r < numParts; r++) {
          std::cout << " Rank " << r << ": " << blocksPerRank[r] << " blocks" << std::endl;
        }
        std::cout << " Total: " << meshBlockNames.size() << " blocks" << std::endl;
      }
      
      // Create content for each block
      std::string texelFormat = dataURL.get("format", dataURL.get("type", "float"));
      if (texelFormat == "f") texelFormat = "float";
      int numChannels = dataURL.get_int("channels", 1);
      float isoValue = NAN;
      std::string isoString = dataURL.get("iso", dataURL.get("isoValue"));
      if (!isoString.empty())
        isoValue = std::stof(isoString);
      std::string isoExtractPath = dataURL.get("iso_extract", "");
      std::string mappedScalarField = dataURL.get("mapped_scalar", dataURL.get("ms", ""));
      
      for (int i = 0; i < meshBlockNames.size(); i++) {
        int rankID = i % numParts;
        // Use block index 'i' as the processor ID for consistent numbering
        loader->addContent(new SiloContent(dataURL.where, i,
                                           box3i(vec3i(0), vec3i(0)), // cellRange not used for multi-mesh
                                           vec3i(0), // fullVolumeDims determined per-block
                                           texelFormat,
                                           numChannels,
                                           isoValue,
                                           variableName,  // Pass the requested variable, not the one used for mesh structure
                                           meshBlockNames[i],
                                           true, // isMultiMesh = true
                                           isoExtractPath,
                                           mappedScalarField));
      }
      return; // Done with multi-mesh handling
    }
    
    // Not a multi-mesh file, continue with single-mesh logic
    DBClose(dbfile);
#endif
    
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
    std::string isoExtractPath = dataURL.get("iso_extract", "");
    std::string mappedScalarField = dataURL.get("mapped_scalar", dataURL.get("ms", ""));
    
    if (variableName.empty())
      variableName = dataURL.get("var", dataURL.get("variable", ""));
    
    for (int i=0;i<dataURL.numParts;i++) {
      loader->addContent(new SiloContent(dataURL.where,i,
                                              regions[i],
                                              dims,texelFormat,//scalarType,
                                              numChannels,
                                              isoValue,
                                              variableName,
                                              "",  // meshBlockName (not used for single-mesh)
                                              false,  // isMultiMesh
                                              isoExtractPath,
                                              mappedScalarField));
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
    // Determine which file to open and which variable to load
    std::string fileToOpen = fileName;
    std::string varToLoad;
    
    if (isMultiMesh) {
      // Multi-mesh: parse block name to get file path and variable name
      // Format: "../p0/0.silo:vel1" or "path/to/file.silo:varname"
      size_t colonPos = meshBlockName.find(':');
      if (colonPos != std::string::npos) {
        std::string relativeBlockFile = meshBlockName.substr(0, colonPos);
        varToLoad = meshBlockName.substr(colonPos + 1);
        
        // Resolve relative path based on the master file's directory
        size_t lastSlash = fileName.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
          std::string masterDir = fileName.substr(0, lastSlash + 1);
          fileToOpen = masterDir + relativeBlockFile;
        } else {
          fileToOpen = relativeBlockFile;
        }
        
        if (verbose) {
          std::cout << "Loading Silo multi-mesh block:" << std::endl;
          std::cout << "  Master file: " << fileName << std::endl;
          std::cout << "  Block reference: " << meshBlockName << std::endl;
          std::cout << "  Resolved file: " << fileToOpen << std::endl;
          std::cout << "  Variable: " << varToLoad << std::endl;
        }
      } else {
        // No colon, treat entire meshBlockName as variable name
        varToLoad = meshBlockName;
        if (verbose) {
          std::cout << "Loading Silo multi-mesh block: " << meshBlockName << std::endl;
        }
      }
    }
    
    // Open the appropriate Silo file
    DBfile *dbfile = DBOpen(fileToOpen.c_str(), DB_UNKNOWN, DB_READ);
    if (!dbfile)
      throw std::runtime_error("hs::SiloContent: could not open Silo file '"+fileToOpen+"'");
    
    // Determine which variable to load
    const char *varname = nullptr;
    std::string requestedVar;
    
    // Check if this is a derived variable
    bool isDerivedVar = false;
    if (isMultiMesh) {
      // Use variableName (passed from constructor) instead of parsing from meshBlockName
      // because for derived variables, meshBlockName contains vel1, not the requested variable
      requestedVar = variableName.empty() ? varToLoad : variableName;
      isDerivedVar = (requestedVar == "lambda2" || requestedVar == "qCriterion" || 
                     requestedVar == "vorticity" || requestedVar == "helicity" ||
                     requestedVar == "vel_mag");
      // For derived vars in multi-mesh, we need vel1 for the mesh structure
      varname = isDerivedVar ? "vel1" : varToLoad.c_str();
    } else {
      // Single mesh: use variableName or first quad variable
      DBtoc *toc = DBGetToc(dbfile);
      if (!toc || toc->nqvar == 0) {
        DBClose(dbfile);
        throw std::runtime_error("hs::SiloContent: no quad variables found in file '"+fileName+"'");
      }
      
      if (!variableName.empty()) {
        requestedVar = variableName;
        isDerivedVar = (requestedVar == "lambda2" || requestedVar == "qCriterion" || 
                       requestedVar == "vorticity" || requestedVar == "helicity" ||
                       requestedVar == "vel_mag");
        varname = isDerivedVar ? "vel1" : variableName.c_str();
        if (verbose) {
          std::cout << "Loading Silo variable (specified): " << requestedVar << std::endl;
        }
      } else {
        varname = toc->qvar_names[0];
        requestedVar = varname;
        if (verbose) {
          std::cout << "Loading Silo variable (first in file): " << varname << std::endl;
        }
      }
    }
    
    // Read the quad variable
    DBquadvar *qvar = DBGetQuadvar(dbfile, varname);
    if (!qvar) {
      DBClose(dbfile);
      throw std::runtime_error("hs::SiloContent: could not read variable '"+std::string(varname)+"'");
    }
    
    // Get dimensions from the variable (important for multi-mesh where each block has different dims)
    vec3i blockDims(qvar->dims[0], qvar->dims[1], qvar->ndims > 2 ? qvar->dims[2] : 1);
    vec3i numVoxels = blockDims;
    
    // Get the spatial coordinates from the associated mesh
    vec3f meshOrigin(0.f);
    vec3f meshSpacing(1.f);
    
    if (qvar->meshname) {
      DBquadmesh *qmesh = DBGetQuadmesh(dbfile, qvar->meshname);
      if (qmesh) {
        // Get coordinate arrays
        if (qmesh->coords && qmesh->coordtype == DB_COLLINEAR) {
          // Collinear coordinates: separate arrays for x, y, z
          float *xCoords = (float*)qmesh->coords[0];
          float *yCoords = (float*)qmesh->coords[1];
          float *zCoords = qmesh->ndims > 2 ? (float*)qmesh->coords[2] : nullptr;
          
          meshOrigin.x = xCoords[0];
          meshOrigin.y = yCoords[0];
          meshOrigin.z = zCoords ? zCoords[0] : 0.f;
          
          if (qmesh->dims[0] > 1) {
            meshSpacing.x = xCoords[1] - xCoords[0];
          }
          if (qmesh->dims[1] > 1) {
            meshSpacing.y = yCoords[1] - yCoords[0];
          }
          if (zCoords && qmesh->dims[2] > 1) {
            meshSpacing.z = zCoords[1] - zCoords[0];
          }
          
          if (verbose) {
            std::cout << "  Mesh origin: " << meshOrigin << std::endl;
            std::cout << "  Mesh spacing: " << meshSpacing << std::endl;
          }
        }
        DBFreeQuadmesh(qmesh);
      }
    }
    
    // For multi-mesh, we load the entire block; for single mesh, we use cellRange
    box3i loadRange;
    if (isMultiMesh) {
      loadRange = box3i(vec3i(0), blockDims - 1);
    } else {
      loadRange = cellRange;
      numVoxels = cellRange.size() + 1;
    }
    
    size_t numScalars = size_t(numVoxels.x) * size_t(numVoxels.y) * size_t(numVoxels.z);
    std::vector<uint8_t> rawData(numScalars * sizeOf(texelFormat));
    
    // Extract data from the quad variable
    void *srcData = qvar->vals[0]; // First component
    int datatype = qvar->datatype;
    
    if (verbose) {
      std::cout << "  Block dimensions: " << blockDims << std::endl;
      std::cout << "  Load range: " << loadRange << std::endl;
      std::cout << "  Num voxels to extract: " << numVoxels << std::endl;
      std::cout << "  Data type: " << qvar->datatype << " (DB_FLOAT=" << DB_FLOAT 
              << ", DB_DOUBLE=" << DB_DOUBLE << ", DB_INT=" << DB_INT << ")" << std::endl;
      std::cout << "  Texel format: " << texelFormat << std::endl;
      std::cout << "  Buffer size: " << rawData.size() << " bytes" << std::endl;
    }
    
    // For single mesh, verify dimensions match
    if (!isMultiMesh && blockDims != fullVolumeDims) {
      DBFreeQuadvar(qvar);
      DBClose(dbfile);
      throw std::runtime_error("hs::SiloContent: dimension mismatch - expected " 
                               + std::to_string(fullVolumeDims.x) + "x" 
                               + std::to_string(fullVolumeDims.y) + "x" 
                               + std::to_string(fullVolumeDims.z) 
                               + " but got " + std::to_string(blockDims.x) + "x" 
                               + std::to_string(blockDims.y) + "x" 
                               + std::to_string(blockDims.z));
    }
    
    // Convert and copy data to our buffer based on data type
    char *dataPtr = (char *)rawData.data();
    size_t texelSize = sizeOf(texelFormat);
    
    for (int iz=loadRange.lower.z;iz<=loadRange.upper.z;iz++) {
      for (int iy=loadRange.lower.y;iy<=loadRange.upper.y;iy++) {
        for (int ix=loadRange.lower.x;ix<=loadRange.upper.x;ix++) {
          size_t srcIdx = ix + iy*size_t(blockDims.x) 
                            + iz*size_t(blockDims.x)*size_t(blockDims.y);
          
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
    
    // Compute derived variables if needed
    if (isDerivedVar) {
      std::cout << "#hs.silo: computing derived variable '" << requestedVar << "' from velocity components" << std::endl;
      
      // Load vel2 and vel3
      std::vector<float> vel1Data, vel2Data, vel3Data;
      
      // vel1 is already loaded in rawData, convert to float vector
      vel1Data.resize(numScalars);
      for (size_t i = 0; i < numScalars; i++) {
        if (texelFormat == "float")
          vel1Data[i] = ((float*)rawData.data())[i];
        else if (texelFormat == "uint8_t")
          vel1Data[i] = ((uint8_t*)rawData.data())[i] / 255.0f;
        else if (texelFormat == "uint16_t")
          vel1Data[i] = ((uint16_t*)rawData.data())[i] / 65535.0f;
      }
      
      // Load vel2
      DBquadvar *qvar2 = DBGetQuadvar(dbfile, "vel2");
      if (!qvar2) throw std::runtime_error("Could not load vel2 for derived variable");
      vel2Data.resize(numScalars);
      void *srcData2 = qvar2->vals[0];
      int datatype2 = qvar2->datatype;
      size_t idx = 0;
      for (int iz=loadRange.lower.z;iz<=loadRange.upper.z;iz++) {
        for (int iy=loadRange.lower.y;iy<=loadRange.upper.y;iy++) {
          for (int ix=loadRange.lower.x;ix<=loadRange.upper.x;ix++) {
            size_t srcIdx = ix + iy*size_t(blockDims.x) + iz*size_t(blockDims.x)*size_t(blockDims.y);
            if (datatype2 == DB_FLOAT) vel2Data[idx] = ((float*)srcData2)[srcIdx];
            else if (datatype2 == DB_DOUBLE) vel2Data[idx] = (float)((double*)srcData2)[srcIdx];
            else if (datatype2 == DB_INT) vel2Data[idx] = (float)((int*)srcData2)[srcIdx];
            idx++;
          }
        }
      }
      DBFreeQuadvar(qvar2);
      
      // Load vel3
      DBquadvar *qvar3 = DBGetQuadvar(dbfile, "vel3");
      if (!qvar3) throw std::runtime_error("Could not load vel3 for derived variable");
      vel3Data.resize(numScalars);
      void *srcData3 = qvar3->vals[0];
      int datatype3 = qvar3->datatype;
      idx = 0;
      for (int iz=loadRange.lower.z;iz<=loadRange.upper.z;iz++) {
        for (int iy=loadRange.lower.y;iy<=loadRange.upper.y;iy++) {
          for (int ix=loadRange.lower.x;ix<=loadRange.upper.x;ix++) {
            size_t srcIdx = ix + iy*size_t(blockDims.x) + iz*size_t(blockDims.x)*size_t(blockDims.y);
            if (datatype3 == DB_FLOAT) vel3Data[idx] = ((float*)srcData3)[srcIdx];
            else if (datatype3 == DB_DOUBLE) vel3Data[idx] = (float)((double*)srcData3)[srcIdx];
            else if (datatype3 == DB_INT) vel3Data[idx] = (float)((int*)srcData3)[srcIdx];
            idx++;
          }
        }
      }
      DBFreeQuadvar(qvar3);
      
      // Build z coordinate array for non-uniform spacing
      std::vector<float> zCoords(numVoxels.z);
      for (int i = 0; i < numVoxels.z; i++) {
        zCoords[i] = meshOrigin.z + i * meshSpacing.z;
      }
      
      // Compute the requested derived variable
      std::vector<float> derivedData(numScalars);
      
      if (requestedVar == "vel_mag") {
        // Velocity magnitude
        for (size_t i = 0; i < numScalars; i++) {
          derivedData[i] = std::sqrt(vel1Data[i]*vel1Data[i] + vel2Data[i]*vel2Data[i] + vel3Data[i]*vel3Data[i]);
        }
      } else {
        // Need gradients for other derived variables
        std::vector<float> dux(numScalars), duy(numScalars), duz(numScalars);
        std::vector<float> dvx(numScalars), dvy(numScalars), dvz(numScalars);
        std::vector<float> dwx(numScalars), dwy(numScalars), dwz(numScalars);
        
        vorticity::computeVelocityGradients(vel1Data, vel2Data, vel3Data,
                                           numVoxels.x, numVoxels.y, numVoxels.z,
                                           meshSpacing.x, meshSpacing.y, zCoords,
                                           dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz);
        
        if (requestedVar == "lambda2") {
          vorticity::computeLambda2(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz, derivedData);
        } else if (requestedVar == "qCriterion") {
          vorticity::computeQCriterion(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz, derivedData);
        } else if (requestedVar == "vorticity") {
          vorticity::computeVorticity(duy, duz, dvx, dvz, dwx, dwy, derivedData);
        } else if (requestedVar == "helicity") {
          vorticity::computeHelicity(vel1Data, vel2Data, vel3Data, duy, duz, dvx, dvz, dwx, dwy, derivedData);
        }
      }
      
      // Convert back to rawData format (derived variables are always float)
      rawData.resize(numScalars * sizeof(float));
      memcpy(rawData.data(), derivedData.data(), numScalars * sizeof(float));
      
      std::cout << "#hs.silo: computed " << requestedVar << std::endl;
    }
    
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
    
    // Use mesh coordinates if available, otherwise fall back to cell range
    vec3f gridOrigin = isMultiMesh ? meshOrigin : vec3f(loadRange.lower);
    vec3f gridSpacing = isMultiMesh ? meshSpacing : vec3f(1.f);
    
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

      // Load mapped scalar field if specified (and not a coordinate specifier)
      std::vector<float> mappedScalarVolume;
      bool haveMappedScalars = false;
      if (!mappedScalarField.empty() && mappedScalarField[0] != ':') {
#ifdef HS_HAVE_SILO
        // Check if this is a derived variable
        bool isMappedDerivedVar = (mappedScalarField == "lambda2" || mappedScalarField == "qCriterion" || 
                                   mappedScalarField == "vorticity" || mappedScalarField == "helicity" ||
                                   mappedScalarField == "vel_mag");
        
        if (isMappedDerivedVar) {
          // Compute derived field for mapped scalar
          std::cout << "#hs.silo: computing derived mapped scalar field '" << mappedScalarField << "' from velocity components" << std::endl;
          
          // Reopen the file to load velocity components
          DBfile *dbfile2 = DBOpen(fileToOpen.c_str(), DB_UNKNOWN, DB_READ);
          if (dbfile2) {
            size_t numScalars = numVoxels.x * numVoxels.y * numVoxels.z;
            std::vector<float> vel1Data(numScalars), vel2Data(numScalars), vel3Data(numScalars);
            
            // Load vel1
            DBquadvar *qvar1 = DBGetQuadvar(dbfile2, "vel1");
            if (qvar1) {
              void *srcData1 = qvar1->vals[0];
              int datatype1 = qvar1->datatype;
              size_t idx = 0;
              for (int iz=loadRange.lower.z;iz<=loadRange.upper.z;iz++) {
                for (int iy=loadRange.lower.y;iy<=loadRange.upper.y;iy++) {
                  for (int ix=loadRange.lower.x;ix<=loadRange.upper.x;ix++) {
                    size_t srcIdx = ix + iy*size_t(blockDims.x) + iz*size_t(blockDims.x)*size_t(blockDims.y);
                    if (datatype1 == DB_FLOAT) vel1Data[idx] = ((float*)srcData1)[srcIdx];
                    else if (datatype1 == DB_DOUBLE) vel1Data[idx] = (float)((double*)srcData1)[srcIdx];
                    else if (datatype1 == DB_INT) vel1Data[idx] = (float)((int*)srcData1)[srcIdx];
                    idx++;
                  }
                }
              }
              DBFreeQuadvar(qvar1);
            }
            
            // Load vel2
            DBquadvar *qvar2 = DBGetQuadvar(dbfile2, "vel2");
            if (qvar2) {
              void *srcData2 = qvar2->vals[0];
              int datatype2 = qvar2->datatype;
              size_t idx = 0;
              for (int iz=loadRange.lower.z;iz<=loadRange.upper.z;iz++) {
                for (int iy=loadRange.lower.y;iy<=loadRange.upper.y;iy++) {
                  for (int ix=loadRange.lower.x;ix<=loadRange.upper.x;ix++) {
                    size_t srcIdx = ix + iy*size_t(blockDims.x) + iz*size_t(blockDims.x)*size_t(blockDims.y);
                    if (datatype2 == DB_FLOAT) vel2Data[idx] = ((float*)srcData2)[srcIdx];
                    else if (datatype2 == DB_DOUBLE) vel2Data[idx] = (float)((double*)srcData2)[srcIdx];
                    else if (datatype2 == DB_INT) vel2Data[idx] = (float)((int*)srcData2)[srcIdx];
                    idx++;
                  }
                }
              }
              DBFreeQuadvar(qvar2);
            }
            
            // Load vel3
            DBquadvar *qvar3 = DBGetQuadvar(dbfile2, "vel3");
            if (qvar3) {
              void *srcData3 = qvar3->vals[0];
              int datatype3 = qvar3->datatype;
              size_t idx = 0;
              for (int iz=loadRange.lower.z;iz<=loadRange.upper.z;iz++) {
                for (int iy=loadRange.lower.y;iy<=loadRange.upper.y;iy++) {
                  for (int ix=loadRange.lower.x;ix<=loadRange.upper.x;ix++) {
                    size_t srcIdx = ix + iy*size_t(blockDims.x) + iz*size_t(blockDims.x)*size_t(blockDims.y);
                    if (datatype3 == DB_FLOAT) vel3Data[idx] = ((float*)srcData3)[srcIdx];
                    else if (datatype3 == DB_DOUBLE) vel3Data[idx] = (float)((double*)srcData3)[srcIdx];
                    else if (datatype3 == DB_INT) vel3Data[idx] = (float)((int*)srcData3)[srcIdx];
                    idx++;
                  }
                }
              }
              DBFreeQuadvar(qvar3);
            }
            
            // Compute the derived field
            mappedScalarVolume.resize(numScalars);
            
            if (mappedScalarField == "vel_mag") {
              // Velocity magnitude
              for (size_t i = 0; i < numScalars; i++) {
                mappedScalarVolume[i] = std::sqrt(vel1Data[i]*vel1Data[i] + vel2Data[i]*vel2Data[i] + vel3Data[i]*vel3Data[i]);
              }
            } else {
              // Need gradients for other derived variables
              std::vector<float> dux(numScalars), duy(numScalars), duz(numScalars);
              std::vector<float> dvx(numScalars), dvy(numScalars), dvz(numScalars);
              std::vector<float> dwx(numScalars), dwy(numScalars), dwz(numScalars);
              
              // Build z coordinate array for non-uniform spacing
              std::vector<float> zCoords(numVoxels.z);
              for (int i = 0; i < numVoxels.z; i++) {
                zCoords[i] = gridOrigin.z + i * gridSpacing.z;
              }
              
              vorticity::computeVelocityGradients(vel1Data, vel2Data, vel3Data,
                                                 numVoxels.x, numVoxels.y, numVoxels.z,
                                                 gridSpacing.x, gridSpacing.y, zCoords,
                                                 dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz);
              
              if (mappedScalarField == "lambda2") {
                vorticity::computeLambda2(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz, mappedScalarVolume);
              } else if (mappedScalarField == "qCriterion") {
                vorticity::computeQCriterion(dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz, mappedScalarVolume);
              } else if (mappedScalarField == "vorticity") {
                vorticity::computeVorticity(duy, duz, dvx, dvz, dwx, dwy, mappedScalarVolume);
              } else if (mappedScalarField == "helicity") {
                vorticity::computeHelicity(vel1Data, vel2Data, vel3Data, duy, duz, dvx, dvz, dwx, dwy, mappedScalarVolume);
              }
            }
            
            haveMappedScalars = true;
            std::cout << "#hs.silo: computed mapped scalar field '" << mappedScalarField << "'" << std::endl;
            DBClose(dbfile2);
          }
        } else {
          // Regular (non-derived) field - load directly from file
          DBfile *dbfile2 = DBOpen(fileToOpen.c_str(), DB_UNKNOWN, DB_READ);
          if (dbfile2) {
            DBquadvar *qvar2 = DBGetQuadvar(dbfile2, mappedScalarField.c_str());
            if (qvar2) {
              std::cout << "#hs.silo: loading mapped scalar field '" << mappedScalarField << "'" << std::endl;
              void *srcData2 = qvar2->vals[0];
              int datatype2 = qvar2->datatype;
              
              mappedScalarVolume.resize(numVoxels.x * numVoxels.y * numVoxels.z);
              size_t idx = 0;
              for (int iz=loadRange.lower.z;iz<=loadRange.upper.z;iz++) {
                for (int iy=loadRange.lower.y;iy<=loadRange.upper.y;iy++) {
                  for (int ix=loadRange.lower.x;ix<=loadRange.upper.x;ix++) {
                    size_t srcIdx = ix + iy*size_t(blockDims.x) + iz*size_t(blockDims.x)*size_t(blockDims.y);
                    
                    if (datatype2 == DB_FLOAT) {
                      mappedScalarVolume[idx] = ((float*)srcData2)[srcIdx];
                    } else if (datatype2 == DB_DOUBLE) {
                      mappedScalarVolume[idx] = (float)((double*)srcData2)[srcIdx];
                    } else if (datatype2 == DB_INT) {
                      mappedScalarVolume[idx] = (float)((int*)srcData2)[srcIdx];
                    }
                    idx++;
                  }
                }
              }
              haveMappedScalars = true;
              DBFreeQuadvar(qvar2);
            } else {
              std::cout << "#hs.silo: WARNING - could not load mapped scalar field '" 
                        << mappedScalarField << "'" << std::endl;
            }
            DBClose(dbfile2);
          }
        }
#endif
      }
      
      // volume = umesh::tetrahedralize(volume,0,0,0,volume->hexes.size());
      // PRINT(volume->toString());
      umesh::UMesh::SP surf = umesh::extractIsoSurface(volume, isoValue);
      surf->finalize();
      PRINT(surf->toString());

      // Save isosurface to file if iso_extract path is specified
      if (!isoExtractPath.empty()) {
        // Create output filename prefix: outPath/iso_field_timestep.processor
        std::string outputFilePrefix = isoExtractPath;
        // Ensure the path ends with a separator
        if (!outputFilePrefix.empty() && outputFilePrefix.back() != '/' && outputFilePrefix.back() != '\\') {
          outputFilePrefix += "/";
        }
        
        // Get field name (variable name or "unknown")
        std::string fieldName = variableName.empty() ? "unknown" : variableName;
        // Remove any path prefix from field name (e.g., "vel1" from "../_p5_4320.silo_vel1")
        size_t lastSlash = fieldName.find_last_of("/\\");
        if (lastSlash != std::string::npos && lastSlash + 1 < fieldName.length()) {
          fieldName = fieldName.substr(lastSlash + 1);
        }
        
        // Extract timestep from meshBlockName or fileName
        std::string timestep = "0";
        // Use thisPartID as the processor number (sequential from 0)
        std::string processorId = std::to_string(thisPartID);
        
        if (isMultiMesh) {
          // For multi-mesh, parse meshBlockName to extract timestep
          // Format: "../p5/4320.silo:vel1" or similar
          std::string blockName = meshBlockName;
          
          // Look for timestep (number before .silo)
          size_t siloPos = blockName.find(".silo");
          if (siloPos != std::string::npos) {
            // Search backwards for a number
            size_t numEnd = siloPos;
            size_t numStart = siloPos;
            while (numStart > 0 && std::isdigit(blockName[numStart - 1])) {
              numStart--;
            }
            if (numStart < numEnd) {
              timestep = blockName.substr(numStart, numEnd - numStart);
            }
          }
        } else {
          // For single-mesh, try to extract timestep from fileName
          std::string baseName = fileName;
          size_t siloPos = baseName.find(".silo");
          if (siloPos != std::string::npos) {
            size_t numEnd = siloPos;
            size_t numStart = siloPos;
            while (numStart > 0 && std::isdigit(baseName[numStart - 1])) {
              numStart--;
            }
            if (numStart < numEnd) {
              timestep = baseName.substr(numStart, numEnd - numStart);
            }
          }
        }
        
        // Construct filename: iso_field_timestep.processor
        outputFilePrefix += "iso_" + fieldName + "_" + timestep + "." + processorId;
        
        // Compute mapped scalars for isosurface vertices
        std::vector<float> mappedScalars;
        if (!mappedScalarField.empty()) {
          if (mappedScalarField == ":x") {
            // Map X coordinate
            for (auto v : surf->vertices)
              mappedScalars.push_back(v.x);
          } else if (mappedScalarField == ":y") {
            // Map Y coordinate
            for (auto v : surf->vertices)
              mappedScalars.push_back(v.y);
          } else if (mappedScalarField == ":z") {
            // Map Z coordinate
            for (auto v : surf->vertices)
              mappedScalars.push_back(v.z);
          } else if (haveMappedScalars) {
            // Trilinear interpolation of the mapped scalar field
            for (auto v : surf->vertices) {
              // Convert world position back to grid coordinates
              umesh::vec3f gridPos = (v - (const umesh::vec3f&)gridOrigin) / (const umesh::vec3f&)gridSpacing;
              
              // Clamp to valid range
              gridPos.x = std::max(0.f, std::min((float)(numVoxels.x - 1), gridPos.x));
              gridPos.y = std::max(0.f, std::min((float)(numVoxels.y - 1), gridPos.y));
              gridPos.z = std::max(0.f, std::min((float)(numVoxels.z - 1), gridPos.z));
              
              // Trilinear interpolation
              int ix0 = (int)std::floor(gridPos.x), ix1 = std::min(ix0 + 1, numVoxels.x - 1);
              int iy0 = (int)std::floor(gridPos.y), iy1 = std::min(iy0 + 1, numVoxels.y - 1);
              int iz0 = (int)std::floor(gridPos.z), iz1 = std::min(iz0 + 1, numVoxels.z - 1);
              
              float fx = gridPos.x - ix0;
              float fy = gridPos.y - iy0;
              float fz = gridPos.z - iz0;
              
              auto getVal = [&](int ix, int iy, int iz) {
                return mappedScalarVolume[ix + iy * numVoxels.x + iz * numVoxels.x * numVoxels.y];
              };
              
              float v000 = getVal(ix0, iy0, iz0);
              float v001 = getVal(ix1, iy0, iz0);
              float v010 = getVal(ix0, iy1, iz0);
              float v011 = getVal(ix1, iy1, iz0);
              float v100 = getVal(ix0, iy0, iz1);
              float v101 = getVal(ix1, iy0, iz1);
              float v110 = getVal(ix0, iy1, iz1);
              float v111 = getVal(ix1, iy1, iz1);
              
              float v00 = v000 * (1 - fx) + v001 * fx;
              float v01 = v010 * (1 - fx) + v011 * fx;
              float v10 = v100 * (1 - fx) + v101 * fx;
              float v11 = v110 * (1 - fx) + v111 * fx;
              
              float v0 = v00 * (1 - fy) + v01 * fy;
              float v1 = v10 * (1 - fy) + v11 * fy;
              
              float interpolated = v0 * (1 - fz) + v1 * fz;
              mappedScalars.push_back(interpolated);
            }
          } else {
            // Fallback to isoValue
            std::cout << "#hs.silo: WARNING - mapped scalar field '" << mappedScalarField 
                      << "' could not be loaded, using isoValue" << std::endl;
            for (int i = 0; i < surf->vertices.size(); i++)
              mappedScalars.push_back(isoValue);
          }
        } else {
          // No mapped scalar specified, use isoValue
          for (int i = 0; i < surf->vertices.size(); i++)
            mappedScalars.push_back(isoValue);
        }
        
        std::cout << "#hs.silo: writing isosurface in raw numpy arrays format to " 
                  << outputFilePrefix << "{.vertex_coords.f3,.vertex_scalars.f1,.triangle_indices.i3}" << std::endl;
        if (!mappedScalarField.empty()) {
          std::cout << "#hs.silo: mapping scalar field: " << mappedScalarField << std::endl;
        }
        
        // Write the three binary files
        std::ofstream vertices(outputFilePrefix + ".vertex_coords.f3", std::ios::binary);
        std::ofstream scalars(outputFilePrefix + ".vertex_scalars.f1", std::ios::binary);
        std::ofstream indices(outputFilePrefix + ".triangle_indices.i3", std::ios::binary);
        
        if (!vertices.is_open() || !scalars.is_open() || !indices.is_open()) {
          std::cerr << "#hs.silo: ERROR - failed to open output files for writing" << std::endl;
        } else {
          // Write triangle indices
          for (auto t : surf->triangles) {
            indices.write((const char *)&t, 3*sizeof(int));
          }
          
          // Write vertex coordinates and mapped scalars
          for (int i = 0; i < surf->vertices.size(); i++) {
            umesh::vec3f v = surf->vertices[i];
            vertices.write((const char *)&v, sizeof(v));
            
            // Write mapped scalar value
            float f = mappedScalars[i];
            scalars.write((const char *)&f, sizeof(f));
          }
          
          vertices.close();
          scalars.close();
          indices.close();
          
          std::cout << "#hs.silo: saved " << surf->triangles.size() << " triangles and " 
                    << surf->vertices.size() << " vertices" << std::endl;
        }
      }

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
    
    if (verbose) {
      std::cout << "  Loaded block with " << (numVoxels.x * numVoxels.y * numVoxels.z) 
                << " cells at origin " << gridOrigin << std::endl;
    }
  }
  
  std::string SiloContent::toString() 
  {
    std::stringstream ss;
    ss << "SiloContext{#" << thisPartID << ",fileName=" << fileName;
    if (isMultiMesh) {
      ss << ",multiMeshBlock=" << meshBlockName;
    } else {
      ss << ",cellRange=" << cellRange;
    }
    ss << "}";
    return ss.str();
  }

  // Vorticity and derived field computations
  namespace vorticity {
    
    // Compute gradient in X direction (periodic boundaries)
    void gradX(const std::vector<float>& u, std::vector<float>& uGrad, int nx, int ny, int nz, float dx) {
      for (int z = 0; z < nz; ++z) {
        size_t off = z * nx * ny;
        for (int row = 0; row < ny; ++row) {
          // Periodic boundaries
          uGrad[off + row*nx] = 0.5f * (u[off + row*nx + 1] - u[off + row*nx + nx - 1]) / (2.0f*dx);
          uGrad[off + row*nx + nx - 1] = 0.5f * (u[off + row*nx] - u[off + row*nx + nx - 2]) / (2.0f*dx);
          // Interior points
          for (int col = 1; col < nx - 1; ++col) {
            uGrad[off + row*nx + col] = 0.5f * (u[off + row*nx + col + 1] - u[off + row*nx + col - 1]) / (2.0f*dx);
          }
        }
      }
    }
    
    // Compute gradient in Y direction (periodic boundaries)
    void gradY(const std::vector<float>& v, std::vector<float>& vGrad, int nx, int ny, int nz, float dy) {
      for (int z = 0; z < nz; ++z) {
        size_t off = z * nx * ny;
        for (int col = 0; col < nx; ++col) {
          // Periodic boundaries
          vGrad[0*nx + col + off] = 0.5f * (v[1*nx + col + off] - v[(ny-1)*nx + col + off]) / (2.0f*dy);
          vGrad[(ny-1)*nx + col + off] = 0.5f * (v[0*nx + col + off] - v[(ny-2)*nx + col + off]) / (2.0f*dy);
          // Interior points
          for (int row = 1; row < ny - 1; ++row) {
            vGrad[row*nx + col + off] = 0.5f * (v[(row+1)*nx + col + off] - v[(row-1)*nx + col + off]) / (2.0f*dy);
          }
        }
      }
    }
    
    // Compute gradient in Z direction (non-uniform spacing)
    void gradZ(const std::vector<float>& w, std::vector<float>& wGrad, int nx, int ny, int nz, const std::vector<float>& z) {
      size_t off = nx * ny;
      for (int row = 0; row < ny; ++row) {
        for (int col = 0; col < nx; ++col) {
          // Boundaries (forward/backward difference)
          wGrad[(nz-1)*off + row*nx + col] = (w[(nz-1)*off + row*nx + col] - w[(nz-2)*off + row*nx + col]) / (z[nz-1] - z[nz-2]);
          wGrad[0*off + row*nx + col] = (w[1*off + row*nx + col] - w[0*off + row*nx + col]) / (z[1] - z[0]);
          // Interior points (central difference)
          for (int zi = 1; zi < nz - 1; ++zi) {
            wGrad[zi*off + row*nx + col] = 0.5f * (w[(zi+1)*off + row*nx + col] - w[(zi-1)*off + row*nx + col]) / (z[zi+1] - z[zi-1]);
          }
        }
      }
    }
    
    void computeVelocityGradients(
      const std::vector<float>& vel1, const std::vector<float>& vel2, const std::vector<float>& vel3,
      int nx, int ny, int nz, float dx, float dy, const std::vector<float>& z,
      std::vector<float>& dux, std::vector<float>& duy, std::vector<float>& duz,
      std::vector<float>& dvx, std::vector<float>& dvy, std::vector<float>& dvz,
      std::vector<float>& dwx, std::vector<float>& dwy, std::vector<float>& dwz)
    {
      gradX(vel1, dux, nx, ny, nz, dx);
      gradX(vel2, dvx, nx, ny, nz, dx);
      gradX(vel3, dwx, nx, ny, nz, dx);
      
      gradY(vel1, duy, nx, ny, nz, dy);
      gradY(vel2, dvy, nx, ny, nz, dy);
      gradY(vel3, dwy, nx, ny, nz, dy);
      
      gradZ(vel1, duz, nx, ny, nz, z);
      gradZ(vel2, dvz, nx, ny, nz, z);
      gradZ(vel3, dwz, nx, ny, nz, z);
    }
    
    void computeLambda2(const std::vector<float>& dux, const std::vector<float>& duy, const std::vector<float>& duz,
                        const std::vector<float>& dvx, const std::vector<float>& dvy, const std::vector<float>& dvz,
                        const std::vector<float>& dwx, const std::vector<float>& dwy, const std::vector<float>& dwz,
                        std::vector<float>& result)
    {
      size_t len = dux.size();
      for (size_t i = 0; i < len; ++i) {
        // Strain rate tensor S = 0.5*(J + J^T)
        float s11 = dux[i];
        float s12 = 0.5f * (duy[i] + dvx[i]);
        float s13 = 0.5f * (duz[i] + dwx[i]);
        float s22 = dvy[i];
        float s23 = 0.5f * (dvz[i] + dwy[i]);
        float s33 = dwz[i];
        
        // Antisymmetric part Omega = 0.5*(J - J^T)
        float o12 = 0.5f * (duy[i] - dvx[i]);
        float o13 = 0.5f * (duz[i] - dwx[i]);
        float o23 = 0.5f * (dvz[i] - dwy[i]);
        
        // S^2 + Omega^2
        float m11 = s11*s11 + s12*s12 + s13*s13 - o12*o12 - o13*o13;
        float m12 = s11*s12 + s12*s22 + s13*s23 + o12*(s11 - s22) + o13*o23;
        float m13 = s11*s13 + s12*s23 + s13*s33 + o13*(s11 - s33) - o12*o23;
        float m22 = s12*s12 + s22*s22 + s23*s23 - o12*o12 - o23*o23;
        float m23 = s12*s13 + s22*s23 + s23*s33 + o23*(s22 - s33) + o12*o13;
        float m33 = s13*s13 + s23*s23 + s33*s33 - o13*o13 - o23*o23;
        
        // Compute eigenvalues of 3x3 symmetric matrix
        float p1 = m12*m12 + m13*m13 + m23*m23;
        float q = (m11 + m22 + m33) / 3.0f;
        float p2 = (m11 - q)*(m11 - q) + (m22 - q)*(m22 - q) + (m33 - q)*(m33 - q) + 2.0f*p1;
        float p = std::sqrt(p2 / 6.0f);
        
        float b11 = (m11 - q) / p;
        float b12 = m12 / p;
        float b13 = m13 / p;
        float b22 = (m22 - q) / p;
        float b23 = m23 / p;
        float b33 = (m33 - q) / p;
        
        float r = (b11*(b22*b33 - b23*b23) - b12*(b12*b33 - b23*b13) + b13*(b12*b23 - b22*b13)) / 2.0f;
        r = std::max(-1.0f, std::min(1.0f, r));
        
        float phi = std::acos(r) / 3.0f;
        float eig1 = q + 2.0f*p*std::cos(phi);
        float eig3 = q + 2.0f*p*std::cos(phi + (2.0f*M_PI/3.0f));
        float eig2 = 3.0f*q - eig1 - eig3; // middle eigenvalue
        
        result[i] = -std::min(eig2, 0.0f);
      }
    }
    
    void computeQCriterion(const std::vector<float>& dux, const std::vector<float>& duy, const std::vector<float>& duz,
                           const std::vector<float>& dvx, const std::vector<float>& dvy, const std::vector<float>& dvz,
                           const std::vector<float>& dwx, const std::vector<float>& dwy, const std::vector<float>& dwz,
                           std::vector<float>& result)
    {
      size_t len = dux.size();
      for (size_t i = 0; i < len; ++i) {
        // Strain rate tensor S = 0.5*(J + J^T)
        float s11 = dux[i];
        float s12 = 0.5f * (duy[i] + dvx[i]);
        float s13 = 0.5f * (duz[i] + dwx[i]);
        float s22 = dvy[i];
        float s23 = 0.5f * (dvz[i] + dwy[i]);
        float s33 = dwz[i];
        
        // Rotation tensor Omega = 0.5*(J - J^T)
        float o12 = 0.5f * (duy[i] - dvx[i]);
        float o13 = 0.5f * (duz[i] - dwx[i]);
        float o23 = 0.5f * (dvz[i] - dwy[i]);
        
        // Q = 0.5 * (||Omega||^2 - ||S||^2)
        float omegaNorm2 = 2.0f * (o12*o12 + o13*o13 + o23*o23);
        float sNorm2 = s11*s11 + s22*s22 + s33*s33 + 2.0f*(s12*s12 + s13*s13 + s23*s23);
        
        result[i] = std::max(0.5f * (omegaNorm2 - sNorm2), 0.0f);
      }
    }
    
    void computeVorticity(const std::vector<float>& duy, const std::vector<float>& duz,
                          const std::vector<float>& dvx, const std::vector<float>& dvz,
                          const std::vector<float>& dwx, const std::vector<float>& dwy,
                          std::vector<float>& result)
    {
      size_t len = duy.size();
      for (size_t i = 0; i < len; ++i) {
        float wx = dwy[i] - dvz[i];
        float wy = duz[i] - dwx[i];
        float wz = dvx[i] - duy[i];
        result[i] = std::sqrt(wx*wx + wy*wy + wz*wz);
      }
    }
    
    void computeHelicity(const std::vector<float>& vel1, const std::vector<float>& vel2, const std::vector<float>& vel3,
                         const std::vector<float>& duy, const std::vector<float>& duz,
                         const std::vector<float>& dvx, const std::vector<float>& dvz,
                         const std::vector<float>& dwx, const std::vector<float>& dwy,
                         std::vector<float>& result)
    {
      size_t len = vel1.size();
      for (size_t i = 0; i < len; ++i) {
        float wx = dwy[i] - dvz[i];
        float wy = duz[i] - dwx[i];
        float wz = dvx[i] - duy[i];
        
        float helicity = std::abs(wx*vel1[i] + wy*vel2[i] + wz*vel3[i]);
        float velMag = std::sqrt(vel1[i]*vel1[i] + vel2[i]*vel2[i] + vel3[i]*vel3[i]);
        float vortMag = std::sqrt(wx*wx + wy*wy + wz*wz);
        
        if (velMag != 0.0f && vortMag != 0.0f) {
          result[i] = helicity / (2.0f * velMag * vortMag);
        } else {
          result[i] = 0.0f;
        }
      }
    }
    
  } // namespace vorticity
  
}
