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

#pragma once

#include "viewer/DataLoader.h"
#include "hayStack/StructuredVolume.h"
#include <map>
#include <memory>
#include <limits>

namespace hs {
  
  // Statistics for Silo data to enable global min/max computation
  struct SiloStats {
    float minVal = std::numeric_limits<float>::max();
    float maxVal = -std::numeric_limits<float>::max();
    int nanCount = 0;
    int infCount = 0;
    int numBlocks = 0;
  };
  
  /*! a file of 'raw' spheres */
  struct SiloContent : public LoadableContent {
    // using ScalarType = StructuredVolume::ScalarType;
    
    SiloContent(const std::string &fileName,
                     int thisPartID,
                     const box3i &cellRange,
                     vec3i fullVolumeDims,
                     const std::string &texelFormat,
                     int numChannels,
                     /*! if not NaN, we'll actually not store the
                         volume, but run iso-value extraction and use
                         the resulting surface(s) */
                     const float isoValue,
                     /*! name of the variable to load from the Silo file */
                     const std::string &variableName = "",
                     /*! mesh block name for multi-mesh files */
                     const std::string &meshBlockName = "",
                     /*! whether this is a multi-mesh file */
                     bool isMultiMesh = false,
                     /*! if not empty, save extracted isosurface to this path (numpy format) */
                     const std::string &isoExtractPath = "",
                     /*! if not empty, save extracted isosurface to this path (AGX format) */
                     const std::string &isoAgxPath = "",
                     /*! if not empty, map this scalar field to isosurface vertices */
                     const std::string &mappedScalarField = "",
                     /*! if true, skip adding isosurface to scene (for export-only mode) */
                     bool noRender = false,
                     /*! number of processors in each dimension for ghost cell trimming */
                     vec3i numProcs = vec3i(1),
                     /*! total number of processor blocks */
                     int totalBlocks = 1);
    
    static void create(DataLoader *loader,
                       const ResourceSpecifier &dataURL);
    size_t projectedSize() override;
    void   executeLoad(DataRank &dataGroup, bool verbose) override;

    std::string toString() override;

    const std::string   fileName;
    const int           thisPartID;
    const vec3i         fullVolumeDims;
    const box3i         cellRange;
    const int           numChannels;
    const std::string   texelFormat;
    const float         isoValue;
    const std::string   variableName;
    const std::string   meshBlockName;
    const bool          isMultiMesh;
    const std::string   isoExtractPath;
    const std::string   isoAgxPath;
    const std::string   mappedScalarField;
    const bool          noRender;
    const vec3i         numProcs;     // number of processors in each dimension
    const int           totalBlocks;  // total number of processor blocks
  };
  
}

namespace umesh {
  UMesh::SP tetrahedralize(UMesh::SP in,
                           int ownedTets,
                           int ownedPyrs,
                           int ownedWedges,
                           int ownedHexes);
}  


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
    
    // Specialized gradient computation for vorticity/helicity that only computes needed gradients
    void computeVorticityGradients(
      const std::vector<float>& vel1, const std::vector<float>& vel2, const std::vector<float>& vel3,
      int nx, int ny, int nz, float dx, float dy, const std::vector<float>& z,
      std::vector<float>& duy, std::vector<float>& duz,
      std::vector<float>& dvx, std::vector<float>& dvz,
      std::vector<float>& dwx, std::vector<float>& dwy);
  }
}
