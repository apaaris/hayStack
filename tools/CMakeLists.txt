# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

add_executable(miniSplitObjectSpace miniSplitObjectSpace.cpp)
target_link_libraries(miniSplitObjectSpace miniScene)

add_executable(miniSetMaterial miniSetMaterial.cpp)
target_link_libraries(miniSetMaterial miniScene)

add_executable(swcMakeBinaries swcMakeBinaries.cpp)
target_link_libraries(swcMakeBinaries miniScene)

# NanoVDB merge tool
find_path(NANOVDB_INCLUDE_DIR 
  NAMES nanovdb/NanoVDB.h nanovdb/util/GridBuilder.h
  PATHS /usr/include /usr/local/include /opt/openvdb/include
)

if(NANOVDB_INCLUDE_DIR)
  add_executable(mergeNanoVDB mergeNanoVDB.cpp)
  target_include_directories(mergeNanoVDB PRIVATE ${NANOVDB_INCLUDE_DIR})
  target_compile_definitions(mergeNanoVDB PRIVATE -DHS_HAVE_NANOVDB=1)
  target_compile_features(mergeNanoVDB PRIVATE cxx_std_17)
  message(STATUS "Building mergeNanoVDB with NanoVDB at: ${NANOVDB_INCLUDE_DIR}")
else()
  message(STATUS "Skipping mergeNanoVDB (NanoVDB not found)")
endif()
