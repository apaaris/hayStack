// ======================================================================== //
// Merge distributed NanoVDB files into a single grid
// Usage: ./mergeNanoVDB <input_dir> <output_file> [--timestep TIMESTEP]
// ======================================================================== //

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cstring>

#ifdef HS_HAVE_NANOVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridBuilder.h>
#endif

namespace fs = std::filesystem;

struct Options {
  std::string inputDir;
  std::string outputFile;
  std::string timestep;
  std::string field;
  bool verbose = false;
};

void printUsage(const char* prog) {
  std::cout << "Usage: " << prog << " <input_dir> <output_file> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --timestep <TS>   Only merge files from specific timestep\n";
  std::cout << "  --field <name>    Only merge files for specific field\n";
  std::cout << "  --verbose, -v     Print detailed progress\n";
  std::cout << "\nExample:\n";
  std::cout << "  " << prog << " /data/nvdb merged_004320.nvdb --timestep 004320\n";
}

Options parseArgs(int argc, char** argv) {
  Options opts;
  
  if (argc < 3) {
    printUsage(argv[0]);
    exit(1);
  }
  
  opts.inputDir = argv[1];
  opts.outputFile = argv[2];
  
  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--timestep" && i + 1 < argc) {
      opts.timestep = argv[++i];
    } else if (arg == "--field" && i + 1 < argc) {
      opts.field = argv[++i];
    } else if (arg == "--verbose" || arg == "-v") {
      opts.verbose = true;
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      printUsage(argv[0]);
      exit(1);
    }
  }
  
  return opts;
}

std::vector<std::string> findNVDBFiles(const Options& opts) {
  std::vector<std::string> files;
  
  try {
    for (const auto& entry : fs::directory_iterator(opts.inputDir)) {
      if (entry.path().extension() != ".nvdb") continue;
      
      std::string filename = entry.path().filename().string();
      
      // Filter by timestep
      if (!opts.timestep.empty()) {
        if (filename.find("_" + opts.timestep + ".") == std::string::npos) {
          continue;
        }
      }
      
      // Filter by field
      if (!opts.field.empty()) {
        if (filename.find("nvdb_" + opts.field + "_") == std::string::npos) {
          continue;
        }
      }
      
      files.push_back(entry.path().string());
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading directory: " << e.what() << std::endl;
    exit(1);
  }
  
  std::sort(files.begin(), files.end());
  return files;
}

#ifdef HS_HAVE_NANOVDB

void mergeGrids(const Options& opts) {
  auto files = findNVDBFiles(opts);
  
  if (files.empty()) {
    std::cerr << "ERROR: No .nvdb files found in " << opts.inputDir << std::endl;
    exit(1);
  }
  
  std::cout << "Found " << files.size() << " NanoVDB files to merge" << std::endl;
  
  if (opts.verbose) {
    for (size_t i = 0; i < std::min(size_t(5), files.size()); i++) {
      std::cout << "  - " << fs::path(files[i]).filename().string() << std::endl;
    }
    if (files.size() > 5) {
      std::cout << "  ... and " << (files.size() - 5) << " more" << std::endl;
    }
  }
  
  std::cout << "\nMerging grids..." << std::endl;
  
  // Create a new grid builder for the merged result
  nanovdb::GridBuilder<float> mergedBuilder(0.0f);
  auto mergedAccessor = mergedBuilder.getAccessor();
  
  size_t totalActiveVoxels = 0;
  double voxelSize = 0.075;  // Will be read from first grid
  nanovdb::Vec3d worldOrigin(0, 0, 0);
  
  // Read and merge each grid
  for (size_t i = 0; i < files.size(); i++) {
    try {
      // Read grid from file
      auto handle = nanovdb::io::readGrid(files[i], "density");
      if (!handle) {
        std::cerr << "  WARNING: Failed to read " << fs::path(files[i]).filename().string() << std::endl;
        continue;
      }
      
      auto* grid = handle.grid<float>();
      if (!grid) {
        std::cerr << "  WARNING: Invalid grid in " << fs::path(files[i]).filename().string() << std::endl;
        continue;
      }
      
      // Get voxel size and origin from first grid
      if (i == 0) {
        voxelSize = grid->voxelSize()[0];
        worldOrigin = grid->worldBBox().min();
      }
      
      // Iterate over active voxels and copy to merged grid
      auto accessor = grid->getAccessor();
      size_t blockActiveCount = 0;
      
      // Get the bounding box and iterate over all coords
      auto bbox = grid->indexBBox();
      for (int z = bbox.min()[2]; z <= bbox.max()[2]; z++) {
        for (int y = bbox.min()[1]; y <= bbox.max()[1]; y++) {
          for (int x = bbox.min()[0]; x <= bbox.max()[0]; x++) {
            nanovdb::Coord ijk(x, y, z);
            if (grid->tree().isActive(ijk)) {
              float value = accessor.getValue(ijk);
              mergedAccessor.setValue(ijk, value);
              blockActiveCount++;
            }
          }
        }
      }
      
      totalActiveVoxels += blockActiveCount;
      
      if (opts.verbose && (i + 1) % 10 == 0) {
        std::cout << "  Merged " << (i + 1) << "/" << files.size() 
                  << " grids (" << totalActiveVoxels << " active voxels)" << std::endl;
      }
      
    } catch (const std::exception& e) {
      std::cerr << "  ERROR processing " << fs::path(files[i]).filename().string() 
                << ": " << e.what() << std::endl;
      continue;
    }
  }
  
  std::cout << "\nBuilding final merged grid..." << std::endl;
  auto finalHandle = mergedBuilder.getHandle<>(voxelSize, worldOrigin, "density");
  auto* finalGrid = finalHandle.grid<float>();
  
  std::cout << "Merged grid statistics:" << std::endl;
  std::cout << "  Active voxels: " << finalGrid->activeVoxelCount() << std::endl;
  
  auto idxBBox = finalGrid->indexBBox();
  std::cout << "  Index bbox: [" << idxBBox.min()[0] << "," << idxBBox.min()[1] << "," << idxBBox.min()[2] 
            << "] to [" << idxBBox.max()[0] << "," << idxBBox.max()[1] << "," << idxBBox.max()[2] << "]" << std::endl;
  
  auto wBBox = finalGrid->worldBBox();
  std::cout << "  World bbox: [" << wBBox.min()[0] << "," << wBBox.min()[1] << "," << wBBox.min()[2] 
            << "] to [" << wBBox.max()[0] << "," << wBBox.max()[1] << "," << wBBox.max()[2] << "]" << std::endl;
  
  // Write merged grid
  std::cout << "\nWriting merged grid to: " << opts.outputFile << std::endl;
  nanovdb::io::writeGrid(opts.outputFile, finalHandle);
  
  size_t fileSize = fs::file_size(opts.outputFile);
  std::cout << "  Output file size: " << (fileSize / (1024.0 * 1024.0)) << " MB" << std::endl;
  std::cout << "✅ Merge complete!" << std::endl;
}

#else

void mergeGrids(const Options& opts) {
  std::cerr << "ERROR: NanoVDB support not compiled in" << std::endl;
  std::cerr << "Install OpenVDB 8.0+ and reconfigure CMake with -DHS_HAVE_NANOVDB=ON" << std::endl;
  exit(1);
}

#endif

int main(int argc, char** argv) {
  Options opts = parseArgs(argc, argv);
  
  if (!fs::exists(opts.inputDir)) {
    std::cerr << "ERROR: Input directory not found: " << opts.inputDir << std::endl;
    return 1;
  }
  
  // Ensure output directory exists
  fs::path outPath(opts.outputFile);
  if (outPath.has_parent_path()) {
    fs::create_directories(outPath.parent_path());
  }
  
  mergeGrids(opts);
  
  return 0;
}

