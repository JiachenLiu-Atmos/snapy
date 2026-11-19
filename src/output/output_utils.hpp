#pragma once

// C/C++
#include <string>
#include <vector>

// base
#include <configure.h>

namespace snap {

class MetadataTable {
 protected:
  using StringTable = std::vector<std::vector<std::string>>;

  //! Protected ctor access thru static member function Instance
  MetadataTable();

 public:
  ~MetadataTable();

  static MetadataTable const* GetInstance();
  static void Destroy();

  std::string GetGridType(std::string name) const;
  std::string GetUnits(std::string name) const;
  std::string GetLongName(std::string name) const;

 private:
  StringTable table_;

  //! Pointer to the single MetadataTable instance
  static MetadataTable* myptr_;
};

//! Units of the x2/x3 axes: the angles of the curvilinear grids are in
//! radians, the cartesian axes are lengths.
inline std::string horizontal_axis_units(std::string const& coord_type) {
  return coord_type == "gnomonic-equiangle" || coord_type == "spherical-polar"
             ? "radian"
             : "m";
}

class MeshBlockImpl;
std::string get_hydro_names(MeshBlockImpl* pmb, std::string prepend = "");

}  // namespace snap
