// external
#include <gtest/gtest.h>

// snap
#include <snap/output/output_utils.hpp>
#include <snap/output/writer_core.hpp>

TEST(NetcdfUtils, keeps_valid_names) {
  EXPECT_EQ(snap::sanitize_netcdf_name("rho"), "rho");
  EXPECT_EQ(snap::sanitize_netcdf_name("theta_v"), "theta_v");
  EXPECT_EQ(snap::sanitize_netcdf_name("path_H2O"), "path_H2O");
}

TEST(NetcdfUtils, sanitizes_species_names) {
  EXPECT_EQ(snap::sanitize_netcdf_name("H2O(l)"), "H2O_l_");
  EXPECT_EQ(snap::sanitize_netcdf_name("H2O(l,p)"), "H2O_l_p_");
  EXPECT_EQ(snap::sanitize_netcdf_name("path_NH4SH(s,p)"), "path_NH4SH_s_p_");
}

TEST(NetcdfUtils, prefixes_invalid_leading_character) {
  EXPECT_EQ(snap::sanitize_netcdf_name("1bad"), "_1bad");
  EXPECT_EQ(snap::sanitize_netcdf_name(""), "_");
}

TEST(NetcdfUtils, flux_suffix_is_radial_face_centered) {
  auto table = snap::MetadataTable::GetInstance();
  EXPECT_EQ(table->GetGridType("net_flux"), "CCF");
  EXPECT_EQ(table->GetGridType("mass_flux"), "CCF");
  EXPECT_EQ(table->GetGridType("_flux"), "CCF");
  EXPECT_EQ(table->GetGridType("net_flux_mean"), "CCF");
  EXPECT_EQ(table->GetGridType("net_flux_std"), "CCF");
  EXPECT_EQ(table->GetGridType("path_flux"), "CCF");
  EXPECT_EQ(table->GetGridType("avg_flux"), "CCF");
  EXPECT_EQ(table->GetGridType(""), "");
  EXPECT_EQ(table->GetGridType("flux"), "");
  EXPECT_EQ(table->GetGridType("net_flux_extra"), "");
  EXPECT_EQ(table->GetGridType("path_species"), "-CC");
  EXPECT_EQ(table->GetGridType("avg_temp"), "--C");
  EXPECT_EQ(table->GetGridType("rho"), "CCC");
}

TEST(NetcdfUtils, flux_suffix_has_energy_flux_units) {
  auto table = snap::MetadataTable::GetInstance();
  EXPECT_EQ(table->GetUnits("net_flux"), "W/m^2");
  EXPECT_EQ(table->GetUnits("stellar_flux"), "W/m^2");
  EXPECT_EQ(table->GetUnits("_flux"), "W/m^2");
  EXPECT_EQ(table->GetUnits("net_flux_mean"), "W/m^2");
  EXPECT_EQ(table->GetUnits("net_flux_std"), "W/m^2");
  EXPECT_EQ(table->GetUnits("path_flux"), "W/m^2");
  EXPECT_EQ(table->GetUnits(""), "");
  EXPECT_EQ(table->GetUnits("flux"), "");
  EXPECT_EQ(table->GetUnits("net_flux_extra"), "");
  EXPECT_EQ(table->GetUnits("path_species"), "kg/m^2");
  EXPECT_EQ(table->GetUnits("rho"), "kg/m^3");
}
