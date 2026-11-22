#pragma once

// base
#include <configure.h>

namespace snap {

#if NMASS > 0  // use legacy Athena++ indexing scheme

//! \brief Indexing scheme for hydro and reconstruction variables
//!
//! These enumerations define the array indices for accessing
//! hydrodynamic and reconstruction variables in legacy Athena++ format.
enum {
  // hydro variables
  IDN = 0,          //!< Density index
  ICY = 1,          //!< Composition/species index (first mass fraction)
  IVX = 1 + NMASS,  //!< X-velocity index
  IVY = 2 + NMASS,  //!< Y-velocity index
  IVZ = 3 + NMASS,  //!< Z-velocity index
  IPR = 4 + NMASS,  //!< Pressure index

  // reconstruction variables
  ILT = 0,  //!< Left interface
  IRT = 1,  //!< Right interface
};

#else  // use new indexing scheme

//! \brief Indexing scheme for hydro and reconstruction variables
//!
//! These enumerations define the array indices for accessing
//! hydrodynamic and reconstruction variables in new format.
enum {
  // hydro variables
  IDN = 0,  //!< Density index
  IVX = 1,  //!< X-velocity index
  IVY = 2,  //!< Y-velocity index
  IVZ = 3,  //!< Z-velocity index
  IPR = 4,  //!< Pressure index
  ICY = 5,  //!< Composition/species index (first mass fraction)

  // reconstruction variables
  ILT = 0,  //!< Left interface
  IRT = 1,  //!< Right interface
};

#endif  // index scheme

// Species start right after the hydro variables; the `size - ICY` species
// counts rely on it, so the legacy NMASS > 0 layout is refused here.
static_assert(
    ICY == IPR + 1,
    "species rows must start right after the 5 hydro rows; NMASS>0 is "
    "not supported");

//! \brief Variable type enumeration
//!
//! Defines different types of variables used in the simulation.
enum {
  // variable type
  kPrimitive = 0,  //!< Primitive variables (rho, v, P)
  kConserved = 1,  //!< Conserved variables (rho, rho*v, E)
  kScalar = 2,     //!< Scalar variables
};

//! \brief Velocity component enumeration
//!
//! Indices for accessing velocity components.
enum {
  VEL1 = 0,  //!< First velocity component
  VEL2 = 1,  //!< Second velocity component
  VEL3 = 2   //!< Third velocity component
};

}  // namespace snap
