#pragma once

// base
#include <configure.h>

// snap
#include <snap/coord/coord_utils_impl.h>
#include <snap/snap.h>

#include "sed_hydro_dispatch.hpp"

namespace snap {

//! The kernel's arrays, x1 last: w, the EOS temperature and species energy
//! (rho y e, rows ICY..), cosine_cell_kj, the particles' radius, density,
//! const_vsed and hydro rows, and the flux and vsed it adds into
template <typename T>
struct SedArrays {
  T const *w, *temp, *energy, *cosine, *radius, *density, *const_vsed;
  T *flux, *vsed;
  int64_t const* ids;
  int np, nc3, nc2, nc1;
};

//! The view of the kernel's tensors (host side)
template <typename T>
SedArrays<T> sed_arrays(torch::Tensor w, torch::Tensor temp,
                        torch::Tensor energy, torch::Tensor flux,
                        torch::Tensor vsed, torch::Tensor cos,
                        torch::Tensor radius, torch::Tensor density,
                        torch::Tensor const_vsed, torch::Tensor ids) {
  return {w.data_ptr<T>(),          temp.data_ptr<T>(),   energy.data_ptr<T>(),
          cos.data_ptr<T>(),        radius.data_ptr<T>(), density.data_ptr<T>(),
          const_vsed.data_ptr<T>(), flux.data_ptr<T>(),   vsed.data_ptr<T>(),
          ids.data_ptr<int64_t>(),  int(radius.size(0)),  int(w.size(1)),
          int(w.size(2)),           int(w.size(3))};
}

//! Adds to x1 face `face` the flux leaving cell `cell` of the particles moving
//! toward it, rising (vsed > 0) if `rising`, else settling, and their velocity
//! into vsed
template <typename T>
DISPATCH_MACRO void sedimentation_donor_impl(SedArrays<T> const& a,
                                             SedParams const& p, int cell,
                                             int face, bool rising) {
  int n = a.nc1 * a.nc2 * a.nc3, j = (cell / a.nc1) % a.nc2;
  T pi = T(p.pi), kboltz = T(p.kboltz), gas_mass = T(p.gas_mass);
  T rho = a.w[IDN * n + cell], pres = a.w[IPR * n + cell], tem = a.temp[cell];
  T v1 = a.w[IVX * n + cell], v2 = a.w[IVY * n + cell],
    v3 = a.w[IVZ * n + cell];
  coord_vec_lower_impl(&v2, &v3, a.cosine[cell / (a.nc1 * a.nc2) * a.nc2 + j]);
  // eta and lambda: on the Stokes branch only, once per cell; with drive_sign
  // a Stokes vsed has the sign of its drive term (or is NaN)
  T eta = T(0), lambda = T(0), upper = T(p.upper_limit);
  bool have_eta = false, drive_sign = pres > T(0) && tem > T(0) && !(upper < 0);
  for (int q = 0; q < a.np; ++q) {
    int id = static_cast<int>(a.ids[q]);
    // a non-zero const_vsed is the whole (signed) velocity, used instead of
    // Stokes; else Stokes, with the SIGNED grav: dense particles settle to -x1
    T vsed = a.const_vsed[q];
    if (vsed == T(0)) {
      T r = a.radius[q],
        drive = T(2) * r * r * T(p.grav) * (a.density[q] - rho);
      // skip a particle moving away from this face before evaluating its vsed
      if (drive_sign && r > 0 && (rising ? !(drive > 0) : !(drive < 0)))
        continue;
      if (!have_eta) {
        T d = T(p.gas_diameter);
        eta = (T(5) / T(16)) * sqrt(pi * kboltz) * sqrt(gas_mass) * sqrt(tem) *
              pow(kboltz / T(p.gas_epsilon_lj) * tem, T(0.16)) /
              (pi * d * d * T(1.22));
        lambda =
            (eta * sqrt(pi * kboltz * tem)) / (pres * sqrt(T(2) * gas_mass));
        have_eta = true;
      }
      T kn = lambda / r;
      T beta = T(1) + kn * (T(1.256) + T(0.4) * exp(-T(1.1) / kn));
      vsed = beta / (T(9) * eta) * drive;
    }
    if (vsed < -upper) vsed = -upper;
    if (vsed > upper) vsed = upper;
    if (rising ? !(vsed > T(0)) : !(vsed < T(0))) continue;
    a.vsed[q * n + face] += vsed;
    T rhos_vsed = rho * a.w[id * n + cell] * vsed;
    a.flux[id * n + face] += rhos_vsed;
    a.flux[IVX * n + face] += v1 * rhos_vsed;
    a.flux[IVY * n + face] += v2 * rhos_vsed;
    a.flux[IVZ * n + face] += v3 * rhos_vsed;
    a.flux[IPR * n + face] += vsed * a.energy[(id - ICY) * n + cell];
  }
}

//! The flux at x1 face `flat` (between cells i-1 and i) from each particle's
//! upwind cell: i if it settles, i-1 if it rises. Faces at or below il and
//! above iu are sealed.
template <typename T>
DISPATCH_MACRO void sedimentation_flux_impl(SedArrays<T> const& a,
                                            SedParams const& p, int flat) {
  int i = flat % a.nc1;
  for (int q = 0; q < a.np; ++q) a.vsed[q * a.nc1 * a.nc2 * a.nc3 + flat] = 0;
  if (i <= p.il || i > p.iu) return;
  for (int up = 0; up < (i > 0 ? 2 : 1); ++up)
    sedimentation_donor_impl(a, p, flat - up, flat, up == 1);
}

}  // namespace snap
