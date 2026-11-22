#pragma once

// Eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include <snap/snap.h>
#include <snap/utils/sqr.h>

namespace snap {

template <typename T>
T DISPATCH_MACRO SoundSpeed(T* prim, T gm1) {
  return sqrt(prim[IPR] * (gm1 + 1.) / prim[IDN]);
}

template <typename T>
void DISPATCH_MACRO CopyPrimitives(T* wl, T* wr, T* prim, int i, int stride1,
                                   int stride2, int ny) {
  (void)ny;
  wl[IDN] = prim[(i - 1) * stride2];
  wr[IDN] = prim[i * stride2];

  for (int n = IVX; n <= IPR; ++n) {
    wl[n] = prim[n * stride1 + (i - 1) * stride2];
    wr[n] = prim[n * stride1 + i * stride2];
  }

  // EOS primitive IDN is total mixture density. The species entries are mass
  // fractions of that total and must not be added to the density again.
}

template <typename T>
void DISPATCH_MACRO RoeAverage(T* prim, T gm1, T* wl, T* wr) {
  T sqrtdl = sqrt(wl[IDN]);
  T sqrtdr = sqrt(wr[IDN]);
  T isdlpdr = 1.0 / (sqrtdl + sqrtdr);

  // Roe average scheme
  // Flux in the interface between i-th and i+1-th cells:
  // A(i+1/2) = [sqrt(rho(i))*A(i) + sqrt(rho(i+1))*A(i+1)]/(sqrt(rho(i)) +
  // sqrt(rho(i+1)))

  prim[IDN] = sqrtdl * sqrtdr;
  prim[IVX] = (sqrtdl * wl[IVX] + sqrtdr * wr[IVX]) * isdlpdr;
  prim[IVY] = (sqrtdl * wl[IVY] + sqrtdr * wr[IVY]) * isdlpdr;
  prim[IVZ] = (sqrtdl * wl[IVZ] + sqrtdr * wr[IVZ]) * isdlpdr;

  // Etot of the left side.
  T el = wl[IPR] / gm1 +
         0.5 * wl[IDN] * (sqr(wl[IVX]) + sqr(wl[IVY]) + sqr(wl[IVZ]));

  // Etot of the right side.
  T er = wr[IPR] / gm1 +
         0.5 * wr[IDN] * (sqr(wr[IVX]) + sqr(wr[IVY]) + sqr(wr[IVZ]));

  // Enthalpy divided by the density.
  T hbar = ((el + wl[IPR]) / sqrtdl + (er + wr[IPR]) / sqrtdr) * isdlpdr;

  // Roe averaged pressure
  prim[IPR] =
      (hbar - 0.5 * (sqr(prim[IVX]) + sqr(prim[IVY]) + sqr(prim[IVZ]))) * gm1 /
      (gm1 + 1.) * prim[IDN];
}

template <typename T>
inline void DISPATCH_MACRO Eigenvalue(Eigen::Matrix<T, 5, 1>& Lambda, T u,
                                      T cs) {
  Lambda << fabs(u - cs), fabs(u), fabs(u + cs), fabs(u), fabs(u);
}

template <typename T>
void DISPATCH_MACRO Eigenvector(Eigen::Matrix<T, 5, 5>& Rmat,
                                Eigen::Matrix<T, 5, 5>& Rimat, T* prim, T cs,
                                T gm1, int dir) {
  T r = prim[IDN];
  T u = prim[IVX + dir];
  T v = prim[IVX + (IVY - IVX + dir) % 3];
  T w = prim[IVX + (IVZ - IVX + dir) % 3];
  T p = prim[IPR];

  T ke = 0.5 * (u * u + v * v + w * w);
  T hp = (gm1 + 1.) / gm1 * p / r;
  T h = hp + ke;

  Rmat << 1., 1., 1., 0., 0.,     //
      u - cs, u, u + cs, 0., 0.,  //
      v, v, v, 1., 0.,            //
      w, w, w, 0., 1.,            //
      h - u * cs, ke, h + u * cs, v, w;

  Rimat << (cs * ke + u * hp) / (2. * cs * hp),
      (-hp - cs * u) / (2. * cs * hp),  //
      -v / (2. * hp), -w / (2. * hp), 1. / (2. * hp), (hp - ke) / hp,
      u / hp,                                                          //
      v / hp, w / hp, -1. / hp, (cs * ke - u * hp) / (2. * cs * hp),   //
      (hp - cs * u) / (2. * cs * hp), -v / (2. * hp), -w / (2. * hp),  //
      1. / (2. * hp), -v, 0., 1., 0., 0., -w, 0., 0., 1., 0.;
}

template <typename T>
void DISPATCH_MACRO FluxJacobian(Eigen::Matrix<T, 5, 5>& dfdq, T gm1, T* w,
                                 int dir) {
  // flux derivative
  // Input variables are density, velocity field and energy.
  // The primitives of cell (n,i)
  T v1 = w[IVX + dir];
  T v2 = w[IVX + (IVY - IVX + dir) % 3];
  T v3 = w[IVX + (IVZ - IVX + dir) % 3];
  T rho = w[IDN];
  T pres = w[IPR];
  T s2 = v1 * v1 + v2 * v2 + v3 * v3;

  T c1 = ((gm1 - 1) * s2 / 2 - (gm1 + 1) / gm1 * pres / rho) * v1;
  T c2 = (gm1 + 1) / gm1 * pres / rho + s2 / 2 - gm1 * v1 * v1;

  dfdq << 0, 1., 0., 0., 0.,                                               //
      gm1 * s2 / 2 - v1 * v1, (2. - gm1) * v1, -gm1 * v2, -gm1 * v3, gm1,  //
      -v1 * v2, v2, v1, 0., 0.,                                            //
      -v1 * v3, v3, 0., v1, 0., c1,                                        //
      c2, -gm1 * v2 * v1, -gm1 * v3 * v1, (gm1 + 1) * v1;
}

}  // namespace snap
