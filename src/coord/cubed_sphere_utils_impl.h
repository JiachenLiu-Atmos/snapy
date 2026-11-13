#pragma once

namespace snap {

template <typename T>
void cs_lonlat_to_ab(int *nP, T *alpha, T *beta, T lon, T lat) {
  // Translate from RLL coordinates to XYZ space
  T xx, yy, zz, pm;

  xx = cos(lon) * cos(lat);
  yy = sin(lon) * cos(lat);
  zz = sin(lat);

  pm = std::max(fabs(xx), std::max(fabs(yy), fabs(zz)));

  // Check maxmality of the x coordinate
  if (pm == fabs(xx)) {
    if (xx > 0) {  // +X
      (*nP) = 0;
    } else {  // -X
      (*nP) = 2;
    }
  }

  // Check maximality of the y coordinate
  if (pm == fabs(yy)) {
    if (yy > 0) {  // +Y
      (*nP) = 1;
    } else {  // -Y
      (*nP) = 4;
    }
  }

  // Check maximality of the z coordinate
  if (pm == fabs(zz)) {
    if (zz > 0) {  // +Z
      (*nP) = 3;
    } else {  // -Z
      (*nP) = 5;
    }
  }

  // Panel assignments
  double sx, sy, sz;
  switch ((*nP)) {
    case 0:
      sx = yy;
      sy = zz;
      sz = xx;
      break;
    case 1:
      sx = -xx;
      sy = zz;
      sz = yy;
      break;
    case 2:
      sx = -yy;
      sy = zz;
      sz = -xx;
      break;
    case 3:
      sx = yy;
      sy = -xx;
      sz = zz;
      break;
    case 4:
      sx = xx;
      sy = zz;
      sz = -yy;
      break;
    case 5:
      sx = yy;
      sy = xx;
      sz = -zz;
      break;
  }

  // Convert to gnomonic coordinates
  (*alpha) = atan2(sx, sz);
  (*beta) = atan2(sy, sz);
}

}  // namespace snap
