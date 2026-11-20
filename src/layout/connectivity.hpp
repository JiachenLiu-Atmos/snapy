#pragma once

// C/C++
#include <cstdint>
#include <cstdlib>

namespace snap {

/* ===========================
 * Bit de-interleaving helpers
 * ===========================
 * compact1by1:  remove every other bit (keep bits at positions 0,2,4,...)
 * compact1by2:  remove two other bits (keep bits at positions 0,3,6,...) for
 * 3-way interleave
 *
 * Based on "Bit Twiddling Hacks" (public domain).
 */

//! \brief De-interleave bits by keeping every other bit
//!
//! Extracts bits at positions 0, 2, 4, ... for decoding 2D Morton codes.
//!
//! \param[in] v Input value with interleaved bits
//! \return Value with every other bit extracted
//!
//! \note Based on "Bit Twiddling Hacks" (public domain)
inline int compact1by1(int v) {
  v &= 0x55555555u;
  v = (v | (v >> 1)) & 0x33333333u;
  v = (v | (v >> 2)) & 0x0F0F0F0Fu;
  v = (v | (v >> 4)) & 0x00FF00FFu;
  v = (v | (v >> 8)) & 0x0000FFFFu;
  return v;
}

/* For 3D Morton codes we need to extract every 3rd bit. Using 64-bit lanes. */
//! \brief De-interleave bits by keeping every third bit
//!
//! Extracts bits at positions 0, 3, 6, ... for decoding 3D Morton codes.
//!
//! \param[in] v Input 64-bit value with interleaved bits
//! \return Value with every third bit extracted
//!
//! \note Uses 64-bit arithmetic for 3D Morton code decoding
inline int compact1by2(uint64_t v) {
  v &= 0x1249249249249249ULL;  // 0b001001.. pattern
  v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ULL;
  v = (v ^ (v >> 4)) & 0x100f00f00f00f00fULL;
  v = (v ^ (v >> 8)) & 0x1f0000ff0000ffULL;
  v = (v ^ (v >> 16)) & 0x1f00000000ffffULL;
  v = (v ^ (v >> 32)) & 0x1fffffULL;
  return (int)v;
}

/* ===========================
 * Morton decode (Z-order)
 * =========================== */

/* Decode 2D Morton code -> (y,x). Note the order: (y,x). */
//! \brief Decode 2D Morton (Z-order) code to coordinates
//!
//! \param[in] code Morton code to decode
//! \param[out] y Y coordinate (first output)
//! \param[out] x X coordinate (second output)
//!
//! \note Output order is (y,x), not (x,y)
inline void morton_decode2(int code, int *y, int *x) {
  *x = compact1by1(code);
  *y = compact1by1(code >> 1);
}

/* Decode 3D Morton code -> (z,y,x). Uses 64-bit Morton codes. */
//! \brief Decode 3D Morton (Z-order) code to coordinates
//!
//! \param[in] code 64-bit Morton code to decode
//! \param[out] z Z coordinate (first output)
//! \param[out] y Y coordinate (second output)
//! \param[out] x X coordinate (third output)
//!
//! \note Uses 64-bit Morton codes for 3D decoding
//! \note Output order is (z,y,x)
inline void morton_decode3(uint64_t code, int *z, int *y, int *x) {
  *x = compact1by2(code);
  *y = compact1by2(code >> 1);
  *z = compact1by2(code >> 2);
}

/* ===========================
 * Coordinate containers
 * =========================== */

//! \brief 2D coordinate container
struct Coord2 {
  int y, x;  //!< Y and X coordinates
};

//! \brief 3D coordinate container
struct Coord3 {
  int z, y, x;  //!< Z, Y, and X coordinates
};

/* ================
 * Z-order builders
 * ================ */

/* Build Py×Px coordinates in Z-order. coords must have length >= py*px. */
//! \brief Build 2D coordinates in Z-order (Morton order)
//!
//! \param[in] px Number of tiles in X direction
//! \param[in] py Number of tiles in Y direction
//! \param[out] coords Output array of coordinates (must have length >= py*px)
//! \return Number of coordinates generated (py*px)
size_t build_zorder_coords2(int px, int py, Coord2 *coords);

/* Build Pz×Py×Px coordinates in Z-order. coords must have length >= pz*py*px.
 */
//! \brief Build 3D coordinates in Z-order (Morton order)
//!
//! \param[in] px Number of tiles in X direction
//! \param[in] py Number of tiles in Y direction
//! \param[in] pz Number of tiles in Z direction
//! \param[out] coords Output array of coordinates (must have length >=
//! pz*py*px)
//! \return Number of coordinates generated (pz*py*px)
size_t build_zorder_coords3(int px, int py, int pz, Coord3 *coords);

/* ======================
 * coords -> rank mapping
 * ======================
 * We build a dense array rank_of with shape:
 *  - 2D: [py][px]
 *  - 3D: [pz][py][px]
 * storing the Z-order rank at that coordinate.
 * Access via linear index helpers below.
 */

//! \brief Compute linear index for 2D coordinate
//!
//! \param[in] px Number of tiles in X direction
//! \param[in] py Number of tiles in Y direction (unused, for consistency)
//! \param[in] y Y coordinate
//! \param[in] x X coordinate
//! \return Linear index for accessing [y][x] in flattened array
inline size_t linear_index2(int px, int /*py*/, int y, int x) {
  return (size_t)y * (size_t)px + (size_t)x;
}

//! \brief Compute linear index for 3D coordinate
//!
//! \param[in] px Number of tiles in X direction
//! \param[in] py Number of tiles in Y direction
//! \param[in] z Z coordinate
//! \param[in] y Y coordinate
//! \param[in] x X coordinate
//! \return Linear index for accessing [z][y][x] in flattened array
inline size_t linear_index3(int px, int py, int z, int y, int x) {
  return ((size_t)z * (size_t)py + (size_t)y) * (size_t)px + (size_t)x;
}

/* rank_of2: array length py*px, filled with rank at (y,x) */
//! \brief Build rank mapping for 2D coordinates
//!
//! \param[in] px Number of tiles in X direction
//! \param[in] py Number of tiles in Y direction
//! \param[in] coords Array of coordinates in Z-order
//! \param[out] rank_of_out Output array (length py*px) storing rank at each
//! (y,x)
void build_rank_of2(int px, int py, const Coord2 *coords, int *rank_of_out);

/* rank_of3: array length pz*py*px, filled with rank at (z,y,x) */
//! \brief Build rank mapping for 3D coordinates
//!
//! \param[in] px Number of tiles in X direction
//! \param[in] py Number of tiles in Y direction
//! \param[in] pz Number of tiles in Z direction
//! \param[in] coords Array of coordinates in Z-order
//! \param[out] rank_of_out Output array (length pz*py*px) storing rank at each
//! (z,y,x)
void build_rank_of3(int px, int py, int pz, const Coord3 *coords,
                    int *rank_of_out);

/* Return logical location of a tile on a face in (-1,0,1) coding.
 *   (-1,0) = left edge, (1,0) = right edge, (0,-1) = bottom edge, (0,1) = top
 * edge
 *   (-1,-1) = bottom-left corner, etc.
 *   (0,0) = interior (not on any edge).
 */
//! \brief Determine logical location of a tile on a face
//!
//! Returns the logical location using (-1,0,1) coding:
//!   - (-1,0) = left edge, (1,0) = right edge
//!   - (0,-1) = bottom edge, (0,1) = top edge
//!   - (-1,-1) = bottom-left corner, etc.
//!   - (0,0) = interior (not on any edge)
//!
//! \param[in] rx Tile X coordinate
//! \param[in] ry Tile Y coordinate
//! \param[in] px Total number of tiles in X direction
//! \param[in] py Total number of tiles in Y direction
//! \param[out] lx Logical X location (-1, 0, or 1)
//! \param[out] ly Logical Y location (-1, 0, or 1)
static inline void logical_loc2(int rx, int ry, int px, int py, int *lx,
                                int *ly) {
  *lx = 0;
  *ly = 0;

  if (rx == 0)
    *lx = -1;
  else if (rx == px - 1)
    *lx = 1;

  if (ry == 0)
    *ly = -1;
  else if (ry == py - 1)
    *ly = 1;
}

}  // namespace snap
