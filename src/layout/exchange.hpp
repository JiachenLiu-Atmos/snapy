#pragma once

// C/C++
#include <cstdint>
#include <utility>
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

class MeshBlockImpl;

//! what a block face is (V3 S step (2); S.rst rev 2.4 section 3.1): one
//! value per face, answered in one place
enum class FaceKind : uint8_t {
  interior,  // a neighbour block of the same panel, or no neighbour and no
             // boundary function (nothing writes the ghosts but the exchange)
  seam_x1,   // an x1 neighbour block (pz > 1)
  panel,     // a cubed-sphere panel seam, rotated
  periodic,  // a seam to self: the layout wraps, or a periodic bc copies
  wall,      // reflecting or fixed temperature: a mirrored or set ghost
  outflow,   // radiating
  custom,    // any other boundary function, or one with no recorded name
};

//! The x1 transport of a block's within-stage exchanges (V3 lane S, step (3),
//! phase 1: transport only). Exchange moves the bytes of the hand-written x1
//! relays that hydro used to carry itself; what a received value means (a
//! ghost row, the mean of a seam flux, a running anchor) stays with the
//! caller. A same-process neighbour goes through a board, a remote one through
//! the process group, as before. Holding an Exchange costs one pointer.
class Exchange {
 public:
  explicit Exchange(MeshBlockImpl const* pmb) : pmb_(pmb) {}

  //! the kind of block face `face`, numbered as bfuncs(): x1 inner, x1
  //! outer, x2 inner, x2 outer, x3 inner, x3 outer
  FaceKind kind(int face) const;

  //! the x1 neighbours across a split column, {below, above}; -1 when the
  //! column is not split (one x1 block, or periodic x1)
  std::pair<int, int> x1_neighbors() const;

  //! x1 ghost rows (rows 4, 5, 8, 10, 13-16, 19, 22-25 of the stage
  //! schedule): overwrite the m ghost rows (faces: ghost faces, past the
  //! shared seam face) of each field with the x1 neighbours' interior rows.
  //! Several fields travel in one message per direction, concatenated along
  //! the last dimension in the order given; they must share their leading
  //! dimensions. The up message carries tag, the down message tag + 1.
  void x1_rows(std::vector<torch::Tensor> fields, int m, bool faces,
               int tag) const;

  //! the peers of the x1 seam slab exchange, {below, above}: the layout's x1
  //! neighbours (a periodic x1 wrap included) when pz > 1; process_only drops
  //! a neighbour in this process. -1: no peer
  std::pair<int, int> x1_seam_peers(bool process_only) const;

  //! the x1 seam slab (row 9, transport only): send `lower` (this block's
  //! lowest x1 face) to the block below and `upper` (its highest face) to the
  //! block above, and return the neighbours' slabs for the same faces,
  //! {from below, from above}; undefined where there is no peer. The up
  //! message carries tag, the down message tag + 1. process_only: only seams
  //! to another process exchange (the rule at main; same-process seams do
  //! not).
  std::pair<torch::Tensor, torch::Tensor> x1_seam_slab(
      torch::Tensor const& lower, torch::Tensor const& upper, int tag,
      bool process_only) const;

  //! the x1 reference anchor relay down the column (row 6): the running
  //! face value from the block above; undefined when the column is not
  //! split or this block is its top
  torch::Tensor unpack_anchor(torch::Tensor const& like, int tag) const;
  //! row 7: hand this block's bottom-face value to the block below
  void pack_anchor(torch::Tensor const& face, int tag) const;

 private:
  MeshBlockImpl const* pmb_;
};

}  // namespace snap
