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

//! The message tags of the within-stage exchanges, in one place so that no
//! two collide. Every exchange adds the receiver's side to tag: an x1 relay
//! (x1_rows, x1_seam_slab) has the sides 0 (below) and 1 (above), so its up
//! message carries tag and its down message tag + 1; a face-slab or panel
//! seam exchange adds the receiver's side L, R, B, T (0..3).
enum ExchangeTag : int {
  kTagWbRef = 0x7715,          // recon x1_reference: the reference anchor
  kTagWbGhost = 0x7717,        // recon x1_reference: its ghost rows
  kTagSeamFlux = 0x7720,       // hydro: x1 seam flux slab
  kTagFacePressure = 0x7722,   // hydro: x1 face-pressure ghost faces
  kTagPlainMeans = 0x7724,     // hydro: x1 plain-mean ghost rows
  kTagGravityWork = 0x7726,    // hydro: x1 gravity-work density rows
  kTagPrimGhosts = 0x772a,     // hydro: x1 primitive ghost layers
  kTagFaceX2 = 0x7730,         // hydro: x2 face states (x1_seam_rows)
  kTagFaceX3 = 0x7734,         // hydro: x3 face states (x1_seam_rows)
  kTagScalarCov = 0x7740,      // hydro: scalar covariance, + 4 for x3
  kTagPanelSeams = 0x7750,     // hydro: reconcile_panel_seams
  kTagDiffusionSeam = 0x7760,  // diffusion cubed sphere: row 18 face slabs
};

//! The transport of a block's within-stage exchanges (V3 lane S, step (3),
//! phase 1: transport only). Exchange moves the bytes; what a received value
//! means (a ghost row, the mean of a seam flux, a running anchor) stays with
//! the caller. Every relay, x1 column and x2/x3 seam alike, is a set of
//! SeamSides moved by seam_exchange: a same-process neighbour goes through a
//! board, a remote one through the process group. Holding an Exchange costs
//! one pointer.
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
  //! message carries tag, the down message tag + 1. A same-process peer goes
  //! through a board, a remote one through the process group. process_only:
  //! only seams to another process exchange.
  std::pair<torch::Tensor, torch::Tensor> x1_seam_slab(
      torch::Tensor const& lower, torch::Tensor const& upper, int tag,
      bool process_only) const;

  //! the x2/x3 face slabs (S step M6, S.rst row M6; row 18 and any face-slab
  //! row): a raw copy across both faces of direction dim (2: x2, 3: x3),
  //! panel seams and same-panel neighbours alike. lower and upper are this
  //! block's slabs next to its lower and upper face of dim, each [C, n, ni]:
  //! C channels, n cells along the edge (x3 on an x2 face, x2 on an x3 face,
  //! in this block's index order), ni cells of x1. Returns the neighbours'
  //! slabs for the same faces, {across the lower face, across the upper
  //! face}, in the same [C, n, ni] layout and in this block's along-edge
  //! order: a panel seam whose edge runs the other way (CS_FACE_EDGES rev)
  //! arrives reversed along n. No sign change and no rotation; a caller that
  //! needs them (row 16r) applies them itself. Undefined where there is no
  //! peer. The message to a peer carries tag + the peer's side (L, R, B, T =
  //! 0..3). A same-process peer goes through a board, a remote one through
  //! the process group; process_only: only peers in another process.
  //! A cubed-sphere panel seam can join an x2 face to an x3 face
  //! (CS_FACE_EDGES: 1B-5R, 1T-3R, 3L-4T, 4B-5L), whose slab then travels in
  //! the peer's call for the other direction: a row calls dim 2, then dim 3,
  //! on every block, with one tag. That order completes: panels 0, 1, 2 and 4
  //! finish their x2 call on x2 posts alone; 3 and 5 wait in it for the x3
  //! posts of 1 and 4, made as their x3 call starts; 0 and 2 wait in their x3
  //! call for 5 and 3.
  std::pair<torch::Tensor, torch::Tensor> seam_slab(int dim,
                                                    torch::Tensor const& lower,
                                                    torch::Tensor const& upper,
                                                    int tag,
                                                    bool process_only) const;

  //! one side of a seam exchange: side (L, R, B, T = 0..3) of dim 2 or 3, or
  //! side 0 (below) or 1 (above) of the x1 column, dim 1; resolve_seam fills
  //! the peer block, the peer's side and whether the edge runs the other way
  //! (CS_FACE_EDGES rev, panel seams only; the caller flips along n);
  //! seam_exchange posts `send` when defined and, when `take`, receives the
  //! peer's into `recv` (a remote peer: the caller's fixed buffer, or one
  //! allocated like `send`) or hands over the peer's posted tensor (a
  //! same-process peer: the same tensor at every replay)
  struct SeamSide {
    int dim, side;
    torch::Tensor send, recv;
    bool take = true;
    int peer = -1, peer_side = -1;
    bool reversed = false;
  };
  //! the peer of a side: false when it has none (process_only: a
  //! same-process peer counts as none)
  bool resolve_seam(SeamSide& s, bool process_only) const;
  //! the host part of every seam exchange: post every send (a board or the
  //! process group, tag + the peer's side), then take every receive (tag +
  //! this side); all posts before any take, so a cross-direction panel seam
  //! completes in one call
  void seam_exchange(std::vector<SeamSide>& sides, int tag) const;

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
