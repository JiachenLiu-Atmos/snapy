#pragma once

// C/C++
#include <map>
#include <string>

// snap
#include "cubed_sphere_constants.h"
#include "layout.hpp"

namespace snap {

extern const char CS_FACE_NAMES[6][3];
extern const CSEdge CS_FACE_EDGES[6][4];
extern const CSVel CS_CART_TO_LOCAL_VEL[6][3];
extern const CSVel CS_LOCAL_TO_CART_VEL[6][3];
//! Given CS_CART_TO_LOCAL_VEL, populate its inverse (CS_LOCAL_TO_CART_VEL)
void populate_cs_l2g_vel(CSVel l2g[6][3]);

class CubedSphereLayoutImpl : public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  CubedSphereLayoutImpl() = default;
  CubedSphereLayoutImpl(const LayoutOptions& opts,
                        MeshBlockImpl* owner = nullptr)
      : LayoutImpl(opts, owner) {
    options->type("cubed-sphere");
    _initialize();
  }

  ~CubedSphereLayoutImpl() = default;
  void pretty_print(std::ostream& os) const;

  int pxy() const { return options->px(); }

  int rank_of(std::tuple<int, int, int> iloc) const override;
  std::tuple<int, int, int> loc_of(int global_rank) const override;

  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;

  void serialize(MeshBlockImpl const* pmb, Variables& vars,
                 SyncOptions const& opts) override;

  void deserialize(MeshBlockImpl const* pmb, Variables& vars,
                   SyncOptions const& opts) const override;

  int num_exchange_buffers() const override { return 9; }

 private:
  //! Constructor-time setup that replaces the old torch-style reset() hook.
  void _initialize();
  std::tuple<int, int, int> _remap_exchange_offset(
      std::tuple<int, int, int> iloc, int dy, int dx,
      int dz = 0) const override;
  std::tuple<int, int, int> _peer_exchange_offset(
      int peer_rank, int target_rank, SyncOptions const& opts,
      std::tuple<int, int, int> offset) const override;
  bool _skips_direction(MeshBlockImpl const* pmb,
                        std::tuple<int, int, int> offset) const override;

  //! \brief Interpolate transmitted variable to local ghost zones
  void _interpolate_to_local(MeshBlockImpl const* pmb,
                             std::tuple<int, int, int> offset,
                             torch::Tensor var) const;

  torch::Tensor const& _velocity_transform(SyncOptions const& opts,
                                           int buffer_id, bool to_spherical,
                                           torch::Tensor alpha,
                                           torch::Tensor beta,
                                           torch::Tensor cosine,
                                           torch::Tensor const& velocity) const;

  mutable std::map<std::string, torch::Tensor> _velocity_transform_cache;

  //! \brief Global rank layout: face-major, Z-order within face
  int _global_rank_from_face_local(int face, int r_local) const {
    int P = pxy() * pxy();
    return face * P + r_local;
  }

  //! \brief Reverse: get (face, r_local) from global rank */
  void _global_rank_to_face_local(int grank, int* face, int* r_local) const {
    int P = pxy() * pxy();
    *face = grank / P;
    *r_local = grank % P;
  }

  //! \brief map local (rx,ry) to per-face Z-order rank */
  int _face_local_rank(int rx, int ry) const {
    return _rankof[linear_index2(pxy(), pxy(), ry, rx)];
  }

  //! \brief Edge stepping helper
  /*!
   * Move off the face by one tile in (dx,dy) \in {-1,0,1}^2.
   * Returns neighbor (nface, nrank) or (-1, -1) on error (should not happen on
   * a closed cube).
   *
   * Logic:
   * - If inside same face: trivial offset of (rx,ry).
   * - If crossing a single edge (|dx|+|dy|==1): use edge table to decide
   *    neighbor face & side, compute the along-edge index (pos), reverse if
   *    needed, and place at neighbor border.
   * - If crossing a corner (|dx|==1 && |dy|==1): do it in two hops.
   *    (dx,0) and (0,dy) through the intermediate face.
   *    If across a panel boundary, do first step inside the panel
   *    and second step outside. This mirrors typical ghost-corner
   *    exchange.
   */
  void _step_one(int face, int rx, int ry, int dx, int dy, int* out_face,
                 int* out_rx, int* out_ry) const;
};

}  // namespace snap
