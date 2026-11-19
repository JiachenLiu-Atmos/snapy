#include "x1_reference.hpp"

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/coord/x1_centroid.hpp>
#include <snap/hydro/hydro_reference.hpp>
#include <snap/hydro/wb_ref4.hpp>
#include <snap/layout/exchange.hpp>
#include <snap/mesh/precision_policy.hpp>

#include "reconstruct.hpp"

namespace snap {

// the relay tags of the reference anchor and its ghost rows (down is +1)
constexpr int kWbRefTag = 0x7715;
constexpr int kWbGhostTag = 0x7717;

ExchangeX1Link::ExchangeX1Link(MeshBlockImpl const* pmb) : pmb_(pmb) {
  std::tie(below_, above_) = Exchange(pmb).x1_neighbors();
}

torch::Tensor ExchangeX1Link::take_anchor(torch::Tensor const& like) {
  return Exchange(pmb_).unpack_anchor(like, kWbRefTag);
}

void ExchangeX1Link::pass_anchor(torch::Tensor const& face) {
  Exchange(pmb_).pack_anchor(face, kWbRefTag);
}

void ExchangeX1Link::exchange_ghost_rows(std::vector<torch::Tensor> rows,
                                         int ng) {
  Exchange(pmb_).x1_rows(rows, ng, false, kWbGhostTag);
}

ReferenceFaces X1Reference::evaluate(X1ReferenceConfig const& cfg,
                                     ReferenceInput const& in, double g,
                                     X1ColumnLink* link) const {
  auto const& w = in.base;
  auto const& dw = in.delta;
  bool paired = dw.defined();
  bool fp32 = cfg.policy->fp32();
  auto pcoord = cfg.coord;
  int is = pcoord->il();
  int iu = pcoord->iu();

  // Global x1 reference across a vertical (nb1>1) decomposition: make the
  // hydrostatic reference continuous over the x1 process seams. The block
  // owning x1-outer anchors at the true domain top; every block below receives
  // the running seam-face pressure from the block above and passes on that
  // value plus its own interior hydrostatic drop (= its bottom-face pressure).
  // A serial top->bottom scan along the x1 block column, whether the blocks
  // sit on other processes or on other threads of this one (#254). nb1 == 1
  // (no x1 neighbor) => an empty anchor tells the backend to compute the
  // local top anchor.
  torch::Tensor anchor;
  // relay only across a SPLIT x1 column (else unmatched send/recv at nb1=1)
  bool x1_split = link && link->split();
  if (x1_split) {
    anchor = link->take_anchor(torch::empty(
        {w.size(1), w.size(2), paired ? (fp32 ? 3 : 2) : 1}, w.options()));
  }

  bool stretched = pcoord->options->stretched_x1();
  if (!op_ || !op_->matches(pcoord->dx1f, iu, stretched, cfg.phys_in,
                            cfg.phys_out, cfg.wall_clamp)) {
    op_ = std::make_shared<HydroReferenceX1Operator>(
        pcoord->dx1f, iu, stretched, cfg.phys_in, cfg.phys_out, cfg.wall_clamp);
  }
  HydroReferenceX1 reference;
  ReferenceFaces out;
  if (paired) {
    HydroReferenceX1 pairs;
    op_->evaluate_split(
        w, dw, g,
        anchor.defined() ? anchor.narrow(-1, 0, 2).unsqueeze(-2) : anchor,
        pairs);
    auto total = [](torch::Tensor const& p) {
      return p.select(-1, 0) + p.select(-1, 1);
    };
    reference.psf_lo = total(pairs.psf_lo);
    reference.psf_hi = total(pairs.psf_hi);
    reference.pref = total(pairs.pref);
    reference.dsf = total(pairs.dsf);
    reference.dref = total(pairs.dref);
    out.face0 = pairs.psf_lo.select(-1, 0).contiguous();
    out.face_delta = pairs.psf_lo.select(-1, 1).contiguous();
    out.cell0 = pairs.pref.select(-1, 0).contiguous();
    out.cell_delta = pairs.pref.select(-1, 1).contiguous();
    if (fp32)
      out.gauge = anchor.defined()
                      ? anchor.narrow(-1, 2, 1).clone()
                      : (w[IPR] - out.cell0).narrow(-1, iu, 1).clone();
  } else {
    reference = HydroReferenceX1(w);
    op_->evaluate(w, g, anchor, reference);
  }
  auto& psf_lo = reference.psf_lo;
  auto& psf_hi = reference.psf_hi;
  auto& pref = reference.pref;
  auto& dsf = reference.dsf;
  auto& dref = reference.dref;

  // dynamics/conversion-order 4 (#289): the only fourth-order part is the
  // face density below, the face value of the kernel's own cell dref -- that
  // is what makes the O(dz^2) face offset vanish, whatever dref's own accuracy
  // (wb-ref4.md section 2), and replacing dref broke the theta bound of a
  // density current at a wall. psf is never changed, so the rest balance is
  // the kernel's. Only the resolution flag is wanted here; the face part
  // follows the seam exchange, so a column split along x1 gets one block's
  // faces.
  torch::Tensor wb4_flag;
  if (cfg.conversion_order == 4) {
    if (!wb_ref4_ || wb_ref4_->key != X1RowsKey(pcoord->x1f, w.options())) {
      wb_ref4_ = std::make_shared<FaceDensityRows>(
          face_density_rows(pcoord->x1f, is, iu, cfg.wall_clamp && cfg.phys_in,
                            cfg.wall_clamp && cfg.phys_out, w.options()));
    }
    wb4_flag = wb_ref4_flag(*wb_ref4_, psf_lo, psf_hi);
  }

  if (link && link->passes_anchor()) {
    auto outgoing = psf_lo.narrow(-1, is, 1);
    if (paired) {
      std::vector<torch::Tensor> fields = {out.face0.narrow(-1, is, 1),
                                           out.face_delta.narrow(-1, is, 1)};
      if (fp32) fields.push_back(out.gauge);
      outgoing = torch::cat(fields, -1);
    }
    link->pass_anchor(outgoing);
  }

  // Ghost-row reference exchange across x1 seams: overwrite this block's
  // ghost rows of (pref, dref) with the NEIGHBOR'S interior rows for the same
  // physical cells. The block-local ghost quadrature (w6e edge rows, and the
  // [lo,hi] guard whose accept/fallback decision is stencil-dependent) does
  // NOT reproduce the owner's interior values -- near a physical wall the
  // mismatch reaches ~1e2 Pa (measured), which enters
  // w' = w - ref as a spurious seam perturbation every step and makes the
  // dynamics nb1-dependent. After this exchange the perturbation field seen
  // by the reconstruction is identical on both sides of every seam, so the
  // seam-face states agree and the single-valued seam flux average becomes a
  // no-op. nb1=1: no seams, bit-unchanged. A same-process neighbor goes
  // through a board, a remote one through the process group (#254).
  if (x1_split) {
    int ng = is;  // il() == nghost
    std::vector<torch::Tensor> fields = {pref, dref};
    if (paired) {
      fields.push_back(out.cell0);
      fields.push_back(out.cell_delta);
    }
    link->exchange_ghost_rows(fields, ng);
  }

  // conversion-order 4: the face density is the fourth-order face value of
  // the (exchanged) cell density reference, so a split column gets the same
  // faces as one block
  if (wb4_flag.defined()) wb_ref4_faces(*wb_ref4_, dref, dsf, wb4_flag);

  // the face density on each side of a face is the x1 reconstruction's own
  // optimal-linear row applied to dref (cp5 = the weno5 candidates combined
  // with their linear weights), so rho_f = R_lin(rho) + N(rho') whatever
  // dref is. Read after the seam exchange; a face whose value leaves
  // [min, max] of its two cells' dref keeps the face density above, which is
  // also what any other reconstruction gets on both sides.
  auto dsf_lr = torch::stack({dsf, dsf});
  auto const& itype = cfg.interp;
  int nc1 = dref.size(-1);
  if ((itype == "weno5" || itype == "cp5") && nc1 >= 6) {
    if (!dsf_lin_ || dsf_lin_key_ != X1RowsKey(pcoord->x1f, w.options())) {
      dsf_lin_key_ = X1RowsKey(pcoord->x1f, w.options());
      auto op = InterpOptionsImpl::create();
      op->type("cp5");
      dsf_lin_ = std::make_shared<InterpImpl>(op);
      if (cfg.interp_x1_coeffs) {
        auto const& co = pcoord->options;
        bool plain =
            x1_centroid_exact_enabled() && co->type() == "spherical-polar";
        auto x1f = !plain               ? pcoord->x1_row_faces(itype)
                   : co->stretched_x1() ? pcoord->x1f
                                        : torch::Tensor();
        if (x1f.defined()) dsf_lin_->set_x1_coeffs(x1f);
      }
      dsf_lin_->to(w.device(), w.scalar_type());
    }
    constexpr int D1 = 3;
    int il = 3, iu5 = nc1 - 3;
    auto lin = torch::stack({dsf, dsf}).unsqueeze(1);
    dsf_lin_->forward(dref.unsqueeze(0), D1,
                      lin[IRT].slice(D1, il - 1, iu5 + 1),
                      lin[ILT].slice(D1, il, iu5 + 2));
    lin = lin.squeeze(1);
    auto dl = dref.narrow(-1, 0, nc1 - 1), dr = dref.narrow(-1, 1, nc1 - 1);
    for (int s : {(int)ILT, (int)IRT}) {
      auto face = lin[s].narrow(-1, il - 1, iu5 + 3 - il);
      auto lo = torch::minimum(dl, dr).narrow(-1, il - 2, iu5 + 3 - il);
      auto hi = torch::maximum(dl, dr).narrow(-1, il - 2, iu5 + 3 - il);
      auto d = dsf_lr[s].narrow(-1, il - 1, iu5 + 3 - il);
      d.copy_(torch::where((face >= lo) & (face <= hi), face, d));
    }
    // a reflecting x1 wall face is a mirror: one face density on both sides,
    // the mean of the two rows. Two values there give a Riemann solver whose
    // wave speeds read the face density (hllc, roe) a nonzero contact speed,
    // which moves mass through the wall.
    auto const& refl = cfg.reflecting;
    if (wb4_flag.defined()) {
      // conversion-order 4: a side whose cp5 row reads a ghost keeps wb_ref4's
      // face density (both sides on the two wall faces, the reading side on
      // the third), all walls in one gather and one index_put on (side, face)
      // rows built once per layout. Both sides then hold dsf at the wall face,
      // where the mean below is that value (0.5 * (x + x) = x), so it is
      // skipped
      std::array<int64_t, 3> at{pcoord->il(), pcoord->iu(),
                                refl[0] + 2 * refl[1]};
      if (at != wb4_lr_at_ || !wb4_lr_col_.defined() ||
          wb4_lr_col_.device() != dsf.device()) {
        std::vector<int64_t> side, col;
        for (int f : {0, 1}) {
          if (!refl[f]) continue;
          int i = f == 0 ? pcoord->il() : pcoord->iu() + 1;
          for (int k = 0; k < 3; ++k)
            for (int s : {(int)ILT, (int)IRT})
              if (k < 2 || s == (f ? IRT : ILT))
                side.push_back(s), col.push_back(i + (f ? -k : k));
        }
        auto lo = torch::TensorOptions().dtype(torch::kLong);
        wb4_lr_side_ = torch::tensor(side, lo).to(dsf.device());
        wb4_lr_col_ = torch::tensor(col, lo).to(dsf.device());
        wb4_lr_at_ = at;
      }
      if (wb4_lr_col_.numel() > 0)  // no reflecting x1 wall: nothing to keep
        dsf_lr.index_put_(
            {wb4_lr_side_, torch::indexing::Ellipsis, wb4_lr_col_},
            dsf.index_select(-1, wb4_lr_col_).movedim(-1, 0));
    } else {
      for (int f : {0, 1}) {
        if (!refl[f]) continue;
        int i = f == 0 ? pcoord->il() : pcoord->iu() + 1;
        auto mean =
            0.5 * (dsf_lr[ILT].select(-1, i) + dsf_lr[IRT].select(-1, i));
        for (int s : {(int)ILT, (int)IRT}) dsf_lr[s].select(-1, i).copy_(mean);
      }
    }
  }

  out.psf_lo = psf_lo;
  out.pref = pref;
  out.dsf_lr = dsf_lr;
  out.dref = dref;
  return out;
}

}  // namespace snap
