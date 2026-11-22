#include "x1_reference.hpp"

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/coord/x1_centroid.hpp>
#include <snap/coord/x1_stencil.hpp>
#include <snap/layout/exchange.hpp>
#include <snap/mesh/precision_policy.hpp>

#include "reconstruct.hpp"

namespace snap {

namespace {

// relative margin of the range guards: on a flat profile the two bounds
// coincide, and an exact test would flip on round-off between a split and a
// one-block column
constexpr double kMargin = 1.e-10;
// |ln(p_lo/p_hi)| above this: fewer than two cells per pressure scale height
constexpr double kMaxLogDrop = 0.5;

torch::Tensor in_range(torch::Tensor const& v, torch::Tensor const& lo,
                       torch::Tensor const& hi) {
  return (v >= lo - kMargin * lo.abs()) & (v <= hi + kMargin * hi.abs());
}

// wb_ref4 (#289): the resolution flag, cells (dilated by two) keeping the
// kernel's face density; local, so it needs no seam exchange
torch::Tensor wb_ref4_flag(FaceDensityRows const& st,
                           torch::Tensor const& psf_lo,
                           torch::Tensor const& psf_hi) {
  if (!st.usable) return torch::ones_like(psf_lo, torch::kBool);
  auto bad = ((psf_lo / psf_hi).log().abs() > kMaxLogDrop) & st.counted;
  auto sizes = bad.sizes();
  return torch::max_pool1d(bad.to(psf_lo.dtype()).reshape({-1, 1, st.nc1}), 5,
                           1, 2)
             .reshape(sizes) > 0.;
}

// wb_ref4 faces, after the seam exchange: dsf on is..iu+1 is the fourth-order
// face value of dref (needs is >= 1)
void wb_ref4_faces(FaceDensityRows const& st, torch::Tensor const& dref,
                   torch::Tensor dsf, torch::Tensor const& flag) {
  if (!st.usable) return;
  int nf = st.iu - st.is + 2;
  auto face = x1_stencil_apply(dref, st.idx, st.wt);
  auto dl = dref.narrow(-1, st.is - 1, nf), dr = dref.narrow(-1, st.is, nf);
  auto ok = in_range(face, torch::minimum(dl, dr), torch::maximum(dl, dr)) &
            ~flag.narrow(-1, st.is - 1, nf) & ~flag.narrow(-1, st.is, nf);
  auto d = dsf.narrow(-1, st.is, nf);
  d.copy_(torch::where(ok, face, d));
}

}  // namespace

ExchangeX1Link::ExchangeX1Link(MeshBlockImpl const* pmb) : pmb_(pmb) {
  std::tie(below_, above_) = Exchange(pmb).x1_neighbors();
}

torch::Tensor ExchangeX1Link::take_anchor(torch::Tensor const& like) {
  return Exchange(pmb_).unpack_anchor(like, kTagWbRef);
}

void ExchangeX1Link::pass_anchor(torch::Tensor const& face) {
  Exchange(pmb_).pack_anchor(face, kTagWbRef);
}

void ExchangeX1Link::exchange_ghost_rows(std::vector<torch::Tensor> rows,
                                         int ng) {
  Exchange(pmb_).x1_rows(rows, ng, false, kTagWbGhost);
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

  // global x1 reference: a top->bottom anchor relay down a split x1 column;
  // empty at nb1 == 1 (the kernel anchors locally). See lane-c-notes.md.
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

  // conversion-order 4 (#289): only the flag here; the faces follow the seam
  // exchange, so a split column gets one block's faces
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

  // ghost rows of (pref, dref) <- the neighbour's owned rows, so w' is the
  // same on both sides of a seam (lane-c-notes.md); nb1 = 1: unchanged
  if (x1_split) {
    int ng = is;  // il() == nghost
    std::vector<torch::Tensor> fields = {pref, dref};
    if (paired) {
      fields.push_back(out.cell0);
      fields.push_back(out.cell_delta);
    }
    link->exchange_ghost_rows(fields, ng);
  }

  if (wb4_flag.defined()) wb_ref4_faces(*wb_ref4_, dref, dsf, wb4_flag);

  // per-side face density: the reconstruction's cp5 row on dref, guarded by
  // [min, max] of the two cells (lane-c-notes.md)
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
    // a reflecting x1 wall face is a mirror: one face density on both sides
    auto const& refl = cfg.reflecting;
    if (wb4_flag.defined()) {
      // key 4: a side whose cp5 row reads a ghost keeps wb_ref4's dsf, so the
      // wall mean is skipped (lane-c-notes.md)
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
