// C/C++ headers
#include <unistd.h>

#include <csignal>
#include <iostream>
#include <mutex>

// snap
#include <snap/layout/distributed.hpp>
#include <snap/mesh/meshblock.hpp>

#include "signal_handler.hpp"

//! \file signal_handler.cpp
/*!
 * Some functions in this file are adapted from Athena++ code
 */

namespace snap {

static std::mutex signal_mutex;

// Static member variable definitions
SignalHandler* SignalHandler::mysignal_ = nullptr;
int SignalHandler::signalflag_[SignalHandler::nsignal] = {0, 0, 0};
sigset_t SignalHandler::mask_;

SignalHandler::SignalHandler() {
  for (int n = 0; n < nsignal; n++) {
    signalflag_[n] = 0;
  }
  // C++11 standard guarantees that <csignal> places C-standard signal.h
  // contents in std:: namespace. POSIX C extensions are likely only placed in
  // global namespace (not std::)
  std::signal(SIGTERM, SignalHandler::SetSignalFlag);
  std::signal(SIGINT, SignalHandler::SetSignalFlag);
  std::signal(SIGALRM, SignalHandler::SetSignalFlag);

  // populate set of signals to block while the handler is running; prevent
  // premption
  sigemptyset(&mask_);
  sigaddset(&mask_, SIGTERM);
  sigaddset(&mask_, SIGINT);
  sigaddset(&mask_, SIGALRM);
}

void SignalHandler::Destroy() {
  if (SignalHandler::mysignal_ != nullptr) {
    delete SignalHandler::mysignal_;
    SignalHandler::mysignal_ = nullptr;
  }
}

SignalHandler* SignalHandler::GetInstance() {
  if (SignalHandler::mysignal_ == nullptr) {
    std::unique_lock<std::mutex> lock(signal_mutex);
    mysignal_ = new SignalHandler();
  }
  return mysignal_;
}

//! \brief Synchronize and check signal flags and return true if any of them is
//! caught
int SignalHandler::CheckSignalFlags(MeshBlockImpl const* pmb) {
  // Currently, only checking for nonzero return code at the end of each
  // timestep in main.cpp; i.e. if an issue prevents a process from reaching the
  // end of a cycle, the signals will never be handled by that process / the
  // solver may hang
  int ret = 0;
  sigprocmask(SIG_BLOCK, &mask_, nullptr);
  for (int n = 0; n < nsignal; n++) ret += signalflag_[n];
  sigprocmask(SIG_UNBLOCK, &mask_, nullptr);

  std::vector<at::Tensor> ret_reduce = {
      torch::tensor({ret}, torch::dtype(torch::kInt32))};
  c10d::AllreduceOptions op;
  op.reduceOp = c10d::ReduceOp::MAX;
  if (pmb != nullptr && pmb->get_layout() != nullptr &&
      pmb->get_layout()->has_process_group()) {
    pmb->get_layout()->comm->allreduce(ret_reduce, op.reduceOp);
  } else if (is_process_group_initialized() && get_world_size() > 1) {
    get_process_group()->allreduce(ret_reduce, op)->wait();
  }

  return ret_reduce[0].item<int>();
}

//! \brief Gets a signal flag assuming the signalflag array is already
//! synchronized.
//!        Returns -1 if the specified signal is not handled.
int SignalHandler::GetSignalFlag(int s) {
  int ret = -1;
  switch (s) {
    case SIGTERM:
      ret = signalflag_[ITERM];
      break;
    case SIGINT:
      ret = signalflag_[IINT];
      break;
    case SIGALRM:
      ret = signalflag_[IALRM];
      break;
    default:
      // nothing
      break;
  }
  return ret;
}

//! \brief Sets signal flags and reinstalls the signal handler function.
void SignalHandler::SetSignalFlag(int s) {
  // Signal handler functions must have C linkage; C++ linkage is
  // implemantation-defined
  switch (s) {
    case SIGTERM:
      signalflag_[ITERM] = 1;
      signal(s, SetSignalFlag);
      break;
    case SIGINT:
      signalflag_[IINT] = 1;
      signal(s, SetSignalFlag);
      break;
    case SIGALRM:
      signalflag_[IALRM] = 1;
      signal(s, SetSignalFlag);
      break;
    default:
      // nothing
      break;
  }
  return;
}

//! \brief Set the wall time limit alarm
void SignalHandler::SetWallTimeAlarm(int t) {
  alarm(t);
  return;
}

//! \brief Cancel the wall time limit alarm
void SignalHandler::CancelWallTimeAlarm() {
  alarm(0);
  return;
}

}  // namespace snap
