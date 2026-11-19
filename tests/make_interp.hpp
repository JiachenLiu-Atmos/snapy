#pragma once

// C/C++
#include <memory>
#include <string>

// snap
#include <snap/recon/reconstruct.hpp>

namespace snap {

//! an unregistered reconstruction module of the scheme named type
inline Interp make_interp(std::string const& type) {
  return std::make_shared<InterpImpl>(
      std::make_shared<InterpOptionsImpl>(type));
}

}  // namespace snap
