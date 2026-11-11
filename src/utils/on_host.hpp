#pragma once

// C/C++
#include <deque>
#include <map>

// torch
#include <ATen/TensorIterator.h>

namespace snap {

//! Runs a CPU kernel for a backend that has none of its own (MPS): each
//! storage the kernel sees is copied to the host once, so views that alias on
//! the device alias on the host, and every storage is copied back afterwards.
class HostCopies {
 public:
  ~HostCopies() noexcept(false) {
    for (auto& [_, s] : store_) s.first.copy_(s.second);
  }

  template <typename T>
  decltype(auto) operator()(T&& x) {
    using U = std::decay_t<T>;
    if constexpr (std::is_same_v<U, at::TensorIterator>) {
      auto& it = iters_.emplace_back(x);
      for (int i = 0; i < x.ntensors(); ++i)
        it.unsafe_replace_operand(i, host_ptr(x.tensor(i), x.data_ptr(i)));
      return it;
    } else if constexpr (std::is_same_v<U, at::Tensor>) {
      if (!x.defined()) return tensors_.emplace_back(x);
      return tensors_.emplace_back(
          host(x).as_strided(x.sizes(), x.strides(), x.storage_offset()));
    } else {
      return std::forward<T>(x);
    }
  }

 private:
  at::Tensor& host(at::TensorBase const& b) {
    at::Tensor t(b);
    auto key = t.storage().unsafeGetStorageImpl();
    if (!store_.count(key)) {
      int64_t n = t.storage().nbytes() / t.element_size();
      auto all = t.as_strided({n}, {1}, 0);
      store_[key] = {all, all.cpu()};
    }
    return store_[key].second;
  }

  void* host_ptr(at::TensorBase const& t, void const* dev) {
    auto off = static_cast<char const*>(dev) -
               static_cast<char const*>(t.storage().data());
    return static_cast<char*>(host(t).data_ptr()) + off;
  }

  std::map<void*, std::pair<at::Tensor, at::Tensor>> store_;
  std::deque<at::Tensor> tensors_;
  std::deque<at::TensorIterator> iters_;
};

template <auto Fn>
struct OnHost;

template <typename R, typename... A, R (*Fn)(A...)>
struct OnHost<Fn> {
  static R run(A... a) {
    HostCopies h;
    return Fn(h(a)...);
  }
};

}  // namespace snap
