#include <asmjit/asmjit.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <sstream>
#include <cassert>

template <class T> inline void Log(const __m256i &value) {
  const size_t n = sizeof(__m256i) / sizeof(T);
  T buffer[n];
  _mm256_storeu_si256((__m256i *)buffer, value);
  for (int i = 0; i < n; i++)
    std::cout << buffer[i] << " ";
}

float extract_float(const __m128 v, const int i) {
  float x;
  if (i == 0) {
    _MM_EXTRACT_FLOAT(x, v, 0);
  }
  if (i == 1) {
    _MM_EXTRACT_FLOAT(x, v, 1);
  }
  if (i == 2) {
    _MM_EXTRACT_FLOAT(x, v, 2);
  }
  if (i == 3) {
    _MM_EXTRACT_FLOAT(x, v, 3);
  }
  return x;
}

void print(const __m128 v) {
  std::cerr << "vec[4]{" << extract_float(v, 0) << ", " << extract_float(v, 1)
            << ", " << extract_float(v, 2) << ", " << extract_float(v, 3)
            << "}\n";
}

void print(const __m256 v) {
  print(_mm256_extractf128_ps(v, 0));
  print(_mm256_extractf128_ps(v, 1));
}

float *genRandomActivations(size_t width) {
  auto *out = (float *)malloc(width * sizeof(float));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  for (int n = 0; n < width; ++n) {
    out[n] = dis(gen);
  }
  return out;
}

// sparsity 1.0 == all zeros, 0.0 == no zeros
float *genRandomSparseWeights(size_t K, size_t N, float sparsity, size_t block=1) {
  auto *out = (float *)malloc(K * N * sizeof(float));
  std::random_device rd;
  std::mt19937 gen(rd());
  float max = 1000;
  std::uniform_real_distribution<float> dis(-max, max);
  float thresh = sparsity * max;
  for (size_t k = 0; k < K; k++) {
    for (size_t n = 0; n < N; n+=block) {
      float candidate = dis(gen);
      for (size_t b = 0; b < block; ++b) {
        if (b + n >= N) continue;
        if (std::abs(candidate) < thresh) {
          out[(n + b) * K + k] = 0;
        } else {
          out[(n + b) * K + k] = candidate;
        }
      }
    }
  }
  return out;
}

void ref_mv(float *inp, float *w, float *out, size_t n, size_t k) {
  // inp is 1xk
  // w is kxn
  // o is 1xn
  for (auto i = 0ULL; i < n; ++i) {
    out[i] = 0;
    for (auto j = 0ULL; j < k; ++j) {
      out[i] += w[i * k + j] * inp[j];
    }
  }
}

struct Nzd {
  std::vector<int64_t> nzi;
  std::vector<float> nz;
};
#define WIDTH 8
void gather_mv(float *inp, float *w, float *out, size_t n, size_t k) {
  // gather 4 elems vertically
  for (auto j = 0ULL; j < n; j += WIDTH) {
    Nzd nzd[WIDTH];
    for (size_t i = 0; i < k; ++i) {
      bool hot = false;
      for (auto j_ = 0; j_ < WIDTH; ++j_) {
        if (w[(j + j_) * k + i]) {
          hot = true;
        }
      }
      if (hot) {
        for (auto j_ = 0; j_ < WIDTH; ++j_) {
          // std::cerr << "val i " << i << " and " << static_cast<float>(w[(j +
          // j_) * k + i])<< "\n";
          nzd[j_].nzi.emplace_back(i);
          nzd[j_].nz.emplace_back(
              static_cast<float>(w[(j + WIDTH - 1 - j_) * k + i]));
        }
      }
    }
    size_t max = 0;
    for (auto j_ = 0; j_ < WIDTH; ++j_) {
      if (nzd[j_].nzi.size() > max) {
        max = nzd[j_].nzi.size();
      }
    }
    for (auto j_ = 0; j_ < WIDTH; ++j_) {
      if (nzd[j_].nzi.size() < max) {
        int64_t last = 0;
        if (nzd[j_].nzi.size()) {
          last = nzd[j_].nzi.back();
        }
        auto extra = max - nzd[j_].nzi.size();
        for (auto k = 0; k < extra; ++k) {
          nzd[j_].nzi.emplace_back(last);
          nzd[j_].nz.emplace_back(0.0);
        }
      }
    }

    // produce a __m128
    __m256 acc = _mm256_set1_ps(0.0);
    for (auto i = 0ULL; i < max; ++i) {
      //__m256i vindex = _mm256_set_epi64x(
      //    nzd[0].nzi[i],
      //    nzd[1].nzi[i],
      //    nzd[2].nzi[i],
      //    nzd[3].nzi[i]
      //    );
      // Log<uint64_t>(vindex);
      // std::cout << "\n";
      //__m128 activations = _mm256_i64gather_ps(inp, vindex, 1);
      __m256 activations = _mm256_set1_ps(inp[nzd[0].nzi[i]]);
      // print(activations);
      // vindex = _mm256_set1_epi64x(nzd[0].nzi[i]);
      __m256 scale =
          _mm256_set_ps(nzd[0].nz[i], nzd[1].nz[i], nzd[2].nz[i], nzd[3].nz[i],
                        nzd[4].nz[i], nzd[5].nz[i], nzd[6].nz[i], nzd[7].nz[i]);
      acc = _mm256_add_ps(_mm256_mul_ps(activations, scale), acc);
    }
    if (j + WIDTH < n + 1) {
      _mm256_storeu_ps(&(out[j]), acc);
    } else {
      // if (n - j == 1) {
      //  out[j] = _mm_extract_ps(acc, 0);
      //} else if (n - j == 2) {
      //  out[j] = _mm_extract_ps(acc, 0);
      //  out[j + 1] = _mm_extract_ps(acc, 1);
      //} else {
      //  out[j] = _mm_extract_ps(acc, 0);
      //  out[j + 1] = _mm_extract_ps(acc, 1);
      //  out[j + 2] = _mm_extract_ps(acc, 2);
      //}
    }
  }
}

using namespace asmjit;
typedef void (*Func)(float *, float *);

Func jit(float *W, size_t N, size_t K, JitRuntime &rt) {

  CodeHolder code;
  code.init(rt.codeInfo());
  StringLogger logger;

  code.setLogger(&logger);

  x86::Compiler cc(&code);
  cc.addFunc(FuncSignatureT<void, const float *, float *>());

  x86::Gp input = cc.newIntPtr("input");
  x86::Gp output = cc.newIntPtr("output");

  x86::Gp k0 = cc.newGpd("k0");
  cc.setArg(0, input);
  cc.setArg(1, output);

  auto d_zero = Data256::fromF32(0.0f);
  x86::Mem zero_ymm = cc.newYmmConst(ConstPool::kScopeLocal, d_zero);
  for (size_t n = 0; n < N; n += 8) {
    // Accumulate into ymm0
    cc.vmovups(x86::ymm0, zero_ymm);
    for (size_t k = 0; k < K; ++k) {

      bool hot = false;
      for (size_t n_ = 0; n_ < 8; ++n_) {
        if (W[(n + n_) * K + k]) {
          hot = true;
          break;
        }
      }

      if (hot) {
        auto d = Data256::fromF32(static_cast<float>(W[(n + 0) * K + k]),
                                  static_cast<float>(W[(n + 1) * K + k]),
                                  static_cast<float>(W[(n + 2) * K + k]),
                                  static_cast<float>(W[(n + 3) * K + k]),
                                  static_cast<float>(W[(n + 4) * K + k]),
                                  static_cast<float>(W[(n + 5) * K + k]),
                                  static_cast<float>(W[(n + 6) * K + k]),
                                  static_cast<float>(W[(n + 7) * K + k]));
        x86::Mem cymm = cc.newYmmConst(ConstPool::kScopeLocal, d);
        cc.vmovups(x86::ymm1, cymm);
        auto offset = k * sizeof(float);
        cc.vbroadcastss(x86::ymm2, x86::ptr(input, offset));
        cc.vmulps(x86::ymm1, x86::ymm1, x86::ymm2);
        cc.vaddps(x86::ymm0, x86::ymm0, x86::ymm1);
      }
    }
    auto offset = n * sizeof(float);
    cc.vmovups(x86::ptr(output, offset), x86::ymm0);
  }

  cc.endFunc();
  cc.finalize();

  Func fn;
  Error err = rt.add(&fn, &code);
  // std::cout << logger.data();
  if (err) {
    std::cerr << "error! " << err << "\n";
    return nullptr;
  }
  return fn;
}

#define REG_WIDTH 8
Func jit_f(float *W, size_t N, size_t K, size_t P, JitRuntime &rt) {

  CodeHolder code;
  code.init(rt.codeInfo());
  StringLogger logger;

  code.setLogger(&logger);

  x86::Compiler cc(&code);
  cc.addFunc(FuncSignatureT<void, const float *, float *>());

  x86::Gp input = cc.newIntPtr("input");
  x86::Gp output = cc.newIntPtr("output");

  x86::Gp k0 = cc.newGpd("k0");
  cc.setArg(0, input);
  cc.setArg(1, output);

  auto d_zero = Data256::fromF32(0.0f);
  x86::Mem zero_ymm = cc.newYmmConst(ConstPool::kScopeLocal, d_zero);

  std::vector<x86::Ymm> pRegs;
  std::vector<x86::Ymm> pRegsAcc;
  std::vector<bool> pHot;
  if (P == 4) {
    pRegs.emplace_back(x86::ymm15);
    pRegs.emplace_back(x86::ymm14);
    pRegs.emplace_back(x86::ymm13);
    pRegs.emplace_back(x86::ymm12);
    pRegsAcc.emplace_back(x86::ymm5);
    pRegsAcc.emplace_back(x86::ymm4);
    pRegsAcc.emplace_back(x86::ymm3);
    pRegsAcc.emplace_back(x86::ymm2);
    pHot.resize(4);
  } else {
    std::cerr << "invalid P\n";
    return nullptr;
  }

  for (size_t n = 0; n < N; n += REG_WIDTH * pRegs.size()) {
    // Accumulate into regs
    for (size_t reg = 0; reg < pRegs.size(); ++reg) {
      cc.vmovups(pRegsAcc[reg], zero_ymm);
    }

    for (size_t k = 0; k < K; ++k) {
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        pHot[reg] = false;
        for (size_t el = 0; el < REG_WIDTH; ++el) {
          auto n_ = el + reg * REG_WIDTH;
          if (W[(n + n_) * K + k]) {
            pHot[reg] = true;
            break;
          }
        }
      }
      bool any_hot = false;
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        any_hot |= pHot[reg];
      }
      if (!any_hot)
        continue;

      // Load activation
      size_t offset = k * sizeof(float);
      cc.vbroadcastss(x86::ymm0, x86::ptr(input, offset));
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        if (pHot[reg]) {
          auto off = reg * REG_WIDTH;
          auto d =
              Data256::fromF32(static_cast<float>(W[(n + off + 0) * K + k]),
                               static_cast<float>(W[(n + off + 1) * K + k]),
                               static_cast<float>(W[(n + off + 2) * K + k]),
                               static_cast<float>(W[(n + off + 3) * K + k]),
                               static_cast<float>(W[(n + off + 4) * K + k]),
                               static_cast<float>(W[(n + off + 5) * K + k]),
                               static_cast<float>(W[(n + off + 6) * K + k]),
                               static_cast<float>(W[(n + off + 7) * K + k]));
          x86::Mem cymm = cc.newYmmConst(ConstPool::kScopeLocal, d);
          cc.vmovups(pRegs[reg], cymm);
        }
      }
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        if (pHot[reg]) {
          cc.vmulps(pRegs[reg], pRegs[reg], x86::ymm0);
        }
      }
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        if (pHot[reg]) {
          cc.vaddps(pRegsAcc[reg], pRegsAcc[reg], pRegs[reg]);
        }
      }
    }
    for (size_t reg = 0; reg < pRegs.size(); ++reg) {
      auto offset = (n + reg * REG_WIDTH) * sizeof(float);
      cc.vmovups(x86::ptr(output, offset), pRegsAcc[reg]);
    }
  }

  cc.endFunc();
  cc.finalize();

  Func fn;
  Error err = rt.add(&fn, &code);
  //std::cout << logger.data();
  if (err) {
    std::cerr << "error! " << err << "\n";
    return nullptr;
  }
  return fn;
}

Func jit_ff(float *W, size_t N, size_t K, size_t P, JitRuntime &rt) {

  CodeHolder code;
  code.init(rt.codeInfo());
  StringLogger logger;

  code.setLogger(&logger);

  x86::Compiler cc(&code);
  cc.addFunc(FuncSignatureT<void, const float *, float *>());

  x86::Gp input = cc.newIntPtr("input");
  x86::Gp output = cc.newIntPtr("output");

  x86::Gp k0 = cc.newGpd("k0");
  cc.setArg(0, input);
  cc.setArg(1, output);

  auto d_zero = Data256::fromF32(0.0f);
  x86::Mem zero_ymm = cc.newYmmConst(ConstPool::kScopeLocal, d_zero);

  std::vector<x86::Ymm> pRegs;
  std::vector<x86::Ymm> pRegsAcc;
  std::vector<bool> pHot;
  if (P == 4) {
    pRegs.emplace_back(x86::ymm15);
    pRegs.emplace_back(x86::ymm14);
    pRegs.emplace_back(x86::ymm13);
    pRegs.emplace_back(x86::ymm12);
    pRegsAcc.emplace_back(x86::ymm5);
    pRegsAcc.emplace_back(x86::ymm4);
    pRegsAcc.emplace_back(x86::ymm3);
    pRegsAcc.emplace_back(x86::ymm2);
    pHot.resize(4);
  } else {
    std::cerr << "invalid P\n";
    return nullptr;
  }

  for (size_t n = 0; n < N; n += REG_WIDTH * pRegs.size()) {
    // Accumulate into regs
    for (size_t reg = 0; reg < pRegs.size(); ++reg) {
      cc.vmovups(pRegsAcc[reg], zero_ymm);
    }

    for (size_t k = 0; k < K; ++k) {
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        pHot[reg] = false;
        for (size_t el = 0; el < REG_WIDTH; ++el) {
          auto n_ = el + reg * REG_WIDTH;
          if (W[(n + n_) * K + k]) {
            pHot[reg] = true;
            break;
          }
        }
      }
      bool any_hot = false;
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        any_hot |= pHot[reg];
      }
      if (!any_hot)
        continue;

      // Load activation
      size_t offset = k * sizeof(float);
      cc.vbroadcastss(x86::ymm0, x86::ptr(input, offset));
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        if (pHot[reg]) {
          auto off = reg * REG_WIDTH;
          auto d =
              Data256::fromF32(static_cast<float>(W[(n + off + 0) * K + k]),
                               static_cast<float>(W[(n + off + 1) * K + k]),
                               static_cast<float>(W[(n + off + 2) * K + k]),
                               static_cast<float>(W[(n + off + 3) * K + k]),
                               static_cast<float>(W[(n + off + 4) * K + k]),
                               static_cast<float>(W[(n + off + 5) * K + k]),
                               static_cast<float>(W[(n + off + 6) * K + k]),
                               static_cast<float>(W[(n + off + 7) * K + k]));
          x86::Mem cymm = cc.newYmmConst(ConstPool::kScopeLocal, d);
          //cc.vmovups(pRegs[reg], cymm);
          cc.vmulps(pRegs[reg], x86::ymm0, cymm);
        }
      }
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        if (pHot[reg]) {
          cc.vaddps(pRegsAcc[reg], pRegsAcc[reg], pRegs[reg]);
        }
      }
    }
    for (size_t reg = 0; reg < pRegs.size(); ++reg) {
      auto offset = (n + reg * REG_WIDTH) * sizeof(float);
      cc.vmovups(x86::ptr(output, offset), pRegsAcc[reg]);
    }
  }

  cc.endFunc();
  cc.finalize();

  Func fn;
  Error err = rt.add(&fn, &code);
  //std::cout << logger.data();
  if (err) {
    std::cerr << "error! " << err << "\n";
    return nullptr;
  }
  return fn;
}

Func jit_fff(float *W, size_t N, size_t K, size_t P, JitRuntime &rt) {

  CodeHolder code;
  code.init(rt.codeInfo());
  StringLogger logger;

  code.setLogger(&logger);

  x86::Compiler cc(&code);
  cc.addFunc(FuncSignatureT<void, const float *, float *>());

  x86::Gp input = cc.newIntPtr("input");
  x86::Gp output = cc.newIntPtr("output");

  x86::Gp k0 = cc.newGpd("k0");
  cc.setArg(0, input);
  cc.setArg(1, output);

  auto d_zero = Data256::fromF32(0.0f);
  x86::Mem zero_ymm = cc.newYmmConst(ConstPool::kScopeLocal, d_zero);

  std::vector<x86::Ymm> pRegs;
  std::vector<x86::Ymm> pRegsAcc;
  std::vector<bool> pHot;
  if (P == 4) {
    pRegs.emplace_back(x86::ymm15);
    pRegs.emplace_back(x86::ymm14);
    pRegs.emplace_back(x86::ymm13);
    pRegs.emplace_back(x86::ymm12);
    pRegsAcc.emplace_back(x86::ymm5);
    pRegsAcc.emplace_back(x86::ymm4);
    pRegsAcc.emplace_back(x86::ymm3);
    pRegsAcc.emplace_back(x86::ymm2);
    pHot.resize(4);
  } else {
    std::cerr << "invalid P\n";
    return nullptr;
  }

  for (size_t n = 0; n < N; n += REG_WIDTH * pRegs.size()) {
    // Accumulate into regs
    for (size_t reg = 0; reg < pRegs.size(); ++reg) {
      cc.vmovups(pRegsAcc[reg], zero_ymm);
    }

    for (size_t k = 0; k < K; ++k) {
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        pHot[reg] = false;
        for (size_t el = 0; el < REG_WIDTH; ++el) {
          auto n_ = el + reg * REG_WIDTH;
          if (W[(n + n_) * K + k]) {
            pHot[reg] = true;
            break;
          }
        }
      }
      bool any_hot = false;
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        any_hot |= pHot[reg];
      }
      if (!any_hot)
        continue;

      // Load activation
      size_t offset = k * sizeof(float);
      cc.vbroadcastss(x86::ymm0, x86::ptr(input, offset));
      std::vector<x86::Mem> regMem(4);
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        if (pHot[reg]) {
          auto off = reg * REG_WIDTH;
          auto d =
              Data256::fromF32(static_cast<float>(W[(n + off + 0) * K + k]),
                               static_cast<float>(W[(n + off + 1) * K + k]),
                               static_cast<float>(W[(n + off + 2) * K + k]),
                               static_cast<float>(W[(n + off + 3) * K + k]),
                               static_cast<float>(W[(n + off + 4) * K + k]),
                               static_cast<float>(W[(n + off + 5) * K + k]),
                               static_cast<float>(W[(n + off + 6) * K + k]),
                               static_cast<float>(W[(n + off + 7) * K + k]));
          x86::Mem cymm = cc.newYmmConst(ConstPool::kScopeLocal, d);
          regMem[reg] = cymm;
        }
      }
      for (size_t reg = 0; reg < pRegs.size(); ++reg) {
        if (pHot[reg]) {
          cc.vfmadd231ps(pRegsAcc[reg], x86::ymm0, regMem[reg]);
        }
      }
    }
    for (size_t reg = 0; reg < pRegs.size(); ++reg) {
      auto offset = (n + reg * REG_WIDTH) * sizeof(float);
      cc.vmovups(x86::ptr(output, offset), pRegsAcc[reg]);
    }
  }

  cc.endFunc();
  cc.finalize();

  Func fn;
  Error err = rt.add(&fn, &code);
  //std::cout << logger.data();
  if (err) {
    std::cerr << "error! " << err << "\n";
    return nullptr;
  }
  return fn;
}

struct YmmPool {
  x86::Ymm getReg() {
    assert(avail_.size() && "ran out of registers");
    auto reg = avail_.back();
    avail_.pop_back();
    return reg;
  }
  static x86::Ymm toYmm(x86::Xmm x) {
    if (x == x86::xmm0) {
      return x86::ymm0;
    }
    return x86::ymm0;
  }
  static x86::Xmm toXmm(x86::Ymm y) {
    if (y == x86::ymm0) {
      return x86::xmm0;
    }
    return x86::xmm0;
  }
  void freeReg(x86::Ymm y) {
    avail_.emplace(avail_.begin(), y);
  }
  std::vector<x86::Ymm> avail_ = {
    x86::ymm0,
    x86::ymm1,
    x86::ymm2,
    x86::ymm3,
    x86::ymm4,
    x86::ymm5,
    x86::ymm6,
    x86::ymm7,
    x86::ymm8,
    x86::ymm9,
    x86::ymm10,
    x86::ymm11,
    x86::ymm12,
    x86::ymm13,
    x86::ymm14,
    x86::ymm15
  };
};

// p: register in question
bool isHot(float *W, size_t n, size_t p, size_t k, size_t K) {
  size_t real_n = n + p * REG_WIDTH;
  for (size_t val = 0; val < REG_WIDTH; ++val) {
    if (W[(real_n + val) * K + k]) {
      return true;
    }
  }
  return false;
}

x86::Mem getData(float *W, size_t n, size_t p, size_t k, size_t K, x86::Compiler& cc) {
  size_t off = p * REG_WIDTH;
  auto d =
    Data256::fromF32(static_cast<float>(W[(n + off + 0) * K + k]),
        static_cast<float>(W[(n + off + 1) * K + k]),
        static_cast<float>(W[(n + off + 2) * K + k]),
        static_cast<float>(W[(n + off + 3) * K + k]),
        static_cast<float>(W[(n + off + 4) * K + k]),
        static_cast<float>(W[(n + off + 5) * K + k]),
        static_cast<float>(W[(n + off + 6) * K + k]),
        static_cast<float>(W[(n + off + 7) * K + k]));
  return cc.newYmmConst(ConstPool::kScopeLocal, d);
}

x86::Mem getDataT(float *W, size_t n, size_t p, size_t k, size_t K, x86::Compiler& cc) {
  size_t off = p;
  auto d =
    Data256::fromF32(
        static_cast<float>(W[(n + off) * K + k + 0]),
        static_cast<float>(W[(n + off) * K + k + 1]),
        static_cast<float>(W[(n + off) * K + k + 2]),
        static_cast<float>(W[(n + off) * K + k + 3]),
        static_cast<float>(W[(n + off) * K + k + 4]),
        static_cast<float>(W[(n + off) * K + k + 5]),
        static_cast<float>(W[(n + off) * K + k + 6]),
        static_cast<float>(W[(n + off) * K + k + 7]));
  return cc.newYmmConst(ConstPool::kScopeLocal, d);
}


Func jit_parameterized(float *W, size_t N, size_t K,
  size_t num_accumulators,
  size_t num_broadcast,
  bool use_fma,
  bool sparse_aware,
  JitRuntime &rt) {

  CodeHolder code;
  code.init(rt.codeInfo());
  StringLogger logger;
  code.setLogger(&logger);

  x86::Compiler cc(&code);
  cc.addFunc(FuncSignatureT<void, const float *, float *>());

  // Only use two GPs
  x86::Gp input = cc.newIntPtr("input");
  x86::Gp output = cc.newIntPtr("output");
  cc.setArg(0, input);
  cc.setArg(1, output);

  YmmPool yp;

  // Parallelism
  std::vector<x86::Ymm> pRegsAcc;
  std::vector<x86::Ymm> pRegsTmp;
  for (auto p = 0; p < num_accumulators; ++p) {
    pRegsAcc.emplace_back(yp.getReg());
    //if (!use_fma) {
    // In the FMA case this is alternated with Acc
    pRegsTmp.emplace_back(yp.getReg());
    //}
  }

  // Broadcasts
  std::vector<x86::Ymm> bRegs;

  size_t n = 0;
  bool alternator = false;

  auto iter = [&](size_t num_acc) {
    // Zero accumulators
    for (size_t p = 0; p < num_acc; ++p) {
      cc.vxorps(pRegsAcc[p], pRegsAcc[p], pRegsAcc[p]);
      if (use_fma) {
        // FMA uses tmp to accumulate as well
        cc.vxorps(pRegsTmp[p], pRegsTmp[p], pRegsTmp[p]);
      }
    }
    size_t k = 0;
    std::vector<size_t> bs;
    // Accumulate broadcasts
    while (k < K) {
      // Check to accumulate broadcasts
      for (size_t p = 0; p < num_acc; ++p) {
        if (isHot(W, n, p, k, K) && sparse_aware) {
          bs.emplace_back(k);
          break;
        }
      }
      ++k;
      // If we've saturated broadcasts or finished, start compute
      if (bs.size() == num_broadcast || k == K) {

        for (size_t b = 0; b < bs.size(); ++b) {
          bRegs.emplace_back(yp.getReg());
          size_t offset = bs[b] * sizeof(float);
          cc.vbroadcastss(bRegs[b], x86::ptr(input, offset));
        }

        // b == k
        for (auto b = 0; b < bs.size(); ++b) {
          alternator = !alternator;
          for (size_t p = 0; p < num_acc; ++p) {
            if (isHot(W, n, p, bs[b], K) && sparse_aware) {
              auto bReg = bRegs[b];
              auto d = getData(W, n, p, bs[b], K, cc);

              if (use_fma) {
                if (alternator) {
                  cc.vfmadd231ps(pRegsAcc[p], bReg, d);
                } else {
                  cc.vfmadd231ps(pRegsTmp[p], bReg, d);
                }
              } else {
                // TODO split
                cc.vmulps(pRegsTmp[p], bReg, d);
              }
            }
          }
          for (size_t p = 0; p < num_acc; ++p) {
            if (isHot(W, n, p, bs[b], K) && sparse_aware) {
              if (!use_fma) {
                cc.vaddps(pRegsAcc[p], pRegsAcc[p], pRegsTmp[p]);
              }
            }
          }
        }

        for (auto reg : bRegs) {
          yp.freeReg(reg);
        }

        bRegs.clear();
        bs.clear();
      }
    }
    for (size_t p = 0; p < num_acc; ++p) {
      if (use_fma) {
        cc.vaddps(pRegsAcc[p], pRegsAcc[p], pRegsTmp[p]);
      }
      size_t offset = (n + p * REG_WIDTH) * sizeof(float);
      cc.vmovups(x86::ptr(output, offset), pRegsAcc[p]);
    }
  };

  if (N >= REG_WIDTH * num_accumulators) {
    for (; n < N; n += REG_WIDTH * num_accumulators) {
      iter(num_accumulators);
    }
    if (n != N) {
      n -= REG_WIDTH * num_accumulators;
    }
  }

  size_t acc_left = (N - n) / REG_WIDTH;
  if (acc_left) {
    iter(acc_left);
    n += acc_left * REG_WIDTH;
  }

  for (auto reg : pRegsTmp) {
    yp.freeReg(reg);
  }
  for (auto reg : pRegsAcc) {
    yp.freeReg(reg);
  }
  pRegsAcc.clear();
  pRegsTmp.clear();

  size_t N_left = N - n;

  // TODO speed this up -- faster code below (unfinished)
  for (size_t n_ = 0; n_ < N_left; ++n_) {
    cc.vxorps(x86::xmm0, x86::xmm0, x86::xmm0);
    for (size_t k = 0; k < K; ++k) {
      size_t offset = k * sizeof(float);
      cc.movss(x86::xmm1, x86::ptr(input, offset));
      auto d = Data128::fromF32(static_cast<float>(W[(n + n_) * K + k]), 0,0,0);
      cc.mulps(x86::xmm1, cc.newXmmConst(ConstPool::kScopeLocal, d));
      cc.addps(x86::xmm0, x86::xmm1);
    }
    size_t offset = (n + n_) * sizeof(float);
    cc.movss(x86::ptr(output, offset), x86::xmm0);
  }

/* UNFINISHED SPEED UP OF EDGE CASE
  std::vector<x86::Ymm> pRegsAccT;
  // For this we accumulate transposed
  for (size_t n_ = 0; n_ < N_left; ++n_) {
    pRegsAccT.emplace_back(yp.getReg());
  }
  for (size_t n_ = 0; n_ < N_left; ++n_) {
    cc.vxorps(pRegsAccT[n_], pRegsAccT[n_], pRegsAccT[n_]);
  }
 
  size_t k = 0; 
  for (; k < K; k+=REG_WIDTH) {
    for (size_t n_ = 0; n_ < N_left; ++n_) {
      // exploit the cyclic nature of the YmmPool
      auto reg = yp.getReg();
      auto d = getDataT(W, n + n_, n_, k, K, cc);
      cc.vmovups(reg, x86::ptr(input, k * sizeof(float)));
      if (use_fma) {
        cc.vfmadd231ps(pRegsAccT[n_], reg, d);
      } else {
        cc.vmulps(reg, reg, d);
        cc.vaddps(pRegsAccT[n_], pRegsAccT[n_], reg);
      }
      yp.freeReg(reg);
    }
  }
  for (size_t n_ = 0; n_ < N_left; ++n_) {
    cc.vhaddps(pRegsAccT[n_], pRegsAccT[n_], pRegsAccT[n_]);
    cc.vhaddps(pRegsAccT[n_], pRegsAccT[n_], pRegsAccT[n_]);
    cc.vhaddps(pRegsAccT[n_], pRegsAccT[n_], pRegsAccT[n_]);
    // store 0 in 4 for xmm
    
  }
  for (; k < K; ++k) {
    for (size_t n_ = 0; n_ < N_left; ++n_) {
      size_t off = k * sizeof(float);
      auto reg = yp.toXmm(yp.getReg());
      cc.movss(reg, x86::ptr(input, off));
      auto d = Data128::fromF32(static_cast<float>(W[(n + n_) * K + k]), 0,0,0);
      cc.mulps(reg, cc.newXmmConst(ConstPool::kScopeLocal, d));
      cc.addps(yp.toXmm(pRegsAccT[n_]), reg);
      yp.freeReg(yp.toYmm(reg));
    }
  }
  for (size_t n_ = 0; n_ < N_left; ++n_) {
    size_t offset = (n + n_) * sizeof(float);
    cc.movd(x86::ptr(output, offset), yp.toXmm(pRegsAccT[n_]));
  }
*/

  cc.endFunc();
  cc.finalize();

  Func fn;
  Error err = rt.add(&fn, &code);
  //std::cout << logger.data();
  if (err) {
    std::cerr << "error! " << err << "\n";
    return nullptr;
  }
  return fn;

}

struct Timer {
  Timer(std::string name, size_t iters, size_t flops=0)
      : name_(name), start_(std::chrono::high_resolution_clock::now()), iters_(iters), flops_(flops) {}
  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start_;
    std::cerr << name_ << ":\t " << diff.count() * 1e3 << "ms \t";
    if (flops_) {
      std::cerr << flops_ * iters_ / diff.count() / 1e9 << " GFlops\n";
    } else {
      std::cerr << iters_ / diff.count() << "iter/s\n";
    }
  }
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  size_t iters_;
  size_t flops_;
};

template <typename T> void dump(size_t n, size_t k, T *arr) {
  std::cerr << "\n---\n";
  for (auto i = 0ULL; i < n; ++i) {
    for (auto j = 0ULL; j < k; ++j) {
      std::cerr << static_cast<float>(arr[i * k + j]) << "\t";
    }
    std::cerr << "\n";
  }
  std::cerr << "\n---\n";
}

bool differ(float a, float b, float rtol, float atol) {
  auto diff = std::abs(a - b);
  auto tol = std::max(rtol * std::abs(a), atol);
  if (diff > tol) {
    std::cerr << "differ by " << diff << "\n";
  }
  return diff > tol;
}

void bench(size_t n, size_t k, float s, size_t b) {
  float *arr = genRandomActivations(k);
  float *w = genRandomSparseWeights(k, n, s, b);

  float *out = (float *)calloc(n, sizeof(float));
  ref_mv(arr, w, out, n, k);
  float *out2 = (float *)calloc(n, sizeof(float));

  JitRuntime rt;
  auto iters = 1000000;
#define CHECK(fn)                                                              \
  {                                                                            \
    out2[0] = 0; out2[1] = 1337; \
    fn(arr, out2);\
    for (auto j = 0ULL; j < n; ++j) {                                          \
      if (differ(out2[j], out[j], 0.001, 0.01)) { \
        std::cerr << "mismatch at " << j << " (" << out2[j] << " vs "          \
                  << out[j] << ")\n";                                          \
        dump(1, n, out2);                                                      \
        dump(1, n, out);                                                       \
        return;                                                                \
      }                                                                        \
    }                                                                          \
  }
#define TIME(fn)                                                               \
{ \
    for (auto i = 0; i < iters/100; ++i) {                                         \
      fn(arr, out2);                                                           \
    }                                                                          \
}\
  {                                                                            \
    std::stringstream ss;                                                      \
    ss << #fn << " " << n << "x" << k << " " << s << " (" << b\
       << " blocksize)";                                                           \
    Timer t(ss.str(), iters, n*k*2);                                           \
    for (auto i = 0; i < iters; ++i) {                                         \
      fn(arr, out2);                                                           \
    }                                                                          \
  }

  //{
  //  auto fn = jit(w, n, k, rt);
  //  fn(arr, out2);
  //  CHECK;
  //  TIME(fn);
  //}
  //{
  //  auto fn2 = jit_f(w, n, k, 4, rt);
  //  fn2(arr, out2);
  //  CHECK;
  //  TIME(fn2);
  //}
  //{
  //  auto fn3 = jit_ff(w, n, k, 4, rt);
  //  fn3(arr, out2);
  //  CHECK;
  //  TIME(fn3);
  //}
  //{
  //  auto fn_4_1 = jit_parameterized(w, n, k, 4, 1, false, true, rt);
  //  CHECK(fn_4_1);
  //  TIME(fn_4_1);
  //}
  //{
  //  auto fn_4_2 = jit_parameterized(w, n, k, 4, 2, false, true, rt);
  //  CHECK(fn_4_2);
  //  TIME(fn_4_2);
  //}
  //{
  //  auto fn_4_1_fma = jit_parameterized(w, n, k, 4, 1, true, true, rt);
  //  CHECK(fn_4_1_fma);
  //  TIME( fn_4_1_fma);
  //}
  {
    auto fn_4_8_fma = jit_parameterized(w, n, k, 4, 8, true, true, rt);
    CHECK(fn_4_8_fma);
    TIME( fn_4_8_fma);
  }
  //{
  //  auto fn_2_4_fma = jit_parameterized(w, n, k, 2, 4, true, true, rt);
  //  CHECK(fn_2_4_fma);
  //  TIME( fn_2_4_fma);
  //}
  //{
  //  auto fn_4_4_fma = jit_parameterized(w, n, k, 4, 4, true, true, rt);
  //  CHECK(fn_4_4_fma);
  //  TIME( fn_4_4_fma);
  //}
  //std::cerr << "--\n";
  free(arr);
  free(w);
  free(out);
  free(out2);
}

int main() {
  for (size_t n : std::vector<size_t>{8, 64, 128, 256}) {
    for (size_t k : std::vector<size_t>{8, 32, 128, 256}) {
      for (float s : std::vector<float>{0.1f, 0.5f, 0.8f, 0.9f, 0.95f}) {
        for (size_t b : std::vector<size_t>{1, 2, 4, 8}) {
          bench(n, k, s, b);
        }
      }
    }
  }
  // bench(n, k, 0.85f);
  // bench(n, k, 0.9f);
  // bench(n, k, 0.95f);
  // bench(n, k, 0.98f);
  return 0;
  size_t n = 128;
  size_t k = 256;
  float *arr = genRandomActivations(k);
  float *w = genRandomSparseWeights(k, n, 0.9f);

  // for (int i = 0; i < k; ++i) {
  //	std::cerr << arr[i] << " " << kout[i] << ", ";
  //}
  // std::cerr << "\n";

  // for (auto j = 0ULL; j < k; ++j) {
  //  std::cerr << static_cast<float>(arr[j]) << "\t";
  //}
  // std::cerr << "\n---\n";
  // for (auto i = 0ULL; i < n; ++i) {
  //  for (auto j = 0ULL; j < k; ++j) {
  //    std::cerr << static_cast<float>(w[i * k + j]) << "\t";
  //  }
  //  std::cerr << "\n";
  //}
  // std::cerr << "\n---\n";

  float *out = (float *)calloc(n, sizeof(float));
  ref_mv(arr, w, out, n, k);
  // for (auto j = 0ULL; j < n; ++j) {
  //  std::cerr << static_cast<float>(out[j]) << "\t";
  //}
  // std::cerr << "\n";

  float *out2 = (float *)calloc(n, sizeof(float));
  JitRuntime rt;
  auto fn = jit(w, n, k, rt);
  // float* kout = (float*)malloc(sizeof(float) * k);
  {
    Timer t("test", 1000);
    for (auto i = 0; i < 1000; ++i) {
      fn(arr, out2);
    }
  }
  // gather_mv(arr, w, out2, n, k);
  // for (auto j = 0ULL; j < n; ++j) {
  //  std::cerr << static_cast<float>(out2[j]) << "\t";
  //}
  // std::cerr << "\n";
  for (auto j = 0ULL; j < n; ++j) {
    if (!(std::abs(out2[j] - out[j]) < 0.00001 * out[j])) {
      std::cerr << "mismatch at " << j << "\n";
    }
  }
  std::cerr << "done\n";

  free(arr);
  free(w);
  free(out);
  free(out2);
  return 0;
}
