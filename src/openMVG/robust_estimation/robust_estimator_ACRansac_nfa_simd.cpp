// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// ----------------------------------------------------------------
// CPU capability gate for the ACRANSAC AVX2 NFA kernel.
//
// !! DO NOT ADD AVX2 CODE, OR AVX2 COMPILE FLAGS, TO THIS FILE !!
//
// This TU is compiled with the project's baseline instruction set on
// purpose. It executes on CPUs whose capabilities are still unknown -- if
// the compiler were allowed to emit VEX-encoded instructions here (which
// /arch:AVX2 lets it do anywhere it likes, including in this function's own
// prologue or in the std::bitset code inside CpuInstructionSet), the process
// would die with an illegal instruction *while performing the very check
// that was supposed to prevent that*. Only
// robust_estimator_ACRansac_nfa_avx2.cpp carries AVX2 flags.
// ----------------------------------------------------------------

#include "openMVG/robust_estimation/robust_estimator_ACRansac_nfa_simd.hpp"

#if OPENMVG_ACRANSAC_NFA_AVX2_KERNEL

#include "openMVG/system/cpu_instruction_set.hpp"

namespace openMVG {
namespace robust {
namespace acransac_nfa_internal {

bool ACRansacNFA_HasAVX2()
{
  // Function-local static: initialisation is thread-safe and CPUID runs
  // exactly once per process, not once per ACRANSAC call.
  static const bool supported = []() -> bool
  {
    const openMVG::system::CpuInstructionSet cpu;
    // The kernel uses both AVX2 integer shifts and vfmadd231pd, and those
    // are two independent CPUID feature bits.
    return cpu.supportAVX2() && cpu.supportFMA();
  }();
  return supported;
}

} // namespace acransac_nfa_internal
} // namespace robust
} // namespace openMVG

#endif // OPENMVG_ACRANSAC_NFA_AVX2_KERNEL
