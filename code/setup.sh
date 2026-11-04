#!/usr/bin/env bash
# Fetch Ramulator (v1, the version used by the PIM-MMU paper), apply our 2-line patch, build pimsim.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d ramulator ]; then
  git clone https://github.com/CMU-SAFARI/ramulator.git ramulator
  git -C ramulator checkout 214f635
  git -C ramulator apply ../patches/ramulator_pimmmu.patch
fi
make -j"$(nproc)"
echo "built ./build/pimsim"
