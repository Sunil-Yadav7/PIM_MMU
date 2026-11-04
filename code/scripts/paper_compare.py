# Paper's reported numbers (for the comparison table in the report)
PAPER = {
  "d2p_speedup_avg": 3.3, "d2p_speedup_max": 4.1,
  "p2d_speedup_avg": 3.8, "p2d_speedup_max": 5.7,
  "overall_tput_avg": 4.1, "overall_tput_max": 6.9,
  "memcpy_avg": 4.9, "memcpy_max": 6.0,
  "locality_vs_mlp_dram": 0.30,  # Fig 8: locality is ~30% of MLP
  "sw_bw_util_d2p": 0.155,       # 15.5% of peak
  "sw_avg_GBps": 8.9, "sw_peak_GBps": 57.6,
}
print(PAPER)
