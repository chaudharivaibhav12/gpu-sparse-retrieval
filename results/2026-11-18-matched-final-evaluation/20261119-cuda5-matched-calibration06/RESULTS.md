# Matched full-collection calibration06

Measured source: `61c8a00d8323d49a011d8909113a8755906323c9`. Full8,841,823passages,500calibration queries,RTX4070,k10,batch1,3processblocks,3warmups and20rounds. Resident parsed-query through ordered host top-k boundary.

| System | Median ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| prunets256 | 0.271837 | 1.015918 | 1.903299 |
| prunets1024 | 0.368284 | 0.811226 | 1.001470 |
| range | 0.690613 | 0.962983 | 1.085530 |
| compact | 5.871430 | 7.994977 | 8.562405 |
| bmp | 12.635464 | 33.178055 | 55.384473 |

Paired geometric speed ratios below use per-query20-round medians and crossed process/query bootstrap,10000draws,seed20261118. They are not ratios of the system medians above. Intervals conditional on this calibration population and three shared-host processblocks; no familywise coverage claim.

| Contrast (first system faster by) | Paired ratio | Conditional95% |
|---|---:|---:|
| prunets256 / compact | 19.6274x | 18.6474–20.6564x |
| prunets256 / bmp | 42.9663x | 41.6703–44.2779x |
| prunets1024 / compact | 16.3265x | 15.7017–16.9921x |
| prunets1024 / bmp | 35.7403x | 34.7378–36.7886x |
| prunets256 / range | 2.3788x | 2.2685–2.4900x |
| prunets1024 / range | 1.9787x | 1.9131–2.0460x |

Independent validation checked1,750,000ranked scores,all complete runtime snapshots,15timing monitor/ownership/CPU eligibility replays,64Git source bindings,178runner artifact hashes and all4compact JIT scoring compilations. Original exact-source binaries/compiler logs from preparation05 reused explicitly.

The12compact/BMP mechanical comparisons remain BLOCKED by contract/arithmetic labels. Scoped comparison is justified only for the independently verified exact integer domain:uint8 impacts,unique cap32terms,<=128terms,255sumweights<2^24,canonical original numeric-ID ties. FP32 partial sums/products are exact there;U32/U64 share the resulting integers. No unrestricted floating-weight or equal physical-arithmetic claim.

Corrected residentCPU-BMP useswideU64bounds/scores,not released unsafeU16BMP. Do not call it the strongest possibleCPUcontrol or a speedup over unmodifiedBMP. CPUstring/GPUfeature representation differences and shared-host sampled exclusivity limits remain disclosed.

Arnav contributed core range/V2 scorer work;Parva contributed pruning/profiling/BMP work. This study supplies matched controls and evidence,not a new retrieval kernel.

Attempt05 and all earlier failures remain preserved;05timings are not pooled with06. W256 remains primary latency choice,W1024secondarytail option; all5systems carry forward unchanged. Heldout6480contents remain untouched at calibration acceptance.

Terminal evidence SHA256: `ba3b39d527e2110b10630639613681d5fadcd59c8f4b63481e86ce39fbe4ed7b`. Records:OBS-20261119-MATCHED06-INDEPENDENT-REVIEW,DEC-20261119-HELDOUT-SUBMISSION-AUTHORIZED.
