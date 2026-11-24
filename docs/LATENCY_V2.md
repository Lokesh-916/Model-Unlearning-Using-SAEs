# Latency of DSG, StreamGuard v1, v2 and v2-R (measurement only)

**Summary.** Reading one multiple-choice answer takes about a third longer with StreamGuard v2 or v2-R (34-38 % over the plain model), because the gate needs an extra pass through the first nine layers, while DSG and v1 add 1-3 %.
For a written answer of 96 tokens every guard adds little (DSG and v1 at most 1.3 %, v2 at most 2.7 %, v2-R at most 6.0 %, reached when it fires on a 2,048-token prompt).

No method, threshold or result changes. Reproduce: `scripts/latency_v2.py bench` then `report` (lab wrapper `scripts/latency_v2_run.sh`; raw cells `docs/LATENCY_V2.json`); the script file is also in the clean repository.

**Setup.** One RTX 2000 Ada (16 GB), batch size 1, bfloat16, TransformerLens, no other compute process on the GPU (checked before every
repetition; idle desktop processes of other users hold some graphics memory). 10 warm-up + 50 timed runs per cell with
`torch.cuda.synchronize`, methods interleaved in rotating order. Harmless prompts only: an MMLU DEV question (hazard-adjacent
subjects excluded) behind WikiText filler, exactly 64 / 256 / 1024 / 2048 tokens in the chat template. Every written answer is
exactly 96 greedy tokens (end tokens ignored).

**Paths.** Decisions are forced: the gate is computed, then "not fired" / "fired" is imposed (harmless prompts rarely fire). Logging
inside the hooks is off. *v2*: pass 1 = prompt through layer 8, layer-8 SAE, CUSUM; pass 2 = plain model (not fired) or the model with the
clamp (fired); in the written-answer workload the gate is checked on the prompt only. *v2-R*: in the multiple-choice workload
like v2 with the refusal vector at layer 10. In the written-answer workload, not fired = the layer-8 gate runs inside the answering pass and after
every token (no separate pass 1); fired = the prompt pass with the gate, then the prompt is fed again with the refusal vector on the prompt and the
first 8 tokens (gate latched after the fire). Fires in the middle of an answer are not timed (one more pass over the sequence).
*v2 cached* reuses pass 1's activations of layers 0-2; **not used in the paper's results**.

**Implementation choices that affect timing, identical for every method.** (i) One plain softmax at the last position replaces the
evaluation scorer, which recomputes the unembedding in chunks twice; same letter on all 12 checked prompts. (ii) The transposed unembedding
matrix is copied once instead of on every call; bit-identical logits. Both would otherwise hide the overheads. (iii) The layers are
unchanged TransformerLens code, slower than an optimised server, so read the percentages, not the absolute times; on a faster server
the base shrinks and the percentages may differ. Peak memory is the allocator peak without the SAEs a method does not use; it includes the precomputed
1.1 GiB unembedding copy. Differences below about 2 % at 1,024 written tokens are within the run-to-run spread (95th percentile up to 236 ms above the median).

**Checks before timing** (3 prompts per length): v2 multiple-choice probabilities equal the frozen `sg2-l08` method bit for bit; cached v2 = v2 (probabilities
and 24 generated tokens); not-fired v2 = base; DSG and v1 streaming tokens equal the development
streaming code (3 prompts each at 64 and 1,024 tokens, 32 tokens, checked once before the unembedding change). 0 mismatches.

**Expected average overhead** = (1 - f) x not fired + f x fired, using medians and the measured fire rate f: v2 and v2-R 0.019 (MMLU false
alarms) and 0.94 (WMDP-Bio); DSG 0.017 / 0.878 and v1 0.046 / 0.926 from the TEST tables (seed 0). Each method is compared at its own rate.

**Caveat on those rates.** 0.019 was measured on plain MMLU questions. In the unforced prompts of this benchmark the v2 gate fired on:

| prompts (60 per cell) | 64 | 256 | 1024 | 2048 tokens |
|---|---|---|---|---|
| multiple choice | 2 | 1 | 6 | 15 |
| written answer | 1 | 2 | 9 | 16 |

So on long harmless prompts the false-alarm rate is probably higher than 0.019 and the average overhead closer to the fired rows (a running-maximum
gate has more chances to fire on longer text). This is an incidental observation on 60 filler-padded prompts per cell that are
partly shared between the workloads, not a designed false-alarm test; the cause was not investigated.

## Multiple choice (one forward pass)

Each cell: median / 95th percentile in ms, overhead of the median against the base model, peak GPU memory.

| method | path | 64 tokens | 256 tokens | 1024 tokens | 2048 tokens |
|---|---|---|---|---|---|
| base model | – | 129.5 / 130.2, 7.22 GiB | 151.4 / 152.0, 7.24 GiB | 309.9 / 310.8, 7.29 GiB | 653.5 / 654.6, 7.51 GiB |
| DSG | not fired | 130.8 / 131.6, +1.0 %, 7.36 GiB | 153.4 / 154.3, +1.3 %, 7.38 GiB | 317.9 / 319.0, +2.6 %, 7.48 GiB | 669.1 / 670.7, +2.4 %, 7.65 GiB |
| DSG | fired | 130.8 / 131.7, +1.0 %, 7.36 GiB | 153.5 / 154.1, +1.4 %, 7.38 GiB | 317.7 / 318.6, +2.5 %, 7.48 GiB | 669.2 / 670.6, +2.4 %, 7.65 GiB |
| StreamGuard v1 | not fired | 130.5 / 131.9, +0.8 %, 7.36 GiB | 152.8 / 153.5, +0.9 %, 7.38 GiB | 315.8 / 316.4, +1.9 %, 7.48 GiB | 665.6 / 666.5, +1.8 %, 7.65 GiB |
| StreamGuard v1 | fired | 130.8 / 132.5, +1.0 %, 7.36 GiB | 153.4 / 154.3, +1.3 %, 7.38 GiB | 318.1 / 319.2, +2.6 %, 7.48 GiB | 669.9 / 671.5, +2.5 %, 7.65 GiB |
| StreamGuard v2 | not fired | 173.7 / 175.8, +34.1 %, 7.50 GiB | 203.1 / 204.0, +34.2 %, 7.52 GiB | 419.9 / 421.0, +35.5 %, 7.59 GiB | 886.4 / 887.5, +35.6 %, 7.79 GiB |
| StreamGuard v2 | fired | 175.1 / 176.6, +35.2 %, 7.50 GiB | 205.4 / 206.3, +35.7 %, 7.52 GiB | 428.0 / 429.1, +38.1 %, 7.62 GiB | 902.6 / 903.8, +38.1 %, 7.79 GiB |
| StreamGuard v2-R | not fired | 173.7 / 174.4, +34.1 %, 7.36 GiB | 203.2 / 204.1, +34.2 %, 7.38 GiB | 419.7 / 420.8, +35.4 %, 7.45 GiB | 886.2 / 887.5, +35.6 %, 7.65 GiB |
| StreamGuard v2-R | fired | 173.8 / 174.4, +34.2 %, 7.36 GiB | 203.3 / 203.9, +34.3 %, 7.38 GiB | 419.8 / 421.1, +35.5 %, 7.45 GiB | 886.3 / 887.8, +35.6 %, 7.65 GiB |
| v2 cached (not used in the paper's results) | not fired | 160.9 / 162.0, +24.2 %, 7.50 GiB | 188.0 / 188.8, +24.2 %, 7.52 GiB | 385.1 / 386.9, +24.3 %, 7.60 GiB | 811.0 / 812.3, +24.1 %, 7.80 GiB |
| v2 cached (not used in the paper's results) | fired | 162.4 / 164.4, +25.4 %, 7.50 GiB | 190.0 / 190.7, +25.5 %, 7.52 GiB | 393.5 / 396.0, +27.0 %, 7.62 GiB | 826.6 / 827.9, +26.5 %, 7.80 GiB |
| DSG | expected average overhead at the MMLU false-alarm rate (0.017) | 130.8 ms, +1.0 % | 153.4 ms, +1.3 % | 317.9 ms, +2.6 % | 669.1 ms, +2.4 % |
| StreamGuard v1 | expected average overhead at the MMLU false-alarm rate (0.046) | 130.6 ms, +0.8 % | 152.8 ms, +0.9 % | 315.9 ms, +1.9 % | 665.8 ms, +1.9 % |
| StreamGuard v2 | expected average overhead at the MMLU false-alarm rate (0.019) | 173.7 ms, +34.1 % | 203.2 ms, +34.2 % | 420.1 ms, +35.5 % | 886.8 ms, +35.7 % |
| StreamGuard v2-R | expected average overhead at the MMLU false-alarm rate (0.019) | 173.7 ms, +34.1 % | 203.2 ms, +34.2 % | 419.7 ms, +35.4 % | 886.2 ms, +35.6 % |
| v2 cached (not used in the paper's results) | expected average overhead at the MMLU false-alarm rate (0.019) | 160.9 ms, +24.3 % | 188.1 ms, +24.2 % | 385.3 ms, +24.3 % | 811.3 ms, +24.1 % |
| DSG | expected average overhead at the WMDP-Bio fire rate (0.878) | 130.8 ms, +1.0 % | 153.5 ms, +1.4 % | 317.8 ms, +2.5 % | 669.2 ms, +2.4 % |
| StreamGuard v1 | expected average overhead at the WMDP-Bio fire rate (0.926) | 130.8 ms, +1.0 % | 153.4 ms, +1.3 % | 317.9 ms, +2.6 % | 669.6 ms, +2.5 % |
| StreamGuard v2 | expected average overhead at the WMDP-Bio fire rate (0.94) | 175.0 ms, +35.1 % | 205.3 ms, +35.6 % | 427.5 ms, +37.9 % | 901.7 ms, +38.0 % |
| StreamGuard v2-R | expected average overhead at the WMDP-Bio fire rate (0.94) | 173.8 ms, +34.2 % | 203.3 ms, +34.3 % | 419.8 ms, +35.5 % | 886.3 ms, +35.6 % |
| v2 cached (not used in the paper's results) | expected average overhead at the WMDP-Bio fire rate (0.94) | 162.4 ms, +25.4 % | 189.9 ms, +25.4 % | 393.0 ms, +26.8 % | 825.7 ms, +26.3 % |

## Written answer (96 generated tokens)

Each cell: median / 95th percentile in ms, overhead of the median against the base model, peak GPU memory.

| method | path | 64 tokens | 256 tokens | 1024 tokens | 2048 tokens |
|---|---|---|---|---|---|
| base model | – | 11476.8 / 11485.9, 7.28 GiB | 11536.7 / 11547.4, 7.57 GiB | 11954.2 / 12098.1, 8.75 GiB | 12751.2 / 12787.1, 10.32 GiB |
| DSG | not fired | 11571.4 / 11582.1, +0.8 %, 7.42 GiB | 11622.5 / 11646.2, +0.7 %, 7.71 GiB | 12034.7 / 12119.7, +0.7 %, 8.89 GiB | 12841.6 / 12890.6, +0.7 %, 10.46 GiB |
| DSG | fired | 11598.2 / 11606.9, +1.1 %, 7.42 GiB | 11651.4 / 11674.3, +1.0 %, 7.71 GiB | 12065.6 / 12099.2, +0.9 %, 8.89 GiB | 12878.7 / 12920.5, +1.0 %, 10.46 GiB |
| StreamGuard v1 | not fired | 11581.4 / 11587.0, +0.9 %, 7.42 GiB | 11633.8 / 11658.6, +0.8 %, 7.71 GiB | 12061.8 / 12134.2, +0.9 %, 8.89 GiB | 12891.0 / 12942.2, +1.1 %, 10.46 GiB |
| StreamGuard v1 | fired | 11608.1 / 11619.4, +1.1 %, 7.42 GiB | 11666.8 / 11685.2, +1.1 %, 7.71 GiB | 12089.0 / 12145.2, +1.1 %, 8.89 GiB | 12917.4 / 12976.0, +1.3 %, 10.46 GiB |
| StreamGuard v2 | not fired | 11518.6 / 11528.6, +0.4 %, 7.56 GiB | 11586.0 / 11596.7, +0.4 %, 7.85 GiB | 12068.2 / 12174.9, +1.0 %, 9.03 GiB | 12979.3 / 13015.1, +1.8 %, 10.60 GiB |
| StreamGuard v2 | fired | 11636.6 / 11648.0, +1.4 %, 7.56 GiB | 11701.4 / 11719.1, +1.4 %, 7.85 GiB | 12171.8 / 12323.8, +1.8 %, 9.03 GiB | 13099.4 / 13145.4, +2.7 %, 10.60 GiB |
| StreamGuard v2-R | not fired | 11536.0 / 11547.5, +0.5 %, 7.42 GiB | 11592.3 / 11604.2, +0.5 %, 7.71 GiB | 12011.1 / 12150.5, +0.5 %, 8.89 GiB | 12806.5 / 12846.8, +0.4 %, 10.46 GiB |
| StreamGuard v2-R | fired | 11618.1 / 11625.8, +1.2 %, 7.42 GiB | 11704.2 / 11717.9, +1.5 %, 7.71 GiB | 12312.4 / 12503.0, +3.0 %, 8.89 GiB | 13517.4 / 13554.6, +6.0 %, 10.46 GiB |
| v2 cached (not used in the paper's results) | not fired | 11506.8 / 11521.5, +0.3 %, 7.56 GiB | 11572.0 / 11585.7, +0.3 %, 7.86 GiB | 12032.2 / 12205.2, +0.7 %, 9.06 GiB | 12905.0 / 12943.1, +1.2 %, 10.66 GiB |
| v2 cached (not used in the paper's results) | fired | 11628.0 / 11638.6, +1.3 %, 7.56 GiB | 11685.2 / 11707.3, +1.3 %, 7.86 GiB | 12137.8 / 12373.3, +1.5 %, 9.06 GiB | 13024.3 / 13073.6, +2.1 %, 10.66 GiB |
| DSG | expected average overhead at the MMLU false-alarm rate (0.017) | 11571.9 ms, +0.8 % | 11623.0 ms, +0.7 % | 12035.2 ms, +0.7 % | 12842.2 ms, +0.7 % |
| StreamGuard v1 | expected average overhead at the MMLU false-alarm rate (0.046) | 11582.6 ms, +0.9 % | 11635.3 ms, +0.9 % | 12063.0 ms, +0.9 % | 12892.2 ms, +1.1 % |
| StreamGuard v2 | expected average overhead at the MMLU false-alarm rate (0.019) | 11520.9 ms, +0.4 % | 11588.2 ms, +0.4 % | 12070.1 ms, +1.0 % | 12981.5 ms, +1.8 % |
| StreamGuard v2-R | expected average overhead at the MMLU false-alarm rate (0.019) | 11537.5 ms, +0.5 % | 11594.4 ms, +0.5 % | 12016.8 ms, +0.5 % | 12820.0 ms, +0.5 % |
| v2 cached (not used in the paper's results) | expected average overhead at the MMLU false-alarm rate (0.019) | 11509.1 ms, +0.3 % | 11574.1 ms, +0.3 % | 12034.2 ms, +0.7 % | 12907.3 ms, +1.2 % |
| DSG | expected average overhead at the WMDP-Bio fire rate (0.878) | 11595.0 ms, +1.0 % | 11647.9 ms, +1.0 % | 12061.8 ms, +0.9 % | 12874.2 ms, +1.0 % |
| StreamGuard v1 | expected average overhead at the WMDP-Bio fire rate (0.926) | 11606.2 ms, +1.1 % | 11664.3 ms, +1.1 % | 12087.0 ms, +1.1 % | 12915.4 ms, +1.3 % |
| StreamGuard v2 | expected average overhead at the WMDP-Bio fire rate (0.94) | 11629.6 ms, +1.3 % | 11694.5 ms, +1.4 % | 12165.6 ms, +1.8 % | 13092.2 ms, +2.7 % |
| StreamGuard v2-R | expected average overhead at the WMDP-Bio fire rate (0.94) | 11613.2 ms, +1.2 % | 11697.5 ms, +1.4 % | 12294.3 ms, +2.8 % | 13474.7 ms, +5.7 % |
| v2 cached (not used in the paper's results) | expected average overhead at the WMDP-Bio fire rate (0.94) | 11620.7 ms, +1.3 % | 11678.4 ms, +1.2 % | 12131.5 ms, +1.5 % | 13017.1 ms, +2.1 % |

