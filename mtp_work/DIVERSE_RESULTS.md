# Diverse MTP Acceptance Results — Swift-27B

Finetuned MTP head (`mtp_head_final.pt`) vs the original `mtp.*` head, measured on
the fixed diverse eval (`diverse_eval.jsonl`, 254 seqs) by `measure_mtp.py`.
Both heads see the identical sequences; the main 27B model is loaded once.

## Per-task acceptance (baseline → finetuned)

| Task | Baseline | Finetuned | Delta | Above noise? |
|------|---------:|----------:|------:|:------------|
| json         | 0.4182 (3150/7533)  | 0.6014 (4530/7533)  | **+0.1832** | yes |
| python       | 0.4321 (3279/7588)  | 0.6203 (4707/7588)  | **+0.1882** | yes |
| tool_call    | 0.5797 (1746/3012)  | 0.8476 (2553/3012)  | **+0.2679** | yes |
| prose        | 0.3450 (2290/6637)  | 0.3612 (2397/6637)  | **+0.0162** | **no — iterate** |
| **OVERALL**  | 0.4225 (10465/24770)| 0.5727 (14187/24770)| **+0.1503** | yes |

## Notes
- Training: two-phase (phase 1 harvests h_i from the frozen 27B on the diverse
  train set; phase 2 trains the 424.7M-param MTP head with CE + margin loss,
  early-stopped on val acceptance). Baseline val 0.4225 → best 0.5727.
- The structured domains (json/python/tool_call) respond strongly to the
  distribution-matched finetune; **prose (summarization proxy) is within noise**
  and needs a better/more diverse prose training corpus (candidate: Nemotron
  post-training text) — see iteration plan.
