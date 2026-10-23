# Depth-6 MTP training: sampler-matched acceptance objective

## Why change the training recipe

The first fine-tune kept the JP branch's native MTP head and trained one next-token step at a time with cross-entropy plus a margin term. Its held-out offline label top-1 rose from **56.96%** for the parent to **69.70%** at the best checkpoint. That proxy gain did not carry over to live speculative decoding: the single-epoch candidate scored below its untrained BF16 parent on every measured workload.

| Workload | Parent acceptance | One-step-trained acceptance | Parent decode tok/s/request | Trained decode tok/s/request |
|---|---:|---:|---:|---:|
| Real, 4K | 63.69% | 27.99% | 72.51 | 41.78 |
| Real, 32K | 72.05% | 25.42% | 62.96 | 31.80 |
| Coding, 4K | 58.06% | 25.07% | 65.51 | 39.27 |
| Real, 192K | 53.84% | 24.56% | 20.13* | 14.22 |

The candidate was trained on **775,899 labels** for one pass. The live candidate measurement was one repetition per workload; the parent had three repetitions for the short workloads and two for 192K, so the comparison has asymmetric repeat counts. The 192K cache-matched repetition had 97.9% prefix hits on both heads. The benchmark measured forced-length decode speed and speculative acceptance, not coding-task correctness. The parent itself produced different greedy output token hashes across repetitions, so text changes cannot be attributed to the fine-tune alone.

*The 192K parent decode average includes a second repetition with no prefix-cache hits; its cache-matched repetition was 24.25 tok/s/request.*

The results show that better one-step top-1 is not enough. The branch recipe uses one-token attention and does not model the deployed six-position recurrent MTP path. Its CE-plus-margin loss also does not directly reward the probability that a draft token survives the verifier's rejection sampler. Repeating that recipe or adding epochs is not supported by these results.

## Objective in this change

The head now exposes the same one-layer recurrent KV-cache path used at serving while preserving all native parameter names and tensor counts. Training unrolls six positions and maximizes expected accepted-prefix length. At each position it computes the exact top-k/top-p filtered target and draft distributions and uses their overlap

`sum_token min(p_target(token), p_draft(token))`

as the one-step rejection-sampling acceptance probability. The product of those probabilities across positions estimates the probability of accepting each longer prefix. A small target top-1 NLL anchor keeps a useful gradient when the initial filtered supports do not overlap. Greedy top-1 comparisons are diagnostics only.

The sampler is taken from the production vLLM model's `/model/generation_config.json`: sampling enabled, temperature **1.0**, top-k **20**, top-p **0.95**. Hermes omits these fields in its request, so vLLM uses that model configuration. The head A/B benchmark explicitly set temperature 0 for controlled repeatability; that was a benchmark override and must not be copied into normal training or serving. The rollout trainer records these values in its result and checkpoint metadata.

The current offline rollout uses aligned continuation tokens and hidden states from the captured corpus. It does not regenerate exact live target trajectories at every sampled prefix, so sampler-overlap metrics are a training proxy. A live matched vLLM acceptance test remains required before drawing a deployment conclusion.

## Scope and follow-up experiments

This change trains draft depth **6**, matching the current vLLM configuration. It does not change the target model, shared vocabulary projection, backbone, serving sampler, or the native MTP parameter set. Keep the original head as the rollback point and evaluate the candidate against it with identical prompts, sampler, concurrency, and output budget.

After a depth-6 result is measured, test whether training at depths above six improves the first six proposals or provides useful behavior when the speculative budget increases. Compare depths 6, 8, and possibly 10 under matched serving budgets; report actual accepted tokens per verification, throughput, and latency. A larger training depth is a hypothesis, not an expected gain, and should not alter the deployed draft depth without its own evaluation.

## Validation

CPU checks cover cached-versus-full causal attention equivalence, blocked projection NLL and gradients, filtered-policy overlap, six-step loss gradients, absolute token positions, and capture alignment. They do not validate XPU training speed, live sampler behavior, or serving acceptance.

## First AI01 run and memory adjustment

The first sampler-matched AI01 launch loaded **128,512 six-step rollouts** from the existing **775,899 training labels** and **675 held-out rollouts** (4,050 covered labels), using the verified production sampler above. The untrained parent baseline measured **1.390 expected accepted tokens per six-token proposal** on the root-disjoint held-out set. This is an offline captured-context proxy, not a live vLLM acceptance result.

That launch used 64 rollouts per optimizer batch and stopped before its first update when host available memory fell to **1.29 GiB**; swap use was **1.64 GB**, while the training container recorded **zero OOM kills**. No checkpoint or trained result was produced by that attempt. The bounded-memory change in `hermes_rollout.py` recomputes vocabulary projection blocks during top-k backward instead of retaining every block's autograd intermediates. The dense-reference CPU check confirms matching top-k values and close gradients. The follow-up run also reduces the rollout batch, so the first launch's resource guard is not mistaken for a training result.

The production vLLM container was restored and its health and model endpoints returned HTTP 200 after the stopped run. Training still needs a successful optimizer update and a matched live serving comparison before the method can be judged.

The next bounded run produced a real optimizer checkpoint at step **32** after **512/128,512 rollouts** had been visited (3,072/771,072 covered train labels). On the same **675 held-out rollouts / 4,050 labels**, the depth-6 sampler-overlap proxy was **1.608 expected accepted tokens per six-token proposal**, versus **1.389** for the untrained parent (+15.7%). This is early offline evidence only; it is not live acceptance. The candidate and optimizer checkpoint were saved before the guard stopped the run. Host full-memory PSI reached **34.44** with **2.16 GiB** available and **zero cgroup OOM kills**, so the stop was caused by sustained pressure rather than an OOM. Production and runner restoration checks passed.

The full captured pass must resume from `rollout-latest.pt`, preserving the optimizer and deterministic rollout order. To reduce repeated validation overhead and HDD write pressure, resumed runs reuse the saved parent baseline instead of recomputing it, checkpoint every 64 optimizer updates, and evaluate the 675-rollout held-out set every 128 updates. A live test must load the exact depth-6 candidate hash and retain the production sampling policy; do not treat this step-32 proxy as a live result.
