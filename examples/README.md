# Examples

Owl ships two example tests. They are registered like any other test and appear in the dashboard labelled "Example". They show the tool working end to end and are not part of the general machinery; the code is in `backend/src/owl/examples/`.

| Test | Data | Scoring |
|---|---|---|
| `prompt_injection` | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections), test split, 116 samples (needs the `hf` extra) | yes/no: is the text a prompt-injection or jailbreak attempt? 1 means injection |
| `sentiment_reviews` | 24 short product reviews written for Owl, half positive (`backend/src/owl/examples/sentiment.csv`) | yes/no: is the review positive? A quick smoke test |

The [`plugins/owl-laya`](plugins/owl-laya/README.md) folder is an example provider plugin.

## Results: prompt injection (116 samples)

One run per model on a Windows laptop (i5-12450HX), with Laya on the CPU and Ollama on the NVIDIA GPU. Latency is not like-for-like between Laya and Ollama. These runs were made with Owl 0.1; Owl 0.2 shows the same runs with 95% intervals.

| model | n | accuracy | 95% CI | precision | recall | F1 | unparseable | median latency | p95 latency | wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| `laya/english` | 116 | 0.698 | 0.609–0.774 | 0.963 | 0.433 | 0.598 | 0 | 305 ms | 579 ms | 40.7 s |
| `ollama/qwen3.5:0.8b` | 116 | 0.802 | 0.720–0.864 | 0.911 | 0.683 | 0.781 | 0 | 145 ms | 177 ms | 17.6 s |
| `ollama/qwen3.5:4b` | 116 | 0.672 | 0.583–0.751 | 1.000 | 0.367 | 0.537 | 0 | 186 ms | 263 ms | 23.2 s |

With intervals this wide, the differences are weak evidence. On the same 116 samples the exact McNemar test gives p = 0.043 for qwen3.5:0.8b against Laya and p = 0.006 for qwen3.5:0.8b against qwen3.5:4b; these p-values are not corrected for making several comparisons. Only one prompt wording was used, and the 0.8b score depends on it: appending ` /no_think` (a Qwen soft switch) to the system prompt changes it from 0.802 to 0.716 on the same samples.

## Smoke checks of the providers (5 samples each, Owl 0.2)

These confirm that each route works. Five samples say nothing about which model is better.

| model | accuracy | note |
|---|---|---|
| `ollama/qwen3.5:0.8b` | 0.800 | native Ollama provider |
| `http/qwen3.5:0.8b` | 0.800 | the generic HTTP provider pointed at Ollama's `/api/chat`; same answers as the native provider |
| `hf/deepset/deberta-v3-base-injection` | 1.000 | `sequence-classification`, `label_map={"INJECTION": "1", "LEGIT": "0"}` |
| `laya/english` | 1.000 | through the `owl-laya` plugin |
