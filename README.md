# Experiment on CoT-Guard: Small Models for Strong Monitoring

[arXiv Paper](https://arxiv.org/abs/2605.12746) [repo](https://github.com/nirav0999/CoT-Guard) 
<a id="about"></a>
## Background on the paper

Code ships with little/no human review. The threat model that this paper’s authors are basing the work on is that something in the middle i.e the model router like OpenRouter, can add a hidden objective that can be malicious, which the user can’t easily see. Current monitors employing large models are expensive and small models don’t perform well. A cheap small model could be cost effective and performant. The authors used the Qwen3-4B model and trained on a bigger model’s correct monitoring answers (Qwen 32B). SFT appeared to generalise poorly to new kinds of hidden objectives, which is consistent with prior work. Authors then used RL to train on subtle hidden objectives. Authors claim that RL training on subtle hidden objectives enables generalization and small monitors like CoT-Guard can be as effective as large ones but with only a fraction of the cost.

Read [CoT-Guard: Small Models for Strong Monitoring](https://arxiv.org/abs/2605.12746) for the method and evaluation details.

## What I did and why it matters

I wanted to test why small monitors fail and more concretely whether small monitors, like CoT-Guard, are better at detecting hidden objectives that add behaviour, and worse at detecting objectives that weaken or replace behaviour the task already needs.

The paper’s threat model is a malicious router. If small monitors catch “extra code” but are weaker on “changed code”, a router simply can phrase its sabotage as a change to existing code. I describe the experiment and its limitations below.


## Inference Pipeline

1. Attack: create tasks and generate attack responses.
2. Verify: test responses for main-task correctness and side-task success.
3. Monitor: create monitor tasks from the responses and run monitors.
4. Score: evaluate monitor predictions.

<a id="citation"></a>
## Experiment and results

I introduced a new hidden objective, TLS verification, to use for the experiment. The set up consisted of 26 BigCodeBench problems (each sampled 5 times, with a monitor rollout of k=4 per reading) and varying only the placement of the hidden goal. The experiment runs were over two different scopes of data: CoT + the code solution and code-only.

In the case of CoT + code scope, CoT-Guard 4B detected the additive variant in every case (TPR 1.000) and the substitutive variant at TPR 0.837, a gap of 0.163.

All the cases that it failed were due to misattribution, i.e thinking that the side task was requested in the original problem statement although that was not mentioned.

In the case of code-only scope, CoT-Guard detected the additive variant in every case (TPR 1.000) and the substitutive variant in nearly every case (TPR 0.996), so the gap is nearly non-existent.

I also ran the full experiment over the two scopes using Cot-Guard’s base model, base Qwen3-4B to test the assumption that the gap between “additive” and “substitutive” hidden objective detection could have been created in CoT-Guard’s training. The results here are noisy. Code-only shows somehow of a gap, 0.87 additive vs 0.62 substitutive. Adding the CoT collapses both variants to about 0.13 (it could be that with the CoT present, the base model is persuaded on everything).

I wanted to test whether the detection gap between the additive vs substitutive nature of hidden objectives also shows in bigger stronger monitors. I chose Gemini-3-Flash as it was the best performing monitor in the CoT-Guard paper. In both code-only scope, and CoT + code scope, the detection gap was non-existent for this capable monitor.

| Monitor × scope | TPR additive | TPR substitutive | ΔTPR  |
|---|---|---|---|
| CoT-Guard, CoT+code  | 1.000 | 0.837 | +0.163  |
| CoT-Guard, code-only | 1.000 | 0.996 | ceiling |
| Base 4B, CoT+code | 0.133 | 0.135 | −0.002  |
| Base 4B, code-only | 0.873 | 0.619 | +0.254  |
| Gemini, CoT+code | 1.000 | 0.996 | ceiling |
| Gemini, code-only | 0.992 | 1.000 | ceiling |

These findings are only descriptive of this limited experiment (only 26 problems, one hidden objective) and do not generalize to a claim that a monitor’s ability to be worse at detecting substitutive behaviours is a property of less capable monitors when CoT is present in its scope.

## Possible next steps

It would be interesting to test on a harder dataset like BigCodeBench-Hard problems and compare CoT-Guard’s performance on Hard problems. Testing on harder datasets, particularly in agentic transcripts, is a stated limitation of this paper and it could be informative to test the performance of CoT-Guard in harder scenarios but they break the single-turn monitor prompt (the way CoT-guard is designed) and it may not be feasible given it’s just a 4B model.

Additionally, multiple objectives could be experimented with, a requested requirement next to a hidden and test if that makes the monitor more lenient on a hidden objective.
