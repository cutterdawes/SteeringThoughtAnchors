# Project: Steering Thought Anchors

**Cutter Dawes, Santiago Aranguri**

Previous work has found evidence of “thought anchors”, reasoning steps that have outsized importance for downstream steps in the model’s chain of thought (CoT; Bogdan+ 2025); separately, others have experimented with steering the CoT towards specific reasoning behaviors, such as backtracking (Venhoff+ 2025). Here, I propose that we apply CoT steering to thought anchors, in order to study their emergence and downstream importance. To investigate, we might steer activations at two points relative to a given thought anchor: just before, to measure the spontaneity of its emergence; and immediately after, to measure the durability of its downstream importance. For both cases, the steering vector could be constructed by subtracting the average activation of sampled counterfactual sentences from the activation of the thought anchor. This research idea directly probes the transience of thought: do we see thought anchors arise smoothly with translations in activation space, or are there phase transitions in which thought anchors (and their downstream impacts on CoT) spontaneously emerge and disappear?

## Background

- “Thought Anchors: Which LLM Reasoning Steps Matter?” (Bogdan+ 2025) investigates the existence of thought anchors, or “reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process”
- Uses three methods: (i) counterfactual importance sampled over CoT rollouts, (ii) aggregating attention patterns by sentence, and (iii) causal attribution by suppressing sentence-level attention
- “Understanding reasoning in thinking language models via steering vectors” (Venhoff+ 2025) experiments with steering CoTs with vectors representing specific reasoning behaviors (e.g., backtracking, expressing uncertainty)
- Finds that reasoning behaviors are mediated by linear directions in latent space, and that steering in these directions can effectively control CoT

## Experiments

1.  **Preliminaries:**
    a. Collect a dataset of (prompt, CoT, answer, activations) tuples—could be from Bogdan+ and/or Venhoff+
    b. Annotate the dataset with the thought anchors in the CoT
2.  **Finding thought anchor steering vectors:**
    a. For a given thought anchor sentence, take its average activation (a_anchor) and the average activations of sampled counterfactual replacement sentences (a_counter)
        i. The counterfactual replacement sentence could be obtained by a similar procedure to Bogdan+: resampling just before the thought anchor sentence, conditioned on a minimum threshold of dissimilarity to the thought anchor
        ii. Choose the steering vector from the most causally relevant layer (test via patching)
    b. The steering vector is then: v = a_anchor - a_counter (in practice, we’ll probably normalize)
3.  **Transience (or not) of thought anchors:**
    a. Just before the thought anchor sentence, if we steer the activations (a) by a + βv for β < 0, then how will the logits change as a function of β? We can measure this concretely using the KL divergence from the original logits (producing the thought anchor). Will we see a sharp jump in KL divergence (indicating a high degree of transience and a possible phase transition) or a gradual rise (indicating some degree of contingence/predictability)
    b. What if we steer with β > 0 at other points in the CoT? Will the same or a similar thought anchor appear there too?
4.  **Effect on downstream CoT:**
    a. This time after the thought anchor sentence (either throughout the remaining CoT, immediately after the thought anchor, or in the final answer) and varying β ∈ [-1, 0], measure the KL divergence between the original and resampled logits (again, either throughout the remaining CoT, immediately after the thought anchor, or in the final answer)
    b. Do we see a gradual rise in KL divergence as we vary β, or do we see a sharp transition?
    c. How does attentional structure change? Is there a gradual reduction in the thought anchor broadcasting, or a sharp transition?

## Notes

In general for this project outline, I thought it better to go into too much detail (which we can backtrack) than too little. This is just one research direction based on your Problem 1, and I am open to a variety of other directions—even including starting from scratch—so please don’t be shy to edit and comment as much as you want!

Also, this idea represents a substantial research project that is likely infeasible for the trial period (through 9/1). As I mentioned, I would rather give too much detail than too little, so I’d love to hear your suggestions on tangible starting points (if you like this idea; if not, feel free to suggest a different starting point). What I’m currently thinking is to start with a relatively simple fixed question, CoT, and thought anchor, and experiment with the ideas above in that limited example. Please let me know what you think!