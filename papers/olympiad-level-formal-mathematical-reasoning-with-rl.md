# Olympiad-level formal mathematical reasoning with reinforcement learning

**Authors:** Thomas Hubert, Rishi Mehta, Laurent Sartran, Miklós Z. Horváth, Goran Žužić, Eric Wieser, Aja Huang, Julian Schrittwieser, Yannick Schroecker, Hussain Masoom, Ottavia Bertolli, Tom Zahavy, Amol Mandhane, Jessica Yung, Iuliya Beloshapka, Borja Ibarz, Vivek Veeriah, Lei Yu, Oliver Nash, Paul Lezeau, Salvatore Mercuri, Calle Sönne, Bhavik Mehta, Alex Davies, Daniel Zheng, Fabian Pedregosa, Yin Li, Ingrid von Glehn, Mark Rowland, Samuel Albanie, Ameya Velingker, Simon Schmitt, Edward Lockhart, Edward Hughes, Henryk Michalewski, Nicolas Sonnerat, Demis Hassabis, Pushmeet Kohli & David Silver

**Published:** Nature | Vol 651 | 19 March 2026

---

## Abstract

A long-standing goal of artificial intelligence (AI) is to build systems capable of complex reasoning in vast domains, a task epitomized by mathematics with its boundless concepts and demand for rigorous proof. Recent AI systems, often reliant on human data, typically lack the formal verification necessary to guarantee correctness. By contrast, formal languages such as Lean offer an interactive environment that grounds reasoning, and reinforcement learning (RL) provides a mechanism for learning in such environments. Here we present AlphaProof, an AlphaZero-inspired agent that learns to find formal proofs through RL by training on millions of auto-formalized problems. For the most difficult problems, it uses test-time RL, a method of generating and learning from millions of related problem variants at inference time to enable deep, problem-specific adaptation. AlphaProof substantially improves state-of-the-art results on historical mathematics competition problems. At the 2024 International Mathematical Olympiad competition, our AI system, with AlphaProof as its core reasoning engine, solved three out of the five non-geometry problems, including the competition’s most difficult problem. Combined with AlphaGeometry 2, this performance, achieved with multi-day computation, resulted in reaching a score equivalent to that of a silver medallist, marking the first time an AI system achieved any medal-level performance, to our knowledge. Our work demonstrates that learning at scale from grounded experience produces agents with complex mathematical reasoning strategies, paving the way for a reliable AI tool in complex mathematical problem solving.

---

## Introduction

One of the grand challenges in artificial intelligence (AI) is to develop agents that can reason effectively and discover solutions in complex, open-ended environments. Mathematics, with its role as a foundation for scientific understanding, serves as a profound and meaningful domain in which to develop these capabilities. As a natural step towards this goal, we focus on developing the necessary reasoning capabilities within the domain of elite mathematics competitions. Although not open-ended themselves, these competitions are renowned for problems that demand a depth of creative and multi-step reasoning, thereby providing a crucial and standardized environment for measuring progress.

The historical arc of mathematics, from Euclid’s foundational axiomatization of geometry to the widespread adoption of symbolic algebraic notation, has been one of increasing formalization. Modern computer-verified systems such as the Lean proof assistant and collaborative libraries such as Mathlib represent the logical continuation of this trajectory, enabling the expression of complex mathematics in a machine-understandable format. In these systems, a formal proof is not just a sequence of arguments, but a specific data structure called a ‘proof term’ that encodes the entire logical argument from axioms to conclusion. Although these terms can be constructed directly, a user typically builds them interactively by applying actions called tactics: small programs that manipulate the current proof state—the set of hypotheses and goals—to advance the proof one logical step at a time. The soundness of this process is guaranteed by Lean’s kernel, which verifies that the generated proof term is a valid construction. These systems offer two transformative capabilities: first, the rigorous, automated verification of every logical step, guaranteeing proof correctness; and second, the transformation of mathematics into an interactive, verifiable domain, allowing mathematical reasoning to be treated as a process that can be simulated, experimented with and learned.

Reinforcement learning (RL) offers a powerful paradigm for learning through interaction and experience, where agents optimize their behaviour through trial and error to achieve specified goals. This approach has proven to be particularly adept at mastering complex domains where optimal strategies are unknown. The AlphaZero family of agents, for instance, demonstrated the ability to achieve superhuman performance in challenging board games such as Go, chess and shogi, optimize quantum dynamics, and discover more efficient algorithms for fundamental computations such as sorting and matrix multiplication. The power of these systems stems from their ability to interact at scale with a verifiable environment and use grounded trial-and-error feedback to continually learn and refine their strategies. RL coupled with formal systems thus represents a particularly promising approach for tackling the challenge of automated mathematical reasoning.

Although formal systems provide verifiable grounding, considerable progress in AI mathematical reasoning has also occurred using large language models (LLMs) trained on vast corpora of informal, natural-language mathematical text. These models have shown impressive capabilities in solving a wide range of problems and generating human-like mathematical discourse, benefiting directly from the scale and breadth of existing human knowledge expressed in text. Rigorously verifying the correctness of their reasoning remains, however, an active research challenge, currently using techniques such as checking final answers against known solutions or comparing, with systems that cannot be fully trusted, generated reasoning steps against reference proofs. This lack of guaranteed correctness limits their reliability for validating mathematical claims or tackling problems without pre-existing reference points. In contrast, the inherent verification capabilities of formal systems provide the necessary foundation for building AI agents whose reasoning process and outputs can be trusted, even when exploring beyond the boundaries of existing human proofs and training data.

AlphaProof combines the rigour of formal systems with the experiential learning of RL to find proofs within the Lean theorem prover environment and to develop powerful mathematical reasoning. AlphaProof markedly improved state-of-the-art results on elite historical mathematics competition problems and, notably, proved three out of five problems at the 2024 International Mathematical Olympiad (IMO) competition. Although its solutions required computational time far exceeding that of human contestants, this success demonstrates the ability to tackle mathematical challenges previously considered beyond the reach of automated systems.

## AlphaProof

AlphaProof is an RL agent designed to discover formal mathematical proofs by interacting with a verifiable environment based on the Lean theorem prover. Its architecture, training and inference integrate several key innovations.

### The Lean RL environment

We model the interactive proving process within Lean as a sequential decision-making problem, a standard formulation for RL tasks. To distinguish our formal RL task from the Lean proof assistant itself, we term this specific formulation the ‘Lean environment’. Each mathematical statement to be proved constitutes a distinct problem instance. We now formally define this environment using the standard RL terminology of states, actions, rewards and returns. At any time step t, the state st is the logical state of the Lean prover, encompassing established hypotheses and remaining goals, observed by the agent as the Lean tactic state. The agent interacts by proposing an action at, a Lean tactic, as a text string. The environment attempts to execute these tactics, transitioning to a new state by updating hypotheses and goals. Each episode starts with a new problem statement and ends when a proof of that statement is successfully found, or a timeout occurs. The agent is incentivized to find short proofs by a reward signal rt = −1 for each tactic applied. The return Gt from a state st is the sum of these rewards until termination. Crucially, for proof states that decompose into multiple independent subgoals that must all be solved, the return is defined as the minimum return over these subgoals (that is, corresponding to the longest proof branch), rather than the more natural sum of returns from each subgoal.

### Prover agent

The AlphaProof agent combines a deep neural network with a powerful search algorithm inspired by AlphaZero. At its core is the proof network, a 3-billion-parameter encoder–decoder transformer model, that learns to interpret the observed Lean tactic state and generate two outputs: a policy, suggesting promising tactics to apply next, and a value function, estimating the expected return Gt. These outputs guide a specialized tree search that executes sequences of tactics and evaluates their consequences. Key adaptations for formal theorem proving include an AND–OR tree structure to handle the decomposition of proofs into multiple independent subgoals that must all be solved. Furthermore, to manage the large, open-ended space of possible tactics, AlphaProof samples actions and incorporates progressive sampling to explore a broader range of proof strategies along critical paths.

### Training

AlphaProof’s capabilities are primarily developed through a multi-stage training process. First, the proof network undergoes pretraining on a large corpus of approximately 300 billion tokens of code and mathematical text using a next-token prediction objective. Next, supervised fine-tuning is performed using approximately 300,000 state–tactic pairs extracted from human-written proofs in the Mathlib library. This stage enables the proof network to understand Lean syntax and internal states, imitate expert Lean tactics, and provide initial estimates for proof difficulty.

The central learning phase, inspired by AlphaZero, is the main RL loop in which AlphaProof learns from self-generated experience. To bridge the gap in manually formalized problems, we developed an auto-formalization process. This process uses a Gemini-based LLM to auto-formalize approximately 1 million natural-language problems into a dataset of around 80 million formal Lean problems.

A matchmaker system assigns auto-formalized problems and adaptive compute budgets to distributed actors, randomly tasking them to either prove or disprove each statement. Lean-verified outcomes—whether a proof, a disproof or a timeout—provide grounded feedback. The proof network is continually improved using experience from both successful proof and disproof attempts.

### Inference

When presented with a new problem, AlphaProof leverages two complementary computational scaling mechanisms:
1.  **Increasing the tree search budget:** allows for a more exhaustive exploration of proof paths.
2.  **Test-time RL (TTRL):** For problems where extensive search may be insufficient, TTRL focuses learning on a bespoke curriculum of synthetic problem variants (e.g., simplifications or generalizations) generated specifically around the target problem.

---

## Benchmarks

We evaluated AlphaProof on a comprehensive suite of formal mathematics benchmarks, all manually formalized in Lean:
1.  **miniF2F benchmark:** high-school mathematics competitions.
2.  **formal-imo:** all non-geometry historical IMO problems internally formalized by experts.
3.  **Putnam benchmark:** undergraduate Putnam Mathematical Competition problems.

### Main RL progress

AlphaProof’s solve rate on held-out benchmarks consistently improved throughout its main RL phase, which spanned approximately 80,000 TPU days. The proportion of problems successfully proved or disproved in its training dataset steadily increased. Training significantly enhanced proof-finding efficiency, requiring fewer simulations for the same solve rate as training progressed.

### Inference-time scaling

Performance can be further enhanced at inference by:
-   **Scaling search compute:** Increasing TPU hours for tree search yielded significant improvements.
-   **TTRL:** Yielded rapid initial gains, solving many new problems and increasing solve rates by an additional 15 absolute percentage points on both formal-imo and PutnamBench-test compared with extensive search.

---

## Final benchmark evaluation

AlphaProof established state-of-the-art performance across all evaluated formal mathematics benchmarks. Even at modest compute budgets (2 TPU minutes), it achieves strong results. For peak performance, TTRL is crucial. On formal-imo, TTRL reached particularly strong performance in number theory (75.7%) and algebra (72.6%).

## Performance at the 2024 IMO

At the 2024 IMO, the combined system (AlphaProof for non-geometry, AlphaGeometry 2 for geometry) solved four out of six problems (P1, P2, P4, P6). This yielded a score of 28 out of 42 points, placing the system within the silver-medal range, one point below the gold-medal threshold. Notably, it solved P6, the most difficult problem of the competition, which was only solved by five human contestants.

---

## Discussion and conclusion

AlphaProof demonstrates a powerful capacity for automated mathematical reasoning by combining AlphaZero-inspired learning with a curriculum of millions of auto-formalized problems and TTRL. While the computational requirements are significant, this work serves as foundational research. Future directions include improving algorithmic efficiency and expanding capabilities toward the frontiers of research mathematics.

---

## Methods

### Related work
The research is grounded in interactive theorem proving (Lean, Isabelle, etc.) and machine learning for theorem proving (LLMs, AlphaZero-like RL).

### Lean environment
-   **States (st):** Pretty-printed string representation of the Lean tactic state.
-   **Actions (at):** Lean tactic as a text string.
-   **Reward signal:** rt = −1 for each tactic applied to find the shortest proof.
-   **Return definition:** Minimum return over subgoals (longest/hardest branch).

### Prover agent details
-   **Proof network:** 3-billion-parameter encoder–decoder transformer. Policy head (decoder) and value head (categorical distribution).
-   **Tree search:** Adapted from AlphaZero and Sampled MuZero. Uses PUCT bound. Includes AND-node handling for multi-goal states and progressive sampling.

### Training details
-   **Pretraining:** 300 billion tokens of code and math text.
-   **Supervised Fine-Tuning (SFT):** 300,000 state–tactic pairs from Mathlib.
-   **Auto-formalization:** Gemini 1.5 Pro fine-tuned to translate natural language to Lean. Iterative refinement using a low-compute AlphaProof to certify equivalence.
-   **Main RL:** Centralized matchmaker, distributed actors, and centralized learner. 10% SFT data, 90% replay buffer data.

### Inference details
-   **Scaling search:** Multiple independent search attempts with increased simulation budget.
-   **TTRL:** 
    -   **Variant generation:** Gemini LLM generates synthetic variants (simplification, generalization, lemma proposal, etc.).
    -   **Focused RL:** Trains a specialist agent on the bespoke curriculum of variants.

### Evaluation details
-   **Standard benchmarks:** formal-imo (258 problems), corrected miniF2F, PutnamBench-test (189 problems).
-   **Data separation:** Explicitly excluded all Lean code and evaluation benchmarks from pretraining and SFT data. Removed similar documents from auto-formalization pipeline.
-   **IMO 2024 Protocol:** Pre-competition freeze, manual formalization of problems, answer guessing (AlphaProof sifts through candidates).

---

## Tables and Figures Highlights

-   **Table 1:** Shows AlphaProof outperforming previous SOTA (GPT-F, Hypertree, InternLM2-Math, Kimina-Prover, DeepSeek-Prover-V2) across miniF2F, formal-imo, and PutnamBench.
-   **Fig 3:** Shows learning progress and proof-finding efficiency gains.
-   **Fig 4:** Shows performance scaling with inference compute (Search vs. TTRL).
-   **Fig 5:** IMO 2024 performance vs. human contestants (Silver medal range).
-   **Extended Data Figs 7-9:** Show complete Lean proofs for IMO 2024 P1, P2, and P6.
