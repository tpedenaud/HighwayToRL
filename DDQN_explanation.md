# Extension Task: Double-DQN vs Standard DQN

## Rigorous Comparison of Overestimation Bias Mitigation

---

## Executive Summary

This extension task investigates whether **Double-DQN** (Van Hasselt et al., 2015) outperforms standard DQN on the highway-v0 benchmark by reducing Q-value overestimation bias.

### Hypothesis
Double-DQN's decoupling mechanism (using the current network to select actions and the target network to evaluate them) reduces overestimation bias, resulting in:
- **Improved stability** during training
- **Higher final performance** on evaluation
- **Increased safety** (lower crash rates)

### Key Question
On a continuous driving task, does theoretical overestimation bias translate to practical performance differences?

---

## Part 1: Theoretical Background

### The Overestimation Problem in DQN

Standard DQN uses the Bellman update:
$$Q_{\text{target}}(s, a) = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

**Problem:** The same target network that evaluates $Q_{\text{target}}(s', a')$ also selects the greedy action $\max_{a'}$. This creates **optimistic bias**:
- If $Q_{\text{target}}$ slightly overestimates some action values
- The $\max$ operation amplifies these overestimates
- Leading to inflated Q-values and suboptimal policies

### Double-DQN Solution

Decouple action selection from action evaluation:
$$Q_{\text{target}}(s, a) = r + \gamma Q_{\text{target}}(s', \underbrace{\arg\max_{a'} Q_{\text{current}}(s', a')}_{\text{selection by current net}})$$

**Why it works:**
- Current network $Q_{\text{current}}$ is slightly different from $Q_{\text{target}}$
- Their overestimation biases are partially independent
- Using one to select and the other to evaluate reduces correlation
- Statistical independence → bias cancellation

### Mathematical Intuition
If $Q_{\text{current}}$ and $Q_{\text{target}}$ are trained on slightly different data:
$$\mathbb{E}[\max_{a'} Q_{\text{target}}(s', a')] \leq \mathbb{E}[Q_{\text{target}}(s', \arg\max_{a'} Q_{\text{current}}(s', a'))]$$

The inequality shows that Double-DQN's target is typically more conservative (lower), reducing cascading overestimation.

---

## Part 2: Experimental Design

### Null Hypothesis vs Alternative Hypothesis
- **$H_0$:** Double-DQN performance ≈ Standard DQN (difference < measurement noise)
- **$H_1$:** Double-DQN performance > Standard DQN (statistically significant)

### Controlled Variables (Identical for Both)
- **Environment:** highway-v0 with shared config (see config.py)
- **Observation type:** Kinematics (presence, x, y, vx, vy)
- **Action type:** DiscreteMetaAction (3 discrete actions: slow, medium, fast)
- **Hyperparameters:**
  - Buffer capacity: 10,000
  - Batch size: 32
  - Learning rate: 0.001 (Adam optimizer)
  - Discount factor γ: 0.99
  - Target update frequency: 1,000 steps
  - Learning starts: 1,000 steps
  - Epsilon schedule: 1.0 → 0.05 (linear decay over 19,000 steps)
- **Network architecture:** 2 hidden layers, 128 units each (same for both agents)
- **Random seeds:** [0, 1, 2] for training reproducibility

### Independent Variable
- **Algorithm:** Standard DQN vs Double-DQN
  - Standard: $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$
  - Double: $y = r + \gamma Q_{\text{target}}(s', \arg\max_{a'} Q_{\text{current}}(s', a'))$

### Dependent Variables (Metrics)

#### 1. **Performance Metrics** (during evaluation)
- **Mean Return** (μ): Average episode return across 50 evaluation seeds
- **Return Std Dev** (σ): Variability of returns (lower = more consistent)
- **Episode Length**: Average steps per episode (longer = better lane-keeping)
- **Crash Rate** (%): Percentage of episodes ending in collision

#### 2. **Training Metrics** (during learning)
- **Episode Returns**: Return at each episode (convergence speed)
- **Loss Curve**: MSE loss over time (learning stability)
- **Coefficient of Variation**: σ/μ of returns (training stability)

### Sample Size & Statistical Power
- **Training:** 3 seeds × 20,000 steps = 60,000 total steps per algorithm
- **Evaluation:** 50 seeds per trained agent = 50 samples for mean/std estimation
- **Total samples:** 3 × 50 = 150 return measurements per algorithm
- **Statistical test:** Independent t-test (α = 0.05)

### Experimental Timeline
1. **Phase 1:** Train both algorithms on seeds [0, 1, 2]
2. **Phase 2:** Evaluate each on 50 held-out seeds [1000-1050]
3. **Phase 3:** Compare training curves and final metrics
4. **Phase 4:** Analyze safety (crashes) and qualitative behavior
5. **Phase 5:** Interpret results and draw conclusions

---

## Part 3: Implementation Details

### Code Architecture

#### Double-DQN Agent (`rl/double_dqn.py`)
The key difference in the update rule:

```python
def update(self, batch):
    # Unpack batch
    states, actions, rewards, next_states, dones = zip(*batch)
    # Convert to tensors (same as standard DQN)
    
    # ===== STANDARD DQN =====
    # next_q_values = self.target_net(next_states).max(1)[0]
    #
    # ===== DOUBLE-DQN (DIFFERENCE) =====
    next_actions = self.net(next_states).argmax(dim=1)  # <-- Use CURRENT net to choose
    next_q_values = self.target_net(next_states).gather(1, next_actions)  # <-- Use TARGET net to evaluate
    ```

**Why this matters:**
- Standard DQN: 1 network makes both decisions (selection & evaluation)
- Double-DQN: 2 networks share responsibility (reduced bias)

#### Training Function (`rl/extension_train.py`)
A unified training loop supports both DQN classes:
- Pass `DQN` class → trains standard DQN
- Pass `DoubleDQN` class → trains Double-DQN
- Identical hyperparameters, environment, and data collection

#### Evaluation (`rl/evaluate.py` + notebook)
- Compare on same 50 evaluation seeds
- Both run deterministically (ε=0)
- Measure: return, length, crash rate in exactly the same way

### Reproducibility
- Fixed random seeds for training
- Deterministic environment resets
- Identical network initialization (both use same seed before training)
- All randomness controlled via `random.seed()`, `np.random.seed()`, `torch.manual_seed()`

---

## Part 4: Expected Results & Interpretation Framework

### Scenario 1: Double-DQN Wins (+10% return improvement)
**Interpretation:** 
- Overestimation bias is significant on highway-v0
- Double-DQN's decoupling reduces bias effectively
- Recommendation: Deploy Double-DQN for better performance

**Evidence:**
- Training curves: Double-DQN curves smoother/higher
- Loss curves: Double-DQN converges more stably
- Evaluation: Mean return significantly higher (t-test p < 0.05)
- Safety: Fewer crashes with Double-DQN

### Scenario 2: Double-DQN Roughly Equal (±3%)
**Interpretation:**
- The environment is not sensitive to overestimation bias
- Highway driving may have strong "bandit" structure (few action-dependent decisions)
- Overestimation doesn't cascade as much on this domain

**Evidence:**
- Similar training curves
- Similar final returns
- Statistical test: p > 0.05 (no significant difference)

### Scenario 3: Standard DQN Wins (Double-DQN -5%)
**Interpretation:**
- Double-DQN's elimination of bias comes at a cost (e.g., higher variance in targets)
- The highway environment may be better solved by optimistic exploration
- Recommendation: Standard DQN is superior for this specific task

**Evidence:**
- Standard DQN training curves higher
- Higher evaluation returns
- Possibly: Lower variance in returns (more consistent)

---

## Part 5: Execution Plan

### What the Notebook Does (9 phases)

| Phase | Action | Measurement |
|-------|--------|-------------|
| 1 | Train Standard DQN (3 seeds) | Training curves, episode returns |
| 2 | Train Double-DQN (3 seeds) | Training curves, episode returns |
| 3 | Plot training curves | Visual convergence comparison |
| 4 | Evaluate both (50 seeds each) | Mean/std return, crash rate |
| 5 | Plot metric comparisons | Bar charts, statistical summary |
| 6 | Analyze training stability | Loss curves, smoothness |
| 7 | Record rollouts | Visual behavior (GIFs) |
| 8 | Find failure cases | Safety analysis, failure modes |
| 9 | Statistical tests | T-test, effect size, interpretation |

### Expected Runtime
- Training: ~10 min per seed on CPU (30 min total per algorithm)
- Evaluation: ~5 min per algorithm (10 min total)
- Analysis: ~5 min
- **Total:** ~50 minutes on standard machine

### Artifacts Generated
```
artifacts/extension/
├── standard_dqn/seed_0/
│   ├── checkpoint.pt          # Trained weights
│   └── metrics.npz             # Training curves
├── standard_dqn/seed_1/
├── standard_dqn/seed_2/
├── double_dqn/seed_0/
├── double_dqn/seed_1/
├── double_dqn/seed_2/
├── training_curves_comparison.png    # 3-panel plot
├── metrics_comparison.png             # 4 metrics side-by-side
├── loss_curves_comparison.png        # Training stability
├── standard_dqn_rollouts/            # Video GIFs
├── double_dqn_rollouts/
├── failures/                          # Crash episodes
└── experiment_report.txt              # Full analysis
```

---

## Part 6: Design Justification

### Why This Extension is Rigorous

1. **Clear Hypothesis:** Testing a specific, falsifiable claim about bias reduction
2. **Controlled Experiment:** All variables identical except the algorithm
3. **Statistical Power:** 50 evaluation samples per algorithm enables reliable comparison
4. **Multiple Metrics:** Examining return, safety, and stability together
5. **Reproducibility:** Fixed seeds, identical hyperparameters, documented code
6. **Qualitative Analysis:** Rollouts and failure cases provide context
7. **Honest Interpretation:** Framework prepared for all three possible outcomes

### Why Not Other Extensions

| Extension | Why Not Chosen |
|-----------|-----------------|
| Continuous actions | Different environment, requires PPO/SAC, not a fair comparison |
| Observation variation | Would require network architecture changes, complicates interpretation |
| Reward shaping | Changes the problem fundamentally; less scientifically interesting |
| Network architecture ablation | Similar to reward shaping; many confounding factors |
| Prioritized Experience Replay | Orthogonal to Double-DQN; would need full factorial design |
| Safety-oriented training | Complex problem, not directly testable with current framework |

**Double-DQN was chosen because:**
- It's a single, well-defined algorithmic modification
- Published research establishes theoretical foundation
- Testable with identical hyperparameters and environment
- Clear mechanism (reduces overestimation) and measurable impact
- Conceptually related to core material from TP4/TP5

---

## Part 7: Key References

1. **Van Hasselt, H., Guez, A., & Silver, D. (2015).** "Deep Reinforcement Learning with Double Q-learning." International Conference on Machine Learning. PMLR.
   - Introduces Double-DQN and empirically validates it on Atari

2. **Mnih, V., et al. (2015).** "Human-level control through deep reinforcement learning." Nature, 529(7587), 529-533.
   - Original DQN paper; establishes the problem Double-DQN solves

3. **Thrun, S., & Schwartz, A. (1993).** "Issues in using function approximation for reinforcement learning." Canadian Conference on AI.
   - Early work on overestimation in Q-learning with function approximation

---

## Part 8: How to Run

### Prerequisites
```bash
cd HighwayToRL
source .venv/bin/activate
pip install -r requirements.txt
```

### Execute Extension Task
Open `extension_analysis.ipynb` and run cells in order:
1. Cell 1: Setup and configuration
2. Cell 2: Train Standard DQN (watch console for progress)
3. Cell 3: Train Double-DQN
4. Cell 4-5: Compare training curves
5. Cell 6-7: Evaluation metrics and plots
6. Cell 8-9: Rollouts and failure analysis
7. Cell 10: Statistical tests and interpretation

### Monitor Progress
- Console output shows: step, episode return, epsilon at each episode end
- Checkpoints auto-saved to `artifacts/extension/`
- Early interrupts are safe (can resume from checkpoints later)

---

## Part 9: Expected Insights

### What We'll Learn

1. **About Double-DQN:**
   - Whether bias reduction translates to practical gains
   - On what types of problems it helps most
   - Hidden costs (if any) of the decoupling mechanism

2. **About Highway Driving:**
   - Whether the task is sensitive to Q-value bias
   - What behaviors emerge from different algorithms
   - Failure modes and how they differ

3. **About RL Methodology:**
   - How to design fair comparisons
   - Statistical tests for small sample sizes (n=3 seeds)
   - Interpreting non-significant results honestly

### Likely Takeaway
Double-DQN probably shows **modest improvement** (0-10%):
- The highway environment is not highly exploration-dependent
- But overestimation still hurts slightly
- The algorithm is a good "free upgrade" (no additional samples)

---

## Appendix: Code Snippets

### Standard DQN Update (for reference)
```python
def update(self, batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    # Convert to tensors...
    
    q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    
    loss = F.mse_loss(q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()
```

### Double-DQN Update (implemented)
```python
def update(self, batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    # Convert to tensors...
    
    q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double-DQN: Two-step process
        next_actions = self.net(next_states).argmax(dim=1)
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    
    loss = F.mse_loss(q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()
```

---

## Summary

This extension task **rigorously compares Double-DQN to Standard DQN** by:
1. ✅ Fixing all confounding variables
2. ✅ Using identical hyperparameters and environment
3. ✅ Running multiple seeds (3) for training reproducibility
4. ✅ Evaluating on large sample (50 seeds) for statistical power
5. ✅ Measuring multiple dimensions (performance, safety, stability)
6. ✅ Preparing interpretation framework before running experiments
7. ✅ Using standard statistical tests (t-test)
8. ✅ Providing code, data, and detailed methodology

**The result will definitively answer:** Does Double-DQN's theoretical advantage translate to practical gains on highway driving?
