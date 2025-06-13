Proper formatting of the README can be found in SDE_Parameter_Estimation_with_Neural_Networks.pdf




# Estimating SDE Parameters with Neural Networks

---

## Introduction

Stock prices, among other physical phenomena, display seemingly random behavior whilst also following a general trend in a certain direction. A common mathematical representation of this behavior is the following stochastic differential equation (SDE):

$$
dS(t) = \mu S(t)\, dt + \sigma S(t)\, dB(t)
$$

Where:
- \( S(t) \): Stock price  
- \( \mu \): Expected return  
- \( \sigma \): Volatility  
- \( B(t) \): Geometric Brownian motion  

Together, \( \mu S(t)\,dt \) is the **drift term**, and \( \sigma S(t)\,dB(t) \) is the **diffusion term**. These parameters are critical for pricing derivatives, portfolio optimization, etc.

...

## Geometric Brownian Motion

GBM is a continuous stochastic process. It satisfies:

- \( B(0) = 0 \)
- \( B(t) \) is continuous
- \( B(t_4) - B(t_3) \) is independent of \( B(t_2) - B(t_1) \)
- \( B(t_2) - B(t_1) \sim \mathcal{N}(0, t_2 - t_1) \)

**Figure 1:**  
![5 sample simulations of GBM](5_Random_walks_sim.png)

...

## Discretization and Dataset Creation

To simulate GBM:

```python
B[0] = 0
B[i] = B[i-1] + sqrt(dt) * Z  # Z ~ N(0, 1)
