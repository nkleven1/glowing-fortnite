---
title: "Circular Restricted Three Body Problem"
date: 2025-10-16
draft: false
---

# 🛰️ Circular Restricted Three Body Problem

### EP 471 — Project Summary

This project explores how **small disturbances in launch angle and velocity** affect a **single Moon flyby and return to Earth orbit**.  
The orbit is simulated numerically using a **Backward Differentiation Formula (BDF)** — chosen because other methods become unstable when trajectories pass near the Moon.

The system is modeled using the **Circular Restricted Three Body Problem (CR3BP)** approximation.

---

###  Methodology
The numerical simulation:
- Integrates the motion of a spacecraft under Earth–Moon gravity.
- Perturbs both **initial velocity** and **launch angle** to observe sensitivity.
- Uses stable numerical integration suitable for close-approach dynamics.

---

###  Code Access
You can **view or download** the full Python code here:

👉 [**CR3BP.py**](../assets/code/CR3BP.py)

---

###  Input Parameters

| Parameter | Description |
|------------|--------------|
| `total_sample` | The total number of sample iterations. |
| `v_mag_nominal` | The nominal velocity magnitude at launch. |
| `phi_nominal` | The launch angle relative to Earth’s position from the Moon (counterclockwise orbit). |
| `phi_range` | The range of perturbation **angle** values the simulation will iterate over. |
| `vel_range` | The range of perturbation **velocity** values the simulation will iterate over. |

---

###  Output
The simulation outputs trajectory data and plots illustrating how small variations in the initial conditions affect the resulting orbit.

---

###  Notes
This project demonstrates:
- The **sensitivity** of orbital trajectories to small perturbations.
- The usefulness of **implicit integration** methods for close approaches.
- The practical application of the **CR3BP** in trajectory planning.

---

