# Offline-to-Online Reinforcement Learning for Excavator Control

## Overview
This project explores simulation-based reinforcement learning using **AGX Dynamics** to train an excavator to pick up and stably lift a stone. The focus is on comparing reward formulations and perception strategies in an offline-to-online RL setting.

<p align="center">
  <img src="/README/output_3x_klein.gif" alt="Excavator picking up stone – triple view" width="720"/>
  <br><em>Trained policy successfully picking up and stabilizing a stone (triple view)</em>
</p>

---

## Task Description
The objective is to learn a control policy that enables an excavator to:
1. Approach a randomly placed stone  
2. Pick it up using the bucket  
3. Lift it while maintaining stability  

---

## Environment Setup

### Simulation
- Physics Engine: **AGX Dynamics**

### Domain Randomization
Each episode is randomized to improve generalization:
- Stone position: *(x, y)*
- Initial joint angles:
  - Boom  
  - Arm  
  - Bucket  

### Observation Spaces

#### 1. State-Based Observations
- Stone position: *(x, y, z)*  
- Joint angles: *(boom, arm, bucket)*  
- **Total dimension:** 6  

#### 2. Vision-Based Observations
- RGB image (camera mounted in cabin)  
- Joint angles: *(boom, arm, bucket)*  
- **Additional dimension:** 3 (joint states)  

### Action Space
- Continuous control of joint velocities:
  - Δ Boom  
  - Δ Arm  
  - Δ Bucket  
- **Action dimension:** 3  

---

## Reward Function Design

Three reward strategies are compared to evaluate learning efficiency and policy quality:

### 1. Sparse Reward
- +1 reward only upon successful completion  
- Success condition: stone is lifted and stable at episode end  
- 0 reward otherwise  

### 2. Intermediate ("Middle") Reward
- Based on stone height:
  - Clipped to range [0, 1.7] meters  
  - Normalized to [0, 1]  
- Additional small bonus for stability  

### 3. Dense Reward
- Includes all components of the intermediate reward  
- Additional shaping term:
  - Distance between bucket and stone  
  - Encourages faster and more directed interaction  

---

## Vision-Based Experiments

Two perception pipelines are evaluated for image-based policies:

### 1. Pretrained Encoder
- **ResNet18** with ImageNet weights  
- Benefits from transfer learning  

### 2. Learned Encoder
- Lightweight CNN trained from scratch  
- Uses **DrQ-v2-style data augmentation**  
- Focus on sample efficiency and robustness  

---

## Key Research Questions
- How does reward shaping influence learning stability and convergence speed?  
- Can vision-based policies match state-based performance?  
- Does pretraining outperform lightweight learned encoders in this control task?  

---

## Results
The showcased policy (see video above) was vision based trained with drqv2-style augemntations and **only 25 demonstrations** (about 14 hours online training with rtx 4060) and achieves:
- **96% success rate** (stable hold condition)  
- **100% rock lifted ratio**

---

## Summary
This project provides a structured comparison of:
- Reward design (sparse vs. dense)  
- Observation modality (state vs. vision)  
- Representation learning (pretrained vs. learned encoders)  

with the goal of improving sim-to-real transfer for robotic excavation tasks.
