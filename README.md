# Offline-to-Online Reinforcement Learning – Excavator Pick Up Stone

Simulation-based reinforcement learning using **AGX Dynamics**  
Task: Excavator picks up a stone and lifts it stably

<p align="center">
  <img src="output_3x_klein.gif" alt="Excavator picking up stone – triple view" width="720"/>
  <br><em>Short demo (~15 seconds): Trained policy picking up a stone (triple view)</em>
</p>

## Environment

- Physics engine: **AGX Dynamics**
- Randomized every episode:
  - Stone position (x, y)
  - Stone rotation around z-axis
  - Initial joint angles (boom, arm, bucket)
- Observation spaces:
  - **State-based**: stone (x, y, θ_z) + joint angles (boom, arm, bucket) → 6 dimensions
  - **Vision-based**: RGB image (camera view from cabin) + joint angles (6 dimensions)
- Actions: joint velocities (Δ boom, Δ arm, Δ bucket) → 3 continuous dimensions

## Reward Functions – Comparison

Three reward variants are evaluated:

- **Sparse**:  
  +1 reward only when success (stone stably lifted) at the very end of the episode, 0 otherwise

- **Middle**:  
  Stone height clipped between 0 and 1.7 m, normalized to [0,1]  
  + additional small bonus when stone is stable

- **Dense**:  
  Same as middle reward  
  + shaped term based on distance between bucket and stone

## Vision-based Experiments

Two approaches compared:

- Pretrained ResNet18 (ImageNet weights)  
- ResNet18 trained end-to-end with **DrQ-v2** style augmentation
