# ⚡ SheepRL 🐑

This repository is forked from [sheeprl](https://github.com/Eclectic-Sheep/sheeprl).

## Input Experiments

The algorithm is dreamer v3 and the environment is the walker from the Deepmind control suite. The configuration files I added are as follows, which contain the configurations for 3 different input: grayscale images, vector values, and both.
```
├── sheeprl
│   ├── sheeprl
│   │   ├── configs
│   │   │   ├── exp
│   │   │   │   ├── dreamer_v3_dmc_walker_walk_vector
│   │   │   │   ├── dreamer_v3_dmc_walker_walk_grayscale
│   │   │   │   ├── dreamer_v3_dmc_walker_walk_grayscale_vector
│   ├── exp_dreamer_v3_dmc.ipynb
```
In each experiment setting, the agent is trained for about 6 hours on my computer with RTX 2080.

The results are as follows:
<div align="center">
  <table>
    <tr>
      <td><img src="assets/exp/value_loss.png" width="400px"></td>
      <td><img src="assets/exp/reward.png" width="400px"></td>
    </tr>
  </table>
</div>
where the dark blue curve is with grayscale images as input, the pink curve is with vector values as input and the light blue curve is with both as input.
