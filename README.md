# ⚡ SheepRL 🐑

This repository is forked from [sheeprl](https://github.com/Eclectic-Sheep/sheeprl).

## Input Experiments

The algorithm is dreamer v3 and the environment is the walker from the Deepmind control suite. The configuration files I added are as follows, which contain the configurations for 3 different input: grayscale image, vector value, and both.
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
The results are as follows:
<div align="center">
  <table>
    <tr>
      <td><img src="assets/exp/value_loss.png" width="200px"></td>
      <td><img src="assets/exp/reward" width="200px"></td>
    </tr>
  </table>
</div>
