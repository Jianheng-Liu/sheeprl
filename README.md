# ⚡ SheepRL 🐑

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

<p align="center">
  <img src="./assets/images/logo.svg" style="width:40%">
</p>

<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/Eclectic-Sheep/sheeprl/assets/18405289/6efd09f0-df91-4da0-971d-92e0213b8835" width="200px"></td>
      <td><img src="https://github.com/Eclectic-Sheep/sheeprl/assets/18405289/dbba57db-6ef5-4db4-9c53-d7b5f303033a" width="200px"></td>
      <td><img src="https://github.com/Eclectic-Sheep/sheeprl/assets/18405289/3f38e5eb-aadd-4402-a698-695d1f99c048" width="200px"></td>
      <td><img src="https://github.com/Eclectic-Sheep/sheeprl/assets/18405289/93749119-fe61-44f1-94bb-fdb89c1869b5" width="200px"></td>
    </tr>
  </table>
</div>

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Environment</th>
        <th>Total frames</th>
        <th>Training time</th>
        <th>Test reward</th>
        <th>Paper reward</th>
        <th>GPUs</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Crafter</td>
        <td>1M</td>
        <td>1d 3h</td>
        <td>12.1</td>
        <td>11.7</td>
        <td>1-V100</td>
      </tr>
      <tr>
        <td>Atari-MsPacman</td>
        <td>100K</td>
        <td>14h</td>
        <td>1542</td>
        <td>1327</td>
        <td>1-3080</td>
      </tr>
      <tr>
        <td> Atari-Boxing</td>
        <td>100K</td>
        <td>14h</td>
        <td>84</td>
        <td>78</td>
        <td>1-3080</td>
      </tr>
      <tr>
        <td>DOA++(w/o optimizations)<sup>1</sup></td>
        <td>7M</td>
        <td>18d 22h</td>
        <td>2726/3328<sup>2</sup></td>
        <td>N.A.</td>
        <td>1-3080</td>
      </tr>
      <tr>
        <td>Minecraft-Nav(w/o optimizations)</td>
        <td>8M</td>
        <td>16d 4h</td>
        <td>27% &gt;= 70<br>14% &gt;= 100</td>
        <td>N.A.</td>
        <td>1-V100</td>
      </tr>
    </tbody>
  </table>
</div>

1. For comparison: 1M in 2d 7h vs 1M in 1d 5h (before and after optimizations resp.)
2. Best [leaderboard score in DIAMBRA](https://diambra.ai/leaderboard) (11/7/2023)
## Run the Code
To run the code, you may follow the instruction in `.\example.ipynb`.

## Modified Code Files
The code files we modified are as follows, which contain the implementation of our proposed method.
```
├── rl-agents
│   ├── rl_agents
│   │   ├── agents
│   │   │   ├── common
│   │   │   │   ├── models.py -> introduce sparse attention into the model
│   │   │   ├── deep_q_network
│   │   │   │   ├── pytorch.py -> introduce learnable weight for double q learning
│   ├── scripts
│   │   ├── configs
│   │   │   ├── HighwayEnv
│   │   │   │   ├── env_obs_attention.json          -> Configure the environment observation state
│   │   │   │   ├── agent
