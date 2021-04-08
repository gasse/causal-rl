# Using Confounded Data in Offline RL

[Openreview](https://openreview.net/forum?id=NGLwjQj4TY) -
[PDF](https://openreview.net/pdf?id=NGLwjQj4TY) -
[Video](https://youtu.be/dHEY_0PgT98)

Using Confounded Data in Offline RL. Maxime Gasse, Damien Grasset, Guillaume Gaudron, Pierre-Yves Oudeyer. Proceedings of the [3rd Offline Reinforcement Learning Workshop](https://offline-rl-neurips.github.io/2022/papers.html) at Neural Information Processing Systems, 2022.

```
@InProceedings{workshop/offlinerl/GGGO22,
  title = {Using Confounded Data in Offline RL},
  author = {Gasse, Maxime and Grasset, Damien and Gaudron, Guillaume and Oudeyer, Pierre-Yves},
  booktitle = {Proceedings of the 3rd Offline Reinforcement Learning Workshop at Neural Information Processing Systems},
  year = {2022},
}
```

## Requirements

You must have Python 3 and the following packages installed:
```
pytorch
gym
scipy
matplotlib
```

# Toy problem 1 (door)

This toy problem is configured in the following file:
```
experiments/toy1/config.json
```

To run this experiment execute the following commands:
```shell
GPU = 0  # -1 for CPU

for EXPERT in noisy_good perfect_good perfect_bad random strong_bad_bias strong_good_bias; do
  for SEED in {0..19}; do
    python experiments/toy1/01_train_models.py $EXPERT -s $SEED -g $GPU
    python experiments/toy1/02_eval_models.py $EXPERT -s $SEED -g $GPU
  done
  python experiments/toy1/03_plots.py $EXPERT
done
```

Results are stored in the following folders:
```
experiments/
  toy1/
    plots/
    results/
    trained_models/
```

# Toy problem 2 (tiger)

This toy problem is configured in the following file:
```
experiments/toy2/config.json
```

To run this experiment execute the following commands:
```shell
GPU = 0  # -1 for CPU

for EXPERT in noisy_good very_good very_bad random strong_bad_bias strong_good_bias; do
  for SEED in {0..19}; do
    python experiments/toy2/01_train_models.py $EXPERT -s $SEED -g $GPU
    python experiments/toy2/02_eval_models.py $EXPERT -s $SEED -g $GPU
    python experiments/toy2/03_train_agents.py $EXPERT -s $SEED -g $GPU
    python experiments/toy2/04_eval_agents.py $EXPERT -s $SEED -g $GPU
  done
  python experiments/toy2/05_plots.py $EXPERT
done
```

Results are stored in the following folders:
```
experiments/
  toy2/
    plots/
    results/
    trained_models/
    trained_agents/
```

# Toy problem 3 (gridworld)

This toy problem is configured in the following file:
```
experiments/toy3/config.json
```

To run this experiment execute the following commands:
```shell
GPU = 0  # -1 for CPU

for EXPERT in noisy_good very_good very_bad random strong_bad_bias strong_good_bias; do
  for SEED in {0..19}; do
    python experiments/toy3/01_train_models.py $EXPERT -s $SEED -g $GPU
    python experiments/toy3/02_eval_models.py $EXPERT -s $SEED -g $GPU
    python experiments/toy3/03_train_agents.py $EXPERT -s $SEED -g $GPU
    python experiments/toy3/04_eval_agents.py $EXPERT -s $SEED -g $GPU
  done
  python experiments/toy3/05_plots.py $EXPERT
done
```

Results are stored in the following folders:
```
experiments/
  toy3/
    plots/
    results/
    trained_models/
    trained_agents/
```
