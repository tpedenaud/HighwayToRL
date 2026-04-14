# Observation Extension Comparison

## Per-run results

| observation_mode   |   train_seed |   num_eval_episodes |   mean_reward |   reward_std |   reward_median |   crash_rate |   mean_episode_length |   mean_speed |
|:-------------------|-------------:|--------------------:|--------------:|-------------:|----------------:|-------------:|----------------------:|-------------:|
| kinematics         |            0 |                  30 |       19.8099 |      2.4602  |         20.4545 |    0.0666667 |               28.8    |      20.5869 |
| kinematics         |            1 |                  30 |       14.1711 |      6.305   |         14.7002 |    0.733333  |               18.7333 |      24.1931 |
| kinematics         |            2 |                  30 |       15.7941 |      6.90085 |         19.9074 |    0.533333  |               21.5    |      23.1722 |
| occupancy_grid     |            0 |                  30 |       20.9773 |      4.98243 |         22.4915 |    0.166667  |               27.6333 |      23.0312 |
| occupancy_grid     |            1 |                  30 |       20.7006 |      0.57231 |         20.4545 |    0         |               30      |      20.3399 |
| occupancy_grid     |            2 |                  30 |       19.2192 |      5.61157 |         21.1693 |    0.166667  |               26.7667 |      21.783  |

## Grouped by observation mode

| observation_mode   |   mean_reward |   reward_seed_std |   mean_eval_std |   median_reward |   crash_rate |   mean_episode_length |   mean_speed |
|:-------------------|--------------:|------------------:|----------------:|----------------:|-------------:|----------------------:|-------------:|
| kinematics         |       16.5917 |          2.90277  |         5.22202 |         18.354  |     0.444444 |               23.0111 |      22.6507 |
| occupancy_grid     |       20.299  |          0.945373 |         3.7221  |         21.3718 |     0.111111 |               28.1333 |      21.718  |
