# Rapport Detaille - Extension Observation (30 evaluations)

## 1. Objectif de l'extension
Ce travail compare deux representations d'observation pour un agent DQN sur highway-v0 :
- mode kinematics : vecteur de variables tabulaires des vehicules proches,
- mode occupancy_grid : grille spatiale locale (representation structuree de l'environnement).

But principal : mesurer l'impact de la representation d'etat sur la performance, la robustesse et la securite de conduite.

## 2. Protocole experimental
### 2.1 Environnement
- Environnement : highway-v0.
- Action space : discret (acceleration/changement de voie pilote par DQN).
- Entrainement/evaluation reproductibles via des seeds fixes.

### 2.2 Budget et seeds (scenario 30 evals)
Le protocole utilise :
- seeds entrainement : [0, 1, 2],
- episodes d'evaluation par run : 30,
- seeds d'evaluation : 4000 a 4029,
- timesteps entrainement : 30 000.

Ces choix sont visibles dans run_observation_extension_long.py et dans les sorties d'evaluation exportees.

## 3. Configurations utilisees
### 3.1 Configuration observation kinematics
Source : observation_extension_config.py.

Parametres principaux :
- type : Kinematics,
- vehicles_count : 10,
- features : [presence, x, y, vx, vy],
- normalize : True,
- clip : True,
- see_behind : True,
- observe_intentions : False.

Interpretation :
- approche compacte, faible cout de calcul,
- structure spatiale implicite seulement (pas de grille explicite).

### 3.2 Configuration observation occupancy_grid
Source : observation_extension_config.py.

Parametres principaux :
- type : OccupancyGrid,
- features : [presence, x, y, vx, vy],
- grid_size : [[-27.5, 27.5], [-27.5, 27.5]],
- grid_step : [5, 5],
- align_to_vehicle_axes : True,
- normalize : True,
- clip : True.

Interpretation :
- representation spatiale explicite de la scene locale,
- plus informative pour les interactions vehiculaires et les zones de conflit,
- plus couteuse en calcul qu'un simple vecteur.

### 3.3 Hyperparametres DQN (base)
Source : common_observation_extension.py (DEFAULT_EXTENSION_SETTINGS).

- gamma : 0.99,
- learning_rate : 5e-4,
- buffer_size : 100000,
- batch_size : 64,
- learning_starts : 10000,
- train_freq : 4,
- target_update_interval : 1000,
- epsilon_start : 1.0,
- epsilon_end : 0.05,
- epsilon_decay_steps : 80000,
- gradient_clip_norm : 10.0,
- checkpoint_interval : 25000.

Architectures reseau :
- kinematics : MLP [256, 256],
- occupancy_grid : CNN (channels [32, 64], kernels [5, 3], strides [2, 1]) + tete fully-connected (hidden 128).

### 3.4 Ajustements occupancy_grid dans le run long
Source : run_observation_extension_long.py.

Par rapport a la base, occupancy_grid a ete entraine avec :
- learning_rate = 1e-4,
- learning_starts = 8000,
- epsilon_decay_steps = 80000,
- target_update_interval = 1000.

Justification :
- un encodeur CNN est plus sensible a la dynamique d'optimisation,
- reduction du learning rate pour stabiliser les mises a jour,
- maintien d'une exploration progressive pour mieux couvrir l'espace d'etats.

## 4. Pourquoi une approche CNN pour occupancy_grid
Un CNN est adapte a occupancy_grid car la donnee est une carte locale en 2D avec canaux. Le reseau exploite :
- la localite spatiale (vehicules proches, zones critiques),
- les motifs de voisinage (densite, alignement, conflits),
- l'invariance partielle a la translation locale.

Concretement, cela aide l'agent a apprendre des regles de conduite plus coherentes (position relative, espace libre, risque de collision) que des descripteurs purement vectoriels lorsqu'il faut raisonner sur la geometrie de scene.

## 5. Resultats obtenus (30 evals par run)
Source : artifacts/observation_extension/comparison_summary.csv.

### 5.1 Resultats par mode (agregation sur 3 seeds train)
- Kinematics :
  - mean_reward = 16.5917
  - reward_seed_std = 2.9028
  - mean_eval_std = 5.2220
  - median_reward = 18.3540
  - crash_rate = 0.4444
  - mean_episode_length = 23.0111
  - mean_speed = 22.6507

- OccupancyGrid :
  - mean_reward = 20.2990
  - reward_seed_std = 0.9454
  - mean_eval_std = 3.7221
  - median_reward = 21.3718
  - crash_rate = 0.1111
  - mean_episode_length = 28.1333
  - mean_speed = 21.7180

### 5.2 Comparaison directe OccupancyGrid - Kinematics
- Gain reward moyen : +3.7073 (environ +22.3%).
- Crash rate : -0.3333 (de 44.44% a 11.11%, soit ~75% de collisions en moins).
- Longueur d'episode : +5.1222 (episodes plus longs, meilleure survie).
- Variabilite inter-seeds (reward_seed_std) : -1.9574 (~67% plus stable).
- Vitesse moyenne : -0.9327 (legerement plus faible).

### 5.3 Lecture par seed
- Kinematics montre une forte sensibilite au seed :
  - seed 1 : mean_reward 14.17 et crash_rate 0.7333,
  - seed 2 : mean_reward 15.79 et crash_rate 0.5333.
- OccupancyGrid reste plus regulier :
  - seed 1 : mean_reward 20.70 et crash_rate 0.0,
  - seed 0 et 2 gardent des crash rates bas (0.1667).

## 6. Interpretation
### 6.1 Ce que montrent les metriques
- OccupancyGrid ameliore simultanement performance et securite.
- La baisse nette du crash_rate, combinee a la hausse de la longueur d'episode, indique une politique plus prudente et mieux controlee.
- La reduction de la variabilite inter-seeds suggere une meilleure robustesse de l'apprentissage.

### 6.2 Compromis observe
- L'agent occupancy_grid roule un peu moins vite en moyenne.
- Cette baisse est compatible avec un comportement plus conservateur qui privilegie la securite et la completion des episodes.

## 7. Limites et points de vigilance
- Le protocole reste limite a 3 seeds train et 30 episodes d'evaluation par seed.
- Les comparaisons sont sensibles a la definition exacte de la grille (taille, pas, canaux) et aux hyperparametres CNN.
- Une analyse complementaire serait utile avec :
  - davantage de seeds train,
  - budget d'entrainement plus long (100k+ timesteps),
  - tests de robustesse sur variantes de trafic.

## 8. Conclusion
Dans ce cadre (30 evals par run), l'extension occupancy_grid + CNN apporte un gain clair par rapport a kinematics + MLP :
- meilleure recompense moyenne,
- nettement moins de collisions,
- comportement plus stable entre seeds.

La representation spatiale et l'encodeur convolutionnel constituent donc une amelioration pertinente pour la conduite sur highway-v0, en echange d'un cout de modelisation et d'optimisation plus eleve.

## 9. Tracabilite des fichiers
- Config observations : observation_extension_config.py
- Hyperparametres DQN et preprocessing : common_observation_extension.py
- Orchestration protocole 30 evals : run_observation_extension_long.py
- Evaluation et export resultats : evaluate_observation_extension.py
- Resultats chiffres : artifacts/observation_extension/comparison_summary.csv
- Resume auto genere : artifacts/observation_extension/comparison_summary.md
