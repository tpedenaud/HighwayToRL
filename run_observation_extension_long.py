from common_observation_extension import DEFAULT_EXTENSION_SETTINGS
from custom_dqn_observation_extension import train_many
from evaluate_observation_extension import evaluate_all


def main():
    train_seeds = [0, 1, 2]
    # Reduced workload for practical CPU runtime while keeping a fair protocol.
    eval_seeds = list(range(4000, 4030))
    total_timesteps = 30_000

    kinematics_settings = dict(DEFAULT_EXTENSION_SETTINGS)

    occupancy_settings = dict(DEFAULT_EXTENSION_SETTINGS)
    occupancy_settings.update(
        {
            "learning_rate": 1e-4,
            "epsilon_decay_steps": 80_000,
            "learning_starts": 8_000,
            "target_update_interval": 1_000,
        }
    )

    print("[1/3] Train kinematics...")
    train_many("kinematics", train_seeds, total_timesteps=total_timesteps, device="cpu", settings=kinematics_settings)

    print("[2/3] Train occupancy_grid...")
    train_many("occupancy_grid", train_seeds, total_timesteps=total_timesteps, device="cpu", settings=occupancy_settings)

    print("[3/3] Evaluate...")
    evaluate_all(
        observation_modes=["kinematics", "occupancy_grid"],
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        device="cpu",
    )

    print("Done.")


if __name__ == "__main__":
    main()
