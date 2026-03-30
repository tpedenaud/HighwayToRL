from common import DEFAULT_TRAIN_SEEDS, PILOT_SEED
from custom_dqn import train_many as train_custom_dqn
from evaluate_models import evaluate_all
from report_results import make_reports
from train_sb3_dqn import train_many as train_sb3_dqn


def main():
    train_custom_dqn([PILOT_SEED], total_timesteps=50_000, device="cpu")
    train_sb3_dqn([PILOT_SEED], total_timesteps=50_000, device="cpu")
    train_custom_dqn(DEFAULT_TRAIN_SEEDS, total_timesteps=200_000, device="cpu")
    train_sb3_dqn(DEFAULT_TRAIN_SEEDS, total_timesteps=200_000, device="cpu")
    evaluate_all(train_seeds=DEFAULT_TRAIN_SEEDS, device="cpu")
    make_reports(train_seeds=DEFAULT_TRAIN_SEEDS, device="cpu")


if __name__ == "__main__":
    main()
