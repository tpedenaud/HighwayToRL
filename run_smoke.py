from common import SMOKE_DQN_SETTINGS, SMOKE_SEED
from custom_dqn import train_many as train_custom_dqn
from evaluate_models import evaluate_all
from report_results import make_reports
from train_sb3_dqn import train_many as train_sb3_dqn


def main():
    train_custom_dqn([SMOKE_SEED], total_timesteps=256, device="cpu", settings=SMOKE_DQN_SETTINGS)
    train_sb3_dqn([SMOKE_SEED], total_timesteps=256, device="cpu", settings=SMOKE_DQN_SETTINGS)
    evaluate_all(train_seeds=[SMOKE_SEED], eval_seeds=list(range(2000, 2003)), device="cpu")
    make_reports(train_seeds=[SMOKE_SEED], device="cpu")


if __name__ == "__main__":
    main()
