"""Unified training entry point for ReSkill."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig


@hydra.main(config_path='../configs', config_name='base', version_base=None)
def main(config: DictConfig):
    from reskill.verl_integration.runner import run_reskill_ppo

    run_reskill_ppo(config)


if __name__ == '__main__':
    main()
