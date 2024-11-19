from src.attacks.data_source import DataSource

from src.attacks.features_extraction.extractor import FeatureExtractor
from src.attacks.features_extraction.denoising_loss import DenoisingLossExtractor
from src.attacks.features_extraction.secmi import SecMIExtractor
from src.attacks.features_extraction.cdi import (
    CDIExtractor,
)
from src.attacks.features_extraction.pia import PIAExtractor
from src.attacks.features_extraction.gradient_masking import GradientMaskingExtractor
from src.attacks.features_extraction.multiple_loss import MultipleLossExtractor
from src.attacks.features_extraction.noise_optim import NoiseOptimExtractor

from src.attacks.scores_computation.computer import ScoreComputer
from src.attacks.scores_computation.denoising_loss import DenoisingLossComputer
from src.attacks.scores_computation.secmi import SecMIStat
from src.attacks.scores_computation.cdi import CDIComputer

from src.attacks.scores_computation.pia import PIAComputer, PIANComputer
from src.attacks.scores_computation.gradient_masking import GradientMaskingComputer
from src.attacks.scores_computation.multiple_loss import MultipleLossComputer
from src.attacks.scores_computation.noise_optim import NoiseOptimComputer

from typing import Dict


feature_extractors: Dict[str, FeatureExtractor] = {
    "denoising_loss": DenoisingLossExtractor,
    "secmi_stat": SecMIExtractor,
    "cdi": CDIExtractor,
    "pia": PIAExtractor,
    "pian": PIAExtractor,
    "gradient_masking": GradientMaskingExtractor,
    "multiple_loss": MultipleLossExtractor,
    "noise_optim": NoiseOptimExtractor,
}

score_computers: Dict[str, ScoreComputer] = {
    "denoising_loss": DenoisingLossComputer,
    "secmi_stat": SecMIStat,
    "cdi": CDIComputer,
    "pia": PIAComputer,
    "pian": PIANComputer,
    "gradient_masking": GradientMaskingComputer,
    "multiple_loss": MultipleLossComputer,
    "noise_optim": NoiseOptimComputer,
}

from src.attacks.utils import load_data, get_datasets_clf
