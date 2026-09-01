"""S&P 500 equity-pairs research package."""

from .config import PipelineConfig, ResearchConfig, StrategyConfig, load_config

__all__ = ["PipelineConfig", "ResearchConfig", "StrategyConfig", "load_config"]
__version__ = "0.1.0"
