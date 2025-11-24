from abc import ABC, abstractmethod
from config.context import Context

class BasePipeline(ABC):
    def __init__(self, ctx: Context):
        self.ctx = ctx

    @abstractmethod
    def load_data(self):
        """Load dataset"""
        pass

    @abstractmethod
    def column_selection(self, df):
        """Select columns or features relevant to the experiment."""
        pass

    @abstractmethod
    def feature_engineering(self, df):
        """Apply feature engineering logic."""
        pass
    
    @abstractmethod
    def preprocessing(self, df):
        """Split, scale, encode, and prepare datasets."""
        pass
    
    @abstractmethod
    def train(self):
        """Full training routine."""
        pass
    
    @abstractmethod
    def evaluate(self):
        """Evaluation logic for validation/test sets."""
        pass
    
    @abstractmethod
    def run(self, until: str = "evaluate"):
        """
        Orchestrate the full experiment.

        `until` defines the execution limit. e.g.:
        "preprocessing", "train", "evaluate", etc.
        """
        ...