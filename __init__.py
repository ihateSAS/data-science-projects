"""Piano difficulty prediction — symbolic-music feature extraction and ordinal models."""
from .features import extract_features, extract_dataset, PieceFeatures
from .models import (OrdinalGBM, ConformalOrdinal, ordinal_metrics)
from .evaluation import (loco_cv, stratified_cv, expected_calibration_error,
                         reliability_curve, label_noise_sensitivity)

try:
    from .models import CORNMLPRegressor
except ImportError:
    pass

__version__ = "0.1.0"
