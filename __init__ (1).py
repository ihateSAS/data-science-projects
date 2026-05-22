"""Composer classification on GiantMIDI-Piano."""
from .features import extract_features, extract_dataset
from .composer_eras import COMPOSER_ERA, era_for
from .models import (CalibratedComposerClassifier, HierarchicalEraComposer,
                     ConformalClassifier, composer_metrics,
                     expected_calibration_error)
from .evaluation import (random_cv, leave_one_work_out, leave_one_era_out,
                         reliability_curve)
__version__ = "0.1.0"
