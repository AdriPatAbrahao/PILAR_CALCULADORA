"""
Demo script: Run inference tests
"""

import logging

# Desativa logs de INFO, mostra apenas ERROS
logging.getLogger("src.feature_engineering").setLevel(logging.ERROR)
logging.getLogger("src.predictor").setLevel(logging.ERROR)
logging.getLogger("src.data_loader").setLevel(logging.ERROR)

from src.inference_examples import (
    example_single_prediction,
    example_batch_test,
    example_compare_variations,
)


if __name__ == "__main__":
    example_single_prediction()
    print("\n" * 2)
    
    example_batch_test()
    print("\n" * 2)
    
    example_compare_variations()