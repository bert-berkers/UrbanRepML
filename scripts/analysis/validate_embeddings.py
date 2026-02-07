"""
Validate embeddings — thin wrapper around stage3_analysis.validation.

Usage:
    python scripts/analysis/validate_embeddings.py
"""

import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main validation pipeline."""
    from stage3_analysis.validation import EmbeddingValidator

    validator = EmbeddingValidator()

    # Generate comprehensive report
    validator.generate_report("data/study_areas/netherlands/validation_report.txt")

    # Validate specific intermediate data
    logger.info("\nValidating POI intermediate data...")
    poi_validation = validator.validate_intermediate_data("poi", "netherlands", 10)
    logger.info(f"POI intermediate files found: {poi_validation['files_found']}")

    logger.info("\nValidating Roads intermediate data...")
    roads_validation = validator.validate_intermediate_data("roads", "netherlands", 10)
    logger.info(f"Roads intermediate files found: {roads_validation['files_found']}")


if __name__ == "__main__":
    main()
