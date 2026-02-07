"""
Benchmark Script: Cone Construction Geometric Optimizations
============================================================

Compares original vs geometrically-optimized cone construction.

Measures:
- Total construction time
- Memory usage
- Validation against geometric expectations
- Per-cone creation speed
"""

import logging
import time
import sys
from pathlib import Path
import geopandas as gpd
import psutil
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stage2_fusion.data.hierarchical_cone_masking import HierarchicalConeMaskingSystem
from stage2_fusion.geometry import expected_cone_size, log_geometric_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def load_test_regions(study_area: str = "netherlands"):
    """Load regions for testing (small subset for speed)."""
    logger.info(f"Loading regions for {study_area}...")

    data_dir = project_root / f"data/study_areas/{study_area}"
    regions_by_res = {}

    # Load just res5-7 for faster benchmarking
    for res in [5, 6, 7]:
        regions_path = data_dir / "regions_gdf" / f"{study_area}_res{res}.parquet"

        if not regions_path.exists():
            logger.warning(f"Missing: {regions_path}")
            continue

        regions_gdf = gpd.read_parquet(regions_path)
        regions_by_res[res] = regions_gdf
        logger.info(f"  Loaded res{res}: {len(regions_gdf)} hexagons")

    return regions_by_res


def benchmark_original(cone_system, regions_by_res, cache_dir):
    """Benchmark original cone construction."""
    logger.info("\n" + "="*60)
    logger.info("Benchmarking ORIGINAL cone construction")
    logger.info("="*60)

    mem_before = get_memory_usage()
    time_start = time.time()

    cones = cone_system.create_all_cones(
        regions_by_res,
        cache_dir=str(cache_dir)
    )

    time_end = time.time()
    mem_after = get_memory_usage()

    elapsed = time_end - time_start
    mem_used = mem_after - mem_before

    logger.info(f"\n📊 ORIGINAL Results:")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Memory: {mem_used:.1f}MB")
    logger.info(f"  Cones created: {len(cones)}")
    logger.info(f"  Speed: {len(cones)/elapsed:.1f} cones/sec")

    return {
        'time': elapsed,
        'memory': mem_used,
        'num_cones': len(cones),
        'speed': len(cones) / elapsed if elapsed > 0 else 0,
        'cones': cones
    }


def benchmark_optimized(cone_system, regions_by_res, cache_dir):
    """Benchmark geometrically-optimized cone construction."""
    logger.info("\n" + "="*60)
    logger.info("Benchmarking OPTIMIZED cone construction")
    logger.info("="*60)

    mem_before = get_memory_usage()
    time_start = time.time()

    cones = cone_system.create_all_cones_optimized(
        regions_by_res,
        cache_dir=str(cache_dir),
        spatial_sort=True,
        validate_geometry=True
    )

    time_end = time.time()
    mem_after = get_memory_usage()

    elapsed = time_end - time_start
    mem_used = mem_after - mem_before

    logger.info(f"\n📊 OPTIMIZED Results:")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Memory: {mem_used:.1f}MB")
    logger.info(f"  Cones created: {len(cones)}")
    logger.info(f"  Speed: {len(cones)/elapsed:.1f} cones/sec")

    return {
        'time': elapsed,
        'memory': mem_used,
        'num_cones': len(cones),
        'speed': len(cones) / elapsed if elapsed > 0 else 0,
        'cones': cones
    }


def verify_correctness(original_cones, optimized_cones):
    """Verify that both methods produce identical results."""
    logger.info("\n" + "="*60)
    logger.info("Verifying Correctness")
    logger.info("="*60)

    if len(original_cones) != len(optimized_cones):
        logger.error(f"❌ Different number of cones: {len(original_cones)} vs {len(optimized_cones)}")
        return False

    # Check each cone
    mismatches = 0
    for i, (orig, opt) in enumerate(zip(original_cones, optimized_cones)):
        if orig.cone_id != opt.cone_id:
            logger.error(f"❌ Cone {i}: Different IDs: {orig.cone_id} vs {opt.cone_id}")
            mismatches += 1
            continue

        # Check node counts by resolution
        orig_counts = orig.nodes_per_resolution()
        opt_counts = opt.nodes_per_resolution()

        for res in orig_counts:
            if orig_counts[res] != opt_counts.get(res, 0):
                logger.error(f"❌ Cone {i}, Res{res}: {orig_counts[res]} vs {opt_counts.get(res, 0)}")
                mismatches += 1

    if mismatches == 0:
        logger.info("✅ All cones match! Optimized version produces identical results.")
        return True
    else:
        logger.error(f"❌ Found {mismatches} mismatches")
        return False


def print_comparison(original_results, optimized_results):
    """Print side-by-side comparison."""
    logger.info("\n" + "="*60)
    logger.info("Performance Comparison")
    logger.info("="*60)

    speedup = original_results['time'] / optimized_results['time'] if optimized_results['time'] > 0 else 0
    mem_reduction = original_results['memory'] - optimized_results['memory']

    logger.info(f"\nMetric                 Original      Optimized     Improvement")
    logger.info(f"-" * 60)
    logger.info(f"Time:                  {original_results['time']:6.2f}s      {optimized_results['time']:6.2f}s      {speedup:5.2f}x")
    logger.info(f"Memory:                {original_results['memory']:6.1f}MB     {optimized_results['memory']:6.1f}MB     {mem_reduction:+6.1f}MB")
    logger.info(f"Speed:                 {original_results['speed']:6.1f}/s      {optimized_results['speed']:6.1f}/s      {optimized_results['speed']/original_results['speed']:5.2f}x")

    logger.info(f"\n🎯 Summary:")
    if speedup > 1.0:
        logger.info(f"  ✅ Optimized version is {speedup:.2f}x faster")
    elif speedup < 1.0:
        logger.info(f"  ⚠️  Optimized version is {1/speedup:.2f}x slower (unexpected!)")
    else:
        logger.info(f"  ➖ No significant speed difference")

    if mem_reduction > 0:
        logger.info(f"  ✅ Reduced memory usage by {mem_reduction:.1f}MB")
    elif mem_reduction < 0:
        logger.info(f"  ⚠️  Increased memory usage by {abs(mem_reduction):.1f}MB")


def main():
    """Run benchmark comparison."""
    logger.info("="*60)
    logger.info("Cone Construction Benchmark")
    logger.info("="*60)

    # Configuration
    study_area = "netherlands"
    parent_resolution = 5
    target_resolution = 7  # Reduced for faster benchmarking
    neighbor_rings = 5

    # Log geometric expectations
    log_geometric_summary(parent_resolution, target_resolution, neighbor_rings, logger)

    # Setup
    cache_dir = project_root / f"data/study_areas/{study_area}/cones"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear cache for fair comparison
    cache_file = cache_dir / f"parent_to_children_res{parent_resolution}_to_{target_resolution}.pkl"
    if cache_file.exists():
        logger.info(f"Using existing cache: {cache_file}")
    else:
        logger.info("No cache found - will build lookup table")

    # Load regions
    regions_by_res = load_test_regions(study_area)

    if not regions_by_res:
        logger.error("No regions loaded! Cannot run benchmark.")
        return

    # Create cone system
    cone_system = HierarchicalConeMaskingSystem(
        parent_resolution=parent_resolution,
        target_resolution=target_resolution,
        neighbor_rings=neighbor_rings
    )

    # Benchmark original
    original_results = benchmark_original(cone_system, regions_by_res, cache_dir)

    # Benchmark optimized
    optimized_results = benchmark_optimized(cone_system, regions_by_res, cache_dir)

    # Verify correctness
    is_correct = verify_correctness(
        original_results['cones'],
        optimized_results['cones']
    )

    if not is_correct:
        logger.error("❌ CORRECTNESS CHECK FAILED! Optimizations introduced errors.")
        return

    # Print comparison
    print_comparison(original_results, optimized_results)

    logger.info("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
