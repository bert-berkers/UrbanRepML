#!/usr/bin/env python3
"""
Migration script to replace direct h3 imports with SRAI usage.
Per CLAUDE.md: ALL H3 operations must use SRAI, never h3-py directly.

This script:
1. Finds all Python files using h3 directly
2. Analyzes their h3 usage patterns
3. Suggests or applies SRAI replacements
4. Documents changes made

Author: Claude Code
Date: January 2025
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common h3 function mappings to SRAI equivalents
H3_TO_SRAI_MAPPINGS = {
    # H3 hex operations -> SRAI H3Regionalizer
    'h3.geo_to_h3': 'Use H3Regionalizer.transform() with point geometry',
    'h3.h3_to_geo': 'Access geometry attribute of regions_gdf',
    'h3.h3_to_geo_boundary': 'Access geometry.boundary of regions_gdf',
    'h3.h3_get_resolution': 'Store resolution when creating H3Regionalizer',
    'h3.h3_is_valid': 'SRAI validates automatically',
    'h3.k_ring': 'Use H3Neighbourhood.get_neighbours()',
    'h3.h3_distance': 'Use H3Neighbourhood for spatial relationships',
    'h3.h3_to_parent': 'Use H3Regionalizer with different resolution',
    'h3.h3_to_children': 'Use H3Regionalizer with different resolution',
    'h3.polyfill': 'Use H3Regionalizer.transform() with polygon',
    'h3.h3_set_to_multi_polygon': 'Use regions_gdf.dissolve()',
}

class H3ToSRAIMigrator:
    """Migrates h3 imports to SRAI usage."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.changes_made = []
        self.files_analyzed = 0
        self.files_modified = 0
        
    def find_h3_files(self, root_dir: Path) -> List[Path]:
        """Find all Python files that import h3."""
        h3_files = []
        for py_file in root_dir.rglob("*.py"):
            # Skip migration scripts, virtual envs, tests, etc.
            str_path = str(py_file)
            if any(exclude in str_path for exclude in [
                'migration', '__pycache__', '.venv', 'venv', 'env',
                'site-packages', '.git'
            ]):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'import h3' in content:
                        h3_files.append(py_file)
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")
                
        return h3_files
    
    def analyze_h3_usage(self, file_path: Path) -> Dict[str, List[int]]:
        """Analyze how h3 is used in a file."""
        usage = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                for h3_func in H3_TO_SRAI_MAPPINGS.keys():
                    if h3_func in line:
                        func_name = h3_func.split('.')[-1]
                        if func_name not in usage:
                            usage[func_name] = []
                        usage[func_name].append(i)
                        
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            
        return usage
    
    def generate_srai_replacement(self, file_path: Path) -> str:
        """Generate SRAI replacement code for a file."""
        replacements = []
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check current imports
        has_h3_import = False
        has_srai_import = False
        import_line_idx = -1
        
        for i, line in enumerate(lines):
            if 'import h3' in line:
                has_h3_import = True
                import_line_idx = i
            if 'from srai' in line:
                has_srai_import = True
        
        # Create new content
        new_lines = lines.copy()
        
        if has_h3_import and not has_srai_import:
            # Replace h3 import with SRAI imports
            srai_imports = [
                "# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)\n",
                "from srai.regionalizers import H3Regionalizer\n",
                "from srai.neighbourhoods import H3Neighbourhood\n",
                "# Note: SRAI provides H3 functionality with additional spatial analysis tools\n"
            ]
            
            if import_line_idx >= 0:
                # Replace the h3 import line
                new_lines[import_line_idx:import_line_idx+1] = srai_imports
                
        return ''.join(new_lines)
    
    def create_migration_report(self, h3_files: List[Path]) -> str:
        """Create a detailed migration report."""
        report = ["# H3 to SRAI Migration Report\n\n"]
        report.append(f"**Total files with h3 imports**: {len(h3_files)}\n\n")
        
        report.append("## Files Requiring Migration\n\n")
        
        for file_path in h3_files:
            try:
                rel_path = file_path.relative_to(Path.cwd())
            except ValueError:
                rel_path = file_path  # Use full path if can't make relative
            report.append(f"### {rel_path}\n")
            
            usage = self.analyze_h3_usage(file_path)
            if usage:
                report.append("**H3 functions used**:\n")
                for func, lines in usage.items():
                    report.append(f"- `{func}` (lines: {', '.join(map(str, lines))})\n")
                    if f"h3.{func}" in H3_TO_SRAI_MAPPINGS:
                        report.append(f"  - **SRAI replacement**: {H3_TO_SRAI_MAPPINGS[f'h3.{func}']}\n")
            report.append("\n")
            
        report.append("## Recommended Actions\n\n")
        report.append("1. Install SRAI if not already installed: `pip install srai[all]`\n")
        report.append("2. Replace h3 imports with SRAI imports\n")
        report.append("3. Update h3 function calls to use SRAI equivalents\n")
        report.append("4. Test all modified files to ensure functionality\n")
        
        return ''.join(report)
    
    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file from h3 to SRAI."""
        try:
            logger.info(f"Migrating {file_path}")
            
            # Generate replacement
            new_content = self.generate_srai_replacement(file_path)
            
            if not self.dry_run:
                # Backup original
                backup_path = file_path.with_suffix('.py.bak')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(file_path.read_text())
                
                # Write new content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
                logger.info(f"✓ Migrated {file_path} (backup: {backup_path})")
                self.files_modified += 1
                return True
            else:
                logger.info(f"[DRY RUN] Would migrate {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Migrate h3 imports to SRAI')
    parser.add_argument('--apply', action='store_true', 
                       help='Actually apply changes (default is dry run)')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report, no changes')
    parser.add_argument('--path', type=str, default='.',
                       help='Root path to search for files')
    args = parser.parse_args()
    
    root_dir = Path(args.path)
    migrator = H3ToSRAIMigrator(dry_run=not args.apply)
    
    # Find all h3 files
    logger.info(f"Searching for h3 imports in {root_dir}")
    h3_files = migrator.find_h3_files(root_dir)
    logger.info(f"Found {len(h3_files)} files with h3 imports")
    
    # Generate report
    report = migrator.create_migration_report(h3_files)
    report_path = Path('H3_MIGRATION_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    if args.report_only:
        logger.info("Report-only mode, no changes made")
        return
    
    # Migrate files if requested
    if args.apply:
        logger.info("Applying migrations...")
        for file_path in h3_files:
            migrator.migrate_file(file_path)
        logger.info(f"Modified {migrator.files_modified} files")
    else:
        logger.info("Dry run mode - use --apply to make changes")
        logger.info("Suggested next step: Review H3_MIGRATION_REPORT.md")

if __name__ == "__main__":
    main()