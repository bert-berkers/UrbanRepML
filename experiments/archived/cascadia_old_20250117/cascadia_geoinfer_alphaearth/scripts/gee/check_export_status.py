"""
Check Google Earth Engine export task status and log completion progress.
"""

import ee
import json
import time
from datetime import datetime
from typing import Dict, List
import logging

# Initialize Earth Engine
try:
    ee.Initialize(project='boreal-union-296021')
    print("Google Earth Engine initialized successfully with project boreal-union-296021")
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class ExportStatusTracker:
    """Track AlphaEarth export task status."""
    
    def __init__(self):
        self.export_folder = "AlphaEarth_Cascadia"
        self.expected_tiles_per_year = 644  # From our dry run
        self.years = list(range(2017, 2025))
        self.total_expected_tiles = len(self.years) * self.expected_tiles_per_year
        
    def get_all_tasks(self) -> List[Dict]:
        """Get all export tasks from GEE."""
        try:
            task_list = ee.batch.Task.list()
            return task_list
        except Exception as e:
            logger.error(f"Failed to get task list: {e}")
            return []
    
    def filter_cascadia_tasks(self, tasks: List) -> List[Dict]:
        """Filter tasks related to Cascadia AlphaEarth export."""
        cascadia_tasks = []
        
        for task in tasks:
            task_config = task.config if hasattr(task, 'config') else {}
            task_id = getattr(task, 'id', 'unknown')
            task_state = getattr(task, 'state', 'UNKNOWN')
            
            # Check if task is related to our Cascadia export
            description = task_config.get('description', '')
            folder = task_config.get('driveFolder', '')
            
            if (self.export_folder in folder or 
                'cascadia' in description.lower() or
                any(str(year) in description for year in self.years)):
                
                cascadia_tasks.append({
                    'id': task_id,
                    'description': description,
                    'state': task_state,
                    'folder': folder,
                    'creation_timestamp_ms': getattr(task, 'creation_timestamp_ms', 0),
                    'update_timestamp_ms': getattr(task, 'update_timestamp_ms', 0),
                    'start_timestamp_ms': getattr(task, 'start_timestamp_ms', 0),
                    'config': task_config
                })
        
        return cascadia_tasks
    
    def analyze_export_progress(self, cascadia_tasks: List[Dict]) -> Dict:
        """Analyze export progress by year and tile."""
        progress = {
            'total_tasks': len(cascadia_tasks),
            'by_state': {},
            'by_year': {},
            'completed_tiles': [],
            'failed_tiles': [],
            'running_tiles': [],
            'pending_tiles': []
        }
        
        # Initialize year tracking
        for year in self.years:
            progress['by_year'][year] = {
                'total': 0,
                'completed': 0,
                'failed': 0,
                'running': 0,
                'pending': 0,
                'tiles': []
            }
        
        # Process each task
        for task in cascadia_tasks:
            state = task['state']
            description = task['description']
            
            # Update state counts
            if state not in progress['by_state']:
                progress['by_state'][state] = 0
            progress['by_state'][state] += 1
            
            # Extract year and tile info from description
            year_found = None
            tile_id = None
            
            for year in self.years:
                if str(year) in description:
                    year_found = year
                    break
            
            # Extract tile ID (format: YYYY_NNNN)
            import re
            tile_match = re.search(r'(\d{4})_(\d{4})', description)
            if tile_match:
                year_from_tile = int(tile_match.group(1))
                tile_number = int(tile_match.group(2))
                tile_id = f"{year_from_tile}_{tile_number:04d}"
                if not year_found:
                    year_found = year_from_tile
            
            if year_found:
                progress['by_year'][year_found]['total'] += 1
                progress['by_year'][year_found]['tiles'].append({
                    'tile_id': tile_id or description,
                    'state': state,
                    'task_id': task['id']
                })
                
                if state == 'COMPLETED':
                    progress['by_year'][year_found]['completed'] += 1
                    progress['completed_tiles'].append(tile_id or description)
                elif state == 'FAILED':
                    progress['by_year'][year_found]['failed'] += 1
                    progress['failed_tiles'].append(tile_id or description)
                elif state == 'RUNNING':
                    progress['by_year'][year_found]['running'] += 1
                    progress['running_tiles'].append(tile_id or description)
                else:  # READY, PENDING, etc.
                    progress['by_year'][year_found]['pending'] += 1
                    progress['pending_tiles'].append(tile_id or description)
        
        return progress
    
    def generate_completion_report(self, progress: Dict) -> str:
        """Generate a detailed completion report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CASCADIA ALPHAEARTH EXPORT STATUS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        # Overall summary
        total_tasks = progress['total_tasks']
        completed = len(progress['completed_tiles'])
        failed = len(progress['failed_tiles'])
        running = len(progress['running_tiles'])
        pending = len(progress['pending_tiles'])
        
        report_lines.append("\nOVERALL PROGRESS:")
        report_lines.append(f"  Total Tasks Found: {total_tasks}")
        report_lines.append(f"  Expected Total: {self.total_expected_tiles} ({len(self.years)} years × {self.expected_tiles_per_year} tiles)")
        report_lines.append(f"  Completed: {completed} ({completed/max(total_tasks,1)*100:.1f}%)")
        report_lines.append(f"  Failed: {failed}")
        report_lines.append(f"  Running: {running}")
        report_lines.append(f"  Pending: {pending}")
        
        # State breakdown
        report_lines.append("\nTASK STATES:")
        for state, count in progress['by_state'].items():
            percentage = (count / max(total_tasks, 1)) * 100
            report_lines.append(f"  {state}: {count} ({percentage:.1f}%)")
        
        # Year-by-year breakdown
        report_lines.append("\nPROGRESS BY YEAR:")
        for year in self.years:
            year_data = progress['by_year'][year]
            total_year = year_data['total']
            completed_year = year_data['completed']
            
            if total_year > 0:
                completion_pct = (completed_year / total_year) * 100
                report_lines.append(f"  {year}: {completed_year}/{total_year} tiles ({completion_pct:.1f}%)")
                if year_data['failed'] > 0:
                    report_lines.append(f"    Failed: {year_data['failed']}")
                if year_data['running'] > 0:
                    report_lines.append(f"    Running: {year_data['running']}")
                if year_data['pending'] > 0:
                    report_lines.append(f"    Pending: {year_data['pending']}")
            else:
                report_lines.append(f"  {year}: No tasks found - NEEDS EXPORT")
        
        # Missing years analysis
        missing_years = [year for year in self.years if progress['by_year'][year]['total'] == 0]
        if missing_years:
            report_lines.append(f"\nMISSING YEARS (need to start export): {missing_years}")
        
        # Recommendations
        report_lines.append("\nRECOMMENDATIONS:")
        if missing_years:
            report_lines.append(f"  1. Start exports for missing years: {missing_years}")
        if failed > 0:
            report_lines.append(f"  2. Retry {failed} failed tasks")
        if completed < self.total_expected_tiles:
            remaining = self.total_expected_tiles - total_tasks
            report_lines.append(f"  3. {remaining} additional tiles still need to be exported")
        
        return "\n".join(report_lines)
    
    def save_detailed_progress(self, progress: Dict, filename: str = None):
        """Save detailed progress to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"export_progress_{timestamp}.json"
        
        # Add metadata
        progress['metadata'] = {
            'check_timestamp': datetime.now().isoformat(),
            'expected_tiles_per_year': self.expected_tiles_per_year,
            'expected_years': self.years,
            'total_expected_tiles': self.total_expected_tiles
        }
        
        with open(filename, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
        
        logger.info(f"Detailed progress saved to: {filename}")
        return filename
    
    def run_status_check(self):
        """Run complete status check and generate reports."""
        logger.info("Checking Google Earth Engine export tasks...")
        
        # Get all tasks
        all_tasks = self.get_all_tasks()
        logger.info(f"Found {len(all_tasks)} total tasks in project")
        
        # Filter for Cascadia tasks
        cascadia_tasks = self.filter_cascadia_tasks(all_tasks)
        logger.info(f"Found {len(cascadia_tasks)} Cascadia-related tasks")
        
        # Analyze progress
        progress = self.analyze_export_progress(cascadia_tasks)
        
        # Generate and display report
        report = self.generate_completion_report(progress)
        print(report)
        
        # Save detailed progress
        progress_file = self.save_detailed_progress(progress)
        
        return progress, report, progress_file

def main():
    tracker = ExportStatusTracker()
    progress, report, progress_file = tracker.run_status_check()
    
    print(f"\nDetailed progress saved to: {progress_file}")
    print("\nTo retry failed tasks or start missing years, use the export script with appropriate parameters.")

if __name__ == "__main__":
    main()