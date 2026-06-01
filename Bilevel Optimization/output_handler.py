"""
Output handler for FIFA 2026 Bilevel Optimization
Exports solution to Excel files matching input data structure
"""

import pandas as pd
from datetime import datetime, timedelta


class OutputHandler:
    """Handles solution output to Excel files."""
    
    def __init__(self, data_loader):
        """
        Initialize output handler.
        
        Args:
            data_loader: DataLoader instance with loaded data
        """
        self.data = data_loader
    
    def export_solution(self, solution, output_dir="output"):
        """
        Export solution to Excel files.
        
        Args:
            solution: Solution dictionary from solver
            output_dir: Directory to save Excel files (will be created)
        
        Returns:
            Tuple of (schedule_file, base_camps_file, metadata_file) paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate Excel files
        schedule_file = self._export_optimized_schedule(solution, output_dir)
        camps_file = self._export_base_camp_assignments(solution, output_dir)
        metadata_file = self._export_solution_metadata(solution, output_dir)
        
        print(f"\n" + "="*70)
        print("SOLUTION EXPORTED")
        print("="*70)
        print(f"Schedule: {schedule_file}")
        print(f"Base Camps: {camps_file}")
        print(f"Metadata: {metadata_file}")
        print("="*70 + "\n")
        
        return schedule_file, camps_file, metadata_file
    
    def _export_optimized_schedule(self, solution, output_dir):
        """
        Export optimized schedule to Excel (matches.csv structure).
        
        Args:
            solution: Solution dictionary
            output_dir: Output directory
        
        Returns:
            Path to Excel file
        """
        # Start with original matches data
        schedule_df = self.data.matches.copy()
        
        # Extract scheduled matches from solution
        if 'schedule' in solution and solution['schedule']:
            # solution['schedule'] format: {match_id: {'slot_idx': idx, 'venue_id': vid, 'slot': (date, hour)}}
            for match_id, schedule_info in solution['schedule'].items():
                mask = schedule_df['match_id'] == match_id
                if mask.any():
                    # Update venue
                    schedule_df.loc[mask, 'venue_id'] = schedule_info.get('venue_id')
                    
                    # Update date and time from slot
                    if 'slot' in schedule_info:
                        date, hour = schedule_info['slot']
                        schedule_df.loc[mask, 'date'] = pd.Timestamp(date)
                        schedule_df.loc[mask, 'kickoff_local'] = f"{int(hour):02d}:00"
        
        # Ensure correct data types and format
        output_cols = [
            'match_id', 'group', 'round', 'team_a_id', 'team_b_id', 
            'venue_id', 'date', 'kickoff_local'
        ]
        schedule_df = schedule_df[output_cols].copy()
        schedule_df['date'] = pd.to_datetime(schedule_df['date']).dt.strftime('%Y-%m-%d')
        
        # Save to Excel
        filename = f"{output_dir}/optimized_schedule.xlsx"
        schedule_df.to_excel(filename, index=False, sheet_name='Schedule')
        
        print(f"  Optimized Schedule: {len(schedule_df)} matches exported")
        
        return filename
    
    def _export_base_camp_assignments(self, solution, output_dir):
        """
        Export base camp assignments to Excel.
        
        Args:
            solution: Solution dictionary
            output_dir: Output directory
        
        Returns:
            Path to Excel file
        """
        # Create base camps assignment data
        assignments = []
        
        # Handle both 'base_camps' and 'camp_assignment' keys
        camps_dict = solution.get('camp_assignment', solution.get('base_camps', {}))
        
        if camps_dict:
            for team_id, camp_id in camps_dict.items():
                # Get team name from teams data
                team_row = self.data.teams[self.data.teams['team_id'] == team_id]
                team_name = team_row['team_name'].values[0] if len(team_row) > 0 else f"Team {team_id}"
                
                # Get camp info from base_camps data (using base_camp_id not camp_id)
                camp_row = self.data.base_camps[self.data.base_camps['base_camp_id'] == camp_id]
                
                if len(camp_row) > 0:
                    assignment = {
                        'team_id': team_id,
                        'team_name': team_name,
                        'base_camp_id': camp_id,
                        'training_site': camp_row['training_site'].values[0],
                        'city': camp_row['city'].values[0],
                        'country': camp_row['country'].values[0],
                        'lat': camp_row['lat'].values[0],
                        'lon': camp_row['lon'].values[0],
                        'utc_offset_june': camp_row['utc_offset_june'].values[0]
                    }
                    assignments.append(assignment)
        
        # Create DataFrame
        assignments_df = pd.DataFrame(assignments)
        
        if len(assignments_df) == 0:
            # Create empty structure if no assignments
            assignments_df = pd.DataFrame(columns=[
                'team_id', 'team_name', 'base_camp_id', 'training_site',
                'city', 'country', 'lat', 'lon', 'utc_offset_june'
            ])
        
        # Save to Excel
        filename = f"{output_dir}/base_camp_assignments.xlsx"
        assignments_df.to_excel(filename, index=False, sheet_name='Assignments')
        
        print(f"  Base Camp Assignments: {len(assignments_df)} teams exported")
        
        return filename
    
    def _export_solution_metadata(self, solution, output_dir):
        """
        Export solution metadata to Excel.
        
        Args:
            solution: Solution dictionary
            output_dir: Output directory
        
        Returns:
            Path to Excel file
        """
        # Create metadata DataFrame
        metadata = {
            'Metric': [
                'KPI Penalty',
                'Iterations',
                'Solve Time (seconds)',
                'Teams Assigned',
                'Matches Scheduled'
            ],
            'Value': [
                f"{solution.get('kpi_penalty', 0):.4f}",
                solution.get('iterations', 0),
                f"{solution.get('solve_time', 0):.2f}",
                len(solution.get('camp_assignment', {})),
                len(solution.get('schedule', {}))
            ]
        }
        metadata_df = pd.DataFrame(metadata)
        
        # Add KPI history if available
        if 'kpi_history' in solution and solution['kpi_history']:
            history_data = []
            for entry in solution['kpi_history']:
                history_data.append({
                    'Iteration': entry.get('iteration', ''),
                    'Penalty': f"{entry.get('penalty', 0):.4f}",
                    'Accepted': 'Yes' if entry.get('accepted', False) else 'No',
                    'Note': entry.get('note', '')
                })
            history_df = pd.DataFrame(history_data)
        else:
            history_df = pd.DataFrame()
        
        # Save to Excel with multiple sheets
        filename = f"{output_dir}/solution_metadata.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            metadata_df.to_excel(writer, sheet_name='Summary', index=False)
            if not history_df.empty:
                history_df.to_excel(writer, sheet_name='KPI History', index=False)
        
        print(f"  Solution Metadata: Summary + History exported")
        
        return filename


def export_solution_excel(data_loader, solution, output_dir="output"):
    """
    Convenience function to export solution to Excel.
    
    Args:
        data_loader: DataLoader instance
        solution: Solution dictionary
        output_dir: Output directory (default "output")
    
    Returns:
        Tuple of (schedule_file, camps_file) paths
    """
    handler = OutputHandler(data_loader)
    return handler.export_solution(solution, output_dir)
