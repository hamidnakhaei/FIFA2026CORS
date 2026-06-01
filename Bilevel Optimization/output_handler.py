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
            Tuple of (schedule_file, base_camps_file) paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate Excel files
        schedule_file = self._export_optimized_schedule(solution, output_dir)
        camps_file = self._export_base_camp_assignments(solution, output_dir)
        
        print(f"\n" + "="*70)
        print("SOLUTION EXPORTED")
        print("="*70)
        print(f"Schedule: {schedule_file}")
        print(f"Base Camps: {camps_file}")
        print("="*70 + "\n")
        
        return schedule_file, camps_file
    
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
        
        # For now, this is a placeholder that preserves original dates/times
        # In a full implementation, you would update date/venue/kickoff_local
        # based on the x variables in the solution
        
        # Example: Extract scheduled matches from solution if available
        if 'schedule' in solution and solution['schedule']:
            # Process schedule changes here
            # schedule format: {(match_id, hour, venue_id): 1, ...}
            for (match_id, hour, venue_id), assigned in solution['schedule'].items():
                if assigned == 1:
                    # Find the row and update venue (for demo)
                    mask = schedule_df['match_id'] == match_id
                    schedule_df.loc[mask, 'venue_id'] = venue_id
        
        # Ensure correct data types
        schedule_df = schedule_df[[
            'match_id', 'group', 'round', 'team_a_id', 'team_b_id', 
            'venue_id', 'date', 'kickoff_local'
        ]]
        
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
        
        if 'base_camps' in solution:
            for team_id, camp_id in solution['base_camps'].items():
                # Get camp information
                camp_info = self.data.get_camp_info(camp_id)
                team_info = self.data.get_team_info(team_id)
                
                if camp_info and team_info:
                    assignment = {
                        'team_id': team_id,
                        'team_name': team_info.get('team_name', ''),
                        'base_camp_id': camp_id,
                        'training_site': camp_info.get('training_site', ''),
                        'city': camp_info.get('city', ''),
                        'country': camp_info.get('country', ''),
                        'lat': camp_info.get('lat', ''),
                        'lon': camp_info.get('lon', ''),
                        'utc_offset_june': camp_info.get('utc_offset_june', '')
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
