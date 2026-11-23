"""
Test script focusing specifically on the fixed code section from the original request.
"""

import os
from pathlib import Path
from refactored_filter_system import (
    ProductionFilterAnalysisSystem, SystemConfig, ResourceMonitor
)

def main():
    print("Testing specific fixed code section...")
    
    # Create a system config with default values
    config = SystemConfig()
    
    # Create an instance of the ProductionFilterAnalysisSystem
    system = ProductionFilterAnalysisSystem(config)
    
    # Find an Excel file in the current directory
    excel_files = list(Path('.').glob('*.xlsx'))
    if not excel_files:
        print("No Excel files found in the current directory.")
        return
    
    data_path = str(excel_files[0])
    print(f"Using data file: {data_path}")
    
    # Test the specific fixed code section
    with ResourceMonitor(system.config, system.logger) as monitor:
        try:
            system.logger.info(f"Starting system training on {data_path}")
            
            # Load and validate data
            monitor.check_limits()
            df = system._load_and_validate_data(data_path)
            print("✓ Data loaded and validated successfully")
            
            # Create targets
            monitor.check_limits()
            binary_target, multiclass_target = system.target_creator.create_targets(df)
            print("✓ Targets created successfully")
            
            # Analyze risk distribution
            monitor.check_limits()
            system.risk_distribution = system.risk_analyzer.analyze_risk_distribution(
                multiclass_target, binary_target, df
            )
            print("✓ Risk distribution analyzed successfully")
            
            # Create visualizations
            create_visualizations = True
            if create_visualizations and system.visualizer:
                system.visualizer.create_data_overview(df, binary_target)
                print("✓ Visualizations created successfully")
            
            # Split data
            monitor.check_limits()
            split_result = system.splitter.split(df, binary_target, multiclass_target)
            df_train, df_test = split_result[0], split_result[1]
            y_binary_train, y_binary_test = split_result[2], split_result[3]
            y_multi_train, y_multi_test = split_result[4], split_result[5]
            print("✓ Data split successfully")
            
            # Engineer features
            monitor.check_limits()
            system.feature_engineer.fit(df_train)
            X_train = system.feature_engineer.transform(df_train)
            X_test = system.feature_engineer.transform(df_test)
            print("✓ Features engineered successfully")
            
            # Ensure targets are numpy arrays
            import numpy as np
            if not isinstance(y_binary_train, np.ndarray):
                y_binary_train = np.array(y_binary_train)
            if not isinstance(y_binary_test, np.ndarray):
                y_binary_test = np.array(y_binary_test)
            if not isinstance(y_multi_train, np.ndarray):
                y_multi_train = np.array(y_multi_train)
            if not isinstance(y_multi_test, np.ndarray):
                y_multi_test = np.array(y_multi_test)
            print("✓ Targets converted to numpy arrays successfully")
            
            print("All steps completed successfully!")
            return True
            
        except Exception as e:
            system.logger.error(f"System training failed: {e}")
            import traceback
            system.logger.error(f"Full error: {traceback.format_exc()}")
            print(f"Error: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    main()
