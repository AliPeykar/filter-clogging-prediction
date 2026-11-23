"""
Diagnostic script to check your filter analysis data file
"""

import pandas as pd
import numpy as np
from pathlib import Path

def diagnose_data_file(file_path='Comprehensive_Filter_Analysis.xlsx'):
    """Diagnose issues with the data file."""
    
    print("=" * 60)
    print("FILTER ANALYSIS DATA DIAGNOSTIC")
    print("=" * 60)
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ ERROR: File '{file_path}' not found!")
        print("Please ensure the file is in the current directory.")
        return False
    
    print(f"✓ File found: {file_path}")
    
    # Try to load the file
    try:
        if file_path.endswith('.xlsx'):
            # Try reading Excel with different engines
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                except:
                    df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
            
        print(f"✓ File loaded successfully")
        
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        print("\nTrying to read sheet names...")
        try:
            xl_file = pd.ExcelFile(file_path)
            print(f"Available sheets: {xl_file.sheet_names}")
            # Try first sheet
            df = pd.read_excel(file_path, sheet_name=0)
        except Exception as e2:
            print(f"❌ Failed to read any sheets: {e2}")
            return False
    
    # Check if DataFrame is empty
    if df.empty:
        print("❌ ERROR: DataFrame is empty!")
        return False
    
    # Display basic info
    print(f"\n📊 DATA SUMMARY:")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    
    # Display columns
    print(f"\n📋 COLUMNS FOUND:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Check for required columns
    print(f"\n🔍 CHECKING FOR EXPECTED COLUMNS:")
    expected_columns = [
        'flowrate', 'dp', 'time', 
        'validation_filter_status', 'filter_status',
        'short_ema', 'long_ema', 
        'adaptive_cusum_thr', 'validation_score'
    ]
    
    for col in expected_columns:
        if col in df.columns:
            print(f"  ✓ {col} - Found")
        else:
            # Check for case variations
            found_similar = False
            for df_col in df.columns:
                if col.lower() in df_col.lower():
                    print(f"  ⚠️  {col} - Not found, but found similar: '{df_col}'")
                    found_similar = True
                    break
            if not found_similar:
                print(f"  ✗ {col} - Missing")
    
    # Check data types
    print(f"\n📊 DATA TYPES:")
    for col in df.columns[:10]:  # Show first 10
        print(f"  - {col}: {df[col].dtype}")
    
    # Check for null values
    print(f"\n🔍 NULL VALUES CHECK:")
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        print("  Columns with null values:")
        for col, count in cols_with_nulls.items():
            pct = (count / len(df)) * 100
            print(f"    - {col}: {count} nulls ({pct:.1f}%)")
    else:
        print("  ✓ No null values found")
    
    # Sample data
    print(f"\n📝 FIRST 5 ROWS:")
    print(df.head())
    
    # Check for label columns
    print(f"\n🏷️  CHECKING FOR LABEL COLUMNS:")
    potential_label_cols = []
    for col in df.columns:
        if any(word in col.lower() for word in ['status', 'label', 'target', 'clog', 'filter']):
            potential_label_cols.append(col)
            unique_vals = df[col].value_counts()
            print(f"\n  Found: '{col}'")
            print(f"  Unique values: {len(unique_vals)}")
            if len(unique_vals) <= 10:
                print("  Distribution:")
                for val, count in unique_vals.items():
                    print(f"    - {val}: {count} ({count/len(df)*100:.1f}%)")
    
    if not potential_label_cols:
        print("  ⚠️  No obvious label columns found")
    
    # Data quality summary
    print(f"\n✅ DIAGNOSTIC SUMMARY:")
    if len(df) < 100:
        print(f"  ⚠️  Low sample count: {len(df)} (minimum 100 recommended)")
    else:
        print(f"  ✓ Sufficient samples: {len(df)}")
    
    has_required = all(col in df.columns for col in ['flowrate', 'dp'])
    if has_required:
        print("  ✓ Has minimum required sensors (flowrate, dp)")
    else:
        print("  ❌ Missing required sensors")
    
    print("\n" + "=" * 60)
    
    return True


def create_sample_data(output_file='sample_filter_data.csv'):
    """Create a sample dataset if needed."""
    
    print("\n📝 Creating sample filter dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate time series
    time = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    # Generate sensor data with clogging progression
    t = np.linspace(0, 1, n_samples)
    
    # Flow rate decreases over time
    flowrate = 50 - 30 * t + np.random.normal(0, 2, n_samples)
    flowrate = np.maximum(flowrate, 5)
    
    # Pressure drop increases over time
    dp = 10 + 40 * t + np.random.normal(0, 3, n_samples)
    dp = np.maximum(dp, 0)
    
    # Create EMAs
    short_ema = pd.Series(dp).ewm(span=10).mean().values
    long_ema = pd.Series(dp).ewm(span=50).mean().values
    
    # Create labels (0 = clean, 1 = clogged)
    # Transition happens around 70% through the data
    validation_filter_status = np.zeros(n_samples, dtype=int)
    validation_filter_status[int(n_samples * 0.7):] = 1
    
    # Add some noise to the transition
    transition_zone = slice(int(n_samples * 0.65), int(n_samples * 0.75))
    validation_filter_status[transition_zone] = np.random.choice([0, 1], 
                                                                 size=int(n_samples * 0.1),
                                                                 p=[0.3, 0.7])
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'flowrate': flowrate,
        'dp': dp,
        'short_ema': short_ema,
        'long_ema': long_ema,
        'adaptive_cusum_thr': np.random.normal(100, 20, n_samples),
        'validation_score': np.random.uniform(0.5, 1.0, n_samples),
        'validation_filter_status': validation_filter_status
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Sample data created: {output_file}")
    print(f"  - {n_samples} samples")
    print(f"  - {len(df.columns)} columns")
    print(f"  - Clogged samples: {validation_filter_status.sum()} ({validation_filter_status.sum()/n_samples*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    # First try to diagnose the existing file
    success = diagnose_data_file()
    
    if not success:
        print("\n" + "="*60)
        print("Would you like to create a sample dataset for testing?")
        print("This will create 'sample_filter_data.csv' with proper structure.")
        
        try:
            response = input("Create sample data? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                df = create_sample_data()
                print("\n✓ Sample data created successfully!")
                print("You can now run the training system with 'sample_filter_data.csv'")
        except:
            pass