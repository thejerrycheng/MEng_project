import os
import pandas as pd
import glob
import argparse
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

def analyze_csv(filepath):
    """
    Reads a single metric CSV and computes statistics.
    Returns a DataFrame row (dict) and the raw dataframe for merging later if needed.
    """
    try:
        df = pd.read_csv(filepath)
        
        required_cols = [
            'visual_alignment', 'success_flag', 'srr_percent', 
            'framing_error_px', 'cartesian_jerk'
        ]
        
        # Filter for columns that actually exist
        available_cols = [c for c in required_cols if c in df.columns]
        
        if not available_cols:
            return None, None

        stats = {}
        
        # 1. Success Rate
        if 'success_flag' in df.columns:
            stats['Success Rate (%)'] = df['success_flag'].mean() * 100
        elif 'status' in df.columns:
            success_count = df[df['status'] == 'SUCCESS'].shape[0]
            stats['Success Rate (%)'] = (success_count / len(df)) * 100
        else:
            stats['Success Rate (%)'] = 0.0

        # 2. Continuous Metrics
        if 'visual_alignment' in df.columns:
            stats['Avg Vis. Align'] = df['visual_alignment'].mean()
        
        if 'cartesian_jerk' in df.columns:
            stats['Avg Jerk (m/s^3)'] = df['cartesian_jerk'].mean()
            
        if 'framing_error_px' in df.columns:
            stats['Avg Framing Err (px)'] = df['framing_error_px'].mean()
            
        if 'srr_percent' in df.columns:
            stats['Avg SRR (%)'] = df['srr_percent'].mean()
            
        if 'path_length_m' in df.columns:
             stats['Avg Path Len (m)'] = df['path_length_m'].mean()

        stats['Num Episodes'] = len(df)
        
        return stats, df

    except Exception as e:
        print(f"{Fore.RED}Error reading {os.path.basename(filepath)}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Folder containing metrics CSVs")
    parser.add_argument("--output", type=str, default="summary_results.csv", help="Output summary file")
    args = parser.parse_args()

    print(f"{Fore.WHITE}--------------------------------------------------")
    print(f"{Fore.YELLOW}  METRICS SUMMARY GENERATOR (With Merge)")
    print(f"{Fore.WHITE}--------------------------------------------------")

    search_pattern = os.path.join(args.root, "*metrics.csv")
    csv_files = sorted(glob.glob(search_pattern))

    if not csv_files:
        print(f"{Fore.RED}No '*metrics.csv' files found in {args.root}")
        return

    print(f"{Fore.CYAN}Found {len(csv_files)} files. Processing...\n")

    # Data Storage
    processed_methods = {} # Key: Method Name, Value: DataFrame of all episodes

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        # Clean Filename
        name_clean = filename.replace("processed_", "").replace("_metrics.csv", "")
        
        # --- MERGE LOGIC ---
        # If name is "policy_full" or "policy_full2", map them both to "policy_full"
        if name_clean in ["policy_full", "policy_full2"]:
            group_name = "policy_full"
        else:
            group_name = name_clean
            
        print(f"Reading: {filename:<35} -> Group: {Fore.GREEN}{group_name}")
        
        stats, raw_df = analyze_csv(filepath)
        
        if raw_df is not None:
            if group_name in processed_methods:
                # Concatenate with existing data for this group
                processed_methods[group_name] = pd.concat([processed_methods[group_name], raw_df], ignore_index=True)
            else:
                processed_methods[group_name] = raw_df
        else:
            print(f"{Fore.RED}  -> Skipped (Invalid Data)")

    # Compute Final Stats from Grouped Dataframes
    final_summary_data = []
    
    for method_name, df in processed_methods.items():
        stats = {}
        stats['Method'] = method_name
        stats['Num Episodes'] = len(df)
        
        # Re-calculate averages on the merged data
        if 'success_flag' in df.columns:
            stats['Success Rate (%)'] = df['success_flag'].mean() * 100
        elif 'status' in df.columns:
            success_count = df[df['status'] == 'SUCCESS'].shape[0]
            stats['Success Rate (%)'] = (success_count / len(df)) * 100
        else:
             stats['Success Rate (%)'] = 0.0
             
        if 'visual_alignment' in df.columns:
            stats['Avg Vis. Align'] = df['visual_alignment'].mean()
        if 'cartesian_jerk' in df.columns:
            stats['Avg Jerk (m/s^3)'] = df['cartesian_jerk'].mean()
        if 'framing_error_px' in df.columns:
            stats['Avg Framing Err (px)'] = df['framing_error_px'].mean()
        if 'srr_percent' in df.columns:
            stats['Avg SRR (%)'] = df['srr_percent'].mean()
        if 'path_length_m' in df.columns:
             stats['Avg Path Len (m)'] = df['path_length_m'].mean()
             
        final_summary_data.append(stats)

    # Display & Save
    if final_summary_data:
        summary_df = pd.DataFrame(final_summary_data)
        
        cols = ['Method', 'Num Episodes', 'Success Rate (%)', 'Avg Vis. Align', 
                'Avg Jerk (m/s^3)', 'Avg Framing Err (px)', 'Avg SRR (%)']
        
        # Add extras if they exist
        extra_cols = [c for c in summary_df.columns if c not in cols]
        final_cols = [c for c in (cols + extra_cols) if c in summary_df.columns]
        
        summary_df = summary_df[final_cols]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        print(f"\n{Fore.WHITE}--------------------------------------------------")
        print(f"{Fore.YELLOW}  FINAL SUMMARY TABLE (Merged)")
        print(f"{Fore.WHITE}--------------------------------------------------")
        print(summary_df.to_string(index=False))

        output_path = os.path.join(args.root, args.output)
        summary_df.to_csv(output_path, index=False)
        print(f"\n{Fore.GREEN}Summary saved to: {output_path}")

    else:
        print(f"{Fore.RED}No valid data processed.")

if __name__ == "__main__":
    main()