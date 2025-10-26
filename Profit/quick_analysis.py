# quick_analysis.py
from store_analysis import StoreSalesAnalyzer
import sys

def quick_analysis(file_path):
    """
    Run a quick analysis for immediate results
    """
    print("🚀 Running Quick Analysis...")
    
    analyzer = StoreSalesAnalyzer(file_path)
    
    # Run essential analyses only
    analyzer.data_preparation()
    analyzer.sales_analysis()
    analyzer.profit_analysis()
    analyzer.save_analysis_results()
    
    print("✅ Quick analysis completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "Superstore.xlsx"  # Default file path
    
    quick_analysis(file_path)