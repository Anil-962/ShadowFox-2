# store_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StoreSalesAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the Store Sales Analyzer
        """
        self.file_path = file_path
        self.df = None
        self.preprocessed = False
        self.analysis_results = {}
        self.setup_directories()
        
    def setup_directories(self):
        """Create directories for saving results"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"analysis_results_{self.timestamp}"
        self.viz_dir = os.path.join(self.results_dir, "visualizations")
        self.data_dir = os.path.join(self.results_dir, "data")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        print(f"📁 Results will be saved in: {self.results_dir}")
    
    def load_data(self):
        """
        Load data from various file formats
        """
        print("📂 Loading data...")
        
        try:
            if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            else:
                raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
            
            print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
            print(f"📊 Columns: {list(self.df.columns)}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.file_path}")
            print("🔄 Creating sample data for demonstration...")
            self.create_sample_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔄 Creating sample data for demonstration...")
            self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create sample data if file is not found
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic sample data
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        categories = ['Furniture', 'Office Supplies', 'Technology']
        sub_categories = {
            'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
            'Office Supplies': ['Paper', 'Binders', 'Storage', 'Art', 'Labels'],
            'Technology': ['Phones', 'Computers', 'Accessories', 'Copiers']
        }
        regions = ['East', 'West', 'North', 'South']
        segments = ['Consumer', 'Corporate', 'Home Office']
        
        data = []
        for i in range(n_samples):
            category = np.random.choice(categories)
            sub_category = np.random.choice(sub_categories[category])
            sales = np.random.uniform(10, 2000)
            profit = sales * np.random.uniform(0.05, 0.3)  # 5-30% profit margin
            quantity = np.random.randint(1, 5)
            
            data.append({
                'Order_ID': f'ORD_{1000 + i}',
                'Order_Date': np.random.choice(dates),
                'Category': category,
                'Sub_Category': sub_category,
                'Sales': round(sales, 2),
                'Profit': round(profit, 2),
                'Quantity': quantity,
                'Region': np.random.choice(regions),
                'Customer_Segment': np.random.choice(segments)
            })
        
        self.df = pd.DataFrame(data)
        print("✅ Sample data created for demonstration")
        print(f"📊 Sample data shape: {self.df.shape}")
    
    def data_preparation(self):
        """
        Comprehensive data preparation and cleaning
        """
        print("\n" + "="*50)
        print("🔧 DATA PREPARATION")
        print("="*50)
        
        if self.df is None:
            self.load_data()
        
        # Display initial info
        print(f"📈 Original dataset shape: {self.df.shape}")
        print("\n📋 First 5 rows:")
        print(self.df.head())
        
        print("\n📊 Basic Statistics:")
        print(self.df[['Sales', 'Profit', 'Quantity']].describe())
        
        # Convert date columns
        self.convert_date_columns()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Handle outliers
        self.handle_outliers()
        
        # Create additional features
        self.create_features()
        
        self.preprocessed = True
        print("✅ Data preparation completed!")
        
        return self.df
    
    def convert_date_columns(self):
        """Convert date columns to datetime format"""
        print("\n📅 Converting date columns...")
        
        date_columns = self.df.select_dtypes(include=['object']).columns
        date_patterns = ['date', 'time', 'year', 'month', 'day']
        
        for col in date_columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in date_patterns):
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    print(f"   ✅ Converted {col} to datetime")
                except Exception as e:
                    print(f"   ⚠️ Could not convert {col}: {e}")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n🔍 Handling missing values...")
        
        print("Missing values before handling:")
        missing_before = self.df.isnull().sum()
        print(missing_before[missing_before > 0])
        
        # Fill numerical missing values
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                print(f"   ✅ Filled missing values in {col} with median")
        
        # Fill categorical missing values
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(
                    self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                )
                print(f"   ✅ Filled missing values in {col} with mode")
        
        print("Missing values after handling:")
        missing_after = self.df.isnull().sum()
        print(missing_after[missing_after > 0])
    
    def handle_outliers(self):
        """Handle outliers using IQR method"""
        print("\n📊 Handling outliers...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in ['Sales', 'Profit', 'Quantity']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                # Cap outliers
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                
                outliers_after = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                if outliers_before > 0:
                    print(f"   ✅ Handled {outliers_before} outliers in {col}")
    
    def create_features(self):
        """Create additional time-based features"""
        print("\n🎯 Creating additional features...")
        
        # Find date column
        date_columns = [col for col in self.df.columns if 'date' in col.lower()]
        
        if date_columns:
            date_col = date_columns[0]
            self.df['Year'] = self.df[date_col].dt.year
            self.df['Month'] = self.df[date_col].dt.month
            self.df['Quarter'] = self.df[date_col].dt.quarter
            self.df['DayOfWeek'] = self.df[date_col].dt.day_name()
            self.df['MonthYear'] = self.df[date_col].dt.to_period('M')
            
            print(f"   ✅ Created time-based features from {date_col}")
        
        # Create profit margin
        if 'Sales' in self.df.columns and 'Profit' in self.df.columns:
            self.df['Profit_Margin'] = (self.df['Profit'] / self.df['Sales'] * 100).round(2)
            print("   ✅ Created Profit_Margin feature")
    
    def sales_analysis(self):
        """
        Comprehensive sales analysis
        """
        print("\n" + "="*50)
        print("💰 SALES ANALYSIS")
        print("="*50)
        
        if not self.preprocessed:
            self.data_preparation()
        
        # 1. Overall Sales Metrics
        total_sales = self.df['Sales'].sum()
        avg_sales = self.df['Sales'].mean()
        total_transactions = len(self.df)
        
        print(f"📈 Total Sales: ${total_sales:,.2f}")
        print(f"📊 Average Sales per Transaction: ${avg_sales:,.2f}")
        print(f"🔢 Total Transactions: {total_transactions:,}")
        
        # 2. Temporal Sales Trends
        self.analyze_temporal_trends()
        
        # 3. Category Analysis
        self.analyze_categories()
        
        # 4. Regional Analysis
        self.analyze_regional_performance()
        
        print("✅ Sales analysis completed!")
    
    def analyze_temporal_trends(self):
        """Analyze sales trends over time"""
        print("\n📅 Analyzing temporal trends...")
        
        if 'Year' in self.df.columns and 'Month' in self.df.columns:
            # Monthly trends
            monthly_data = self.df.groupby(['Year', 'Month']).agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Quantity': 'sum'
            }).reset_index()
            
            monthly_data['Date'] = pd.to_datetime(
                monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str) + '-01'
            )
            
            # Plot monthly sales trends
            fig = px.line(monthly_data, x='Date', y='Sales', 
                         title='📈 Monthly Sales Trends',
                         template='plotly_white',
                         labels={'Sales': 'Total Sales ($)', 'Date': 'Month'})
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, "monthly_sales_trends.html"))
            print("   ✅ Saved monthly_sales_trends.html")
            
            # Seasonal analysis
            if 'Month' in self.df.columns:
                monthly_avg = self.df.groupby('Month').agg({
                    'Sales': 'mean',
                    'Profit': 'mean'
                }).reset_index()
                
                fig = px.line(monthly_avg, x='Month', y='Sales',
                             title='📊 Average Monthly Sales Pattern',
                             template='plotly_white')
                fig.update_layout(title_x=0.5)
                fig.write_html(os.path.join(self.viz_dir, "seasonal_sales_pattern.html"))
                print("   ✅ Saved seasonal_sales_pattern.html")
    
    def analyze_categories(self):
        """Analyze sales by categories"""
        print("\n📊 Analyzing category performance...")
        
        category_cols = [col for col in self.df.columns if 'categ' in col.lower()]
        
        for cat_col in category_cols:
            category_performance = self.df.groupby(cat_col).agg({
                'Sales': ['sum', 'mean', 'count'],
                'Profit': 'sum',
                'Profit_Margin': 'mean'
            }).round(2)
            
            # Flatten column names
            category_performance.columns = [
                'Total_Sales', 'Average_Sales', 'Transaction_Count', 
                'Total_Profit', 'Avg_Profit_Margin'
            ]
            
            category_performance = category_performance.sort_values('Total_Sales', ascending=False)
            
            print(f"\n🏆 Top performing {cat_col}:")
            print(category_performance.head())
            
            # Save to results
            self.analysis_results[f'sales_by_{cat_col}'] = category_performance
            
            # Visualization
            top_categories = category_performance.head(10)
            fig = px.bar(top_categories, x=top_categories.index, y='Total_Sales',
                        title=f'🏆 Top 10 {cat_col} by Sales',
                        template='plotly_white',
                        labels={'Total_Sales': 'Total Sales ($)', 'index': cat_col})
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, f"top_{cat_col}_sales.html"))
            print(f"   ✅ Saved top_{cat_col}_sales.html")
    
    def analyze_regional_performance(self):
        """Analyze regional sales performance"""
        print("\n🌍 Analyzing regional performance...")
        
        region_cols = [col for col in self.df.columns if 'region' in col.lower() or 'segment' in col.lower()]
        
        for region_col in region_cols:
            regional_performance = self.df.groupby(region_col).agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Profit_Margin': 'mean'
            }).round(2).sort_values('Sales', ascending=False)
            
            print(f"\n📍 Performance by {region_col}:")
            print(regional_performance)
            
            # Visualization
            fig = px.bar(regional_performance, x=regional_performance.index, y='Sales',
                        title=f'🌍 Sales by {region_col}',
                        color='Profit',
                        template='plotly_white')
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, f"sales_by_{region_col}.html"))
            print(f"   ✅ Saved sales_by_{region_col}.html")
    
    def profit_analysis(self):
        """
        Comprehensive profit analysis
        """
        print("\n" + "="*50)
        print("💵 PROFIT ANALYSIS")
        print("="*50)
        
        if not self.preprocessed:
            self.data_preparation()
        
        # 1. Overall Profit Metrics
        total_profit = self.df['Profit'].sum()
        avg_profit = self.df['Profit'].mean()
        overall_profit_margin = (total_profit / self.df['Sales'].sum() * 100).round(2)
        
        print(f"💰 Total Profit: ${total_profit:,.2f}")
        print(f"📊 Average Profit per Transaction: ${avg_profit:,.2f}")
        print(f"🎯 Overall Profit Margin: {overall_profit_margin}%")
        
        # 2. Profit Trends
        self.analyze_profit_trends()
        
        # 3. Most Profitable Items
        self.analyze_profitable_items()
        
        # 4. Profitability by Segments
        self.analyze_profitability_segments()
        
        print("✅ Profit analysis completed!")
    
    def analyze_profit_trends(self):
        """Analyze profit trends over time"""
        print("\n📈 Analyzing profit trends...")
        
        if 'Year' in self.df.columns and 'Month' in self.df.columns:
            monthly_profit = self.df.groupby(['Year', 'Month']).agg({
                'Profit': 'sum',
                'Sales': 'sum'
            }).reset_index()
            
            monthly_profit['Date'] = pd.to_datetime(
                monthly_profit['Year'].astype(str) + '-' + monthly_profit['Month'].astype(str) + '-01'
            )
            monthly_profit['Profit_Margin'] = (monthly_profit['Profit'] / monthly_profit['Sales'] * 100).round(2)
            
            # Plot profit trends
            fig = px.line(monthly_profit, x='Date', y='Profit',
                         title='💵 Monthly Profit Trends',
                         template='plotly_white')
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, "monthly_profit_trends.html"))
            print("   ✅ Saved monthly_profit_trends.html")
    
    def analyze_profitable_items(self):
        """Identify most profitable products/categories"""
        print("\n🏆 Analyzing most profitable items...")
        
        category_cols = [col for col in self.df.columns if 'categ' in col.lower()]
        
        for cat_col in category_cols:
            profitability = self.df.groupby(cat_col).agg({
                'Profit': 'sum',
                'Sales': 'sum',
                'Profit_Margin': 'mean',
                'Quantity': 'sum'
            }).round(2).sort_values('Profit', ascending=False)
            
            profitability['Contribution_Percentage'] = (profitability['Profit'] / profitability['Profit'].sum() * 100).round(2)
            
            print(f"\n💎 Most profitable {cat_col}:")
            print(profitability.head(10))
            
            # Save to results
            self.analysis_results[f'profit_by_{cat_col}'] = profitability
            
            # Visualization
            top_profitable = profitability.head(10)
            fig = px.bar(top_profitable, x=top_profitable.index, y='Profit',
                        title=f'💎 Top 10 {cat_col} by Profit',
                        color='Profit_Margin',
                        template='plotly_white')
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, f"top_{cat_col}_profit.html"))
            print(f"   ✅ Saved top_{cat_col}_profit.html")
    
    def analyze_profitability_segments(self):
        """Analyze profitability by customer segments"""
        print("\n👥 Analyzing profitability by segments...")
        
        segment_cols = [col for col in self.df.columns if 'segment' in col.lower()]
        
        for seg_col in segment_cols:
            segment_profitability = self.df.groupby(seg_col).agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Profit_Margin': 'mean',
                'Quantity': 'sum'
            }).round(2)
            
            segment_profitability = segment_profitability.sort_values('Profit_Margin', ascending=False)
            
            print(f"\n🎯 Profitability by {seg_col}:")
            print(segment_profitability)
            
            # Visualization
            fig = px.scatter(segment_profitability, x='Sales', y='Profit', 
                           size='Profit_Margin', color=segment_profitability.index,
                           title=f'📊 Sales vs Profit by {seg_col}',
                           template='plotly_white',
                           size_max=60)
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, f"profitability_{seg_col}.html"))
            print(f"   ✅ Saved profitability_{seg_col}.html")
    
    def operational_insights(self):
        """
        Generate operational insights
        """
        print("\n" + "="*50)
        print("⚙️ OPERATIONAL INSIGHTS")
        print("="*50)
        
        if not self.preprocessed:
            self.data_preparation()
        
        # 1. Sales-to-Profit Efficiency
        self.analyze_efficiency()
        
        # 2. Customer Segment Performance
        self.analyze_customer_segments()
        
        print("✅ Operational insights generated!")
    
    def analyze_efficiency(self):
        """Analyze sales-to-profit efficiency"""
        print("\n📊 Analyzing sales-to-profit efficiency...")
        
        category_cols = [col for col in self.df.columns if 'categ' in col.lower()]
        
        for cat_col in category_cols:
            efficiency = self.df.groupby(cat_col).agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Quantity': 'sum'
            }).round(2)
            
            efficiency['Sales_to_Profit_Ratio'] = (efficiency['Profit'] / efficiency['Sales'] * 100).round(2)
            efficiency['Efficiency_Score'] = (efficiency['Sales_to_Profit_Ratio'] / efficiency['Sales_to_Profit_Ratio'].max() * 100).round(2)
            
            efficiency = efficiency.sort_values('Sales_to_Profit_Ratio', ascending=False)
            
            print(f"\n🎯 Efficiency Analysis for {cat_col}:")
            print(efficiency[['Sales', 'Profit', 'Sales_to_Profit_Ratio', 'Efficiency_Score']].head())
            
            # Visualization
            fig = px.scatter(efficiency, x='Sales', y='Profit', 
                           size='Sales_to_Profit_Ratio', color='Sales_to_Profit_Ratio',
                           hover_name=efficiency.index,
                           title=f'⚡ Sales vs Profit Efficiency by {cat_col}',
                           template='plotly_white')
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, f"efficiency_{cat_col}.html"))
            print(f"   ✅ Saved efficiency_{cat_col}.html")
    
    def analyze_customer_segments(self):
        """Analyze customer segment performance"""
        print("\n👥 Analyzing customer segment performance...")
        
        segment_cols = [col for col in self.df.columns if 'segment' in col.lower() and 'customer' in col.lower()]
        
        for seg_col in segment_cols:
            segment_analysis = self.df.groupby(seg_col).agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Profit_Margin': 'mean',
                'Quantity': 'sum'
            }).round(2)
            
            segment_analysis['Customer_Value_Score'] = (
                (segment_analysis['Sales'] / segment_analysis['Sales'].sum() * 40) +
                (segment_analysis['Profit_Margin'] / segment_analysis['Profit_Margin'].max() * 60)
            ).round(2)
            
            segment_analysis = segment_analysis.sort_values('Customer_Value_Score', ascending=False)
            
            print(f"\n⭐ Customer Segment Analysis for {seg_col}:")
            print(segment_analysis)
            
            # Visualization
            fig = px.bar(segment_analysis, x=segment_analysis.index, y='Customer_Value_Score',
                        title=f'⭐ Customer Value Score by {seg_col}',
                        color='Profit_Margin',
                        template='plotly_white')
            fig.update_layout(title_x=0.5)
            fig.write_html(os.path.join(self.viz_dir, f"customer_value_{seg_col}.html"))
            print(f"   ✅ Saved customer_value_{seg_col}.html")
    
    def create_comprehensive_dashboard(self):
        """
        Create a comprehensive dashboard
        """
        print("\n" + "="*50)
        print("📊 CREATING COMPREHENSIVE DASHBOARD")
        print("="*50)
        
        # Create summary metrics
        summary_metrics = {
            'Metric': [
                'Total Sales', 'Total Profit', 'Overall Profit Margin', 
                'Total Transactions', 'Average Sale Value', 'Average Profit per Transaction'
            ],
            'Value': [
                f"${self.df['Sales'].sum():,.2f}",
                f"${self.df['Profit'].sum():,.2f}",
                f"{(self.df['Profit'].sum() / self.df['Sales'].sum() * 100):.2f}%",
                f"{len(self.df):,}",
                f"${self.df['Sales'].mean():.2f}",
                f"${self.df['Profit'].mean():.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_metrics)
        print("\n📈 KEY PERFORMANCE INDICATORS:")
        print(summary_df)
        
        # Create interactive dashboard
        self.create_interactive_dashboard()
        
        print("✅ Comprehensive dashboard created!")
    
    def create_interactive_dashboard(self):
        """Create an interactive dashboard with multiple charts"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Monthly Sales Trend', 'Monthly Profit Trend',
                'Top Categories by Sales', 'Top Categories by Profit',
                'Sales vs Profit Correlation', 'Profit Margin Distribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Monthly Sales Trend
        if 'Year' in self.df.columns and 'Month' in self.df.columns:
            monthly_data = self.df.groupby(['Year', 'Month']).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
            monthly_data['Date'] = pd.to_datetime(monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str) + '-01')
            
            fig.add_trace(
                go.Scatter(x=monthly_data['Date'], y=monthly_data['Sales'], 
                          name='Sales', line=dict(color='blue')),
                row=1, col=1
            )
        
        # 2. Monthly Profit Trend
        if 'Year' in self.df.columns and 'Month' in self.df.columns:
            fig.add_trace(
                go.Scatter(x=monthly_data['Date'], y=monthly_data['Profit'], 
                          name='Profit', line=dict(color='green')),
                row=1, col=2
            )
        
        # 3. Top Categories by Sales
        category_cols = [col for col in self.df.columns if 'categ' in col.lower()]
        if category_cols:
            cat_col = category_cols[0]
            top_categories = self.df.groupby(cat_col)['Sales'].sum().nlargest(5)
            
            fig.add_trace(
                go.Bar(x=top_categories.index, y=top_categories.values, 
                      name='Top Sales', marker_color='lightblue'),
                row=2, col=1
            )
        
        # 4. Top Categories by Profit
        if category_cols:
            top_profit = self.df.groupby(cat_col)['Profit'].sum().nlargest(5)
            
            fig.add_trace(
                go.Bar(x=top_profit.index, y=top_profit.values, 
                      name='Top Profit', marker_color='lightgreen'),
                row=2, col=2
            )
        
        # 5. Sales vs Profit Correlation
        fig.add_trace(
            go.Scatter(x=self.df['Sales'], y=self.df['Profit'], 
                      mode='markers', name='Sales vs Profit',
                      marker=dict(size=5, opacity=0.6)),
            row=3, col=1
        )
        
        # 6. Profit Margin Distribution
        if 'Profit_Margin' in self.df.columns:
            fig.add_trace(
                go.Histogram(x=self.df['Profit_Margin'], name='Profit Margin',
                           nbinsx=30, marker_color='orange'),
                row=3, col=2
            )
        
        fig.update_layout(
            height=1200,
            title_text="📊 Comprehensive Store Performance Dashboard",
            showlegend=True,
            title_x=0.5
        )
        
        fig.write_html(os.path.join(self.viz_dir, "comprehensive_dashboard.html"))
        print("   ✅ Saved comprehensive_dashboard.html")
    
    def save_analysis_results(self):
        """
        Save all analysis results to files
        """
        print("\n" + "="*50)
        print("💾 SAVING ANALYSIS RESULTS")
        print("="*50)
        
        # 1. Save cleaned dataset
        cleaned_file = os.path.join(self.data_dir, "cleaned_dataset.csv")
        self.df.to_csv(cleaned_file, index=False)
        print(f"✅ Saved cleaned dataset: {cleaned_file}")
        
        # 2. Save analysis results to Excel
        excel_file = os.path.join(self.results_dir, "detailed_analysis_report.xlsx")
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Save cleaned data
            self.df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
            
            # Save various analyses
            category_cols = [col for col in self.df.columns if 'categ' in col.lower()]
            
            for cat_col in category_cols:
                # Sales analysis
                sales_analysis = self.df.groupby(cat_col).agg({
                    'Sales': ['sum', 'mean', 'count', 'std'],
                    'Profit': ['sum', 'mean'],
                    'Profit_Margin': 'mean',
                    'Quantity': 'sum'
                }).round(2)
                
                sales_analysis.columns = [
                    'Total_Sales', 'Avg_Sales', 'Transaction_Count', 'Sales_Std',
                    'Total_Profit', 'Avg_Profit', 'Avg_Profit_Margin', 'Total_Quantity'
                ]
                
                sales_analysis.to_excel(writer, sheet_name=f'Sales_by_{cat_col}')
                
                # Profit analysis
                profit_analysis = self.df.groupby(cat_col).agg({
                    'Profit': ['sum', 'mean', 'std'],
                    'Sales': 'sum',
                    'Profit_Margin': ['mean', 'std'],
                    'Quantity': 'sum'
                }).round(2)
                
                profit_analysis.columns = [
                    'Total_Profit', 'Avg_Profit', 'Profit_Std',
                    'Total_Sales', 'Avg_Profit_Margin', 'Profit_Margin_Std', 'Total_Quantity'
                ]
                
                profit_analysis.to_excel(writer, sheet_name=f'Profit_by_{cat_col}')
            
            # Temporal analysis
            if 'Year' in self.df.columns and 'Month' in self.df.columns:
                temporal_analysis = self.df.groupby(['Year', 'Month']).agg({
                    'Sales': 'sum',
                    'Profit': 'sum',
                    'Quantity': 'sum',
                    'Profit_Margin': 'mean'
                }).reset_index()
                
                temporal_analysis.to_excel(writer, sheet_name='Monthly_Performance', index=False)
            
            # Summary statistics
            summary_stats = self.df.describe()
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Correlation matrix
            numerical_df = self.df.select_dtypes(include=[np.number])
            correlation_matrix = numerical_df.corr().round(3)
            correlation_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        
        print(f"✅ Saved detailed analysis report: {excel_file}")
        
        # 3. Save key metrics to CSV
        key_metrics = {
            'Total_Sales': [self.df['Sales'].sum()],
            'Total_Profit': [self.df['Profit'].sum()],
            'Overall_Profit_Margin': [(self.df['Profit'].sum() / self.df['Sales'].sum() * 100)],
            'Total_Transactions': [len(self.df)],
            'Average_Sale_Value': [self.df['Sales'].mean()],
            'Average_Profit_per_Transaction': [self.df['Profit'].mean()],
            'Date_Range_Start': [self.df['Order_Date'].min() if 'Order_Date' in self.df.columns else 'N/A'],
            'Date_Range_End': [self.df['Order_Date'].max() if 'Order_Date' in self.df.columns else 'N/A']
        }
        
        metrics_df = pd.DataFrame(key_metrics)
        metrics_file = os.path.join(self.data_dir, "key_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✅ Saved key metrics: {metrics_file}")
        
        # 4. Create a summary report
        self.create_summary_report()
        
        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 All results saved in: {self.results_dir}")
        print(f"📊 Visualizations: {self.viz_dir}")
        print(f"📈 Data files: {self.data_dir}")
        print(f"📋 Main report: {excel_file}")
    
    def create_summary_report(self):
        """Create a text summary report"""
        report_file = os.path.join(self.results_dir, "analysis_summary.txt")
        
        with open(report_file, 'w') as f:
            f.write("STORE SALES AND PROFIT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Shape: {self.df.shape}\n\n")
            
            f.write("KEY METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Sales: ${self.df['Sales'].sum():,.2f}\n")
            f.write(f"Total Profit: ${self.df['Profit'].sum():,.2f}\n")
            f.write(f"Overall Profit Margin: {(self.df['Profit'].sum() / self.df['Sales'].sum() * 100):.2f}%\n")
            f.write(f"Total Transactions: {len(self.df):,}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("📊 detailed_analysis_report.xlsx - Detailed analysis in Excel format\n")
            f.write("📈 comprehensive_dashboard.html - Interactive dashboard\n")
            f.write("📋 Various HTML files - Individual visualizations\n")
            f.write("📊 key_metrics.csv - Key performance indicators\n")
            f.write("💾 cleaned_dataset.csv - Cleaned dataset for further analysis\n")
        
        print(f"✅ Saved summary report: {report_file}")

def main():
    """
    Main function to run the complete analysis
    """
    print("🏪 STORE SALES AND PROFIT ANALYSIS")
    print("=" * 50)
    
    # File path - UPDATE THIS WITH YOUR ACTUAL FILE PATH
    file_path = "Superstore.xlsx"  # Change this to your file path
    
    # Initialize analyzer
    analyzer = StoreSalesAnalyzer(file_path)
    
    try:
        # Run complete analysis pipeline
        analyzer.data_preparation()
        analyzer.sales_analysis()
        analyzer.profit_analysis()
        analyzer.operational_insights()
        analyzer.create_comprehensive_dashboard()
        analyzer.save_analysis_results()
        
        print("\n" + "="*50)
        print("🎯 ANALYSIS COMPLETED!")
        print("="*50)
        print("Next steps:")
        print("1. Open the HTML files in your browser to view interactive visualizations")
        print("2. Check the Excel file for detailed analysis results")
        print("3. Review the summary report for key insights")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()