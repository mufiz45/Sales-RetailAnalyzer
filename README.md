# Sales-RetailAnalyzer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class RetailAnalyzer:
    """
    A comprehensive retail sales data analyzer that processes, analyzes, 
    and visualizes sales data using OOP principles.
    """
    
    def _init_(self):
        """Initialize the RetailAnalyzer with empty dataset."""
        self.data = None
        self.file_path = None
        
    def load_data(self, file_path):
        """
        Load retail sales data from a CSV file.
        
        Parameters:
            file_path (str): Path to the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                print(f"Error: File '{file_path}' not found.")
                return False
            
            # Validate file format
            if not file_path.endswith('.csv'):
                print("Error: File must be in CSV format.")
                return False
            
            # Load data
            self.data = pd.read_csv(file_path)
            self.file_path = file_path
            
            # Validate required columns
            required_columns = ['Date', 'Product', 'Category', 'Price', 'Quantity Sold', 'Total Sales']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return False
            
            # Convert Date column to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Check for missing values
            if self.data.isnull().any().any():
                print("\nWarning: Dataset contains missing values.")
                print(self.data.isnull().sum())
                
                # Handle missing values
                self._handle_missing_values()
            
            print(f"\nData loaded successfully from '{file_path}'")
            print(f"Total records: {len(self.data)}")
            print(f"\nDataset Preview:")
            print(self.data.head())
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        # Fill missing numerical values with median
        numerical_cols = ['Price', 'Quantity Sold', 'Total Sales']
        for col in numerical_cols:
            if col in self.data.columns and self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = ['Product', 'Category']
        for col in categorical_cols:
            if col in self.data.columns and self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        print("Missing values handled successfully.")
    
    def calculate_metrics(self):
        """
        Calculate key sales metrics including total sales, average sales,
        and most popular product.
        
        Returns:
            dict: Dictionary containing calculated metrics
        """
        if self.data is None:
            print("Error: No data loaded. Please load data first.")
            return None
        
        try:
            # Calculate metrics using NumPy
            total_sales = np.sum(self.data['Total Sales'])
            average_sales = np.mean(self.data['Total Sales'])
            total_quantity = np.sum(self.data['Quantity Sold'])
            average_quantity = np.mean(self.data['Quantity Sold'])
            
            # Find most popular product (by quantity sold)
            product_sales = self.data.groupby('Product')['Quantity Sold'].sum()
            most_popular_product = product_sales.idxmax()
            most_popular_quantity = product_sales.max()
            
            # Calculate growth rate (if multiple dates exist)
            if len(self.data['Date'].unique()) > 1:
                monthly_sales = self.data.groupby(self.data['Date'].dt.to_period('M'))['Total Sales'].sum()
                if len(monthly_sales) > 1:
                    growth_rate = ((monthly_sales.iloc[-1] - monthly_sales.iloc[0]) / monthly_sales.iloc[0]) * 100
                else:
                    growth_rate = 0
            else:
                growth_rate = 0
            
            metrics = {
                'total_sales': total_sales,
                'average_sales': average_sales,
                'total_quantity': total_quantity,
                'average_quantity': average_quantity,
                'most_popular_product': most_popular_product,
                'most_popular_quantity': most_popular_quantity,
                'growth_rate': growth_rate,
                'total_transactions': len(self.data)
            }
            
            print("\n" + "="*50)
            print("SALES METRICS SUMMARY")
            print("="*50)
            print(f"Total Sales Revenue: ${metrics['total_sales']:,.2f}")
            print(f"Average Sales per Transaction: ${metrics['average_sales']:,.2f}")
            print(f"Total Quantity Sold: {metrics['total_quantity']:,.0f} units")
            print(f"Average Quantity per Transaction: {metrics['average_quantity']:.2f} units")
            print(f"Most Popular Product: {metrics['most_popular_product']}")
            print(f"  - Total Quantity Sold: {metrics['most_popular_quantity']:,.0f} units")
            print(f"Sales Growth Rate: {metrics['growth_rate']:.2f}%")
            print(f"Total Transactions: {metrics['total_transactions']}")
            print("="*50 + "\n")
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None
    
    def filter_data(self, condition):
        """
        Filter sales data based on user-defined conditions.
        
        Parameters:
            condition (str): Filter condition ('category', 'date_range', or 'product')
            
        Returns:
            DataFrame: Filtered data
        """
        if self.data is None:
            print("Error: No data loaded. Please load data first.")
            return None
        
        try:
            if condition == 'category':
                print("\nAvailable categories:")
                categories = self.data['Category'].unique()
                for i, cat in enumerate(categories, 1):
                    print(f"{i}. {cat}")
                
                choice = input("\nEnter category name to filter: ").strip()
                filtered_data = self.data[self.data['Category'].str.lower() == choice.lower()]
                
            elif condition == 'date_range':
                print("\nEnter date range (YYYY-MM-DD format)")
                start_date = input("Start date: ").strip()
                end_date = input("End date: ").strip()
                
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                filtered_data = self.data[(self.data['Date'] >= start_date) & 
                                         (self.data['Date'] <= end_date)]
                
            elif condition == 'product':
                print("\nAvailable products:")
                products = self.data['Product'].unique()
                for i, prod in enumerate(products, 1):
                    print(f"{i}. {prod}")
                
                choice = input("\nEnter product name to filter: ").strip()
                filtered_data = self.data[self.data['Product'].str.lower() == choice.lower()]
            
            else:
                print("Invalid filter condition. Use 'category', 'date_range', or 'product'.")
                return None
            
            print(f"\nFiltered data contains {len(filtered_data)} records.")
            print(filtered_data.head())
            
            return filtered_data
            
        except Exception as e:
            print(f"Error filtering data: {str(e)}")
            return None
    
    def visualize_data(self):
        """
        Create comprehensive visualizations including bar charts, 
        line graphs, and heatmaps.
        """
        if self.data is None:
            print("Error: No data loaded. Please load data first.")
            return
        
        try:
            # Set style for better aesthetics
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (15, 10)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Retail Sales Data Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Bar Chart: Total Sales by Product Category
            category_sales = self.data.groupby('Category')['Total Sales'].sum().sort_values(ascending=False)
            axes[0, 0].bar(category_sales.index, category_sales.values, color='steelblue', edgecolor='black')
            axes[0, 0].set_title('Total Sales by Product Category', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Category')
            axes[0, 0].set_ylabel('Total Sales ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(category_sales.values):
                axes[0, 0].text(i, v, f'${v:,.0f}', ha='center', va='bottom')
            
            # 2. Line Graph: Sales Trend Over Time
            daily_sales = self.data.groupby('Date')['Total Sales'].sum().sort_index()
            axes[0, 1].plot(daily_sales.index, daily_sales.values, marker='o', 
                           linewidth=2, markersize=4, color='green')
            axes[0, 1].set_title('Sales Trend Over Time', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Total Sales ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Heatmap: Correlation between Price and Quantity Sold
            # Create pivot table for heatmap
            pivot_data = self.data.pivot_table(
                values='Total Sales', 
                index='Category', 
                columns=self.data['Date'].dt.month_name(),
                aggfunc='sum',
                fill_value=0
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                       ax=axes[1, 0], cbar_kws={'label': 'Total Sales ($)'})
            axes[1, 0].set_title('Sales Heatmap: Category vs Month', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Category')
            
            # 4. Top 10 Products by Quantity Sold
            top_products = self.data.groupby('Product')['Quantity Sold'].sum().nlargest(10).sort_values()
            axes[1, 1].barh(top_products.index, top_products.values, color='coral', edgecolor='black')
            axes[1, 1].set_title('Top 10 Products by Quantity Sold', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Quantity Sold')
            axes[1, 1].set_ylabel('Product')
            
            # Add value labels
            for i, v in enumerate(top_products.values):
                axes[1, 1].text(v, i, f' {v:,.0f}', va='center')
            
            plt.tight_layout()
            plt.show()
            
            print("\nVisualizations generated successfully!")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")


def main():
    """Main function to run the Retail Sales Data Analyzer."""
    print("="*60)
    print("RETAIL SALES DATA ANALYZER".center(60))
    print("="*60)
    print("\nWelcome to the Retail Sales Data Analyzer!")
    print("This tool helps you analyze and visualize retail sales data.\n")
    
    # Create analyzer instance
    analyzer = RetailAnalyzer()
    
    # Get file path from user
    while True:
        file_path = input("Enter the path to your CSV file (or 'quit' to exit): ").strip()
        
        if file_path.lower() == 'quit':
            print("Thank you for using Retail Sales Data Analyzer!")
            return
        
        # Try to load the data
        if analyzer.load_data(file_path):
            break
        else:
            print("\nPlease try again with a valid file path.\n")
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("MAIN MENU".center(60))
        print("="*60)
        print("1. Calculate Sales Metrics")
        print("2. Filter Data")
        print("3. Generate Visualizations")
        print("4. Load New Dataset")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            analyzer.calculate_metrics()
            
        elif choice == '2':
            print("\nFilter Options:")
            print("1. Filter by Category")
            print("2. Filter by Date Range")
            print("3. Filter by Product")
            
            filter_choice = input("\nEnter filter option (1-3): ").strip()
            
            if filter_choice == '1':
                analyzer.filter_data('category')
            elif filter_choice == '2':
                analyzer.filter_data('date_range')
            elif filter_choice == '3':
                analyzer.filter_data('product')
            else:
                print("Invalid choice. Please try again.")
                
        elif choice == '3':
            analyzer.visualize_data()
            
        elif choice == '4':
            file_path = input("\nEnter the path to your CSV file: ").strip()
            analyzer.load_data(file_path)
            
        elif choice == '5':
            print("\nThank you for using Retail Sales Data Analyzer!")
            print("Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")


if _name_ == "_main_":
    main()
