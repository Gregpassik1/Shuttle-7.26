import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta
import math
import io

# Set page config
st.set_page_config(
    page_title="Shuttle Tech Platform",
    page_icon="🚐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def generate_sample_data():
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)
    
    # Generate data for 30 days
    data = []
    pickup_locations = ["Hotel Downtown A", "Hotel Downtown B", "Hotel Downtown C"]
    
    # Create time blocks (30-minute intervals)
    time_blocks = []
    for hour in range(24):
        for minute in [0, 30]:
            time_blocks.append(f"{hour:02d}:{minute:02d}")
    
    base_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        current_date = base_date + timedelta(days=day)
        day_of_week = current_date.strftime('%A')
        
        # Different patterns based on day of week
        if day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            # Weekday pattern - higher morning and evening peaks
            daily_multiplier = 1.2
            peak_hours = [6, 7, 8, 17, 18, 19]
        elif day_of_week == 'Saturday':
            daily_multiplier = 0.9
            peak_hours = [9, 10, 11, 15, 16, 17]
        else:  # Sunday
            daily_multiplier = 0.7
            peak_hours = [10, 11, 16, 17, 18]
        
        daily_total = 0
        for time_block in time_blocks:
            hour = int(time_block.split(':')[0])
            
            # Base passenger count with peaks
            if hour in peak_hours:
                base_passengers = np.random.poisson(8) * daily_multiplier
            elif hour in range(5, 22):  # Day hours
                base_passengers = np.random.poisson(3) * daily_multiplier
            else:  # Night hours
                base_passengers = np.random.poisson(1) * daily_multiplier
            
            # Distribute among pickup locations
            for location in pickup_locations:
                # Different locations have different popularity
                location_multiplier = {
                    "Hotel Downtown A": 1.3,
                    "Hotel Downtown B": 1.0,
                    "Hotel Downtown C": 0.8
                }[location]
                
                passenger_count = max(0, int(base_passengers * location_multiplier))
                daily_total += passenger_count
                
                if passenger_count > 0:  # Only add records with passengers
                    data.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'time_block': time_block,
                        'pickup_location': location,
                        'passenger_count': passenger_count
                    })
        
        # Ensure daily total is within range 120-220
        if daily_total < 120:
            # Add random passengers to reach minimum
            additional_needed = 120 - daily_total
            for _ in range(additional_needed):
                random_time = np.random.choice(time_blocks)
                random_location = np.random.choice(pickup_locations)
                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'time_block': random_time,
                    'pickup_location': random_location,
                    'passenger_count': 1
                })
    
    return pd.DataFrame(data)

def process_data(df):
    """Process raw data into analysis-ready format"""
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.strftime('%Y-%m')
    
    # Ensure proper time block format
    df['time_block'] = df['time_block'].astype(str)
    
    return df

def calculate_shuttle_optimization(df, buffer_factor=1.2, baseline_shuttles=4, shuttle_capacity=12):
    """Calculate optimized shuttle counts with operational constraints"""
    # Group by time block and day of week, taking the AVERAGE across all dates
    grouped = df.groupby(['time_block', 'day_of_week'])['passenger_count'].mean().reset_index()
    
    # For planning purposes, use 75th percentile for peak demand planning
    grouped_75th = df.groupby(['time_block', 'day_of_week'])['passenger_count'].quantile(0.75).reset_index()
    grouped_75th.columns = ['time_block', 'day_of_week', 'passenger_count_75th']
    
    # Merge the datasets
    grouped = grouped.merge(grouped_75th, on=['time_block', 'day_of_week'])
    
    # Calculate demand-driven shuttles needed
    grouped['demand_driven_shuttles'] = np.ceil(grouped['passenger_count_75th'] * buffer_factor / shuttle_capacity)
    
    # Operational shuttle calculation:
    # - Minimum baseline shuttles for coverage (one can reach any location within 15 min)
    # - Add extra shuttles only when demand exceeds baseline capacity
    # - Consider that shuttles take ~45 minutes for full loop (pickup + travel + airport + return)
    
    baseline_capacity_per_30min = baseline_shuttles * shuttle_capacity * 0.67  # Account for travel time efficiency
    
    grouped['additional_shuttles_needed'] = np.maximum(
        0, 
        np.ceil((grouped['passenger_count_75th'] * buffer_factor - baseline_capacity_per_30min) / shuttle_capacity)
    )
    
    # Total shuttles = baseline + additional when needed
    grouped['required_shuttles'] = baseline_shuttles + grouped['additional_shuttles_needed']
    grouped['required_shuttles'] = grouped['required_shuttles'].astype(int)
    
    # Cap at reasonable maximum (e.g., 8 shuttles max)
    grouped['required_shuttles'] = np.minimum(grouped['required_shuttles'], 8)
    
    return grouped

def create_time_day_heatmap(df):
    """Create time block x day of week heatmap"""
    pivot_data = df.groupby(['time_block', 'day_of_week'])['passenger_count'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='time_block', columns='day_of_week', values='passenger_count').fillna(0)
    
    # Reorder columns to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(columns=day_order, fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Blues',
        showscale=True,
        hovertemplate='Day: %{x}<br>Time: %{y}<br>Avg Passengers: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Average Passenger Volume: Time Block × Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Time Block",
        height=600,
        font=dict(size=12),
        showlegend=False
    )
    
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    
    return fig

def create_monthly_trends(df):
    """Create monthly trend analysis"""
    # Monthly passenger totals
    monthly_passengers = df.groupby(['month', 'day_of_week'])['passenger_count'].sum().reset_index()
    monthly_avg = df.groupby('month')['passenger_count'].sum().reset_index()
    monthly_avg['daily_average'] = monthly_avg['passenger_count'] / df.groupby('month')['date'].nunique().values
    
    # Monthly trends by day of week
    fig_monthly_trends = px.line(
        monthly_passengers,
        x='month',
        y='passenger_count',
        color='day_of_week',
        title="Monthly Passenger Trends by Day of Week",
        labels={'passenger_count': 'Total Passengers', 'month': 'Month'}
    )
    
    # Monthly daily averages
    fig_monthly_avg = px.bar(
        monthly_avg,
        x='month',
        y='daily_average',
        title="Monthly Daily Average Passengers",
        labels={'daily_average': 'Daily Average Passengers', 'month': 'Month'}
    )
    
    return fig_monthly_trends, fig_monthly_avg, monthly_avg

def create_location_analysis(df):
    """Create time block x pickup location analysis"""
    pivot_data = df.groupby(['time_block', 'pickup_location', 'day_of_week'])['passenger_count'].mean().reset_index()
    
    fig = px.bar(
        pivot_data,
        x='time_block',
        y='passenger_count',
        color='pickup_location',
        facet_col='day_of_week',
        facet_col_wrap=4,
        title="Average Passenger Volume: Time Block × Pickup Location by Day",
        labels={'passenger_count': 'Average Passengers', 'time_block': 'Time Block'}
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig

def calculate_cost_analysis(shuttle_counts, hourly_cost=84):
    """Calculate cost analysis based on shuttle optimization"""
    # Each time block is 30 minutes = 0.5 hours
    shuttle_counts['hourly_cost'] = shuttle_counts['required_shuttles'] * (hourly_cost * 0.5)
    
    # Calculate daily and monthly costs
    daily_costs = shuttle_counts.groupby('day_of_week')['hourly_cost'].sum()
    
    # Assume 4.33 weeks per month on average
    monthly_costs = daily_costs * 4.33
    
    return shuttle_counts, daily_costs, monthly_costs

# Main App Header
st.markdown("""
<div class="main-header">
    <h1>Shuttle Tech Platform</h1>
    <p style="color: white; text-align: center; margin: 0;">
        Optimizing crew transportation between downtown hotels and ANC Airport
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Data Input Section
st.sidebar.subheader("Data Management")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload shuttle data (CSV/Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload file with columns: date, time_block, pickup_location, passenger_count"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.data = process_data(df)
        st.sidebar.success("Data uploaded successfully!")
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

# Generate sample data button
if st.sidebar.button("Generate Sample Data"):
    st.session_state.data = process_data(generate_sample_data())
    st.sidebar.success("Sample data generated!")

# Cost configuration
st.sidebar.subheader("Cost Parameters")
hourly_cost = st.sidebar.slider(
    "Hourly shuttle cost ($)",
    min_value=73,
    max_value=95,
    value=84,
    step=1,
    help="Cost per shuttle per hour"
)

buffer_factor = st.sidebar.slider(
    "Buffer factor",
    min_value=1.0,
    max_value=2.0,
    value=1.2,
    step=0.1,
    help="Safety buffer for shuttle capacity planning"
)

# Operational parameters
st.sidebar.subheader("Operational Parameters")
baseline_shuttles = st.sidebar.number_input(
    "Baseline shuttle count (24/7 coverage)",
    min_value=1,
    max_value=6,
    value=4,
    help="Minimum shuttles needed for continuous coverage across all locations"
)

shuttle_capacity = st.sidebar.number_input(
    "Shuttle capacity (passengers)",
    min_value=8,
    max_value=15,
    value=12,
    help="Maximum passengers per shuttle"
)

# Main content
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Calculate optimization
    shuttle_optimization = calculate_shuttle_optimization(df, buffer_factor, baseline_shuttles, shuttle_capacity)
    cost_analysis, daily_costs, monthly_costs = calculate_cost_analysis(shuttle_optimization, hourly_cost)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_passengers = df['passenger_count'].sum()
        daily_avg = total_passengers / df['date'].nunique()
        st.metric(
            label="Daily Average Passengers",
            value=f"{daily_avg:.0f}",
            delta=f"Total: {total_passengers:,}"
        )
    
    with col2:
        total_shuttles = shuttle_optimization['required_shuttles'].sum()
        st.metric(
            label="Total Daily Shuttles",
            value=f"{total_shuttles}",
            delta="Across all time blocks"
        )
    
    with col3:
        monthly_cost_total = monthly_costs.sum()
        st.metric(
            label="Monthly Cost Estimate",
            value=f"${monthly_cost_total:,.0f}",
            delta=f"@ ${hourly_cost}/hr"
        )
    
    with col4:
        peak_time = df.groupby('time_block')['passenger_count'].sum().idxmax()
        peak_passengers = df.groupby('time_block')['passenger_count'].sum().max()
        st.metric(
            label="Peak Time Block",
            value=peak_time,
            delta=f"{peak_passengers} passengers"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analytics Dashboard", "Monthly Trends", "Route Optimization", "Cost Analysis", "Data Summary"])
    
    with tab1:
        st.header("Analytics Dashboard")
        
        # Time x Day heatmap
        fig_heatmap = create_time_day_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # Location analysis
        fig_location = create_location_analysis(df)
        st.plotly_chart(fig_location, use_container_width=True)
        
        # Summary table
        st.subheader("Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average by time block
            time_summary = df.groupby('time_block')['passenger_count'].agg(['mean', 'sum']).round(1)
            time_summary.columns = ['Average Passengers', 'Total Passengers']
            st.write("**Average Passengers by Time Block**")
            st.dataframe(time_summary, height=300)
        
        with col2:
            # Average by pickup location
            location_summary = df.groupby('pickup_location')['passenger_count'].agg(['mean', 'sum']).round(1)
            location_summary.columns = ['Average Passengers', 'Total Passengers']
            st.write("**Average Passengers by Pickup Location**")
            st.dataframe(location_summary)
    
    with tab2:
        st.header("Monthly Trends")
        
        # Create monthly trends
        fig_monthly_trends, fig_monthly_avg, monthly_avg = create_monthly_trends(df)
        
        # Monthly summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(monthly_avg) > 1:
                trend_change = ((monthly_avg['daily_average'].iloc[-1] - monthly_avg['daily_average'].iloc[0]) / monthly_avg['daily_average'].iloc[0]) * 100
                st.metric(
                    "Trend Direction",
                    f"{trend_change:+.1f}%",
                    "vs first month"
                )
            else:
                st.metric("Trend Direction", "N/A", "Need multiple months")
        
        with col2:
            peak_month = monthly_avg.loc[monthly_avg['daily_average'].idxmax(), 'month']
            peak_avg = monthly_avg['daily_average'].max()
            st.metric(
                "Peak Month",
                peak_month,
                f"{peak_avg:.0f} daily avg"
            )
        
        with col3:
            low_month = monthly_avg.loc[monthly_avg['daily_average'].idxmin(), 'month']
            low_avg = monthly_avg['daily_average'].min()
            st.metric(
                "Low Month",
                low_month,
                f"{low_avg:.0f} daily avg"
            )
        
        # Monthly trends visualization
        st.plotly_chart(fig_monthly_trends, use_container_width=True)
        
        st.plotly_chart(fig_monthly_avg, use_container_width=True)
        
        # Monthly data table
        st.subheader("Monthly Summary Table")
        
        # Enhanced monthly summary with shuttle implications
        monthly_summary = df.groupby('month').agg({
            'passenger_count': ['sum', 'mean'],
            'date': 'nunique'
        }).round(1)
        
        monthly_summary.columns = ['Total Passengers', 'Avg per Record', 'Days in Dataset']
        monthly_summary['Daily Average'] = (monthly_summary['Total Passengers'] / monthly_summary['Days in Dataset']).round(1)
        
        # Calculate estimated monthly shuttle needs
        monthly_summary['Est. Monthly Shuttles'] = monthly_summary['Daily Average'] * 30 * 24 / shuttle_capacity
        monthly_summary['Est. Monthly Shuttles'] = monthly_summary['Est. Monthly Shuttles'].round(0).astype(int)
        
        st.dataframe(monthly_summary, use_container_width=True)

    with tab3:
        st.header("Route Optimization")
        
        # Show planning methodology
        st.info(f"Operational Planning: Starting with {baseline_shuttles} shuttles for 24/7 coverage (≤15min response time). Additional shuttles added only during peak demand periods.")
        
        # Shuttle optimization table
        shuttle_pivot = shuttle_optimization.pivot(index='time_block', columns='day_of_week', values='required_shuttles').fillna(0)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        shuttle_pivot = shuttle_pivot.reindex(columns=day_order, fill_value=0).astype(int)
        
        st.subheader("Optimized Shuttle Count by Time Block and Day")
        st.dataframe(shuttle_pivot, use_container_width=True)
        
        # Show operational analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Baseline vs peak analysis
            baseline_hours = (shuttle_pivot == baseline_shuttles).sum().sum()
            total_hours = shuttle_pivot.shape[0] * shuttle_pivot.shape[1]
            baseline_percentage = (baseline_hours / total_hours) * 100
            
            st.metric(
                "Baseline Operation", 
                f"{baseline_percentage:.0f}%", 
                f"{baseline_hours}/{total_hours} time blocks"
            )
            
        with col2:
            # Peak periods
            peak_blocks = (shuttle_pivot > baseline_shuttles).sum().sum()
            peak_percentage = (peak_blocks / total_hours) * 100
            
            st.metric(
                "Peak Periods", 
                f"{peak_percentage:.0f}%", 
                f"+{peak_blocks} time blocks need extra shuttles"
            )
            
        with col3:
            # Maximum shuttles needed
            max_shuttles = shuttle_pivot.max().max()
            max_increase = max_shuttles - baseline_shuttles
            
            st.metric(
                "Peak Shuttle Count", 
                f"{max_shuttles} shuttles", 
                f"+{max_increase} above baseline"
            )
        
        # Demand vs capacity analysis
        st.subheader("Demand vs Capacity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average demand table
            avg_demand = df.groupby(['time_block', 'day_of_week'])['passenger_count'].mean().reset_index()
            avg_pivot = avg_demand.pivot(index='time_block', columns='day_of_week', values='passenger_count').fillna(0)
            avg_pivot = avg_pivot.reindex(columns=day_order, fill_value=0).round(1)
            
            st.write("**Average Passenger Demand**")
            st.dataframe(avg_pivot, height=300)
            
        with col2:
            # Capacity utilization
            capacity_pivot = shuttle_pivot * shuttle_capacity * 0.67  # Account for operational efficiency
            utilization_pivot = (avg_pivot / capacity_pivot * 100).fillna(0).round(1)
            
            st.write("**Capacity Utilization (%)**")
            st.dataframe(utilization_pivot, height=300)
        
        # Visualization
        fig_shuttles = go.Figure(data=go.Heatmap(
            z=shuttle_pivot.values,
            x=shuttle_pivot.columns,
            y=shuttle_pivot.index,
            colorscale='RdYlGn_r',
            showscale=True,
            hovertemplate='Day: %{x}<br>Time: %{y}<br>Shuttles Needed: %{z}<extra></extra>',
            zmin=baseline_shuttles,
            zmax=shuttle_pivot.max().max()
        ))
        
        fig_shuttles.update_layout(
            title="Required Shuttles: Time Block × Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Time Block",
            height=600,
            showlegend=False
        )
        
        fig_shuttles.update_xaxes(fixedrange=True)
        fig_shuttles.update_yaxes(fixedrange=True)
        
        st.plotly_chart(fig_shuttles, use_container_width=True)
        
        # Optimization parameters
        total_daily_shuttles = shuttle_pivot.sum(axis=0)
        avg_daily_shuttles = total_daily_shuttles.mean()
        
        st.subheader("Operational Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Daily Shuttle-Hours", f"{avg_daily_shuttles:.0f}")
        with col2:
            peak_day = total_daily_shuttles.idxmax()
            peak_shuttles = total_daily_shuttles.max()
            st.metric("Peak Day", f"{peak_day}")
            st.write(f"{peak_shuttles} shuttle-hours")
        with col3:
            light_day = total_daily_shuttles.idxmin()
            light_shuttles = total_daily_shuttles.min()
            st.metric("Light Day", f"{light_day}")
            st.write(f"{light_shuttles} shuttle-hours")
        with col4:
            extra_shuttle_cost = ((total_daily_shuttles - (baseline_shuttles * 48)).sum() / 7) * hourly_cost * 0.5
            st.metric("Daily Extra Cost", f"${extra_shuttle_cost:.0f}")
            st.write("Above baseline")
        
        st.info(f"**Operational Constraints:** {baseline_shuttles} shuttles baseline | {shuttle_capacity} passenger capacity | 20-25min travel between locations | ≤15min response time | Buffer: {buffer_factor:.1f}x")
    
    with tab4:
        st.header("Cost Analysis")
        
        # Cost breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Cost Breakdown")
            daily_cost_df = pd.DataFrame({
                'Day': daily_costs.index,
                'Daily Cost': daily_costs.values
            })
            
            fig_daily_cost = px.bar(
                daily_cost_df,
                x='Day',
                y='Daily Cost',
                title="Daily Shuttle Costs",
                labels={'Daily Cost': 'Cost ($)'}
            )
            fig_daily_cost.update_layout(showlegend=False)
            fig_daily_cost.update_xaxes(fixedrange=True)
            fig_daily_cost.update_yaxes(fixedrange=True)
            st.plotly_chart(fig_daily_cost, use_container_width=True)
        
        with col2:
            st.subheader("Monthly Cost Projection")
            monthly_cost_df = pd.DataFrame({
                'Day': monthly_costs.index,
                'Monthly Cost': monthly_costs.values
            })
            
            fig_monthly_cost = px.bar(
                monthly_cost_df,
                x='Day',
                y='Monthly Cost',
                title="Monthly Shuttle Costs by Day Type",
                labels={'Monthly Cost': 'Cost ($)'}
            )
            fig_monthly_cost.update_xaxes(fixedrange=True)
            fig_monthly_cost.update_yaxes(fixedrange=True)
            st.plotly_chart(fig_monthly_cost, use_container_width=True)
        
        # Cost summary metrics
        st.subheader("Cost Summary")
        
        cost_col1, cost_col2, cost_col3 = st.columns(3)
        
        with cost_col1:
            avg_daily_cost = daily_costs.mean()
            st.metric("Average Daily Cost", f"${avg_daily_cost:,.0f}")
        
        with cost_col2:
            total_monthly_cost = monthly_costs.sum()
            st.metric("Total Monthly Cost", f"${total_monthly_cost:,.0f}")
        
        with cost_col3:
            cost_per_passenger = total_monthly_cost / (total_passengers * 4.33)
            st.metric("Cost per Passenger", f"${cost_per_passenger:.2f}")
    
    with tab5:
        st.header("Data Summary")
        
        # Data overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Total Records:** {len(df):,}")
            st.write(f"**Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Unique Days:** {df['date'].nunique()}")
            st.write(f"**Pickup Locations:** {df['pickup_location'].nunique()}")
            st.write(f"**Time Blocks:** {df['time_block'].nunique()}")
        
        with col2:
            st.subheader("Data Quality")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Zero Passenger Records:** {(df['passenger_count'] == 0).sum()}")
            st.write(f"**Average Records per Day:** {len(df) / df['date'].nunique():.1f}")
        
        # Raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Export functionality
        st.subheader("Export Data")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Export processed data
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Processed Data (CSV)",
                data=csv_data,
                file_name=f"shuttle_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Export optimization results
            optimization_csv = shuttle_optimization.to_csv(index=False)
            st.download_button(
                label="Download Optimization Results (CSV)",
                data=optimization_csv,
                file_name=f"shuttle_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Shuttle Tech Platform
    
    This platform optimizes crew transportation between 3 downtown hotels and ANC Airport, handling 120-220 passengers daily.
    
    ### Key Features:
    - **Data Ingestion**: Upload CSV/Excel files or generate sample data
    - **Route Optimization**: 30-minute time blocks with buffer logic
    - **Analytics Dashboard**: Time × Day and Location analysis
    - **Cost Modeling**: Configurable hourly rates ($73-95/hour)
    - **Executive Reporting**: Clean, modern visualizations
    
    ### To Get Started:
    1. **Upload your data** using the sidebar file uploader, or
    2. **Generate sample data** to explore the platform features
    3. **Configure cost parameters** in the sidebar
    4. **Explore the analytics** across different tabs
    
    ### Expected Data Format:
    Your CSV/Excel file should contain these columns:
    - `date`: Date in YYYY-MM-DD format
    - `time_block`: Time in HH:MM format (30-minute blocks)
    - `pickup_location`: Hotel name or location identifier
    - `passenger_count`: Number of passengers for that time/location
    
    ---
    
    **Ready to optimize your shuttle operations? Use the sidebar to get started!**
    """)
    
    # Sample data format
    st.subheader("Sample Data Format")
    sample_df = pd.DataFrame({
        'date': ['2024-01-15', '2024-01-15', '2024-01-15'],
        'time_block': ['06:00', '06:30', '07:00'],
        'pickup_location': ['Hotel Downtown A', 'Hotel Downtown B', 'Hotel Downtown A'],
        'passenger_count': [8, 5, 12]
    })
    st.dataframe(sample_df, use_container_width=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.8em;">
    Shuttle Tech Platform | Optimizing crew transportation operations
</div>
""", unsafe_allow_html=True)