"""
Standalone Forecast Editor for fundamental assumptions.

Philosophy (Ousterhout's Deep Module):
    - Works independently OR integrated with MPT calculator
    - Hides complexity of data loading/saving behind simple UI
    - Single responsibility: Edit and export forecasts
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Forecast Editor", layout="wide")

st.title("📊 Forecast Editor")
st.markdown("""
Edit your fundamental assumptions. Upload a CSV, make changes, and export when ready.
""")

# --- Template for new forecasts ---
TEMPLATE_COLUMNS = [
    "Ticker", "Current Price", "Sales/Share", "Current Margin", 
    "Target Margin", "Adjusted Growth Rate", "Exit PE"
]

COLUMN_CONFIG = {
    "Ticker": st.column_config.TextColumn("Ticker", help="Stock ticker symbol"),
    "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f", help="Current stock price"),
    "Sales/Share": st.column_config.NumberColumn("Sales/Share", format="$%.2f", help="Sales per share (TTM)"),
    "Current Margin": st.column_config.NumberColumn("Current Margin", format="%.4f", min_value=-1.0, max_value=1.0, help="Current net profit margin (decimal)"),
    "Target Margin": st.column_config.NumberColumn("Target Margin", format="%.4f", min_value=-1.0, max_value=1.0, help="Expected margin at exit"),
    "Adjusted Growth Rate": st.column_config.NumberColumn("Growth Rate", format="%.4f", min_value=-1.0, max_value=2.0, help="Annual sales growth rate"),
    "Exit PE": st.column_config.NumberColumn("Exit PE", format="%.1f", min_value=0, help="Expected P/E ratio at exit"),
    # Optional columns
    "Organic Growth": st.column_config.NumberColumn("Organic Growth", format="%.4f"),
    "Dividend Yield": st.column_config.NumberColumn("Dividend Yield", format="%.4f"),
    "Buyback Yield": st.column_config.NumberColumn("Buyback Yield", format="%.4f"),
    "SBC Yield": st.column_config.NumberColumn("SBC Yield", format="%.4f"),
}


def create_empty_template() -> pd.DataFrame:
    """Create an empty template DataFrame."""
    return pd.DataFrame(columns=TEMPLATE_COLUMNS).set_index("Ticker")


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv().encode('utf-8')


# --- Sidebar: Data Source ---
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Load data from:",
    ["Upload CSV", "Session (from Home)", "New Template"],
    help="Choose where to load your forecast data"
)

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Fundamentals CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, index_col=0)
            st.sidebar.success(f"Loaded {len(df)} assets")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")

elif data_source == "Session (from Home)":
    if 'fundamentals' in st.session_state and st.session_state['fundamentals'] is not None:
        df = st.session_state['fundamentals'].copy()
        st.sidebar.success(f"Loaded {len(df)} assets from session")
    else:
        st.sidebar.warning("No data in session. Upload on Home page first, or use 'Upload CSV'.")

elif data_source == "New Template":
    df = create_empty_template()
    st.sidebar.info("Starting with empty template. Add rows below.")

# --- Main Editor ---
if df is not None:
    st.subheader("Edit Assumptions")
    
    # Sorting controls
    col_sort1, col_sort2 = st.columns([1, 1])
    with col_sort1:
        sort_by = st.selectbox(
            "Sort by",
            ["Ticker (A-Z)", "Ticker (Z-A)"] + [f"{col} ↑" for col in df.columns] + [f"{col} ↓" for col in df.columns],
            index=0
        )
    
    # Apply sorting
    if sort_by == "Ticker (A-Z)":
        df = df.sort_index(ascending=True)
    elif sort_by == "Ticker (Z-A)":
        df = df.sort_index(ascending=False)
    elif sort_by.endswith(" ↑"):
        col_name = sort_by[:-2]
        if col_name in df.columns:
            df = df.sort_values(by=col_name, ascending=True)
    elif sort_by.endswith(" ↓"):
        col_name = sort_by[:-2]
        if col_name in df.columns:
            df = df.sort_values(by=col_name, ascending=False)
    
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        column_config=COLUMN_CONFIG,
        num_rows="dynamic",
        key="forecast_editor"
    )
    
    # Sync back to session state for integration with MPT calculator
    if not df.equals(edited_df):
        st.session_state['fundamentals'] = edited_df
        st.toast("✓ Changes saved to session", icon="💾")
    
    # --- Export Section ---
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Export edited data
        csv_data = convert_df_to_csv(edited_df)
        filename = f"fundamentals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Download your edited forecasts"
        )
    
    with col2:
        # Download blank template
        template_csv = convert_df_to_csv(create_empty_template())
        st.download_button(
            label="📋 Download Template",
            data=template_csv,
            file_name="fundamentals_template.csv",
            mime="text/csv",
            help="Download a blank template to fill in"
        )
    
    # Show summary stats
    with st.expander("Data Summary", expanded=False):
        if len(edited_df) > 0 and "Current Price" in edited_df.columns:
            summary_cols = ["Current Price", "Current Margin", "Target Margin", "Adjusted Growth Rate", "Exit PE"]
            available_cols = [c for c in summary_cols if c in edited_df.columns]
            if available_cols:
                st.dataframe(edited_df[available_cols].describe().T.style.format("{:.2f}"))

else:
    st.info("👈 Select a data source from the sidebar to begin editing.")
    
    # Show expected format
    with st.expander("Expected CSV Format"):
        st.markdown("""
**Required columns:**
- `Ticker` (index): Stock symbol (e.g., AAPL, MSFT)
- `Current Price`: Current stock price
- `Sales/Share`: Sales per share (TTM)
- `Current Margin`: Current net profit margin (decimal, e.g., 0.20 for 20%)
- `Target Margin`: Expected margin at investment horizon
- `Adjusted Growth Rate`: Annual sales growth rate (decimal)
- `Exit PE`: Expected P/E ratio at exit

**Optional columns:**
- `Organic Growth`, `Dividend Yield`, `Buyback Yield`, `SBC Yield`
        """)
