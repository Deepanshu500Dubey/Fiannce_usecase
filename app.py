"""
Financial Close Agent API - FastAPI Implementation
SINGLE ENDPOINT: Triggers complete financial close pipeline
Reads from predefined local files, no upload functionality
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import json
import uuid
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - EXACTLY MATCHING YOUR SCRIPT
# ============================================================================

class Config:
    """Configuration settings matching the original script"""
    RAW_DATA_PATH = "Raw_GL_Export.csv"
    MASTER_DATA_PATH = "Master_Data/"
    REFERENCE_PATH = "Reference/"
    BUDGET_PATH = "Budget/"
    OUTPUT_PATH = "working/"
    REPORTS_PATH = "reports/"
    
    # Fiscal period settings
    CURRENT_FISCAL_PERIOD = "2026-02"
    CURRENT_MONTH = 2
    CURRENT_YEAR = 2026
    
    # Anomaly thresholds
    HIGH_VALUE_THRESHOLD = 50000
    EXTREME_OUTLIER_MULTIPLIER = 5
    SUSPICIOUS_HOUR_START = 22
    SUSPICIOUS_HOUR_END = 6

# Create directories if they don't exist
for path in [Config.OUTPUT_PATH, Config.REPORTS_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Financial Close Agent API",
    description="Single endpoint triggers complete financial close pipeline - reads from local files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PipelineResponse(BaseModel):
    """Response model for pipeline execution"""
    run_id: str
    status: str
    message: str
    start_time: str
    summary: Optional[Dict[str, Any]] = None

class PipelineStatus(BaseModel):
    """Status model for pipeline run"""
    run_id: str
    status: str
    progress: int
    current_task: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ============================================================================
# BACKGROUND TASK MANAGER
# ============================================================================

class PipelineManager:
    """Manage pipeline runs"""
    
    def __init__(self):
        self.runs = {}
    
    def create_run(self, run_id: str):
        """Create a new pipeline run"""
        self.runs[run_id] = {
            "run_id": run_id,
            "status": "starting",
            "progress": 0,
            "current_task": "initializing",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "summary": None,
            "error": None
        }
        return run_id
    
    def update_run(self, run_id: str, **kwargs):
        """Update pipeline run status"""
        if run_id in self.runs:
            self.runs[run_id].update(kwargs)
    
    def get_run(self, run_id: str):
        """Get pipeline run by ID"""
        return self.runs.get(run_id)
    
    def complete_run(self, run_id: str, summary: Dict[str, Any]):
        """Mark run as completed"""
        if run_id in self.runs:
            end_time = datetime.now()
            start_time = datetime.fromisoformat(self.runs[run_id]["start_time"])
            duration = (end_time - start_time).total_seconds()
            
            self.runs[run_id].update({
                "status": "completed",
                "progress": 100,
                "current_task": "complete",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "summary": summary
            })
    
    def fail_run(self, run_id: str, error: str):
        """Mark run as failed"""
        if run_id in self.runs:
            end_time = datetime.now()
            start_time = datetime.fromisoformat(self.runs[run_id]["start_time"])
            duration = (end_time - start_time).total_seconds()
            
            self.runs[run_id].update({
                "status": "failed",
                "progress": self.runs[run_id].get("progress", 0),
                "current_task": self.runs[run_id].get("current_task", "unknown"),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "error": error
            })

pipeline_manager = PipelineManager()

def generate_run_id():
    """Generate unique run ID"""
    return f"close_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============================================================================
# T001: WRANGLE RAW GL DATA
# ============================================================================

class T001_DataWrangler:
    """Task 1: Parse and standardize raw GL export data"""
    
    def __init__(self):
        self.raw_df = None
        self.standardized_df = None
        self.anomaly_log = []
        
    def load_raw_data(self, filepath):
        """Load raw CSV file"""
        print("📂 T001: Loading raw GL data...")
        self.raw_df = pd.read_csv(filepath)
        print(f"   Loaded {len(self.raw_df)} rows")
        return self
    
    def standardize_column_names(self):
        """Convert column names to snake_case"""
        column_mapping = {
            'Txn_ID': 'transaction_id',
            'Posting_Date_Raw': 'posting_date_raw',
            'Invoice_Date_Raw': 'invoice_date_raw',
            'Fiscal_Period': 'fiscal_period',
            'Entity': 'entity_code',
            'Account_Code_Raw': 'account_code_raw',
            'Cost_Center_Raw': 'cost_center_raw',
            'Vendor_Name_Raw': 'vendor_name_raw',
            'Invoice_Number': 'invoice_number',
            'PO_Number': 'po_number',
            'Currency': 'currency_code',
            'Amount': 'amount_raw',
            'Tax_Code': 'tax_code',
            'Narrative': 'narrative',
            'Source_System': 'source_system'
        }
        self.standardized_df = self.raw_df.rename(columns=column_mapping)
        print("   ✓ Column names standardized")
        return self
    
    def standardize_dates(self):
        """Convert all dates to consistent format YYYY-MM-DD"""
        df = self.standardized_df
        
        def parse_date(date_str, txn_id, column_name):
            if pd.isna(date_str) or date_str in ['INVALID', '99/99/9999', '32/13/2026', '2026-13-45']:
                self.anomaly_log.append({
                    'transaction_id': txn_id,
                    'anomaly_type': 'INVALID_DATE',
                    'severity': 'CRITICAL',
                    'description': f"Invalid date value: {date_str}",
                    'column': column_name
                })
                return None
            
            # Try different date formats
            formats = [
                '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
                '%d/%m/%y', '%m/%d/%y', '%d-%m-%y', '%y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(str(date_str), fmt)
                except:
                    continue
            
            # If all formats fail
            self.anomaly_log.append({
                'transaction_id': txn_id,
                'anomaly_type': 'UNPARSABLE_DATE',
                'severity': 'CRITICAL',
                'description': f"Cannot parse date: {date_str}",
                'column': column_name
            })
            return None
        
        # Apply date parsing with transaction_id
        df['posting_date'] = df.apply(
            lambda row: parse_date(row['posting_date_raw'], row['transaction_id'], 'posting_date_raw'), 
            axis=1
        )
        df['invoice_date'] = df.apply(
            lambda row: parse_date(row['invoice_date_raw'], row['transaction_id'], 'invoice_date_raw'), 
            axis=1
        )
        
        # Extract fiscal year and month
        df['fiscal_year'] = df['fiscal_period'].str[:4]
        df['fiscal_month'] = df['fiscal_period'].str[-2:]
        
        # Check fiscal period consistency
        for idx, row in df.iterrows():
            if pd.notna(row['posting_date']):
                posting_month = row['posting_date'].month
                fiscal_month = int(row['fiscal_month']) if pd.notna(row['fiscal_month']) else None
                
                if fiscal_month and posting_month != fiscal_month:
                    self.anomaly_log.append({
                        'transaction_id': row['transaction_id'],
                        'anomaly_type': 'FISCAL_PERIOD_MISMATCH',
                        'severity': 'HIGH',
                        'description': f"Posting date month ({posting_month}) != fiscal period month ({fiscal_month})",
                        'posting_date': row['posting_date'],
                        'fiscal_period': row['fiscal_period']
                    })
        
        print(f"   ✓ Dates standardized. Invalid dates: {sum(df['posting_date'].isna())}")
        return self
    
    def clean_amounts(self):
        """Convert amount strings to floats"""
        df = self.standardized_df
        
        def parse_amount(amt_str, txn_id):
            if pd.isna(amt_str):
                return None
            
            # Remove currency symbols, commas, spaces
            cleaned = str(amt_str).replace('$', '').replace(',', '').strip()
            
            # Handle negative numbers in parentheses
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            try:
                return float(cleaned)
            except:
                self.anomaly_log.append({
                    'transaction_id': txn_id,
                    'anomaly_type': 'INVALID_AMOUNT',
                    'severity': 'HIGH',
                    'description': f"Cannot parse amount: {amt_str}"
                })
                return None
        
        df['amount'] = df.apply(
            lambda row: parse_amount(row['amount_raw'], row['transaction_id']), 
            axis=1
        )
        
        # Flag negative amounts
        df['amount_is_negative'] = df['amount'] < 0
        for idx, row in df[df['amount_is_negative']].iterrows():
            self.anomaly_log.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'NEGATIVE_AMOUNT',
                'severity': 'MEDIUM',
                'description': f"Negative amount: {row['amount']}",
                'amount': row['amount']
            })
        
        print(f"   ✓ Amounts cleaned. Negative amounts: {df['amount_is_negative'].sum()}")
        return self
    
    def detect_embedded_exceptions(self):
        """Look for obvious exceptions in raw data"""
        df = self.standardized_df
        keywords = ['error', 'flag', 'review', 'urgent', 'exception', 'invalid']
        
        df['narrative_lower'] = df['narrative'].str.lower().fillna('')
        
        for idx, row in df.iterrows():
            # Check narrative for keywords
            if any(keyword in str(row['narrative_lower']) for keyword in keywords):
                self.anomaly_log.append({
                    'transaction_id': row['transaction_id'],
                    'anomaly_type': 'NARRATIVE_SUGGESTS_EXCEPTION',
                    'severity': 'MEDIUM',
                    'description': f"Narrative contains exception keywords: {row['narrative']}",
                    'narrative': row['narrative']
                })
            
            # Check for placeholder vendor names
            if row['vendor_name_raw'] in ['Unlisted Company', 'Unknown Vendor LLC', 
                                           'New Vendor XYZ', 'Unregistered Supplier', 
                                           'Mystery Corp']:
                self.anomaly_log.append({
                    'transaction_id': row['transaction_id'],
                    'anomaly_type': 'PLACEHOLDER_VENDOR',
                    'severity': 'HIGH',
                    'description': f"Placeholder vendor name: {row['vendor_name_raw']}",
                    'vendor': row['vendor_name_raw']
                })
        
        print(f"   ✓ Embedded exceptions detected: {len([a for a in self.anomaly_log if a['anomaly_type'] == 'NARRATIVE_SUGGESTS_EXCEPTION'])}")
        return self
    
    def add_metadata(self):
        """Add processing metadata"""
        df = self.standardized_df
        df['processing_timestamp'] = datetime.now()
        df['source_file'] = 'Raw_GL_Export.csv'
        df['data_quality_score'] = 100 - (len(self.anomaly_log) / len(df) * 100) if len(df) > 0 else 100
        df['anomaly_count'] = df.apply(lambda row: len([a for a in self.anomaly_log 
                                                          if a.get('transaction_id') == row['transaction_id']]), axis=1)
        return self
    
    def save_output(self):
        """Save standardized data and anomaly log"""
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        os.makedirs(Config.REPORTS_PATH, exist_ok=True)
        
        # Save standardized data
        output_cols = ['transaction_id', 'posting_date_raw', 'posting_date', 'invoice_date_raw',
                       'invoice_date', 'fiscal_period', 'fiscal_year', 'fiscal_month',
                       'entity_code', 'account_code_raw', 'cost_center_raw', 'vendor_name_raw',
                       'invoice_number', 'po_number', 'currency_code', 'amount_raw', 'amount',
                       'amount_is_negative', 'tax_code', 'narrative', 'source_system',
                       'processing_timestamp', 'data_quality_score', 'anomaly_count']
        
        # Only include columns that exist
        available_cols = [col for col in output_cols if col in self.standardized_df.columns]
        self.standardized_df[available_cols].to_csv(
            f"{Config.OUTPUT_PATH}GL_Standardized.csv", index=False
        )
        
        # Save anomaly log
        if self.anomaly_log:
            pd.DataFrame(self.anomaly_log).to_csv(
                f"{Config.REPORTS_PATH}Input_Anomalies_Detected.csv", index=False
            )
        
        print(f"   💾 Saved {len(self.standardized_df)} rows to {Config.OUTPUT_PATH}GL_Standardized.csv")
        print(f"   💾 Saved {len(self.anomaly_log)} anomalies to {Config.REPORTS_PATH}Input_Anomalies_Detected.csv")
        
        return self.standardized_df, self.anomaly_log
    
    def run(self, filepath):
        """Execute all T001 steps"""
        print("\n" + "="*60)
        print("🚀 T001: Wrangling Raw GL Data")
        print("="*60)
        
        self.load_raw_data(filepath)
        self.standardize_column_names()
        self.standardize_dates()
        self.clean_amounts()
        self.detect_embedded_exceptions()
        self.add_metadata()
        df, anomalies = self.save_output()
        
        print(f"\n✅ T001 Complete. Processed {len(df)} rows, found {len(anomalies)} anomalies.")
        return df, anomalies


# ============================================================================
# T002: MAP ENTITIES AND ACCOUNTS
# ============================================================================

class T002_EntityAccountMapper:
    """Task 2: Resolve entity codes and account codes against master data"""
    
    def __init__(self, working_df):
        self.df = working_df.copy()
        self.entity_master = None
        self.account_master = None
        self.cost_center_master = None
        self.mapping_anomalies = []
        
    def load_master_data(self):
        """Load master reference files"""
        print("\n📂 T002: Loading master data...")
        
        try:
            self.entity_master = pd.read_csv(f"{Config.MASTER_DATA_PATH}Master_Entity.csv")
            print(f"   Loaded {len(self.entity_master)} entities")
        except:
            print("   ⚠️ Entity master not found, creating default")
            self.entity_master = pd.DataFrame({'entity_code': ['AUS01']})
        
        try:
            self.account_master = pd.read_csv(f"{Config.MASTER_DATA_PATH}Master_COA.csv")
            print(f"   Loaded {len(self.account_master)} accounts")
            
            # Standardize column names
            self.account_master.columns = [col.lower().strip() for col in self.account_master.columns]
            
            # Map the account code column
            if 'account_code' not in self.account_master.columns:
                if 'account_code' in self.account_master.columns:
                    self.account_master.rename(columns={'account_code': 'account_code'}, inplace=True)
                elif 'account' in self.account_master.columns:
                    self.account_master.rename(columns={'account': 'account_code'}, inplace=True)
                elif 'code' in self.account_master.columns:
                    self.account_master.rename(columns={'code': 'account_code'}, inplace=True)
                else:
                    first_col = self.account_master.columns[0]
                    self.account_master.rename(columns={first_col: 'account_code'}, inplace=True)
            
        except Exception as e:
            print(f"   ⚠️ Account master not found: {e}")
            self.account_master = pd.DataFrame({'account_code': [f"{i:04d}" for i in range(5000, 5029)]})
        
        try:
            self.cost_center_master = pd.read_csv(f"{Config.MASTER_DATA_PATH}Master_CostCenters.csv")
            print(f"   Loaded {len(self.cost_center_master)} cost centers")
            
            self.cost_center_master.columns = [col.lower().strip() for col in self.cost_center_master.columns]
            
            if 'cost_center' not in self.cost_center_master.columns:
                if 'costcenter' in self.cost_center_master.columns:
                    self.cost_center_master.rename(columns={'costcenter': 'cost_center'}, inplace=True)
                elif 'cc' in self.cost_center_master.columns:
                    self.cost_center_master.rename(columns={'cc': 'cost_center'}, inplace=True)
                else:
                    first_col = self.cost_center_master.columns[0]
                    self.cost_center_master.rename(columns={first_col: 'cost_center'}, inplace=True)
                    
        except Exception as e:
            print(f"   ⚠️ Cost center master not found: {e}")
            self.cost_center_master = pd.DataFrame({'cost_center': ['CC' + str(i).zfill(4) for i in range(1000, 1010)]})
        
        return self
    
    def map_entities(self):
        """Map entity codes against master"""
        if 'entity_code' not in self.entity_master.columns:
            for col in self.entity_master.columns:
                if 'entity' in col.lower() or 'code' in col.lower():
                    self.entity_master.rename(columns={col: 'entity_code'}, inplace=True)
                    break
        
        valid_entities = self.entity_master['entity_code'].tolist() if 'entity_code' in self.entity_master.columns else ['AUS01']
        
        self.df['entity_valid'] = self.df['entity_code'].isin(valid_entities)
        self.df['entity_code_mapped'] = np.where(
            self.df['entity_valid'], 
            self.df['entity_code'], 
            None
        )
        
        for idx, row in self.df[~self.df['entity_valid']].iterrows():
            self.mapping_anomalies.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'INVALID_ENTITY',
                'severity': 'CRITICAL',
                'description': f"Entity code '{row['entity_code']}' not in master",
                'original_value': row['entity_code']
            })
        
        print(f"   ✓ Entities mapped. Invalid: {(~self.df['entity_valid']).sum()}")
        return self
    
    def map_accounts(self):
        """Map account codes against master"""
        
        if 'account_code' in self.account_master.columns:
            valid_accounts = [str(acct).strip() for acct in self.account_master['account_code'].tolist()]
            valid_accounts = list(set(valid_accounts))
        else:
            valid_accounts = []
        
        # Clean raw account codes
        self.df['account_code_clean'] = self.df['account_code_raw'].astype(str).str.strip()
        
        self.df['account_valid'] = False
        
        # Direct match
        direct_match = self.df['account_code_raw'].isin(valid_accounts)
        self.df.loc[direct_match, 'account_valid'] = True
        
        # Clean match
        clean_match = (~direct_match) & self.df['account_code_clean'].isin(valid_accounts)
        self.df.loc[clean_match, 'account_valid'] = True
        
        # Assign mapped account codes
        def find_matching_account(row):
            if row['account_valid']:
                if row['account_code_raw'] in valid_accounts:
                    return row['account_code_raw']
                elif row['account_code_clean'] in valid_accounts:
                    return row['account_code_clean']
                else:
                    return row['account_code_raw']
            return None
        
        self.df['account_code_mapped'] = self.df.apply(find_matching_account, axis=1)
        
        # Get account names if available
        if 'account_name' in self.account_master.columns:
            account_desc_map = {}
            for _, row in self.account_master.iterrows():
                acct = str(row['account_code']).strip()
                desc = row['account_name']
                account_desc_map[acct] = desc
            
            self.df['account_description'] = self.df['account_code_mapped'].map(account_desc_map)
        
        # Log anomalies
        invalid_count = (~self.df['account_valid']).sum()
        for idx, row in self.df[~self.df['account_valid']].iterrows():
            self.mapping_anomalies.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'INVALID_ACCOUNT',
                'severity': 'CRITICAL' if str(row['account_code_raw']) == 'INVALID_ACCT' else 'HIGH',
                'description': f"Account code '{row['account_code_raw']}' not in Chart of Accounts",
                'original_value': row['account_code_raw'],
                'amount': row['amount']
            })
        
        print(f"   ✓ Accounts mapped. Valid: {self.df['account_valid'].sum()}, Invalid: {invalid_count}")
        return self
    
    def map_cost_centers(self):
        """Map cost centers against master"""
        if 'cost_center' in self.cost_center_master.columns:
            valid_centers = self.cost_center_master['cost_center'].tolist()
        else:
            valid_centers = []
        
        self.df['cost_center_present'] = self.df['cost_center_raw'].notna() & (self.df['cost_center_raw'] != '')
        self.df['cost_center_valid'] = self.df['cost_center_raw'].isin(valid_centers) if valid_centers else self.df['cost_center_present']
        self.df['cost_center_mapped'] = np.where(
            self.df['cost_center_valid'],
            self.df['cost_center_raw'],
            None
        )
        
        for idx, row in self.df[~self.df['cost_center_present']].iterrows():
            self.mapping_anomalies.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'MISSING_COST_CENTER',
                'severity': 'MEDIUM',
                'description': "Cost center is missing",
                'amount': row['amount']
            })
        
        for idx, row in self.df[self.df['cost_center_present'] & ~self.df['cost_center_valid']].iterrows():
            self.mapping_anomalies.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'INVALID_COST_CENTER',
                'severity': 'HIGH',
                'description': f"Cost center '{row['cost_center_raw']}' not in master",
                'original_value': row['cost_center_raw']
            })
        
        print(f"   ✓ Cost centers mapped. Missing: {(~self.df['cost_center_present']).sum()}, Invalid: {(self.df['cost_center_present'] & ~self.df['cost_center_valid']).sum()}")
        return self
    
    def save_output(self):
        """Save mapped data"""
        existing_anomalies = pd.read_csv(f"{Config.REPORTS_PATH}Input_Anomalies_Detected.csv") if os.path.exists(f"{Config.REPORTS_PATH}Input_Anomalies_Detected.csv") else pd.DataFrame()
        
        all_anomalies = pd.concat([
            existing_anomalies, 
            pd.DataFrame(self.mapping_anomalies)
        ], ignore_index=True)
        
        all_anomalies.to_csv(f"{Config.REPORTS_PATH}Exceptions_Log.csv", index=False)
        
        self.df.to_csv(f"{Config.OUTPUT_PATH}GL_WithMappings.csv", index=False)
        
        print(f"   💾 Saved to {Config.OUTPUT_PATH}GL_WithMappings.csv")
        print(f"   💾 Updated exceptions log with {len(self.mapping_anomalies)} new anomalies")
        
        return self.df
    
    def run(self):
        """Execute all T002 steps"""
        print("\n" + "="*60)
        print("🚀 T002: Mapping Entities and Accounts")
        print("="*60)
        
        self.load_master_data()
        self.map_entities()
        self.map_accounts()
        self.map_cost_centers()
        df = self.save_output()
        
        print(f"\n✅ T002 Complete. Mapped {len(df)} transactions.")
        return df


# ============================================================================
# T003: RESOLVE VENDOR NAMES
# ============================================================================

class T003_VendorResolver:
    """Task 3: Map vendor aliases to canonical vendor names"""
    
    def __init__(self, working_df):
        self.df = working_df.copy()
        self.vendor_master = None
        self.alias_map = None
        self.vendor_anomalies = []
        
    def load_vendor_data(self):
        """Load vendor master and alias mapping"""
        print("\n📂 T003: Loading vendor data...")
        
        try:
            self.vendor_master = pd.read_csv(f"{Config.MASTER_DATA_PATH}Master_Vendors.csv")
            print(f"   Loaded {len(self.vendor_master)} canonical vendors")
            
            self.vendor_master.columns = [col.lower().strip() for col in self.vendor_master.columns]
            
            if 'vendor_name_canonical' in self.vendor_master.columns:
                self.vendor_master.rename(columns={'vendor_name_canonical': 'canonical_vendor'}, inplace=True)
            elif 'vendor_name' in self.vendor_master.columns:
                self.vendor_master.rename(columns={'vendor_name': 'canonical_vendor'}, inplace=True)
            
        except Exception as e:
            print(f"   ⚠️ Vendor master not found: {e}")
            self.vendor_master = pd.DataFrame({'canonical_vendor': ['Unknown']})
        
        try:
            self.alias_map = pd.read_csv(f"{Config.MASTER_DATA_PATH}Vendor_Alias_Map.csv")
            print(f"   Loaded {len(self.alias_map)} alias mappings")
            
            self.alias_map.columns = [col.lower().strip() for col in self.alias_map.columns]
            
            if 'vendor_name_raw' in self.alias_map.columns:
                self.alias_map.rename(columns={'vendor_name_raw': 'alias'}, inplace=True)
            
            if 'vendor_name_canonical' in self.alias_map.columns:
                self.alias_map.rename(columns={'vendor_name_canonical': 'canonical_vendor'}, inplace=True)
            
        except Exception as e:
            print(f"   ⚠️ Alias map not found: {e}")
            self.alias_map = pd.DataFrame({'alias': [], 'canonical_vendor': []})
        
        return self
    
    def build_alias_dict(self):
        """Create lookup dictionary from aliases to canonical names"""
        alias_dict = {}
        
        if self.alias_map is not None and len(self.alias_map) > 0:
            if 'alias' in self.alias_map.columns and 'canonical_vendor' in self.alias_map.columns:
                for _, row in self.alias_map.iterrows():
                    alias_raw = str(row['alias']).strip()
                    alias_lower = alias_raw.lower()
                    alias_dict[alias_lower] = row['canonical_vendor']
                    
                    for suffix in [' pty', ' ltd', ' inc', ' corp', ' llc', ' australia', ' usa', ' uk']:
                        if alias_lower.endswith(suffix):
                            alias_dict[alias_lower[:-len(suffix)]] = row['canonical_vendor']
        
        if self.vendor_master is not None and 'canonical_vendor' in self.vendor_master.columns:
            for vendor in self.vendor_master['canonical_vendor'].dropna():
                vendor_lower = vendor.lower()
                alias_dict[vendor_lower] = vendor
        
        print(f"   Built alias dictionary with {len(alias_dict)} entries")
        return alias_dict
    
    def resolve_vendors(self):
        """Apply vendor mapping"""
        alias_dict = self.build_alias_dict()
        
        if 'canonical_vendor' in self.vendor_master.columns:
            canonical_list = self.vendor_master['canonical_vendor'].dropna().unique().tolist()
        else:
            canonical_list = []
        
        def resolve(vendor_raw):
            if pd.isna(vendor_raw) or vendor_raw == '':
                return None, 'MISSING'
            
            vendor_original = str(vendor_raw).strip()
            vendor_lower = vendor_original.lower()
            
            if vendor_lower in alias_dict:
                return alias_dict[vendor_lower], 'MAPPED'
            
            if vendor_original in canonical_list:
                return vendor_original, 'CANONICAL'
            
            import re
            vendor_clean = re.sub(r'[^\w\s]', '', vendor_lower)
            if vendor_clean in alias_dict:
                return alias_dict[vendor_clean], 'CLEANED_MATCH'
            
            for canonical in canonical_list:
                canonical_lower = canonical.lower()
                if canonical_lower in vendor_lower:
                    return canonical, 'PARTIAL_MATCH'
                if len(vendor_lower) > 5 and vendor_lower in canonical_lower:
                    return canonical, 'PARTIAL_MATCH'
            
            return None, 'UNMAPPED'
        
        results = self.df['vendor_name_raw'].apply(resolve)
        self.df['vendor_canonical'] = [r[0] for r in results]
        self.df['vendor_resolution_status'] = [r[1] for r in results]
        
        for idx, row in self.df.iterrows():
            if row['vendor_resolution_status'] == 'MISSING':
                self.vendor_anomalies.append({
                    'transaction_id': row['transaction_id'],
                    'anomaly_type': 'MISSING_VENDOR',
                    'severity': 'HIGH',
                    'description': 'Vendor name is missing',
                    'amount': row['amount']
                })
            elif row['vendor_resolution_status'] == 'UNMAPPED':
                self.vendor_anomalies.append({
                    'transaction_id': row['transaction_id'],
                    'anomaly_type': 'UNMAPPED_VENDOR',
                    'severity': 'HIGH',
                    'description': f"Vendor '{row['vendor_name_raw']}' not found in alias map",
                    'original_value': row['vendor_name_raw'],
                    'amount': row['amount']
                })
        
        mapped_count = self.df['vendor_resolution_status'].isin(['MAPPED', 'CANONICAL', 'CLEANED_MATCH', 'PARTIAL_MATCH']).sum()
        unmapped_count = (self.df['vendor_resolution_status'] == 'UNMAPPED').sum()
        missing_count = (self.df['vendor_resolution_status'] == 'MISSING').sum()
        
        print(f"\n   📊 Vendor Resolution Results:")
        print(f"   • Mapped: {mapped_count}")
        print(f"   • Unmapped: {unmapped_count}")
        print(f"   • Missing: {missing_count}")
        
        return self
    
    def save_output(self):
        """Save vendor-resolved data"""
        exceptions_path = f"{Config.REPORTS_PATH}Exceptions_Log.csv"
        if os.path.exists(exceptions_path):
            existing = pd.read_csv(exceptions_path)
            all_exceptions = pd.concat([existing, pd.DataFrame(self.vendor_anomalies)], ignore_index=True)
        else:
            all_exceptions = pd.DataFrame(self.vendor_anomalies)
        
        all_exceptions.to_csv(exceptions_path, index=False)
        self.df.to_csv(f"{Config.OUTPUT_PATH}GL_VendorsResolved.csv", index=False)
        
        print(f"   💾 Saved to {Config.OUTPUT_PATH}GL_VendorsResolved.csv")
        return self.df
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T003: Resolving Vendor Names")
        print("="*60)
        
        self.load_vendor_data()
        self.resolve_vendors()
        df = self.save_output()
        
        print(f"\n✅ T003 Complete. Processed {len(df)} transactions.")
        return df


# ============================================================================
# T004: APPLY FX CONVERSION
# ============================================================================

class T004_FXConverter:
    """Task 4: Convert all transactions to AUD"""
    
    def __init__(self, working_df):
        self.df = working_df.copy()
        self.fx_rates = None
        self.fx_anomalies = []
        
    def load_fx_rates(self):
        """Load foreign exchange rates"""
        print("\n📂 T004: Loading FX rates...")
        
        try:
            self.fx_rates = pd.read_csv(f"{Config.REFERENCE_PATH}FX_Rates.csv")
            print(f"   Loaded {len(self.fx_rates)} FX rates")
            
            self.fx_rates.columns = [col.lower().strip() for col in self.fx_rates.columns]
            
            if 'fiscal_period' in self.fx_rates.columns:
                self.fx_rates.rename(columns={'fiscal_period': 'period'}, inplace=True)
            
            if 'rate_to_aud' in self.fx_rates.columns:
                self.fx_rates.rename(columns={'rate_to_aud': 'rate'}, inplace=True)
            
            self.fx_rates['period'] = self.fx_rates['period'].astype(str)
            
            current_rates = self.fx_rates[self.fx_rates['period'] == Config.CURRENT_FISCAL_PERIOD]
            if not current_rates.empty:
                for _, row in current_rates.iterrows():
                    print(f"   • {row['currency']}: 1 {row['currency']} = {row['rate']:.4f} AUD")
            
        except Exception as e:
            print(f"   ⚠️ Error loading FX rates: {e}")
            periods = self.df['fiscal_period'].unique()
            currencies = self.df['currency_code'].unique()
            
            rates_data = []
            for period in periods:
                for currency in currencies:
                    if currency == 'AUD':
                        rate = 1.0
                    elif currency == 'USD':
                        rate = 1.5250
                    elif currency == 'GBP':
                        rate = 1.9550
                    elif currency == 'NZD':
                        rate = 0.9320
                    else:
                        rate = 1.0
                    
                    rates_data.append({
                        'period': period,
                        'currency': currency,
                        'rate': rate
                    })
            
            self.fx_rates = pd.DataFrame(rates_data)
        
        return self
    
    def convert_to_aud(self):
        """Convert amounts to AUD"""
        self.df['fx_key'] = self.df['fiscal_period'] + '_' + self.df['currency_code']
        self.fx_rates['fx_key'] = self.fx_rates['period'].astype(str) + '_' + self.fx_rates['currency']
        
        rate_dict = dict(zip(self.fx_rates['fx_key'], self.fx_rates['rate']))
        
        def get_rate(row):
            if row['currency_code'] == 'AUD':
                return 1.0
            
            key = row['fx_key']
            if key in rate_dict:
                return rate_dict[key]
            
            currency_rates = {k: v for k, v in rate_dict.items() if k.endswith('_' + row['currency_code'])}
            if currency_rates:
                sorted_rates = sorted(currency_rates.items(), key=lambda x: x[0], reverse=True)
                return sorted_rates[0][1]
            
            self.fx_anomalies.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'MISSING_FX_RATE',
                'severity': 'CRITICAL',
                'description': f"No FX rate found for {row['currency_code']} in period {row['fiscal_period']}",
                'currency': row['currency_code'],
                'period': row['fiscal_period'],
                'amount': row['amount']
            })
            return None
        
        self.df['fx_rate'] = self.df.apply(get_rate, axis=1)
        self.df['amount_aud'] = np.where(
            self.df['fx_rate'].notna(),
            self.df['amount'] * self.df['fx_rate'],
            None
        )
        
        self.df['conversion_status'] = np.where(
            self.df['currency_code'] == 'AUD', 'DOMESTIC',
            np.where(self.df['fx_rate'].notna(), 'CONVERTED', 'FAILED')
        )
        
        converted = (self.df['conversion_status'] == 'CONVERTED').sum()
        failed = (self.df['conversion_status'] == 'FAILED').sum()
        domestic = (self.df['conversion_status'] == 'DOMESTIC').sum()
        
        print(f"\n   ✓ FX conversion complete. Domestic: {domestic}, Converted: {converted}, Failed: {failed}")
        
        return self
    
    def save_output(self):
        """Save converted data"""
        exceptions_path = f"{Config.REPORTS_PATH}Exceptions_Log.csv"
        if os.path.exists(exceptions_path):
            existing = pd.read_csv(exceptions_path)
            all_exceptions = pd.concat([existing, pd.DataFrame(self.fx_anomalies)], ignore_index=True)
        else:
            all_exceptions = pd.DataFrame(self.fx_anomalies)
        
        all_exceptions.to_csv(exceptions_path, index=False)
        self.df.to_csv(f"{Config.OUTPUT_PATH}GL_Converted.csv", index=False)
        
        print(f"\n   💾 Saved to {Config.OUTPUT_PATH}GL_Converted.csv")
        return self.df
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T004: Applying FX Conversion")
        print("="*60)
        
        self.load_fx_rates()
        self.convert_to_aud()
        df = self.save_output()
        
        print(f"\n✅ T004 Complete. Processed {len(df)} transactions.")
        return df


# ============================================================================
# T005: DETECT EXCEPTIONS
# ============================================================================

class T005_ExceptionDetector:
    """Task 5: Run exception rules and flag violations"""
    
    def __init__(self, working_df):
        self.df = working_df.copy()
        self.rulebook = None
        self.exception_results = []
        
    def load_rulebook(self):
        """Load exception rules"""
        print("\n📂 T005: Loading exception rulebook...")
        
        try:
            self.rulebook = pd.read_csv(f"{Config.REFERENCE_PATH}Exception_Rulebook.csv")
            print(f"   Loaded {len(self.rulebook)} exception rules")
            
            if 'rule_id' not in self.rulebook.columns:
                self.rulebook['rule_id'] = [f'EX{i+1:03d}' for i in range(len(self.rulebook))]
                
        except Exception as e:
            print(f"   ⚠️ Rulebook not found: {e}")
            self.rulebook = pd.DataFrame([
                {'rule_id': 'EX001', 'rule_name': 'Missing PO Number', 
                 'severity': 'HIGH', 'description': 'Transaction has no purchase order number'},
                {'rule_id': 'EX002', 'rule_name': 'Missing Cost Center',
                 'severity': 'MEDIUM', 'description': 'Transaction has no cost center allocation'},
                {'rule_id': 'EX003', 'rule_name': 'Invalid Account',
                 'severity': 'CRITICAL', 'description': 'Account code not in Chart of Accounts'},
                {'rule_id': 'EX004', 'rule_name': 'High Value Transaction',
                 'severity': 'MEDIUM', 'description': f'Transaction exceeds ${Config.HIGH_VALUE_THRESHOLD:,}'},
                {'rule_id': 'EX005', 'rule_name': 'Negative Amount',
                 'severity': 'MEDIUM', 'description': 'Transaction has negative amount'},
                {'rule_id': 'EX006', 'rule_name': 'Unmapped Vendor',
                 'severity': 'HIGH', 'description': 'Vendor not found in master data'},
                {'rule_id': 'EX007', 'rule_name': 'Future Dated Transaction',
                 'severity': 'HIGH', 'description': 'Transaction date is in future but in current period'},
                {'rule_id': 'EX008', 'rule_name': 'Invalid Date',
                 'severity': 'CRITICAL', 'description': 'Posting date is invalid or missing'},
                {'rule_id': 'EX009', 'rule_name': 'Missing Tax Code',
                 'severity': 'MEDIUM', 'description': 'Tax code is missing'},
            ])
        
        required_cols = ['rule_id', 'rule_name', 'severity', 'description']
        for col in required_cols:
            if col not in self.rulebook.columns:
                if col == 'rule_id':
                    self.rulebook['rule_id'] = [f'EX{i+1:03d}' for i in range(len(self.rulebook))]
                elif col == 'rule_name':
                    self.rulebook['rule_name'] = [f'Rule {i+1}' for i in range(len(self.rulebook))]
                elif col == 'severity':
                    self.rulebook['severity'] = 'MEDIUM'
                elif col == 'description':
                    self.rulebook['description'] = self.rulebook.get('rule_name', 'No description')
        
        return self
    
    def detect_outliers(self):
        """Statistical outlier detection"""
        account_stats = self.df.groupby('account_code_mapped')['amount_aud'].agg(['mean', 'std', 'count']).reset_index()
        account_stats.columns = ['account_code_mapped', 'mean_amount', 'std_amount', 'txn_count']
        
        self.df = self.df.merge(account_stats, on='account_code_mapped', how='left')
        
        self.df['is_outlier'] = np.where(
            (self.df['std_amount'] > 0) & 
            (self.df['amount_aud'].notna()) &
            (abs(self.df['amount_aud'] - self.df['mean_amount']) > Config.EXTREME_OUTLIER_MULTIPLIER * self.df['std_amount']),
            True,
            False
        )
        
        print(f"   ✓ Outlier detection complete. Found {self.df['is_outlier'].sum()} outliers")
        return self
    
    def apply_rules(self):
        """Apply all exception rules"""
        current_date = datetime(Config.CURRENT_YEAR, Config.CURRENT_MONTH, 28)
        
        rule_functions = {
            'EX001': lambda row: pd.isna(row['po_number']) or row['po_number'] == '',
            'EX002': lambda row: pd.isna(row['cost_center_mapped']),
            'EX003': lambda row: pd.isna(row['account_code_mapped']),
            'EX004': lambda row: row['amount_aud'] > Config.HIGH_VALUE_THRESHOLD if pd.notna(row['amount_aud']) else False,
            'EX005': lambda row: row.get('amount_is_negative', False),
            'EX006': lambda row: row.get('vendor_resolution_status') == 'UNMAPPED',
            'EX007': lambda row: (pd.notna(row['posting_date']) and 
                                  row['posting_date'] > current_date and 
                                  row['fiscal_period'] == Config.CURRENT_FISCAL_PERIOD),
            'EX008': lambda row: pd.isna(row['posting_date']),
            'EX009': lambda row: pd.isna(row['tax_code']) or row['tax_code'] == '',
        }
        
        for _, rule in self.rulebook.iterrows():
            rule_id = rule['rule_id']
            rule_name = rule.get('rule_name', f'Rule {rule_id}')
            severity = rule.get('severity', 'MEDIUM')
            description = rule.get('description', rule_name)
            
            rule_func = rule_functions.get(rule_id)
            if rule_func is None:
                continue
            
            for idx, row in self.df.iterrows():
                try:
                    if rule_func(row):
                        self.exception_results.append({
                            'transaction_id': row['transaction_id'],
                            'rule_id': rule_id,
                            'rule_name': rule_name,
                            'severity': severity,
                            'description': description,
                            'amount': row.get('amount_aud', 0),
                            'vendor': row.get('vendor_name_raw', ''),
                            'account': row.get('account_code_raw', '')
                        })
                except:
                    continue
        
        print(f"   ✓ Applied rules, found {len(self.exception_results)} exceptions")
        return self
    
    def save_output(self):
        """Save exception results"""
        exception_txns = [e['transaction_id'] for e in self.exception_results]
        self.df['has_exception'] = self.df['transaction_id'].isin(exception_txns)
        
        self.df.to_csv(f"{Config.OUTPUT_PATH}GL_WithExceptions.csv", index=False)
        
        if self.exception_results:
            exceptions_df = pd.DataFrame(self.exception_results)
            exceptions_df.to_csv(f"{Config.REPORTS_PATH}Exceptions_Detailed.csv", index=False)
        
        master_exceptions_path = f"{Config.REPORTS_PATH}Exceptions_Log.csv"
        
        new_exceptions = []
        for e in self.exception_results:
            new_exceptions.append({
                'transaction_id': e['transaction_id'],
                'anomaly_type': e['rule_id'],
                'severity': e['severity'],
                'description': e['description'],
                'amount': e.get('amount', 0)
            })
        
        if os.path.exists(master_exceptions_path):
            existing = pd.read_csv(master_exceptions_path)
            all_exceptions = pd.concat([existing, pd.DataFrame(new_exceptions)], ignore_index=True)
        else:
            all_exceptions = pd.DataFrame(new_exceptions)
        
        all_exceptions.to_csv(master_exceptions_path, index=False)
        
        return self.df, self.exception_results
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T005: Detecting Exceptions")
        print("="*60)
        
        self.load_rulebook()
        self.detect_outliers()
        self.apply_rules()
        df, exceptions = self.save_output()
        
        severity_counts = {}
        for e in exceptions:
            sev = e.get('severity', 'UNKNOWN')
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print(f"\n✅ T005 Complete. Exceptions by severity:")
        for severity, count in severity_counts.items():
            print(f"   {severity}: {count}")
        
        return df, exceptions


# ============================================================================
# T006: REVIEW EXCEPTIONS (Automated)
# ============================================================================

class T006_ExceptionReviewer:
    """Task 6: Review and categorize exceptions (automated)"""
    
    def __init__(self, df, exceptions):
        self.df = df.copy()
        self.exceptions = exceptions
        self.critical_exceptions = []
        self.high_exceptions = []
        
    def categorize_exceptions(self):
        """Split exceptions by severity"""
        for e in self.exceptions:
            if e['severity'] == 'CRITICAL':
                self.critical_exceptions.append(e)
            elif e['severity'] == 'HIGH':
                self.high_exceptions.append(e)
        
        print(f"\n📊 T006: Exception Summary")
        print(f"   Critical: {len(self.critical_exceptions)}")
        print(f"   High: {len(self.high_exceptions)}")
        print(f"   Medium/Low: {len(self.exceptions) - len(self.critical_exceptions) - len(self.high_exceptions)}")
        
        return self
    
    def create_review_package(self):
        """Create automated review summary"""
        
        critical_summary = {}
        for e in self.critical_exceptions:
            e_type = e.get('anomaly_type', e.get('rule_id', 'UNKNOWN'))
            if e_type not in critical_summary:
                critical_summary[e_type] = {'count': 0, 'total_amount': 0, 'examples': []}
            
            critical_summary[e_type]['count'] += 1
            critical_summary[e_type]['total_amount'] += e.get('amount', 0)
            
            if len(critical_summary[e_type]['examples']) < 3:
                critical_summary[e_type]['examples'].append({
                    'transaction_id': e['transaction_id'],
                    'amount': e.get('amount', 0),
                    'description': e.get('description', '')
                })
        
        review_data = {
            'timestamp': datetime.now().isoformat(),
            'total_critical': len(self.critical_exceptions),
            'total_high': len(self.high_exceptions),
            'critical_summary': critical_summary,
            'auto_approved': True,
            'note': 'Automated processing - no human review required'
        }
        
        import json
        with open(f"{Config.REPORTS_PATH}Exception_Review_Summary.json", 'w') as f:
            json.dump(review_data, f, indent=2, default=str)
        
        with open(f"{Config.REPORTS_PATH}Exception_Review_Summary.txt", 'w') as f:
            f.write("EXCEPTION REVIEW SUMMARY (Automated)\n")
            f.write("="*50 + "\n\n")
            f.write(f"Review Date: {datetime.now()}\n")
            f.write(f"Status: AUTO-APPROVED\n\n")
            f.write(f"CRITICAL EXCEPTIONS: {len(self.critical_exceptions)}\n")
            for e_type, data in critical_summary.items():
                f.write(f"  • {e_type}: {data['count']} occurrences, ${data['total_amount']:,.2f}\n")
            f.write(f"\nHIGH EXCEPTIONS: {len(self.high_exceptions)}\n")
        
        return review_data
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T006: Reviewing High Severity Exceptions")
        print("="*60)
        print("   ⚡ Automated mode - no human review required")
        
        self.categorize_exceptions()
        review_data = self.create_review_package()
        
        print(f"\n✅ T006 Complete. Proceeding with pipeline.")
        
        return self.df, review_data


# ============================================================================
# T007: COMPUTE BUDGET VARIANCE
# ============================================================================

class T007_BudgetVariance:
    """Task 7: Calculate actual vs budget variance"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.budget_data = None
        self.variance_results = {}
        
    def load_budget(self):
        """Load budget data"""
        print("\n📂 T007: Loading budget data...")
        
        try:
            self.budget_data = pd.read_csv(f"{Config.BUDGET_PATH}Budget_2026.csv")
            print(f"   Loaded budget data with {len(self.budget_data)} rows")
            
            self.budget_data.columns = [col.lower().strip() for col in self.budget_data.columns]
            
            period_col = None
            for col in ['fiscal_period', 'period', 'month', 'reporting_period']:
                if col in self.budget_data.columns:
                    period_col = col
                    break
            
            if period_col:
                self.budget_data.rename(columns={period_col: 'period'}, inplace=True)
            else:
                self.budget_data['period'] = Config.CURRENT_FISCAL_PERIOD
            
            account_col = None
            for col in ['account_code', 'account', 'gl_account', 'coa']:
                if col in self.budget_data.columns:
                    account_col = col
                    break
            
            if account_col:
                self.budget_data.rename(columns={account_col: 'account_code'}, inplace=True)
            
            budget_col = None
            for col in ['budget_amount_aud', 'budget_amount', 'budget', 'amount', 'planned_amount']:
                if col in self.budget_data.columns:
                    budget_col = col
                    break
            
            if budget_col:
                self.budget_data.rename(columns={budget_col: 'budget_amount'}, inplace=True)
                self.budget_data['budget_amount'] = pd.to_numeric(
                    self.budget_data['budget_amount'].astype(str).str.replace('$', '').str.replace(',', ''),
                    errors='coerce'
                )
            else:
                self.budget_data['budget_amount'] = 100000
            
            self.budget_data['period'] = self.budget_data['period'].astype(str)
            self.budget_data['account_code'] = self.budget_data['account_code'].astype(str)
            self.budget_data['budget_amount'] = self.budget_data['budget_amount'].replace(0, 0.01)
            
        except Exception as e:
            print(f"   ⚠️ Budget data not found: {e}")
            accounts = self.df['account_code_mapped'].dropna().unique() if 'account_code_mapped' in self.df.columns else ['5000']
            
            budget_rows = []
            for account in accounts[:30]:
                budget_rows.append({
                    'account_code': str(account),
                    'period': Config.CURRENT_FISCAL_PERIOD,
                    'budget_amount': 100000
                })
            
            self.budget_data = pd.DataFrame(budget_rows)
        
        return self
    
    def calculate_variance(self):
        """Calculate variance"""
        current_period_df = self.df[
            (self.df['fiscal_period'] == Config.CURRENT_FISCAL_PERIOD) &
            (self.df['amount_aud'].notna())
        ].copy()
        
        account_actuals = current_period_df.groupby('account_code_mapped').agg({
            'amount_aud': 'sum',
            'transaction_id': 'count'
        }).rename(columns={
            'amount_aud': 'actual_amount',
            'transaction_id': 'transaction_count'
        }).reset_index()
        
        account_actuals['account_code_mapped'] = account_actuals['account_code_mapped'].astype(str)
        
        feb_budget = self.budget_data[self.budget_data['period'] == Config.CURRENT_FISCAL_PERIOD].copy()
        
        if feb_budget.empty:
            feb_budget = self.budget_data.copy()
        
        feb_budget['account_code'] = feb_budget['account_code'].astype(str)
        
        if not account_actuals.empty and not feb_budget.empty:
            account_variance = pd.merge(
                account_actuals,
                feb_budget[['account_code', 'budget_amount']],
                left_on='account_code_mapped',
                right_on='account_code',
                how='outer'
            )
            
            account_variance['budget_amount'] = account_variance['budget_amount'].fillna(0.01)
            account_variance['actual_amount'] = account_variance['actual_amount'].fillna(0)
            account_variance['variance'] = account_variance['actual_amount'] - account_variance['budget_amount']
            
            def safe_variance_pct(row):
                if row['budget_amount'] > 0:
                    return (row['variance'] / row['budget_amount']) * 100
                elif row['actual_amount'] > 0:
                    return 999999
                else:
                    return 0
            
            account_variance['variance_pct'] = account_variance.apply(safe_variance_pct, axis=1)
            
            account_variance = account_variance.drop(columns=['account_code'], errors='ignore')
            account_variance = account_variance.rename(columns={'account_code_mapped': 'account_code'})
        else:
            account_variance = pd.DataFrame()
        
        suspense_amount = current_period_df[
            current_period_df['account_code_mapped'].isna()
        ]['amount_aud'].sum()
        
        current_date = datetime(Config.CURRENT_YEAR, Config.CURRENT_MONTH, 28)
        future_amount = current_period_df[
            current_period_df['posting_date'] > current_date
        ]['amount_aud'].sum()
        
        total_actual = current_period_df['amount_aud'].sum()
        total_budget = feb_budget['budget_amount'].sum() if not feb_budget.empty else 0.01
        
        total_variance = total_actual - total_budget
        if total_budget > 0:
            total_variance_pct = (total_variance / total_budget) * 100
        elif total_actual > 0:
            total_variance_pct = 999999
        else:
            total_variance_pct = 0
        
        self.variance_results = {
            'by_account': account_variance.to_dict('records') if not account_variance.empty else [],
            'suspense_amount': suspense_amount,
            'future_dated_amount': future_amount,
            'total_actual': total_actual,
            'total_budget': total_budget,
            'total_variance': total_variance,
            'total_variance_pct': total_variance_pct,
            'transaction_count': len(current_period_df),
            'exception_count': current_period_df['has_exception'].sum() if 'has_exception' in current_period_df.columns else 0
        }
        
        print(f"\n   Variance Summary:")
        print(f"   Total Actual: ${total_actual:,.2f}")
        print(f"   Total Budget: ${total_budget:,.2f}")
        print(f"   Variance: ${total_variance:,.2f} ({total_variance_pct:.1f}%)")
        print(f"   Suspense: ${suspense_amount:,.2f}")
        print(f"   Future dated: ${future_amount:,.2f}")
        
        return self
    
    def save_output(self):
        """Save variance results"""
        if self.variance_results['by_account']:
            pd.DataFrame(self.variance_results['by_account']).to_csv(
                f"{Config.REPORTS_PATH}Budget_Variance_By_Account.csv", index=False
            )
        
        summary_df = pd.DataFrame([{
            'metric': 'Total Actual',
            'value': self.variance_results['total_actual']
        }, {
            'metric': 'Total Budget',
            'value': self.variance_results['total_budget']
        }, {
            'metric': 'Variance',
            'value': self.variance_results['total_variance']
        }, {
            'metric': 'Variance %',
            'value': self.variance_results['total_variance_pct']
        }, {
            'metric': 'Suspense Amount',
            'value': self.variance_results['suspense_amount']
        }, {
            'metric': 'Future Dated Amount',
            'value': self.variance_results['future_dated_amount']
        }, {
            'metric': 'Transaction Count',
            'value': self.variance_results['transaction_count']
        }, {
            'metric': 'Exception Count',
            'value': self.variance_results['exception_count']
        }])
        
        summary_df.to_csv(f"{Config.REPORTS_PATH}Budget_Variance_Summary.csv", index=False)
        
        return self.variance_results
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T007: Computing Budget Variance")
        print("="*60)
        
        self.load_budget()
        self.calculate_variance()
        results = self.save_output()
        
        print(f"\n✅ T007 Complete.")
        return results


# ============================================================================
# T008: CLOSE PACK REPORT (Simplified)
# ============================================================================

class T008_ClosePackReport:
    """Task 8: Generate close pack report"""
    
    def __init__(self, df, variance_results, exceptions):
        self.df = df
        self.variance = variance_results
        self.exceptions = exceptions
        self.report_data = {}
        
    def generate_report(self):
        """Generate report data"""
        
        # Currency summary
        currency_summary = self.df.groupby('currency_code').agg({
            'amount_aud': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        currency_summary.columns = ['currency_code', 'amount_aud', 'transaction_count']
        
        # Entity summary
        entity_summary = self.df.groupby('entity_code').agg({
            'amount_aud': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        entity_summary.columns = ['entity_code', 'amount_aud', 'transaction_count']
        
        # Exception summary by type
        exception_types = {}
        for e in self.exceptions:
            e_type = e.get('anomaly_type', e.get('rule_id', 'UNKNOWN'))
            if e_type not in exception_types:
                exception_types[e_type] = 0
            exception_types[e_type] += 1
        
        self.report_data = {
            'generated_at': datetime.now().isoformat(),
            'fiscal_period': Config.CURRENT_FISCAL_PERIOD,
            'total_transactions': len(self.df),
            'total_spend': self.variance['total_actual'],
            'budget_variance': self.variance['total_variance'],
            'variance_percentage': self.variance['total_variance_pct'],
            'total_exceptions': len(self.exceptions),
            'critical_exceptions': len([e for e in self.exceptions if e.get('severity') == 'CRITICAL']),
            'high_exceptions': len([e for e in self.exceptions if e.get('severity') == 'HIGH']),
            'suspense_amount': self.variance['suspense_amount'],
            'future_dated_amount': self.variance['future_dated_amount'],
            'currency_summary': currency_summary.to_dict('records'),
            'entity_summary': entity_summary.to_dict('records'),
            'exception_types': exception_types
        }
        
        return self
    
    def save_report(self):
        """Save report"""
        # Save as JSON
        with open(f"{Config.REPORTS_PATH}Close_Pack_Report_{Config.CURRENT_FISCAL_PERIOD.replace('-', '')}.json", 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        # Create simple Excel
        try:
            with pd.ExcelWriter(f"{Config.REPORTS_PATH}Close_Pack_Report_{Config.CURRENT_FISCAL_PERIOD.replace('-', '')}.xlsx", engine='openpyxl') as writer:
                pd.DataFrame([{
                    'Metric': 'Total Spend',
                    'Value': self.variance['total_actual']
                }]).to_excel(writer, sheet_name='Summary', index=False)
                
                if 'by_account' in self.variance and self.variance['by_account']:
                    pd.DataFrame(self.variance['by_account']).to_excel(writer, sheet_name='Variance by Account', index=False)
                
                if self.exceptions:
                    pd.DataFrame(self.exceptions[:1000]).to_excel(writer, sheet_name='Exceptions', index=False)
        except:
            pass
        
        return self.report_data
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T008: Generating Close Pack Report")
        print("="*60)
        
        self.generate_report()
        report_data = self.save_report()
        
        print(f"\n✅ T008 Complete.")
        return report_data


# ============================================================================
# T009: EXECUTIVE NARRATIVE
# ============================================================================

class T009_ExecutiveNarrative:
    """Task 9: Create natural language summary"""
    
    def __init__(self, variance_results, report_data, exceptions):
        self.variance = variance_results
        self.report = report_data
        self.exceptions = exceptions
        self.narrative = ""
        
    def generate_narrative(self):
        """Generate narrative using templates"""
        lines = []
        
        lines.append("="*80)
        lines.append(f"EXECUTIVE NARRATIVE - {Config.CURRENT_FISCAL_PERIOD}")
        lines.append("="*80)
        lines.append("")
        
        variance_pct = self.variance['total_variance_pct']
        if abs(variance_pct) < 2:
            variance_desc = "in line with"
        elif variance_pct > 0:
            if variance_pct > 10:
                variance_desc = "significantly above"
            else:
                variance_desc = "moderately above"
        else:
            if variance_pct < -10:
                variance_desc = "significantly below"
            else:
                variance_desc = "moderately below"
        
        lines.append(f"Total spend for {Config.CURRENT_FISCAL_PERIOD} was ${self.variance['total_actual']:,.2f}, "
                    f"which is {variance_desc} budget of ${self.variance['total_budget']:,.2f}. "
                    f"The variance is ${abs(self.variance['total_variance']):,.2f} ({variance_pct:.1f}%).")
        lines.append("")
        
        lines.append("EXCEPTION SUMMARY")
        lines.append("-"*40)
        
        critical_count = len([e for e in self.exceptions if e.get('severity') == 'CRITICAL'])
        high_count = len([e for e in self.exceptions if e.get('severity') == 'HIGH'])
        
        lines.append(f"Total exceptions: {len(self.exceptions)}")
        lines.append(f"  • Critical: {critical_count}")
        lines.append(f"  • High: {high_count}")
        lines.append("")
        
        lines.append("DATA QUALITY IMPACT")
        lines.append("-"*40)
        
        suspense_amount = self.variance.get('suspense_amount', 0)
        future_amount = self.variance.get('future_dated_amount', 0)
        
        lines.append(f"Suspense (invalid accounts): ${suspense_amount:,.2f}")
        lines.append(f"Future-dated transactions: ${future_amount:,.2f}")
        lines.append("")
        
        lines.append("RECOMMENDATIONS")
        lines.append("-"*40)
        
        if suspense_amount > 10000:
            lines.append("• Review and remap transactions with invalid account codes")
        if future_amount > 10000:
            lines.append("• Reclassify future-dated transactions to correct period")
        if critical_count > 0:
            lines.append("• Investigate critical exceptions before next close")
        
        self.narrative = "\n".join(lines)
        
        return self
    
    def save_narrative(self):
        """Save narrative to file"""
        filename = f"{Config.REPORTS_PATH}Executive_Narrative_{Config.CURRENT_FISCAL_PERIOD.replace('-', '')}.txt"
        with open(filename, 'w') as f:
            f.write(self.narrative)
        
        return self.narrative
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T009: Generating Executive Narrative")
        print("="*60)
        
        self.generate_narrative()
        narrative = self.save_narrative()
        
        print(f"\n✅ T009 Complete.")
        return narrative


# ============================================================================
# T010: FORECAST NEXT PERIOD
# ============================================================================

class T010_Forecast:
    """Task 10: Generate forecast for next period"""
    
    def __init__(self, df, variance_results):
        self.df = df
        self.variance = variance_results
        self.historical_data = None
        self.forecast = {}
        
    def load_historical(self):
        """Load historical KPI data"""
        print("\n📂 T010: Loading historical data...")
        
        try:
            self.historical_data = pd.read_csv(f"{Config.REFERENCE_PATH}KPI_Monthly_History.csv")
            print(f"   Loaded {len(self.historical_data)} rows of historical data")
            
            self.historical_data.columns = [col.lower().strip() for col in self.historical_data.columns]
            
            period_col = None
            for col in ['period', 'month', 'fiscal_period', 'reporting_period']:
                if col in self.historical_data.columns:
                    period_col = col
                    break
            
            if period_col:
                if period_col != 'period':
                    self.historical_data.rename(columns={period_col: 'period'}, inplace=True)
            else:
                self.historical_data['period'] = [f"2025-{i:02d}" for i in range(1, len(self.historical_data) + 1)]
            
            spend_col = None
            for col in ['total_spend', 'spend', 'amount', 'actual', 'value']:
                if col in self.historical_data.columns:
                    spend_col = col
                    break
            
            if spend_col:
                if spend_col != 'total_spend':
                    self.historical_data.rename(columns={spend_col: 'total_spend'}, inplace=True)
            else:
                base_spend = self.variance.get('total_actual', 1000000)
                self.historical_data['total_spend'] = [
                    base_spend * (0.8 + 0.4 * np.random.random()) 
                    for _ in range(len(self.historical_data))
                ]
            
        except Exception as e:
            print(f"   ⚠️ Historical data not found: {e}")
            months = []
            base_spend = self.variance.get('total_actual', 1000000)
            
            for i in range(1, 13):
                month_num = Config.CURRENT_MONTH - (12 - i)
                year = Config.CURRENT_YEAR
                if month_num <= 0:
                    month_num += 12
                    year -= 1
                
                month = f"{year}-{month_num:02d}"
                months.append({
                    'period': month,
                    'total_spend': base_spend * (0.8 + 0.4 * np.random.random()),
                })
            self.historical_data = pd.DataFrame(months)
        
        self.historical_data['period'] = self.historical_data['period'].astype(str)
        return self
    
    def calculate_trends(self):
        """Calculate trends from historical data"""
        
        try:
            self.historical_data = self.historical_data.sort_values('period')
        except:
            pass
        
        if len(self.historical_data) >= 3:
            self.historical_data['spend_ma_3'] = self.historical_data['total_spend'].rolling(3, min_periods=1).mean()
        else:
            self.historical_data['spend_ma_3'] = self.historical_data['total_spend']
        
        recent_data = self.historical_data.tail(min(3, len(self.historical_data)))
        recent_avg = recent_data['total_spend'].mean()
        
        if Config.CURRENT_MONTH < 12:
            next_period = f"{Config.CURRENT_YEAR}-{Config.CURRENT_MONTH+1:02d}"
        else:
            next_period = f"{Config.CURRENT_YEAR+1}-01"
        
        current_actual = self.variance.get('total_actual', recent_avg)
        blended_forecast = 0.7 * recent_avg + 0.3 * current_actual * 1.05
        
        std_dev = self.historical_data['total_spend'].std() if len(self.historical_data) > 1 else blended_forecast * 0.1
        margin = 1.96 * std_dev / np.sqrt(len(self.historical_data)) if len(self.historical_data) > 1 else blended_forecast * 0.2
        
        lower_bound = max(0, blended_forecast - margin)
        upper_bound = blended_forecast + margin
        
        self.forecast = {
            'next_period': next_period,
            'forecast_amount': blended_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': 0.95,
            'method': 'Blended (70% trend, 30% current)',
            'historical_months_used': len(self.historical_data),
            'current_actual': current_actual,
            'recent_avg': recent_avg
        }
        
        print(f"\n   Forecast for {next_period}:")
        print(f"   • Point forecast: ${self.forecast['forecast_amount']:,.2f}")
        print(f"   • 95% CI: (${self.forecast['lower_bound']:,.2f} - ${self.forecast['upper_bound']:,.2f})")
        
        return self
    
    def save_forecast(self):
        """Save forecast results"""
        forecast_df = pd.DataFrame([self.forecast])
        forecast_df.to_csv(f"{Config.REPORTS_PATH}Forecast_{self.forecast['next_period'].replace('-', '')}.csv", index=False)
        
        return self.forecast
    
    def run(self):
        print("\n" + "="*60)
        print("🚀 T010: Forecasting Next Period")
        print("="*60)
        
        self.load_historical()
        self.calculate_trends()
        forecast = self.save_forecast()
        
        print(f"\n✅ T010 Complete.")
        return forecast


# ============================================================================
# DATA VALIDATOR
# ============================================================================

class DataValidator:
    """Validate that all required data files exist"""
    
    @staticmethod
    def validate_all():
        """Run all validations"""
        issues = []
        
        required_files = {
            f"{Config.MASTER_DATA_PATH}Master_COA.csv": "Chart of Accounts",
            f"{Config.MASTER_DATA_PATH}Master_Entity.csv": "Entity Master",
            f"{Config.MASTER_DATA_PATH}Master_CostCenters.csv": "Cost Center Master",
            f"{Config.BUDGET_PATH}Budget_2026.csv": "Budget Data",
            Config.RAW_DATA_PATH: "Raw GL Export"
        }
        
        print("\n📊 DATA VALIDATION")
        print("-" * 40)
        
        for filepath, description in required_files.items():
            if not os.path.exists(filepath):
                issues.append(f"❌ Missing {description}: {filepath}")
            else:
                try:
                    df = pd.read_csv(filepath)
                    print(f"✅ {description}: {len(df)} rows")
                except Exception as e:
                    issues.append(f"❌ Cannot read {description}: {e}")
        
        if issues:
            print("\n⚠️ DATA VALIDATION ISSUES FOUND:")
            for issue in issues:
                print(issue)
            print("\n✅ Pipeline will continue but may use synthetic data where needed.\n")
            return False
        else:
            print("\n✅ All master data files validated successfully.\n")
            return True


# ============================================================================
# MAIN PIPELINE EXECUTOR
# ============================================================================

class FinancialCloseAgent:
    """Main agent orchestrating all tasks"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def run_pipeline(self):
        """Execute all tasks in sequence"""
        print("\n" + "="*80)
        print("🚀 FINANCIAL CLOSE AGENT PIPELINE")
        print(f"   Started: {self.start_time}")
        print("="*80 + "\n")

        # Validate data files
        validator = DataValidator()
        validator.validate_all()
        
        # Task 001: Wrangle Raw Data
        wrangler = T001_DataWrangler()
        df, anomalies = wrangler.run(Config.RAW_DATA_PATH)
        self.results['df_t001'] = df
        self.results['anomalies'] = anomalies
        
        # Task 002: Map Entities and Accounts
        mapper = T002_EntityAccountMapper(df)
        df = mapper.run()
        self.results['df_t002'] = df
        
        # Task 003: Resolve Vendors
        resolver = T003_VendorResolver(df)
        df = resolver.run()
        self.results['df_t003'] = df
        
        # Task 004: FX Conversion
        converter = T004_FXConverter(df)
        df = converter.run()
        self.results['df_t004'] = df
        
        # Task 005: Detect Exceptions
        detector = T005_ExceptionDetector(df)
        df, exceptions = detector.run()
        self.results['df_t005'] = df
        self.results['exceptions'] = exceptions
        
        # Task 006: Review Exceptions (Automated)
        reviewer = T006_ExceptionReviewer(df, exceptions)
        df, review = reviewer.run()
        self.results['df_t006'] = df
        self.results['review'] = review
        
        # Task 007: Budget Variance
        variance = T007_BudgetVariance(df)
        variance_results = variance.run()
        self.results['variance'] = variance_results
        
        # Task 008: Close Pack Report
        report = T008_ClosePackReport(df, variance_results, exceptions)
        report_data = report.run()
        self.results['report'] = report_data
        
        # Task 009: Executive Narrative
        narrative = T009_ExecutiveNarrative(variance_results, report_data, exceptions)
        narrative_text = narrative.run()
        self.results['narrative'] = narrative_text
        
        # Task 010: Forecast
        forecast = T010_Forecast(df, variance_results)
        forecast_data = forecast.run()
        self.results['forecast'] = forecast_data
        
        # Completion
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE")
        print(f"   Finished: {end_time}")
        print(f"   Duration: {duration:.2f} seconds")
        print("="*80)
        
        return self.results


# ============================================================================
# API ENDPOINTS - SIMPLIFIED
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "service": "Financial Close Agent API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Single endpoint triggers complete financial close pipeline",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    files_status = {}
    required_paths = [
        Config.RAW_DATA_PATH,
        f"{Config.MASTER_DATA_PATH}Master_Entity.csv",
        f"{Config.MASTER_DATA_PATH}Master_COA.csv",
        f"{Config.MASTER_DATA_PATH}Master_CostCenters.csv",
        f"{Config.MASTER_DATA_PATH}Master_Vendors.csv",
        f"{Config.MASTER_DATA_PATH}Vendor_Alias_Map.csv",
        f"{Config.REFERENCE_PATH}FX_Rates.csv",
        f"{Config.REFERENCE_PATH}Exception_Rulebook.csv",
        f"{Config.REFERENCE_PATH}KPI_Monthly_History.csv",
        f"{Config.BUDGET_PATH}Budget_2026.csv"
    ]
    
    for path in required_paths:
        files_status[path] = os.path.exists(path)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "directories": {
            "master_data": Config.MASTER_DATA_PATH,
            "reference": Config.REFERENCE_PATH,
            "budget": Config.BUDGET_PATH,
            "working": Config.OUTPUT_PATH,
            "reports": Config.REPORTS_PATH
        },
        "files_exist": files_status,
        "fiscal_period": Config.CURRENT_FISCAL_PERIOD
    }

@app.post("/pipeline/run", response_model=PipelineResponse, tags=["Pipeline"])
async def run_pipeline(
    background_tasks: BackgroundTasks,
    validate_first: bool = True,
    fiscal_period: str = "2026-02"
):
    """
    Execute the complete financial close pipeline.
    
    Reads from local files:
    - Raw_GL_Export.csv
    - Master_Data/*.csv
    - Reference/*.csv
    - Budget/Budget_2026.csv
    
    Writes to:
    - working/*.csv
    - reports/*.csv, *.json, *.txt, *.xlsx
    
    Returns a run_id for status checking.
    """
    # Update fiscal period if provided
    if fiscal_period != Config.CURRENT_FISCAL_PERIOD:
        Config.CURRENT_FISCAL_PERIOD = fiscal_period
        year, month = fiscal_period.split('-')
        Config.CURRENT_YEAR = int(year)
        Config.CURRENT_MONTH = int(month)
    
    run_id = generate_run_id()
    pipeline_manager.create_run(run_id)
    
    def execute_pipeline():
        try:
            pipeline_manager.update_run(run_id, status="running", progress=5, current_task="validation")
            
            # Optional validation
            if validate_first:
                validator = DataValidator()
                validator.validate_all()
            
            pipeline_manager.update_run(run_id, progress=10, current_task="T001")
            
            # Create and run the agent
            agent = FinancialCloseAgent()
            results = agent.run_pipeline()
            
            # Extract summary for response
            summary = {
                "total_transactions": len(results.get('df_t001', pd.DataFrame())),
                "total_exceptions": len(results.get('exceptions', [])),
                "critical_exceptions": len([e for e in results.get('exceptions', []) 
                                           if e.get('severity') == 'CRITICAL']),
                "total_spend": results.get('variance', {}).get('total_actual', 0),
                "budget_variance": results.get('variance', {}).get('total_variance', 0),
                "variance_percentage": results.get('variance', {}).get('total_variance_pct', 0),
                "forecast_next_period": results.get('forecast', {}).get('forecast_amount', 0)
            }
            
            pipeline_manager.complete_run(run_id, summary)
            
        except Exception as e:
            pipeline_manager.fail_run(run_id, str(e))
    
    background_tasks.add_task(execute_pipeline)
    
    return PipelineResponse(
        run_id=run_id,
        status="started",
        message="Pipeline execution started",
        start_time=datetime.now().isoformat(),
        summary=None
    )

@app.get("/pipeline/status/{run_id}", response_model=PipelineStatus, tags=["Pipeline"])
async def get_pipeline_status(run_id: str):
    """
    Check the status of a pipeline run.
    
    Returns progress, current task, and final results when complete.
    """
    run = pipeline_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found")
    
    return PipelineStatus(**run)

@app.post("/pipeline/validate", tags=["Pipeline"])
async def validate_files():
    """
    Validate that all required input files exist and are readable.
    Does not execute the pipeline.
    """
    issues = []
    files_checked = []
    
    required_files = {
        f"{Config.MASTER_DATA_PATH}Master_COA.csv": "Chart of Accounts",
        f"{Config.MASTER_DATA_PATH}Master_Entity.csv": "Entity Master",
        f"{Config.MASTER_DATA_PATH}Master_CostCenters.csv": "Cost Center Master",
        f"{Config.MASTER_DATA_PATH}Master_Vendors.csv": "Vendor Master",
        f"{Config.MASTER_DATA_PATH}Vendor_Alias_Map.csv": "Vendor Alias Map",
        f"{Config.REFERENCE_PATH}FX_Rates.csv": "FX Rates",
        f"{Config.REFERENCE_PATH}Exception_Rulebook.csv": "Exception Rulebook",
        f"{Config.REFERENCE_PATH}KPI_Monthly_History.csv": "KPI History",
        f"{Config.BUDGET_PATH}Budget_2026.csv": "Budget Data",
        Config.RAW_DATA_PATH: "Raw GL Export"
    }
    
    for filepath, description in required_files.items():
        file_info = {
            "path": filepath,
            "exists": os.path.exists(filepath),
            "readable": False,
            "row_count": None
        }
        
        if file_info["exists"]:
            try:
                df = pd.read_csv(filepath)
                file_info["readable"] = True
                file_info["row_count"] = len(df)
            except:
                issues.append(f"Cannot read {description}: {filepath}")
        else:
            issues.append(f"Missing {description}: {filepath}")
        
        files_checked.append(file_info)
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "files_checked": files_checked,
        "fiscal_period": Config.CURRENT_FISCAL_PERIOD
    }

from fastapi.responses import FileResponse, JSONResponse
import json
import pandas as pd
from pathlib import Path

# ============================================================================
# DATA ACCESS ENDPOINTS FOR WATSONX ORCHESTRATE
# ============================================================================

@app.get("/data/summary", tags=["Watsonx"])
async def get_summary_data():
    """Get summary data in JSON format for Watsonx Orchestrate"""
    try:
        # Read the variance summary
        variance_path = Path(Config.REPORTS_PATH) / "Budget_Variance_Summary.csv"
        if not variance_path.exists():
            return {"error": "No summary data available"}
        
        df = pd.read_csv(variance_path)
        summary = df.to_dict('records')
        
        # Read exceptions count
        exceptions_path = Path(Config.REPORTS_PATH) / "Exceptions_Detailed.csv"
        exception_count = len(pd.read_csv(exceptions_path)) if exceptions_path.exists() else 0
        
        # Read narrative
        narrative_path = Path(Config.REPORTS_PATH) / f"Executive_Narrative_{Config.CURRENT_FISCAL_PERIOD.replace('-', '')}.txt"
        narrative = ""
        if narrative_path.exists():
            with open(narrative_path, 'r') as f:
                narrative = f.read()
        
        # Read forecast
        forecast_path = Path(Config.REPORTS_PATH) / f"Forecast_{Config.CURRENT_YEAR}{Config.CURRENT_MONTH+1:02d}.csv"
        forecast = []
        if forecast_path.exists():
            forecast_df = pd.read_csv(forecast_path)
            forecast = forecast_df.to_dict('records')
        
        return JSONResponse({
            "fiscal_period": Config.CURRENT_FISCAL_PERIOD,
            "summary": summary,
            "exception_count": exception_count,
            "narrative": narrative[:500] + "..." if len(narrative) > 500 else narrative,
            "forecast": forecast[0] if forecast else None,
            "generated_at": datetime.now().isoformat()
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/data/variance", tags=["Watsonx"])
async def get_variance_data(format: str = "json"):
    """Get budget variance data"""
    file_path = Path(Config.REPORTS_PATH) / "Budget_Variance_By_Account.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Variance data not found")
    
    if format == "csv":
        return FileResponse(path=file_path, filename="budget_variance.csv")
    else:
        df = pd.read_csv(file_path)
        return JSONResponse(df.to_dict('records'))

@app.get("/data/exceptions", tags=["Watsonx"])
async def get_exceptions_data(severity: Optional[str] = None, limit: int = 100):
    """Get exceptions data with optional filtering"""
    file_path = Path(Config.REPORTS_PATH) / "Exceptions_Detailed.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Exceptions data not found")
    
    df = pd.read_csv(file_path)
    
    if severity:
        df = df[df['severity'] == severity.upper()]
    
    # Limit results
    df = df.head(limit)
    
    return JSONResponse(df.to_dict('records'))

@app.get("/data/narrative", tags=["Watsonx"])
async def get_narrative():
    """Get the executive narrative"""
    file_path = Path(Config.REPORTS_PATH) / f"Executive_Narrative_{Config.CURRENT_FISCAL_PERIOD.replace('-', '')}.txt"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Narrative not found")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    return JSONResponse({
        "fiscal_period": Config.CURRENT_FISCAL_PERIOD,
        "narrative": content,
        "length": len(content)
    })

@app.get("/data/forecast", tags=["Watsonx"])
async def get_forecast():
    """Get forecast data"""
    file_path = Path(Config.REPORTS_PATH) / f"Forecast_{Config.CURRENT_YEAR}{Config.CURRENT_MONTH+1:02d}.csv"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Forecast not found")
    
    df = pd.read_csv(file_path)
    return JSONResponse(df.to_dict('records')[0])

@app.get("/data/latest-run", tags=["Watsonx"])
async def get_latest_run():
    """Get information about the latest pipeline run"""
    runs = pipeline_manager.runs
    if not runs:
        return JSONResponse({"status": "no_runs"})
    
    latest_run_id = max(runs.keys(), key=lambda k: runs[k]["start_time"])
    latest_run = runs[latest_run_id]
    
    return JSONResponse({
        "run_id": latest_run_id,
        "status": latest_run["status"],
        "start_time": latest_run["start_time"],
        "end_time": latest_run["end_time"],
        "duration_seconds": latest_run["duration_seconds"],
        "summary": latest_run["summary"],
        "fiscal_period": Config.CURRENT_FISCAL_PERIOD
    })

@app.get("/data/files", tags=["Watsonx"])
async def list_available_data():
    """List all available data files for Watsonx to consume"""
    files = []
    
    # List all CSV files in reports
    for file_path in Path(Config.REPORTS_PATH).glob("*.csv"):
        files.append({
            "name": file_path.name,
            "type": "csv",
            "size": file_path.stat().st_size,
            "endpoint": f"/data/{file_path.stem.lower().replace('_', '-')}",
            "generated": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        })
    
    # List all text files
    for file_path in Path(Config.REPORTS_PATH).glob("*.txt"):
        files.append({
            "name": file_path.name,
            "type": "text",
            "size": file_path.stat().st_size,
            "endpoint": f"/data/{file_path.stem.lower().replace('_', '-')}",
            "generated": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        })
    
    return JSONResponse({
        "base_url": "https://finance-usecase.onrender.com",
        "fiscal_period": Config.CURRENT_FISCAL_PERIOD,
        "available_data": files,
        "endpoints": {
            "summary": "/data/summary",
            "variance": "/data/variance",
            "exceptions": "/data/exceptions",
            "narrative": "/data/narrative",
            "forecast": "/data/forecast",
            "latest_run": "/data/latest-run"
        }
    })


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 FINANCIAL CLOSE AGENT API")
    print("="*60)
    print(f"Master Data:   {Config.MASTER_DATA_PATH}")
    print(f"Reference:     {Config.REFERENCE_PATH}")
    print(f"Budget:        {Config.BUDGET_PATH}")
    print(f"Working:       {Config.OUTPUT_PATH}")
    print(f"Reports:       {Config.REPORTS_PATH}")
    print("="*60)
    print("Endpoint: POST /pipeline/run - triggers complete pipeline")
    print("Endpoint: GET  /pipeline/status/{run_id} - check progress")
    print("Endpoint: POST /pipeline/validate - validate files only")
    print("="*60 + "\n")
    
    uvicorn.run(
        "financial_close_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
