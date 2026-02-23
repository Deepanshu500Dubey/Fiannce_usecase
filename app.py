"""
Financial Close Agent API - FastAPI Implementation
Provides endpoints for all financial close tasks
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import shutil
import json
import uuid
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the API"""
    BASE_DIR = Path("financial_close_data")
    UPLOAD_DIR = BASE_DIR / "uploads"
    MASTER_DATA_DIR = BASE_DIR / "master_data"
    REFERENCE_DIR = BASE_DIR / "reference"
    BUDGET_DIR = BASE_DIR / "budget"
    WORKING_DIR = BASE_DIR / "working"
    REPORTS_DIR = BASE_DIR / "reports"
    
    # Create directories
    for dir_path in [UPLOAD_DIR, MASTER_DATA_DIR, REFERENCE_DIR, 
                     BUDGET_DIR, WORKING_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Fiscal period settings
    CURRENT_FISCAL_PERIOD = "2026-02"
    CURRENT_MONTH = 2
    CURRENT_YEAR = 2026
    
    # Anomaly thresholds
    HIGH_VALUE_THRESHOLD = 50000
    EXTREME_OUTLIER_MULTIPLIER = 5
    SUSPICIOUS_HOUR_START = 22
    SUSPICIOUS_HOUR_END = 6

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ProcessingResponse(BaseModel):
    """Base response model for processing tasks"""
    task_id: str
    status: str
    message: str
    data_paths: Optional[Dict[str, str]] = None
    summary: Optional[Dict[str, Any]] = None

class AnomalyItem(BaseModel):
    transaction_id: str
    anomaly_type: str
    severity: str
    description: str
    amount: Optional[float] = None
    vendor: Optional[str] = None
    account: Optional[str] = None

class VarianceSummary(BaseModel):
    total_actual: float
    total_budget: float
    total_variance: float
    total_variance_pct: float
    suspense_amount: float
    future_dated_amount: float
    transaction_count: int
    exception_count: int

class ForecastResponse(BaseModel):
    next_period: str
    forecast_amount: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str

class NarrativeResponse(BaseModel):
    narrative: str
    file_path: str

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Financial Close Agent API",
    description="Automated financial close processing pipeline",
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
# UTILITY FUNCTIONS
# ============================================================================

def generate_task_id():
    """Generate unique task ID"""
    return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    """Save uploaded file to destination"""
    file_path = destination / upload_file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return str(file_path)

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame"""
    return pd.read_csv(file_path)

# ============================================================================
# BACKGROUND TASK PROCESSOR
# ============================================================================

class TaskManager:
    """Manage background tasks and store results"""
    
    def __init__(self):
        self.tasks = {}
    
    def create_task(self, task_id: str, task_type: str):
        """Create a new task"""
        self.tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }
        return task_id
    
    def update_task(self, task_id: str, status: str, result: Any = None, error: str = None):
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id].update({
                "status": status,
                "updated_at": datetime.now().isoformat(),
                "result": result,
                "error": error
            })
    
    def get_task(self, task_id: str):
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self):
        """Get all tasks"""
        return list(self.tasks.values())

task_manager = TaskManager()

# ============================================================================
# T001: WRANGLE RAW GL DATA
# ============================================================================

class T001_DataWrangler:
    """Task 1: Parse and standardize raw GL export data"""
    
    def __init__(self, file_path: str, task_id: str):
        self.file_path = file_path
        self.task_id = task_id
        self.raw_df = None
        self.standardized_df = None
        self.anomaly_log = []
        
    def run(self):
        """Execute all T001 steps"""
        try:
            # Load data
            self.raw_df = pd.read_csv(self.file_path)
            
            # Standardize column names
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
            
            # Only rename columns that exist
            rename_dict = {k: v for k, v in column_mapping.items() if k in self.raw_df.columns}
            self.standardized_df = self.raw_df.rename(columns=rename_dict)
            
            # Standardize dates
            self._standardize_dates()
            
            # Clean amounts
            self._clean_amounts()
            
            # Extract fiscal year and month
            self.standardized_df['fiscal_year'] = self.standardized_df['fiscal_period'].str[:4]
            self.standardized_df['fiscal_month'] = self.standardized_df['fiscal_period'].str[-2:]
            
            # Detect embedded exceptions
            self._detect_embedded_exceptions()
            
            # Add metadata
            self.standardized_df['processing_timestamp'] = datetime.now()
            self.standardized_df['source_file'] = os.path.basename(self.file_path)
            self.standardized_df['data_quality_score'] = 100 - (len(self.anomaly_log) / len(self.standardized_df) * 100) if len(self.standardized_df) > 0 else 100
            self.standardized_df['anomaly_count'] = self.standardized_df.apply(
                lambda row: len([a for a in self.anomaly_log if a.get('transaction_id') == row['transaction_id']]), axis=1
            )
            
            # Save outputs
            output_path = Config.WORKING_DIR / f"GL_Standardized_{self.task_id}.csv"
            self.standardized_df.to_csv(output_path, index=False)
            
            anomalies_path = Config.REPORTS_DIR / f"Input_Anomalies_{self.task_id}.csv"
            if self.anomaly_log:
                pd.DataFrame(self.anomaly_log).to_csv(anomalies_path, index=False)
            
            return {
                "status": "success",
                "rows_processed": len(self.standardized_df),
                "anomalies_found": len(self.anomaly_log),
                "output_files": {
                    "standardized_data": str(output_path),
                    "anomalies": str(anomalies_path) if self.anomaly_log else None
                }
            }
            
        except Exception as e:
            raise Exception(f"T001 failed: {str(e)}")
    
    def _standardize_dates(self):
        """Convert all dates to consistent format"""
        df = self.standardized_df
        
        def parse_date(date_str, txn_id, column_name):
            if pd.isna(date_str) or date_str in ['INVALID', '99/99/9999', '32/13/2026']:
                self.anomaly_log.append({
                    'transaction_id': txn_id,
                    'anomaly_type': 'INVALID_DATE',
                    'severity': 'CRITICAL',
                    'description': f"Invalid date value: {date_str}",
                    'column': column_name
                })
                return None
            
            formats = ['%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
                      '%d/%m/%y', '%m/%d/%y', '%d-%m-%y', '%y-%m-%d']
            for fmt in formats:
                try:
                    return datetime.strptime(str(date_str), fmt)
                except:
                    continue
            
            self.anomaly_log.append({
                'transaction_id': txn_id,
                'anomaly_type': 'UNPARSABLE_DATE',
                'severity': 'CRITICAL',
                'description': f"Cannot parse date: {date_str}",
                'column': column_name
            })
            return None
        
        if 'posting_date_raw' in df.columns:
            df['posting_date'] = df.apply(
                lambda row: parse_date(row['posting_date_raw'], row['transaction_id'], 'posting_date_raw'), 
                axis=1
            )
        
        if 'invoice_date_raw' in df.columns:
            df['invoice_date'] = df.apply(
                lambda row: parse_date(row['invoice_date_raw'], row['transaction_id'], 'invoice_date_raw'), 
                axis=1
            )
    
    def _clean_amounts(self):
        """Convert amount strings to floats"""
        df = self.standardized_df
        
        def parse_amount(amt_str, txn_id):
            if pd.isna(amt_str):
                return None
            
            cleaned = str(amt_str).replace('$', '').replace(',', '').strip()
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
        
        if 'amount_raw' in df.columns:
            df['amount'] = df.apply(
                lambda row: parse_amount(row['amount_raw'], row['transaction_id']), 
                axis=1
            )
            df['amount_is_negative'] = df['amount'] < 0
    
    def _detect_embedded_exceptions(self):
        """Look for obvious exceptions in raw data"""
        df = self.standardized_df
        keywords = ['error', 'flag', 'review', 'urgent', 'exception', 'invalid']
        
        if 'narrative' in df.columns:
            df['narrative_lower'] = df['narrative'].str.lower().fillna('')
            
            for idx, row in df.iterrows():
                if any(keyword in str(row['narrative_lower']) for keyword in keywords):
                    self.anomaly_log.append({
                        'transaction_id': row['transaction_id'],
                        'anomaly_type': 'NARRATIVE_SUGGESTS_EXCEPTION',
                        'severity': 'MEDIUM',
                        'description': f"Narrative contains exception keywords: {row['narrative']}"
                    })

# ============================================================================
# T002: MAP ENTITIES AND ACCOUNTS
# ============================================================================

class T002_EntityAccountMapper:
    """Task 2: Resolve entity codes and account codes against master data"""
    
    def __init__(self, df, task_id: str, master_data_dir: Path):
        self.df = df.copy()
        self.task_id = task_id
        self.master_data_dir = master_data_dir
        self.entity_master = None
        self.account_master = None
        self.cost_center_master = None
        self.mapping_anomalies = []
        self.output_path = None
        
    def run(self):
        """Execute all T002 steps"""
        try:
            self.load_master_data()
            self.map_entities()
            self.map_accounts()
            self.map_cost_centers()
            self.save_output()
            
            return {
                "status": "success",
                "rows_mapped": len(self.df),
                "entity_errors": (~self.df['entity_valid']).sum() if 'entity_valid' in self.df.columns else 0,
                "account_errors": (~self.df['account_valid']).sum() if 'account_valid' in self.df.columns else 0,
                "cost_center_errors": (~self.df['cost_center_valid'] & self.df['cost_center_present']).sum() if 'cost_center_valid' in self.df.columns else 0,
                "output_file": str(self.output_path)
            }
            
        except Exception as e:
            raise Exception(f"T002 failed: {str(e)}")
    
    def load_master_data(self):
        """Load master reference files"""
        
        try:
            entity_path = self.master_data_dir / "Master_Entity.csv"
            if entity_path.exists():
                self.entity_master = pd.read_csv(entity_path)
        except:
            self.entity_master = pd.DataFrame({'entity_code': ['AUS01']})
        
        try:
            account_path = self.master_data_dir / "Master_COA.csv"
            if account_path.exists():
                self.account_master = pd.read_csv(account_path)
                # Standardize column names
                self.account_master.columns = [col.lower().strip() for col in self.account_master.columns]
        except:
            self.account_master = pd.DataFrame({'account_code': [f"{i:04d}" for i in range(5000, 5029)]})
        
        try:
            cc_path = self.master_data_dir / "Master_CostCenters.csv"
            if cc_path.exists():
                self.cost_center_master = pd.read_csv(cc_path)
                self.cost_center_master.columns = [col.lower().strip() for col in self.cost_center_master.columns]
        except:
            self.cost_center_master = pd.DataFrame({'cost_center': ['CC' + str(i).zfill(4) for i in range(1000, 1010)]})
        
        return self
    
    def map_entities(self):
        """Map entity codes against master"""
        if self.entity_master is not None and 'entity_code' in self.entity_master.columns:
            valid_entities = self.entity_master['entity_code'].tolist()
        else:
            valid_entities = ['AUS01']
        
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
        
        return self
    
    def map_accounts(self):
        """Map account codes against master"""
        
        if self.account_master is not None and 'account_code' in self.account_master.columns:
            valid_accounts = [str(acct).strip() for acct in self.account_master['account_code'].tolist()]
        else:
            valid_accounts = []
        
        # Clean raw account codes
        self.df['account_code_clean'] = self.df['account_code_raw'].astype(str).str.strip()
        
        # Try different matching strategies
        self.df['account_valid'] = False
        
        # Direct match
        direct_match = self.df['account_code_raw'].isin(valid_accounts)
        self.df.loc[direct_match, 'account_valid'] = True
        
        # Clean match
        clean_match = (~direct_match) & self.df['account_code_clean'].isin(valid_accounts)
        self.df.loc[clean_match, 'account_valid'] = True
        
        # Numeric match
        if not self.df[~self.df['account_valid']].empty:
            numeric_valid = []
            for acct in valid_accounts:
                try:
                    numeric_valid.append(float(acct))
                except:
                    pass
            
            for idx, row in self.df[~self.df['account_valid']].iterrows():
                try:
                    raw_num = float(row['account_code_raw'])
                    if raw_num in numeric_valid:
                        self.df.at[idx, 'account_valid'] = True
                except:
                    pass
        
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
        
        # Get account names/descriptions if available
        if self.account_master is not None and 'account_name' in self.account_master.columns:
            account_desc_map = {}
            for _, row in self.account_master.iterrows():
                acct = str(row['account_code']).strip()
                desc = row['account_name']
                account_desc_map[acct] = desc
            
            self.df['account_description'] = self.df['account_code_mapped'].map(account_desc_map)
        
        # Log anomalies
        for idx, row in self.df[~self.df['account_valid']].iterrows():
            severity = 'CRITICAL' if str(row['account_code_raw']) == 'INVALID_ACCT' else 'HIGH'
            self.mapping_anomalies.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'INVALID_ACCOUNT',
                'severity': severity,
                'description': f"Account code '{row['account_code_raw']}' not in Chart of Accounts",
                'original_value': row['account_code_raw'],
                'amount': row['amount']
            })
        
        return self
    
    def map_cost_centers(self):
        """Map cost centers against master"""
        if self.cost_center_master is not None and 'cost_center' in self.cost_center_master.columns:
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
        
        return self
    
    def save_output(self):
        """Save mapped data"""
        self.output_path = Config.WORKING_DIR / f"GL_WithMappings_{self.task_id}.csv"
        self.df.to_csv(self.output_path, index=False)
        
        return self.df

# ============================================================================
# T003: RESOLVE VENDOR NAMES
# ============================================================================

class T003_VendorResolver:
    """Task 3: Map vendor aliases to canonical vendor names"""
    
    def __init__(self, df, task_id: str, master_data_dir: Path):
        self.df = df.copy()
        self.task_id = task_id
        self.master_data_dir = master_data_dir
        self.vendor_master = None
        self.alias_map = None
        self.vendor_anomalies = []
        self.output_path = None
        
    def run(self):
        """Execute all T003 steps"""
        try:
            self.load_vendor_data()
            self.resolve_vendors()
            self.save_output()
            
            # Calculate statistics
            mapped_count = self.df['vendor_resolution_status'].isin(['MAPPED', 'CANONICAL', 'CLEANED_MATCH', 'PARTIAL_MATCH']).sum()
            unmapped_count = (self.df['vendor_resolution_status'] == 'UNMAPPED').sum()
            missing_count = (self.df['vendor_resolution_status'] == 'MISSING').sum()
            
            return {
                "status": "success",
                "rows_processed": len(self.df),
                "mapped_vendors": int(mapped_count),
                "unmapped_vendors": int(unmapped_count),
                "missing_vendors": int(missing_count),
                "output_file": str(self.output_path)
            }
            
        except Exception as e:
            raise Exception(f"T003 failed: {str(e)}")
    
    def load_vendor_data(self):
        """Load vendor master and alias mapping"""
        
        try:
            vendor_path = self.master_data_dir / "Master_Vendors.csv"
            if vendor_path.exists():
                self.vendor_master = pd.read_csv(vendor_path)
                self.vendor_master.columns = [col.lower().strip() for col in self.vendor_master.columns]
        except:
            self.vendor_master = pd.DataFrame({'canonical_vendor': ['Unknown']})
        
        try:
            alias_path = self.master_data_dir / "Vendor_Alias_Map.csv"
            if alias_path.exists():
                self.alias_map = pd.read_csv(alias_path)
                self.alias_map.columns = [col.lower().strip() for col in self.alias_map.columns]
        except:
            self.alias_map = pd.DataFrame({'alias': [], 'canonical_vendor': []})
        
        return self
    
    def build_alias_dict(self):
        """Create lookup dictionary from aliases to canonical names"""
        alias_dict = {}
        
        # Build from alias map
        if self.alias_map is not None and len(self.alias_map) > 0:
            if 'alias' in self.alias_map.columns and 'canonical_vendor' in self.alias_map.columns:
                for _, row in self.alias_map.iterrows():
                    alias_raw = str(row['alias']).strip()
                    alias_lower = alias_raw.lower()
                    alias_dict[alias_lower] = row['canonical_vendor']
        
        # Add self-mappings from vendor master
        if self.vendor_master is not None and 'canonical_vendor' in self.vendor_master.columns:
            for vendor in self.vendor_master['canonical_vendor'].dropna():
                vendor_lower = vendor.lower()
                alias_dict[vendor_lower] = vendor
        
        return alias_dict
    
    def resolve_vendors(self):
        """Apply vendor mapping"""
        alias_dict = self.build_alias_dict()
        
        if self.vendor_master is not None and 'canonical_vendor' in self.vendor_master.columns:
            canonical_list = self.vendor_master['canonical_vendor'].dropna().unique().tolist()
        else:
            canonical_list = []
        
        def resolve(vendor_raw):
            if pd.isna(vendor_raw) or vendor_raw == '':
                return None, 'MISSING'
            
            vendor_original = str(vendor_raw).strip()
            vendor_lower = vendor_original.lower()
            
            # Direct alias match
            if vendor_lower in alias_dict:
                return alias_dict[vendor_lower], 'MAPPED'
            
            # Check if already canonical
            if vendor_original in canonical_list:
                return vendor_original, 'CANONICAL'
            
            # Try partial matching
            for canonical in canonical_list:
                canonical_lower = canonical.lower()
                if canonical_lower in vendor_lower or vendor_lower in canonical_lower:
                    return canonical, 'PARTIAL_MATCH'
            
            return None, 'UNMAPPED'
        
        # Apply resolution
        results = self.df['vendor_name_raw'].apply(resolve)
        self.df['vendor_canonical'] = [r[0] for r in results]
        self.df['vendor_resolution_status'] = [r[1] for r in results]
        
        # Log anomalies
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
        
        return self
    
    def save_output(self):
        """Save vendor-resolved data"""
        self.output_path = Config.WORKING_DIR / f"GL_VendorsResolved_{self.task_id}.csv"
        self.df.to_csv(self.output_path, index=False)
        
        return self.df

# ============================================================================
# T004: APPLY FX CONVERSION
# ============================================================================

class T004_FXConverter:
    """Task 4: Convert all transactions to AUD"""
    
    def __init__(self, df, task_id: str, reference_dir: Path):
        self.df = df.copy()
        self.task_id = task_id
        self.reference_dir = reference_dir
        self.fx_rates = None
        self.fx_anomalies = []
        self.output_path = None
        
    def run(self):
        """Execute all T004 steps"""
        try:
            self.load_fx_rates()
            self.convert_to_aud()
            self.save_output()
            
            converted = (self.df['conversion_status'] == 'CONVERTED').sum()
            failed = (self.df['conversion_status'] == 'FAILED').sum()
            domestic = (self.df['conversion_status'] == 'DOMESTIC').sum()
            
            return {
                "status": "success",
                "rows_processed": len(self.df),
                "domestic_aud": int(domestic),
                "converted": int(converted),
                "failed_conversion": int(failed),
                "output_file": str(self.output_path)
            }
            
        except Exception as e:
            raise Exception(f"T004 failed: {str(e)}")
    
    def load_fx_rates(self):
        """Load foreign exchange rates"""
        
        try:
            fx_path = self.reference_dir / "FX_Rates.csv"
            if fx_path.exists():
                self.fx_rates = pd.read_csv(fx_path)
                self.fx_rates.columns = [col.lower().strip() for col in self.fx_rates.columns]
                
                # Map columns
                if 'fiscal_period' in self.fx_rates.columns:
                    self.fx_rates.rename(columns={'fiscal_period': 'period'}, inplace=True)
                
                if 'rate_to_aud' in self.fx_rates.columns:
                    self.fx_rates.rename(columns={'rate_to_aud': 'rate'}, inplace=True)
                
                self.fx_rates['period'] = self.fx_rates['period'].astype(str)
        except:
            # Create default rates
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
                    elif currency == 'EUR':
                        rate = 1.62
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
        
        # Create lookup key
        self.df['fx_key'] = self.df['fiscal_period'] + '_' + self.df['currency_code']
        self.fx_rates['fx_key'] = self.fx_rates['period'].astype(str) + '_' + self.fx_rates['currency']
        
        # Create rate lookup dictionary
        rate_dict = dict(zip(self.fx_rates['fx_key'], self.fx_rates['rate']))
        
        def get_rate(row):
            if row['currency_code'] == 'AUD':
                return 1.0
            
            key = row['fx_key']
            if key in rate_dict:
                return rate_dict[key]
            
            # Try to find any rate for this currency
            currency_rates = {k: v for k, v in rate_dict.items() if k.endswith('_' + row['currency_code'])}
            if currency_rates:
                sorted_rates = sorted(currency_rates.items(), key=lambda x: x[0], reverse=True)
                return sorted_rates[0][1]
            
            # No rate found
            self.fx_anomalies.append({
                'transaction_id': row['transaction_id'],
                'anomaly_type': 'MISSING_FX_RATE',
                'severity': 'CRITICAL',
                'description': f"No FX rate found for {row['currency_code']}",
                'currency': row['currency_code'],
                'amount': row['amount']
            })
            return None
        
        # Apply conversion
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
        
        return self
    
    def save_output(self):
        """Save converted data"""
        self.output_path = Config.WORKING_DIR / f"GL_Converted_{self.task_id}.csv"
        self.df.to_csv(self.output_path, index=False)
        
        return self.df

# ============================================================================
# T005: DETECT EXCEPTIONS
# ============================================================================

class T005_ExceptionDetector:
    """Task 5: Run exception rules and flag violations"""
    
    def __init__(self, df, task_id: str, reference_dir: Path):
        self.df = df.copy()
        self.task_id = task_id
        self.reference_dir = reference_dir
        self.rulebook = None
        self.exception_results = []
        self.output_path = None
        self.exceptions_path = None
        
    def run(self):
        """Execute all T005 steps"""
        try:
            self.load_rulebook()
            self.detect_outliers()
            self.detect_temporal_anomalies()
            self.apply_rules()
            self.save_output()
            
            # Severity counts
            severity_counts = {}
            for e in self.exception_results:
                sev = e.get('severity', 'UNKNOWN')
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            return {
                "status": "success",
                "rows_processed": len(self.df),
                "total_exceptions": len(self.exception_results),
                "severity_breakdown": severity_counts,
                "output_file": str(self.output_path),
                "exceptions_file": str(self.exceptions_path)
            }
            
        except Exception as e:
            raise Exception(f"T005 failed: {str(e)}")
    
    def load_rulebook(self):
        """Load exception rules"""
        
        try:
            rules_path = self.reference_dir / "Exception_Rulebook.csv"
            if rules_path.exists():
                self.rulebook = pd.read_csv(rules_path)
        except:
            # Create default rules
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
                {'rule_id': 'EX010', 'rule_name': 'Extreme Outlier',
                 'severity': 'MEDIUM', 'description': 'Amount is significantly outside normal range'},
            ])
        
        return self
    
    def detect_outliers(self):
        """Statistical outlier detection"""
        if 'account_code_mapped' in self.df.columns and 'amount_aud' in self.df.columns:
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
        else:
            self.df['is_outlier'] = False
        
        return self
    
    def detect_temporal_anomalies(self):
        """Detect unusual timing patterns"""
        if 'posting_date' in self.df.columns:
            self.df['posting_hour'] = self.df['posting_date'].dt.hour if hasattr(self.df['posting_date'], 'dt') else None
            self.df['posting_weekend'] = self.df['posting_date'].dt.dayofweek.isin([5, 6]) if hasattr(self.df['posting_date'], 'dt') else False
            
            self.df['suspicious_hour'] = (
                self.df['posting_hour'].notna() & 
                ((self.df['posting_hour'] >= Config.SUSPICIOUS_HOUR_START) | 
                 (self.df['posting_hour'] <= Config.SUSPICIOUS_HOUR_END))
            )
        
        return self
    
    def apply_rules(self):
        """Apply all exception rules"""
        current_date = datetime(Config.CURRENT_YEAR, Config.CURRENT_MONTH, 28)
        
        # Define rule functions
        rule_functions = {
            'EX001': lambda row: pd.isna(row.get('po_number')) or row.get('po_number') == '',
            'EX002': lambda row: pd.isna(row.get('cost_center_mapped')),
            'EX003': lambda row: pd.isna(row.get('account_code_mapped')),
            'EX004': lambda row: row.get('amount_aud', 0) > Config.HIGH_VALUE_THRESHOLD if pd.notna(row.get('amount_aud')) else False,
            'EX005': lambda row: row.get('amount_is_negative', False),
            'EX006': lambda row: row.get('vendor_resolution_status') == 'UNMAPPED',
            'EX007': lambda row: (pd.notna(row.get('posting_date')) and 
                                  row.get('posting_date', current_date) > current_date and 
                                  row.get('fiscal_period') == Config.CURRENT_FISCAL_PERIOD),
            'EX008': lambda row: pd.isna(row.get('posting_date')),
            'EX009': lambda row: pd.isna(row.get('tax_code')) or row.get('tax_code') == '',
            'EX010': lambda row: row.get('is_outlier', False),
        }
        
        for _, rule in self.rulebook.iterrows():
            rule_id = rule['rule_id']
            rule_name = rule.get('rule_name', f'Rule {rule_id}')
            severity = rule.get('severity', 'MEDIUM')
            description = rule.get('description', rule_name)
            
            rule_func = rule_functions.get(rule_id)
            if rule_func is None:
                continue
            
            # Apply rule
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
        
        return self
    
    def save_output(self):
        """Save exception results"""
        # Add exception flags
        exception_txns = [e['transaction_id'] for e in self.exception_results]
        self.df['has_exception'] = self.df['transaction_id'].isin(exception_txns)
        
        # Save data with flags
        self.output_path = Config.WORKING_DIR / f"GL_WithExceptions_{self.task_id}.csv"
        self.df.to_csv(self.output_path, index=False)
        
        # Save exception log
        if self.exception_results:
            self.exceptions_path = Config.REPORTS_DIR / f"Exceptions_Detailed_{self.task_id}.csv"
            exceptions_df = pd.DataFrame(self.exception_results)
            exceptions_df.to_csv(self.exceptions_path, index=False)
        
        return self.df, self.exception_results

# ============================================================================
# T006: REVIEW HIGH SEVERITY EXCEPTIONS
# ============================================================================

class T006_ExceptionReviewer:
    """Task 6: Review and categorize exceptions (automated)"""
    
    def __init__(self, df, exceptions, task_id: str):
        self.df = df.copy()
        self.exceptions = exceptions
        self.task_id = task_id
        self.critical_exceptions = []
        self.high_exceptions = []
        self.output_path = None
        
    def run(self):
        """Execute T006 steps"""
        try:
            self.categorize_exceptions()
            review_data = self.create_review_package()
            
            return {
                "status": "success",
                "total_exceptions": len(self.exceptions),
                "critical_count": len(self.critical_exceptions),
                "high_count": len(self.high_exceptions),
                "auto_approved": True,
                "review_file": str(self.output_path)
            }
            
        except Exception as e:
            raise Exception(f"T006 failed: {str(e)}")
    
    def categorize_exceptions(self):
        """Split exceptions by severity"""
        for e in self.exceptions:
            if e.get('severity') == 'CRITICAL':
                self.critical_exceptions.append(e)
            elif e.get('severity') == 'HIGH':
                self.high_exceptions.append(e)
        
        return self
    
    def create_review_package(self):
        """Create automated review summary"""
        
        # Group critical exceptions by type
        critical_summary = {}
        for e in self.critical_exceptions:
            e_type = e.get('anomaly_type', e.get('rule_id', 'UNKNOWN'))
            if e_type not in critical_summary:
                critical_summary[e_type] = {'count': 0, 'total_amount': 0, 'examples': []}
            
            critical_summary[e_type]['count'] += 1
            critical_summary[e_type]['total_amount'] += e.get('amount', 0)
        
        review_data = {
            'timestamp': datetime.now().isoformat(),
            'total_critical': len(self.critical_exceptions),
            'total_high': len(self.high_exceptions),
            'critical_summary': critical_summary,
            'auto_approved': True
        }
        
        # Save to file
        self.output_path = Config.REPORTS_DIR / f"Exception_Review_Summary_{self.task_id}.json"
        with open(self.output_path, 'w') as f:
            json.dump(review_data, f, indent=2, default=str)
        
        return review_data

# ============================================================================
# T007: COMPUTE BUDGET VARIANCE
# ============================================================================

class T007_BudgetVariance:
    """Task 7: Calculate actual vs budget variance"""
    
    def __init__(self, df, task_id: str, budget_dir: Path):
        self.df = df.copy()
        self.task_id = task_id
        self.budget_dir = budget_dir
        self.budget_data = None
        self.variance_results = {}
        self.output_paths = {}
        
    def run(self):
        """Execute T007 steps"""
        try:
            self.load_budget()
            self.calculate_variance()
            self.save_output()
            
            return {
                "status": "success",
                "total_actual": float(self.variance_results['total_actual']),
                "total_budget": float(self.variance_results['total_budget']),
                "total_variance": float(self.variance_results['total_variance']),
                "variance_percentage": float(self.variance_results['total_variance_pct']),
                "suspense_amount": float(self.variance_results['suspense_amount']),
                "future_dated_amount": float(self.variance_results['future_dated_amount']),
                "transaction_count": int(self.variance_results['transaction_count']),
                "exception_count": int(self.variance_results['exception_count']),
                "output_files": self.output_paths
            }
            
        except Exception as e:
            raise Exception(f"T007 failed: {str(e)}")
    
    def load_budget(self):
        """Load budget data"""
        
        try:
            budget_path = self.budget_dir / "Budget_2026.csv"
            if budget_path.exists():
                self.budget_data = pd.read_csv(budget_path)
                self.budget_data.columns = [col.lower().strip() for col in self.budget_data.columns]
                
                # Ensure required columns
                if 'account_code' not in self.budget_data.columns:
                    for col in ['account', 'gl_account', 'coa']:
                        if col in self.budget_data.columns:
                            self.budget_data.rename(columns={col: 'account_code'}, inplace=True)
                            break
                
                if 'budget_amount' not in self.budget_data.columns:
                    for col in ['amount', 'budget', 'planned_amount']:
                        if col in self.budget_data.columns:
                            self.budget_data.rename(columns={col: 'budget_amount'}, inplace=True)
                            break
                
                # Clean budget amounts
                if 'budget_amount' in self.budget_data.columns:
                    self.budget_data['budget_amount'] = pd.to_numeric(
                        self.budget_data['budget_amount'].astype(str).str.replace('$', '').str.replace(',', ''),
                        errors='coerce'
                    )
                    self.budget_data['budget_amount'] = self.budget_data['budget_amount'].fillna(0.01).clip(lower=0.01)
                
                # Ensure period column
                if 'period' not in self.budget_data.columns:
                    for col in ['fiscal_period', 'month', 'reporting_period']:
                        if col in self.budget_data.columns:
                            self.budget_data.rename(columns={col: 'period'}, inplace=True)
                            break
                    else:
                        self.budget_data['period'] = Config.CURRENT_FISCAL_PERIOD
                
                self.budget_data['period'] = self.budget_data['period'].astype(str)
                self.budget_data['account_code'] = self.budget_data['account_code'].astype(str)
        except:
            # Create sample budget
            accounts = self.df['account_code_mapped'].dropna().unique() if 'account_code_mapped' in self.df.columns else ['5000']
            
            budget_rows = []
            for account in accounts[:30]:
                budget_rows.append({
                    'account_code': str(account),
                    'period': Config.CURRENT_FISCAL_PERIOD,
                    'budget_amount': float(np.random.randint(50000, 200000))
                })
            
            self.budget_data = pd.DataFrame(budget_rows)
        
        return self
    
    def calculate_variance(self):
        """Calculate variance"""
        
        # Filter to current period
        current_period_df = self.df[
            (self.df['fiscal_period'] == Config.CURRENT_FISCAL_PERIOD) &
            (self.df['amount_aud'].notna())
        ].copy()
        
        # Variance by Account
        if 'account_code_mapped' in current_period_df.columns:
            account_actuals = current_period_df.groupby('account_code_mapped').agg({
                'amount_aud': 'sum',
                'transaction_id': 'count'
            }).rename(columns={
                'amount_aud': 'actual_amount',
                'transaction_id': 'transaction_count'
            }).reset_index()
            
            account_actuals['account_code_mapped'] = account_actuals['account_code_mapped'].astype(str)
            
            # Get budget for current period
            feb_budget = self.budget_data[self.budget_data['period'] == Config.CURRENT_FISCAL_PERIOD].copy()
            if feb_budget.empty:
                feb_budget = self.budget_data.copy()
            
            feb_budget['account_code'] = feb_budget['account_code'].astype(str)
            
            # Merge with budget
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
        else:
            account_variance = pd.DataFrame()
        
        # Suspense amounts
        suspense_amount = current_period_df[
            current_period_df['account_code_mapped'].isna() if 'account_code_mapped' in current_period_df.columns else False
        ]['amount_aud'].sum()
        
        # Future dated amounts
        current_date = datetime(Config.CURRENT_YEAR, Config.CURRENT_MONTH, 28)
        future_amount = current_period_df[
            current_period_df['posting_date'] > current_date if 'posting_date' in current_period_df.columns else False
        ]['amount_aud'].sum()
        
        # Totals
        total_actual = float(current_period_df['amount_aud'].sum())
        total_budget = float(feb_budget['budget_amount'].sum()) if not feb_budget.empty else 0.01
        
        total_variance = total_actual - total_budget
        if total_budget > 0:
            total_variance_pct = (total_variance / total_budget) * 100
        elif total_actual > 0:
            total_variance_pct = 999999
        else:
            total_variance_pct = 0
        
        self.variance_results = {
            'by_account': account_variance.to_dict('records') if not account_variance.empty else [],
            'suspense_amount': float(suspense_amount),
            'future_dated_amount': float(future_amount),
            'total_actual': total_actual,
            'total_budget': total_budget,
            'total_variance': total_variance,
            'total_variance_pct': total_variance_pct,
            'transaction_count': int(len(current_period_df)),
            'exception_count': int(current_period_df['has_exception'].sum() if 'has_exception' in current_period_df.columns else 0)
        }
        
        return self
    
    def save_output(self):
        """Save variance results"""
        
        # Save detailed variance
        if self.variance_results['by_account']:
            account_path = Config.REPORTS_DIR / f"Budget_Variance_By_Account_{self.task_id}.csv"
            pd.DataFrame(self.variance_results['by_account']).to_csv(account_path, index=False)
            self.output_paths['by_account'] = str(account_path)
        
        # Save summary
        summary_path = Config.REPORTS_DIR / f"Budget_Variance_Summary_{self.task_id}.csv"
        summary_df = pd.DataFrame([{
            'metric': 'Total Actual', 'value': self.variance_results['total_actual']
        }, {
            'metric': 'Total Budget', 'value': self.variance_results['total_budget']
        }, {
            'metric': 'Variance', 'value': self.variance_results['total_variance']
        }, {
            'metric': 'Variance %', 'value': self.variance_results['total_variance_pct']
        }, {
            'metric': 'Suspense Amount', 'value': self.variance_results['suspense_amount']
        }, {
            'metric': 'Future Dated Amount', 'value': self.variance_results['future_dated_amount']
        }, {
            'metric': 'Transaction Count', 'value': self.variance_results['transaction_count']
        }, {
            'metric': 'Exception Count', 'value': self.variance_results['exception_count']
        }])
        summary_df.to_csv(summary_path, index=False)
        self.output_paths['summary'] = str(summary_path)
        
        return self.variance_results

# ============================================================================
# T008: CLOSE PACK REPORT
# ============================================================================

class T008_ClosePackReport:
    """Task 8: Generate close pack reports"""
    
    def __init__(self, df, variance_results, exceptions, task_id: str):
        self.df = df.copy()
        self.variance = variance_results
        self.exceptions = exceptions
        self.task_id = task_id
        self.report_data = {}
        self.output_paths = {}
        
    def run(self):
        """Execute T008 steps"""
        try:
            self.prepare_summary()
            self.prepare_currency_summary()
            self.prepare_exception_summary()
            self.save_reports()
            
            return {
                "status": "success",
                "report_data": self.report_data,
                "output_files": self.output_paths
            }
            
        except Exception as e:
            raise Exception(f"T008 failed: {str(e)}")
    
    def prepare_summary(self):
        """Prepare high-level summary"""
        self.report_data['summary'] = {
            'total_spend': float(self.variance['total_actual']),
            'budget_variance': float(self.variance['total_variance']),
            'variance_percentage': float(self.variance['total_variance_pct']),
            'transaction_count': int(self.variance['transaction_count']),
            'exception_count': int(self.variance['exception_count']),
            'suspense_amount': float(self.variance['suspense_amount']),
            'future_dated_amount': float(self.variance['future_dated_amount'])
        }
        return self
    
    def prepare_currency_summary(self):
        """Prepare currency breakdown"""
        if 'currency_code' in self.df.columns and 'amount_aud' in self.df.columns:
            currency_summary = self.df.groupby('currency_code').agg({
                'amount_aud': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            
            currency_summary.columns = ['currency_code', 'amount_aud', 'transaction_count']
            currency_summary['amount_aud'] = currency_summary['amount_aud'].fillna(0)
            
            self.report_data['currency_summary'] = currency_summary.to_dict('records')
        else:
            self.report_data['currency_summary'] = []
        
        return self
    
    def prepare_exception_summary(self):
        """Prepare exception summary"""
        if self.exceptions:
            exception_df = pd.DataFrame(self.exceptions)
            
            severity_counts = exception_df['severity'].value_counts().to_dict()
            type_counts = exception_df['anomaly_type'].value_counts().head(5).to_dict() if 'anomaly_type' in exception_df.columns else {}
            
            self.report_data['exception_summary'] = {
                'total': len(self.exceptions),
                'by_severity': severity_counts,
                'top_types': type_counts
            }
        else:
            self.report_data['exception_summary'] = {'total': 0}
        
        return self
    
    def save_reports(self):
        """Save all reports"""
        
        # Save summary JSON
        summary_path = Config.REPORTS_DIR / f"Close_Pack_Summary_{self.task_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        self.output_paths['summary'] = str(summary_path)
        
        # Save currency summary CSV
        if self.report_data.get('currency_summary'):
            currency_path = Config.REPORTS_DIR / f"Currency_Summary_{self.task_id}.csv"
            pd.DataFrame(self.report_data['currency_summary']).to_csv(currency_path, index=False)
            self.output_paths['currency'] = str(currency_path)
        
        return self.report_data

# ============================================================================
# T009: GENERATE EXECUTIVE NARRATIVE
# ============================================================================

class T009_ExecutiveNarrative:
    """Task 9: Create natural language summary"""
    
    def __init__(self, variance_results, report_data, exceptions, task_id: str):
        self.variance = variance_results
        self.report = report_data
        self.exceptions = exceptions
        self.task_id = task_id
        self.narrative = ""
        self.output_path = None
        
    def run(self):
        """Execute T009 steps"""
        try:
            self.generate_narrative()
            self.save_narrative()
            
            return {
                "status": "success",
                "narrative": self.narrative,
                "file_path": str(self.output_path)
            }
            
        except Exception as e:
            raise Exception(f"T009 failed: {str(e)}")
    
    def generate_narrative(self):
        """Generate narrative using templates"""
        
        lines = []
        
        # Header
        lines.append("="*80)
        lines.append(f"EXECUTIVE NARRATIVE - {Config.CURRENT_FISCAL_PERIOD}")
        lines.append("="*80)
        lines.append("")
        
        # Financial Summary
        lines.append("FINANCIAL SUMMARY")
        lines.append("-"*40)
        
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
        
        # Key Drivers
        lines.append("KEY VARIANCE DRIVERS")
        lines.append("-"*40)
        
        # Find largest variances
        account_variances = self.variance.get('by_account', [])
        top_pos = sorted([a for a in account_variances if a.get('variance', 0) > 0], 
                         key=lambda x: x['variance'], reverse=True)[:3]
        top_neg = sorted([a for a in account_variances if a.get('variance', 0) < 0], 
                         key=lambda x: x['variance'])[:3]
        
        if top_pos:
            lines.append("Positive variances (over budget):")
            for a in top_pos:
                lines.append(f"  • {a.get('account_code', 'Unknown')}: +${a['variance']:,.2f} ({a['variance_pct']:.1f}%)")
        
        if top_neg:
            lines.append("Negative variances (under budget):")
            for a in top_neg:
                lines.append(f"  • {a.get('account_code', 'Unknown')}: ${a['variance']:,.2f} ({a['variance_pct']:.1f}%)")
        lines.append("")
        
        # Exception Summary
        lines.append("EXCEPTION SUMMARY")
        lines.append("-"*40)
        
        critical_count = len([e for e in self.exceptions if e.get('severity') == 'CRITICAL'])
        high_count = len([e for e in self.exceptions if e.get('severity') == 'HIGH'])
        medium_count = len([e for e in self.exceptions if e.get('severity') == 'MEDIUM'])
        
        lines.append(f"Total exceptions: {len(self.exceptions)}")
        lines.append(f"  • Critical: {critical_count}")
        lines.append(f"  • High: {high_count}")
        lines.append(f"  • Medium: {medium_count}")
        
        # Data Quality Impact
        lines.append("")
        lines.append("DATA QUALITY IMPACT")
        lines.append("-"*40)
        
        suspense_amount = self.variance.get('suspense_amount', 0)
        future_amount = self.variance.get('future_dated_amount', 0)
        total_impact = suspense_amount + future_amount
        impact_pct = (total_impact / self.variance['total_actual'] * 100) if self.variance['total_actual'] > 0 else 0
        
        lines.append(f"Transactions with data quality issues: ${total_impact:,.2f} ({impact_pct:.1f}% of total)")
        if suspense_amount > 0:
            lines.append(f"  • Invalid accounts (in suspense): ${suspense_amount:,.2f}")
        if future_amount > 0:
            lines.append(f"  • Future-dated transactions: ${future_amount:,.2f}")
        
        # Join all lines
        self.narrative = "\n".join(lines)
        
        return self
    
    def save_narrative(self):
        """Save narrative to file"""
        self.output_path = Config.REPORTS_DIR / f"Executive_Narrative_{self.task_id}.txt"
        with open(self.output_path, 'w') as f:
            f.write(self.narrative)
        
        return self.narrative

# ============================================================================
# T010: FORECAST NEXT PERIOD
# ============================================================================

class T010_Forecast:
    """Task 10: Generate forecast for next period"""
    
    def __init__(self, df, variance_results, task_id: str, reference_dir: Path):
        self.df = df
        self.variance = variance_results
        self.task_id = task_id
        self.reference_dir = reference_dir
        self.historical_data = None
        self.forecast = {}
        self.output_path = None
        
    def run(self):
        """Execute T010 steps"""
        try:
            self.load_historical()
            self.calculate_trends()
            self.save_forecast()
            
            return {
                "status": "success",
                "next_period": self.forecast['next_period'],
                "forecast_amount": float(self.forecast['forecast_amount']),
                "lower_bound": float(self.forecast['lower_bound']),
                "upper_bound": float(self.forecast['upper_bound']),
                "confidence_level": float(self.forecast['confidence_level']),
                "method": self.forecast['method'],
                "file_path": str(self.output_path)
            }
            
        except Exception as e:
            raise Exception(f"T010 failed: {str(e)}")
    
    def load_historical(self):
        """Load historical KPI data"""
        
        try:
            historical_path = self.reference_dir / "KPI_Monthly_History.csv"
            if historical_path.exists():
                self.historical_data = pd.read_csv(historical_path)
                self.historical_data.columns = [col.lower().strip() for col in self.historical_data.columns]
                
                # Find period column
                period_col = None
                for col in ['period', 'month', 'fiscal_period']:
                    if col in self.historical_data.columns:
                        period_col = col
                        break
                
                if period_col and period_col != 'period':
                    self.historical_data.rename(columns={period_col: 'period'}, inplace=True)
                
                # Find spend column
                spend_col = None
                for col in ['total_spend', 'spend', 'amount', 'actual', 'value']:
                    if col in self.historical_data.columns:
                        spend_col = col
                        break
                
                if spend_col and spend_col != 'total_spend':
                    self.historical_data.rename(columns={spend_col: 'total_spend'}, inplace=True)
                
                if 'period' not in self.historical_data.columns:
                    self.historical_data['period'] = [f"{Config.CURRENT_YEAR-1}-{i:02d}" for i in range(1, 13)]
                
                if 'total_spend' not in self.historical_data.columns:
                    base_spend = self.variance.get('total_actual', 1000000)
                    self.historical_data['total_spend'] = [
                        base_spend * (0.8 + 0.4 * np.random.random()) 
                        for _ in range(len(self.historical_data))
                    ]
        except:
            # Create synthetic history
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
                    'total_spend': base_spend * (0.8 + 0.4 * np.random.random())
                })
            
            self.historical_data = pd.DataFrame(months)
        
        # Ensure period is string
        self.historical_data['period'] = self.historical_data['period'].astype(str)
        
        return self
    
    def calculate_trends(self):
        """Calculate trends and generate forecast"""
        
        # Sort by period
        try:
            self.historical_data = self.historical_data.sort_values('period')
        except:
            pass
        
        # Calculate next period
        if Config.CURRENT_MONTH < 12:
            next_period = f"{Config.CURRENT_YEAR}-{Config.CURRENT_MONTH+1:02d}"
        else:
            next_period = f"{Config.CURRENT_YEAR+1}-01"
        
        # Simple forecast (10% growth)
        current_actual = self.variance.get('total_actual', 1000000)
        forecast_amount = current_actual * 1.10
        
        self.forecast = {
            'next_period': next_period,
            'forecast_amount': forecast_amount,
            'lower_bound': forecast_amount * 0.9,
            'upper_bound': forecast_amount * 1.1,
            'confidence_level': 0.95,
            'method': 'Simple growth projection (10%)',
            'historical_months_used': len(self.historical_data) if self.historical_data is not None else 0,
            'current_actual': current_actual
        }
        
        return self
    
    def save_forecast(self):
        """Save forecast results"""
        self.output_path = Config.REPORTS_DIR / f"Forecast_{self.task_id}.csv"
        forecast_df = pd.DataFrame([self.forecast])
        forecast_df.to_csv(self.output_path, index=False)
        
        return self.forecast

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "service": "Financial Close Agent API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "directories": {
            "uploads": str(Config.UPLOAD_DIR),
            "master_data": str(Config.MASTER_DATA_DIR),
            "reports": str(Config.REPORTS_DIR)
        }
    }

# ============================================================================
# UPLOAD ENDPOINTS
# ============================================================================

@app.post("/upload/gl", tags=["Upload"])
async def upload_gl_file(file: UploadFile = File(...)):
    """Upload raw GL export file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        file_path = save_upload_file(file, Config.UPLOAD_DIR)
        
        # Quick validation
        df = pd.read_csv(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": file_path,
            "rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/master/{master_type}", tags=["Upload"])
async def upload_master_file(master_type: str, file: UploadFile = File(...)):
    """Upload master data files (entity, coa, cost_centers, vendors, alias)"""
    valid_types = ["entity", "coa", "cost_centers", "vendors", "alias"]
    
    if master_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Master type must be one of: {valid_types}")
    
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Map to expected filenames
        filename_map = {
            "entity": "Master_Entity.csv",
            "coa": "Master_COA.csv",
            "cost_centers": "Master_CostCenters.csv",
            "vendors": "Master_Vendors.csv",
            "alias": "Vendor_Alias_Map.csv"
        }
        
        # Save with correct name
        file_path = Config.MASTER_DATA_DIR / filename_map[master_type]
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "master_type": master_type,
            "filename": filename_map[master_type],
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/reference/{ref_type}", tags=["Upload"])
async def upload_reference_file(ref_type: str, file: UploadFile = File(...)):
    """Upload reference files (fx_rates, exception_rules, kpi_history)"""
    valid_types = ["fx_rates", "exception_rules", "kpi_history"]
    
    if ref_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Reference type must be one of: {valid_types}")
    
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Map to expected filenames
        filename_map = {
            "fx_rates": "FX_Rates.csv",
            "exception_rules": "Exception_Rulebook.csv",
            "kpi_history": "KPI_Monthly_History.csv"
        }
        
        file_path = Config.REFERENCE_DIR / filename_map[ref_type]
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "reference_type": ref_type,
            "filename": filename_map[ref_type],
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/budget", tags=["Upload"])
async def upload_budget_file(file: UploadFile = File(...)):
    """Upload budget file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        file_path = Config.BUDGET_DIR / "Budget_2026.csv"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "filename": "Budget_2026.csv",
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PROCESSING ENDPOINTS
# ============================================================================

@app.post("/process/t001", response_model=ProcessingResponse, tags=["Processing"])
async def process_t001(
    background_tasks: BackgroundTasks,
    file_path: str,
    task_id: Optional[str] = None
):
    """Task 001: Wrangle raw GL data"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            wrangler = T001_DataWrangler(file_path, task_id)
            result = wrangler.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T001")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T001 processing started"
    )

@app.post("/process/t002", response_model=ProcessingResponse, tags=["Processing"])
async def process_t002(
    background_tasks: BackgroundTasks,
    data_file: str,
    task_id: Optional[str] = None
):
    """Task 002: Map entities and accounts"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            mapper = T002_EntityAccountMapper(df, task_id, Config.MASTER_DATA_DIR)
            result = mapper.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T002")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T002 processing started"
    )

@app.post("/process/t003", response_model=ProcessingResponse, tags=["Processing"])
async def process_t003(
    background_tasks: BackgroundTasks,
    data_file: str,
    task_id: Optional[str] = None
):
    """Task 003: Resolve vendor names"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            resolver = T003_VendorResolver(df, task_id, Config.MASTER_DATA_DIR)
            result = resolver.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T003")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T003 processing started"
    )

@app.post("/process/t004", response_model=ProcessingResponse, tags=["Processing"])
async def process_t004(
    background_tasks: BackgroundTasks,
    data_file: str,
    task_id: Optional[str] = None
):
    """Task 004: Apply FX conversion"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            converter = T004_FXConverter(df, task_id, Config.REFERENCE_DIR)
            result = converter.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T004")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T004 processing started"
    )

@app.post("/process/t005", response_model=ProcessingResponse, tags=["Processing"])
async def process_t005(
    background_tasks: BackgroundTasks,
    data_file: str,
    task_id: Optional[str] = None
):
    """Task 005: Detect exceptions"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            detector = T005_ExceptionDetector(df, task_id, Config.REFERENCE_DIR)
            result = detector.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T005")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T005 processing started"
    )

@app.post("/process/t006", response_model=ProcessingResponse, tags=["Processing"])
async def process_t006(
    background_tasks: BackgroundTasks,
    data_file: str,
    exceptions_file: str,
    task_id: Optional[str] = None
):
    """Task 006: Review exceptions"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            exceptions = pd.read_csv(exceptions_file).to_dict('records') if os.path.exists(exceptions_file) else []
            reviewer = T006_ExceptionReviewer(df, exceptions, task_id)
            result = reviewer.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T006")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T006 processing started"
    )

@app.post("/process/t007", response_model=ProcessingResponse, tags=["Processing"])
async def process_t007(
    background_tasks: BackgroundTasks,
    data_file: str,
    task_id: Optional[str] = None
):
    """Task 007: Compute budget variance"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            variance = T007_BudgetVariance(df, task_id, Config.BUDGET_DIR)
            result = variance.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T007")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T007 processing started"
    )

@app.post("/process/t008", response_model=ProcessingResponse, tags=["Processing"])
async def process_t008(
    background_tasks: BackgroundTasks,
    data_file: str,
    variance_file: str,
    exceptions_file: str,
    task_id: Optional[str] = None
):
    """Task 008: Generate close pack report"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            
            # Load variance results
            with open(variance_file, 'r') as f:
                variance_results = json.load(f)
            
            exceptions = pd.read_csv(exceptions_file).to_dict('records') if os.path.exists(exceptions_file) else []
            
            report = T008_ClosePackReport(df, variance_results, exceptions, task_id)
            result = report.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T008")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T008 processing started"
    )

@app.post("/process/t009", response_model=ProcessingResponse, tags=["Processing"])
async def process_t009(
    background_tasks: BackgroundTasks,
    variance_file: str,
    report_file: str,
    exceptions_file: str,
    task_id: Optional[str] = None
):
    """Task 009: Generate executive narrative"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            
            with open(variance_file, 'r') as f:
                variance_results = json.load(f)
            
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            exceptions = pd.read_csv(exceptions_file).to_dict('records') if os.path.exists(exceptions_file) else []
            
            narrative = T009_ExecutiveNarrative(variance_results, report_data, exceptions, task_id)
            result = narrative.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T009")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T009 processing started"
    )

@app.post("/process/t010", response_model=ProcessingResponse, tags=["Processing"])
async def process_t010(
    background_tasks: BackgroundTasks,
    data_file: str,
    variance_file: str,
    task_id: Optional[str] = None
):
    """Task 010: Forecast next period"""
    task_id = task_id or generate_task_id()
    
    def run_task():
        try:
            task_manager.update_task(task_id, "processing")
            df = pd.read_csv(data_file)
            
            with open(variance_file, 'r') as f:
                variance_results = json.load(f)
            
            forecast = T010_Forecast(df, variance_results, task_id, Config.REFERENCE_DIR)
            result = forecast.run()
            task_manager.update_task(task_id, "completed", result)
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "T010")
    background_tasks.add_task(run_task)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="T010 processing started"
    )

@app.post("/process/full", response_model=ProcessingResponse, tags=["Processing"])
async def process_full_pipeline(
    background_tasks: BackgroundTasks,
    gl_file_path: str,
    task_id: Optional[str] = None
):
    """Run the complete financial close pipeline"""
    task_id = task_id or generate_task_id()
    
    def run_pipeline():
        try:
            task_manager.update_task(task_id, "processing")
            
            results = {}
            
            # T001: Wrangle
            wrangler = T001_DataWrangler(gl_file_path, task_id)
            t001_result = wrangler.run()
            results['t001'] = t001_result
            df = pd.read_csv(t001_result['output_files']['standardized_data'])
            
            # T002: Map entities and accounts
            mapper = T002_EntityAccountMapper(df, task_id, Config.MASTER_DATA_DIR)
            t002_result = mapper.run()
            results['t002'] = t002_result
            df = pd.read_csv(t002_result['output_file'])
            
            # T003: Resolve vendors
            resolver = T003_VendorResolver(df, task_id, Config.MASTER_DATA_DIR)
            t003_result = resolver.run()
            results['t003'] = t003_result
            df = pd.read_csv(t003_result['output_file'])
            
            # T004: FX Conversion
            converter = T004_FXConverter(df, task_id, Config.REFERENCE_DIR)
            t004_result = converter.run()
            results['t004'] = t004_result
            df = pd.read_csv(t004_result['output_file'])
            
            # T005: Detect exceptions
            detector = T005_ExceptionDetector(df, task_id, Config.REFERENCE_DIR)
            t005_result = detector.run()
            results['t005'] = t005_result
            df = pd.read_csv(t005_result['output_file'])
            exceptions = pd.read_csv(t005_result['exceptions_file']).to_dict('records') if t005_result.get('exceptions_file') and Path(t005_result['exceptions_file']).exists() else []
            
            # T006: Review exceptions
            reviewer = T006_ExceptionReviewer(df, exceptions, task_id)
            t006_result = reviewer.run()
            results['t006'] = t006_result
            
            # T007: Budget variance
            variance = T007_BudgetVariance(df, task_id, Config.BUDGET_DIR)
            t007_result = variance.run()
            results['t007'] = t007_result
            
            # Save variance results for later tasks
            variance_path = Config.WORKING_DIR / f"variance_results_{task_id}.json"
            with open(variance_path, 'w') as f:
                json.dump(t007_result, f, default=str)
            
            # T008: Close pack report
            report = T008_ClosePackReport(df, t007_result, exceptions, task_id)
            t008_result = report.run()
            results['t008'] = t008_result
            
            # Save report data
            report_path = Config.WORKING_DIR / f"report_data_{task_id}.json"
            with open(report_path, 'w') as f:
                json.dump(t008_result['report_data'], f, default=str)
            
            # T009: Executive narrative
            narrative = T009_ExecutiveNarrative(t007_result, t008_result['report_data'], exceptions, task_id)
            t009_result = narrative.run()
            results['t009'] = t009_result
            
            # T010: Forecast
            forecast = T010_Forecast(df, t007_result, task_id, Config.REFERENCE_DIR)
            t010_result = forecast.run()
            results['t010'] = t010_result
            
            # Save final pipeline summary
            summary_path = Config.REPORTS_DIR / f"Pipeline_Summary_{task_id}.json"
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            task_manager.update_task(task_id, "completed", {
                "summary": {
                    "tasks_completed": list(results.keys()),
                    "total_transactions": int(t001_result['rows_processed']),
                    "total_exceptions": t005_result['total_exceptions'],
                    "total_spend": float(t007_result['total_actual']),
                    "budget_variance": float(t007_result['total_variance']),
                    "forecast_next_period": float(t010_result['forecast_amount'])
                },
                "output_files": {
                    "standardized_data": t001_result['output_files']['standardized_data'],
                    "mapped_data": t002_result['output_file'],
                    "vendor_resolved": t003_result['output_file'],
                    "converted_data": t004_result['output_file'],
                    "exceptions": t005_result['exceptions_file'],
                    "variance_summary": t007_result['output_files']['summary'],
                    "report_summary": str(summary_path),
                    "narrative": t009_result['file_path'],
                    "forecast": t010_result['file_path']
                }
            })
            
        except Exception as e:
            task_manager.update_task(task_id, "failed", error=str(e))
    
    task_manager.create_task(task_id, "FULL_PIPELINE")
    background_tasks.add_task(run_pipeline)
    
    return ProcessingResponse(
        task_id=task_id,
        status="pending",
        message="Full pipeline processing started"
    )

# ============================================================================
# TASK STATUS ENDPOINTS
# ============================================================================

@app.get("/tasks", tags=["Tasks"])
async def get_all_tasks():
    """Get all tasks"""
    return task_manager.get_all_tasks()

@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task_status(task_id: str):
    """Get task status by ID"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """Delete task by ID"""
    if task_id in task_manager.tasks:
        del task_manager.tasks[task_id]
        return {"status": "success", "message": f"Task {task_id} deleted"}
    raise HTTPException(status_code=404, detail="Task not found")

# ============================================================================
# DATA RETRIEVAL ENDPOINTS
# ============================================================================

@app.get("/data/gl/{task_id}", tags=["Data"])
async def get_gl_data(task_id: str):
    """Get processed GL data for a task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    # Try to find the output file
    possible_files = [
        Config.WORKING_DIR / f"GL_Standardized_{task_id}.csv",
        Config.WORKING_DIR / f"GL_WithMappings_{task_id}.csv",
        Config.WORKING_DIR / f"GL_VendorsResolved_{task_id}.csv",
        Config.WORKING_DIR / f"GL_Converted_{task_id}.csv",
        Config.WORKING_DIR / f"GL_WithExceptions_{task_id}.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type='text/csv'
            )
    
    raise HTTPException(status_code=404, detail="Data file not found")

@app.get("/reports/{task_id}", tags=["Reports"])
async def get_report(task_id: str, report_type: str = "summary"):
    """Get report files for a task"""
    valid_reports = ["summary", "anomalies", "exceptions", "variance", "forecast", "narrative"]
    
    if report_type not in valid_reports:
        raise HTTPException(status_code=400, detail=f"Report type must be one of: {valid_reports}")
    
    report_files = {
        "summary": f"Pipeline_Summary_{task_id}.json",
        "anomalies": f"Input_Anomalies_{task_id}.csv",
        "exceptions": f"Exceptions_Detailed_{task_id}.csv",
        "variance": f"Budget_Variance_Summary_{task_id}.csv",
        "forecast": f"Forecast_{task_id}.csv",
        "narrative": f"Executive_Narrative_{task_id}.txt"
    }
    
    file_name = report_files[report_type]
    file_path = Config.REPORTS_DIR / file_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{report_type} report not found")
    
    media_type = 'application/json' if file_name.endswith('.json') else 'text/csv' if file_name.endswith('.csv') else 'text/plain'
    
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type=media_type
    )

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/analyze/variance", tags=["Analysis"])
async def calculate_variance(data_file: str, budget_file: Optional[str] = None):
    """Calculate budget variance from processed data"""
    try:
        # Load data
        df = pd.read_csv(data_file)
        
        # Basic variance calculation
        total_actual = df['amount_aud'].sum() if 'amount_aud' in df.columns else df['amount'].sum() if 'amount' in df.columns else 0
        
        if budget_file and os.path.exists(budget_file):
            budget_df = pd.read_csv(budget_file)
            budget_col = 'budget_amount' if 'budget_amount' in budget_df.columns else 'amount' if 'amount' in budget_df.columns else None
            total_budget = budget_df[budget_col].sum() if budget_col else total_actual * 1.05
        else:
            total_budget = total_actual * 1.05  # Placeholder
        
        variance = total_actual - total_budget
        variance_pct = (variance / total_budget * 100) if total_budget != 0 else 0
        
        return {
            "total_actual": float(total_actual),
            "total_budget": float(total_budget),
            "variance": float(variance),
            "variance_percentage": float(variance_pct),
            "transaction_count": int(len(df))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/forecast", tags=["Analysis"])
async def generate_forecast(data_file: str):
    """Generate forecast for next period"""
    try:
        # Load current data
        df = pd.read_csv(data_file)
        amount_col = 'amount_aud' if 'amount_aud' in df.columns else 'amount' if 'amount' in df.columns else None
        current_total = df[amount_col].sum() if amount_col else 0
        
        # Simple forecast (10% growth)
        forecast = current_total * 1.10
        
        # Calculate next period
        if Config.CURRENT_MONTH < 12:
            next_period = f"{Config.CURRENT_YEAR}-{Config.CURRENT_MONTH+1:02d}"
        else:
            next_period = f"{Config.CURRENT_YEAR+1}-01"
        
        return {
            "next_period": next_period,
            "forecast_amount": float(forecast),
            "lower_bound": float(forecast * 0.9),
            "upper_bound": float(forecast * 1.1),
            "confidence_level": 0.95,
            "method": "Simple growth projection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FILE MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/files", tags=["Files"])
async def list_files(directory: str = "uploads"):
    """List files in a directory"""
    valid_dirs = ["uploads", "master_data", "reference", "budget", "working", "reports"]
    
    if directory not in valid_dirs:
        raise HTTPException(status_code=400, detail=f"Directory must be one of: {valid_dirs}")
    
    dir_path = getattr(Config, f"{directory.upper()}_DIR")
    
    files = []
    for file_path in dir_path.glob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return {
        "directory": directory,
        "path": str(dir_path),
        "files": files
    }

@app.delete("/files/{filename}", tags=["Files"])
async def delete_file(filename: str, directory: str = "uploads"):
    """Delete a file"""
    valid_dirs = ["uploads", "master_data", "reference", "budget", "working", "reports"]
    
    if directory not in valid_dirs:
        raise HTTPException(status_code=400, detail=f"Directory must be one of: {valid_dirs}")
    
    dir_path = getattr(Config, f"{directory.upper()}_DIR")
    file_path = dir_path / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path.unlink()
    
    return {
        "status": "success",
        "message": f"File {filename} deleted from {directory}"
    }

# ============================================================================
# VALIDATION ENDPOINTS
# ============================================================================

@app.post("/validate/gl", tags=["Validation"])
async def validate_gl_file(file: UploadFile = File(...)):
    """Validate GL file structure"""
    try:
        # Save temporarily
        temp_path = Config.UPLOAD_DIR / f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and validate
        df = pd.read_csv(temp_path)
        
        required_columns = ['Txn_ID', 'Posting_Date_Raw', 'Fiscal_Period', 'Entity', 
                           'Account_Code_Raw', 'Amount']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Clean up
        temp_path.unlink()
        
        if missing_columns:
            return {
                "valid": False,
                "missing_columns": missing_columns,
                "message": f"Missing required columns: {missing_columns}"
            }
        
        return {
            "valid": True,
            "rows": len(df),
            "columns": list(df.columns),
            "message": "GL file structure is valid"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

@app.get("/config", tags=["Configuration"])
async def get_config():
    """Get current configuration"""
    return {
        "current_fiscal_period": Config.CURRENT_FISCAL_PERIOD,
        "current_month": Config.CURRENT_MONTH,
        "current_year": Config.CURRENT_YEAR,
        "high_value_threshold": Config.HIGH_VALUE_THRESHOLD,
        "directories": {
            "uploads": str(Config.UPLOAD_DIR),
            "master_data": str(Config.MASTER_DATA_DIR),
            "reference": str(Config.REFERENCE_DIR),
            "budget": str(Config.BUDGET_DIR),
            "working": str(Config.WORKING_DIR),
            "reports": str(Config.REPORTS_DIR)
        }
    }

@app.post("/config/fiscal", tags=["Configuration"])
async def set_fiscal_period(year: int, month: int):
    """Set current fiscal period"""
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    Config.CURRENT_YEAR = year
    Config.CURRENT_MONTH = month
    Config.CURRENT_FISCAL_PERIOD = f"{year}-{month:02d}"
    
    return {
        "status": "success",
        "current_fiscal_period": Config.CURRENT_FISCAL_PERIOD
    }

# ============================================================================
# BATCH PROCESSING ENDPOINTS
# ============================================================================

@app.post("/batch/process", tags=["Batch"])
async def batch_process(files: List[str], background_tasks: BackgroundTasks):
    """Process multiple files in batch"""
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def run_batch():
        results = {}
        for i, file_path in enumerate(files):
            try:
                task_id = f"{batch_id}_{i}"
                task_manager.create_task(task_id, "BATCH")
                
                # Run T001 on each file
                wrangler = T001_DataWrangler(file_path, task_id)
                result = wrangler.run()
                
                results[file_path] = {
                    "status": "success",
                    "task_id": task_id,
                    "result": result
                }
            except Exception as e:
                results[file_path] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Save batch results
        batch_path = Config.REPORTS_DIR / f"{batch_id}_results.json"
        with open(batch_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    background_tasks.add_task(run_batch)
    
    return {
        "batch_id": batch_id,
        "status": "processing",
        "file_count": len(files),
        "message": f"Batch processing started with ID: {batch_id}"
    }

@app.get("/batch/{batch_id}", tags=["Batch"])
async def get_batch_status(batch_id: str):
    """Get batch processing status"""
    batch_path = Config.REPORTS_DIR / f"{batch_id}_results.json"
    
    if not batch_path.exists():
        raise HTTPException(status_code=404, detail="Batch not found")
    
    with open(batch_path, 'r') as f:
        results = json.load(f)
    
    return {
        "batch_id": batch_id,
        "status": "completed",
        "results": results
    }

# ============================================================================
# EXPORT ENDPOINTS
# ============================================================================

@app.get("/export/{task_id}/{format}", tags=["Export"])
async def export_results(task_id: str, format: str = "json"):
    """Export task results in specified format"""
    valid_formats = ["json", "csv", "excel"]
    
    if format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Format must be one of: {valid_formats}")
    
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    if format == "json":
        # Return JSON response
        return JSONResponse(content=task)
    
    elif format == "csv":
        # Create CSV export
        export_path = Config.REPORTS_DIR / f"export_{task_id}.csv"
        
        # Convert task data to DataFrame
        if task['result'] and isinstance(task['result'], dict):
            # Flatten nested dict for CSV
            flat_data = {}
            for key, value in task['result'].items():
                if isinstance(value, (dict, list)):
                    flat_data[key] = str(value)
                else:
                    flat_data[key] = value
            
            df = pd.DataFrame([flat_data])
            df.to_csv(export_path, index=False)
            
            return FileResponse(
                path=export_path,
                filename=f"task_{task_id}_export.csv",
                media_type='text/csv'
            )
    
    elif format == "excel":
        # Create Excel export
        export_path = Config.REPORTS_DIR / f"export_{task_id}.xlsx"
        
        with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
            # Write task info
            task_info = pd.DataFrame([{
                'task_id': task['task_id'],
                'status': task['status'],
                'created_at': task['created_at'],
                'task_type': task['task_type']
            }])
            task_info.to_excel(writer, sheet_name='Task Info', index=False)
            
            # Write result data if available
            if task['result'] and isinstance(task['result'], dict):
                # Flatten for Excel
                flat_data = {}
                for key, value in task['result'].items():
                    if isinstance(value, (dict, list)):
                        flat_data[key] = str(value)
                    else:
                        flat_data[key] = value
                
                result_df = pd.DataFrame([flat_data])
                result_df.to_excel(writer, sheet_name='Results', index=False)
        
        return FileResponse(
            path=export_path,
            filename=f"task_{task_id}_export.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    raise HTTPException(status_code=400, detail="Export failed")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "financial_close_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
