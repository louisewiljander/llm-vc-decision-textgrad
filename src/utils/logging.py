"""
Logging utilities for API call analysis and debugging.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class APILogger:
    """Query and analyze API call logs."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            db_path: Path to SQLite database (defaults to results/logs/api_calls.db)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "results" / "logs" / "api_calls.db"
        self.db_path = Path(db_path)
    
    def get_dataframe(
        self,
        agent_name: Optional[str] = None,
        hours: Optional[int] = None,
        include_errors: bool = False
    ) -> pd.DataFrame:
        """
        Get logs as pandas DataFrame for analysis.
        
        Args:
            agent_name: Filter by agent name
            hours: Filter to last N hours
            include_errors: Whether to include error records
            
        Returns:
            DataFrame with log records
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM api_calls WHERE 1=1"
            params = []
            
            if not include_errors:
                query += " AND error_message IS NULL"
            
            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)
            
            if hours:
                cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
                query += " AND timestamp > ?"
                params.append(cutoff_time)
            
            query += " ORDER BY timestamp DESC"
            
            return pd.read_sql_query(query, conn, params=params)
    
    def print_summary(
        self,
        agent_name: Optional[str] = None,
        hours: Optional[int] = None
    ):
        """Print summary statistics."""
        df = self.get_dataframe(agent_name=agent_name, hours=hours)
        
        if df.empty:
            print("No API calls found.")
            return
        
        print("\n" + "="*80)
        print("API CALL SUMMARY")
        print("="*80)
        
        print(f"\nTotal calls: {len(df)}")
        print(f"Total cost: ${df['total_cost_usd'].sum():.4f}")
        print(f"\nTokens:")
        print(f"  Input: {df['input_tokens'].sum():,}")
        print(f"  Output: {df['output_tokens'].sum():,}")
        print(f"  Cache created: {df['cache_creation_input_tokens'].sum():,}")
        print(f"  Cache read: {df['cache_read_input_tokens'].sum():,}")
        
        # Calculate savings
        cache_read = df['cache_read_input_tokens'].sum()
        savings = (cache_read * 3 * 0.9) / 1_000_000
        print(f"\nEstimated cache savings: ${savings:.4f}")
        
        if df['cache_read_input_tokens'].sum() > 0:
            cache_pct = (df['cache_read_input_tokens'].sum() / (df['input_tokens'].sum() + df['cache_read_input_tokens'].sum())) * 100
            print(f"Cache hit rate: {cache_pct:.1f}%")
        
        # By agent
        if 'agent_name' in df.columns:
            print(f"\nBy agent:")
            by_agent = df.groupby('agent_name').agg({
                'id': 'count',
                'total_cost_usd': 'sum',
                'input_tokens': 'sum',
                'output_tokens': 'sum',
                'cache_read_input_tokens': 'sum'
            }).rename(columns={'id': 'calls'})
            print(by_agent)
        
        print("\n" + "="*80 + "\n")
    
    def print_recent(self, limit: int = 10, agent_name: Optional[str] = None):
        """Print recent API calls."""
        df = self.get_dataframe(agent_name=agent_name)
        df = df.head(limit)
        
        if df.empty:
            print("No API calls found.")
            return
        
        print("\n" + "="*80)
        print(f"RECENT API CALLS (latest {limit})")
        print("="*80 + "\n")
        
        for idx, row in df.iterrows():
            print(f"Timestamp: {row['timestamp']}")
            print(f"Agent: {row['agent_name']}")
            print(f"Model: {row['model']}")
            print(f"Tokens: {row['input_tokens']} in → {row['output_tokens']} out")
            
            if row['cache_read_input_tokens'] > 0:
                print(f"Cache read: {row['cache_read_input_tokens']} tokens (10% cost)")
            if row['cache_creation_input_tokens'] > 0:
                print(f"Cache created: {row['cache_creation_input_tokens']} tokens")
            
            print(f"Cost: ${row['total_cost_usd']:.6f}")
            
            if row['error_message']:
                print(f"Error: {row['error_message']}")
            else:
                user_msg = row['user_message'][:100].replace('\n', ' ')
                response = row['assistant_response'][:100].replace('\n', ' ')
                print(f"Message: {user_msg}...")
                print(f"Response: {response}...")
            
            print("-" * 80)
        
        print()
    
    def cost_breakdown(self) -> dict:
        """Get cost breakdown."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            result = cursor.execute("""
                SELECT 
                    SUM(input_tokens * 3.0 / 1000000) as input_cost,
                    SUM(output_tokens * 15.0 / 1000000) as output_cost,
                    SUM(cache_creation_input_tokens * 3.75 / 1000000) as cache_creation_cost,
                    SUM(cache_read_input_tokens * 0.30 / 1000000) as cache_read_cost,
                    SUM(total_cost_usd) as total_cost
                FROM api_calls
                WHERE error_message IS NULL
            """).fetchone()
            
            input_cost, output_cost, cache_creation, cache_read, total = result
            
            return {
                "input_cost": input_cost or 0.0,
                "output_cost": output_cost or 0.0,
                "cache_creation_cost": cache_creation or 0.0,
                "cache_read_cost": cache_read or 0.0,
                "total_cost": total or 0.0
            }


if __name__ == "__main__":
    logger = APILogger()
    
    # Print summary
    logger.print_summary()
    
    # Print recent calls
    logger.print_recent(limit=5)
    
    # Cost breakdown
    breakdown = logger.cost_breakdown()
    print("Cost Breakdown:")
    print(f"  Input: ${breakdown['input_cost']:.4f}")
    print(f"  Output: ${breakdown['output_cost']:.4f}")
    print(f"  Cache creation: ${breakdown['cache_creation_cost']:.4f}")
    print(f"  Cache read: ${breakdown['cache_read_cost']:.4f}")
    print(f"  Total: ${breakdown['total_cost']:.4f}")
