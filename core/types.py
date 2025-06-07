"""Type definitions for nephrotoxicity analysis pipeline."""

import pandas as pd
from typing import Dict, Any


class ComponentResult:
    """Container for component execution results."""
    
    def __init__(self, component_name: str, success: bool = True, 
                 data: Dict[str, Any] = None, error: str = None):
        """
        Initialize component result.
        
        Args:
            component_name: Name of the component
            success: Whether execution was successful
            data: Result data dictionary
            error: Error message if execution failed
        """
        self.component_name = component_name
        self.success = success
        self.data = data or {}
        self.error = error
        self.timestamp = pd.Timestamp.now()
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"ComponentResult({self.component_name}: {status})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'component_name': self.component_name,
            'success': self.success,
            'data_keys': list(self.data.keys()) if self.data else [],
            'error': self.error,
            'timestamp': self.timestamp.isoformat()
        }
