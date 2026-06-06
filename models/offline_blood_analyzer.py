"""
Offline Blood Report Analyzer - No API Quota Required
Uses pattern matching and reference ranges for fast analysis
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple


class OfflineBloodReportAnalyzer:
    """Analyze blood reports using local pattern matching - NO API calls required"""
    
    def __init__(self):
        """Initialize blood markers reference with normal ranges"""
        self.blood_markers = {
            'hemoglobin': {'aliases': ['hgb', 'hb'], 'normal_min': 13.5, 'normal_max': 17.5, 'unit': 'g/dL'},
            'hematocrit': {'aliases': ['hct'], 'normal_min': 41, 'normal_max': 53, 'unit': '%'},
            'rbc': {'aliases': ['red blood cell', 'red blood cells'], 'normal_min': 4.5, 'normal_max': 5.9, 'unit': 'million/mcL'},
            'wbc': {'aliases': ['white blood cell', 'white blood cells'], 'normal_min': 4.5, 'normal_max': 11.0, 'unit': 'thousand/mcL'},
            'platelets': {'aliases': ['plt'], 'normal_min': 150, 'normal_max': 400, 'unit': 'thousand/mcL'},
            'glucose': {'aliases': [], 'normal_min': 70, 'normal_max': 100, 'unit': 'mg/dL'},
            'creatinine': {'aliases': [], 'normal_min': 0.7, 'normal_max': 1.3, 'unit': 'mg/dL'},
            'bun': {'aliases': ['blood urea nitrogen'], 'normal_min': 7, 'normal_max': 20, 'unit': 'mg/dL'},
            'sodium': {'aliases': ['na'], 'normal_min': 136, 'normal_max': 145, 'unit': 'mEq/L'},
            'potassium': {'aliases': ['k'], 'normal_min': 3.5, 'normal_max': 5.0, 'unit': 'mEq/L'},
            'calcium': {'aliases': ['ca'], 'normal_min': 8.5, 'normal_max': 10.2, 'unit': 'mg/dL'},
            'alt': {'aliases': ['alanine aminotransferase'], 'normal_min': 7, 'normal_max': 35, 'unit': 'U/L'},
            'ast': {'aliases': ['aspartate aminotransferase'], 'normal_min': 10, 'normal_max': 40, 'unit': 'U/L'},
            'cholesterol': {'aliases': ['total cholesterol'], 'normal_min': 0, 'normal_max': 200, 'unit': 'mg/dL'},
            'triglycerides': {'aliases': ['trig'], 'normal_min': 0, 'normal_max': 150, 'unit': 'mg/dL'},
            'ldl': {'aliases': ['ldl cholesterol', 'low density lipoprotein'], 'normal_min': 0, 'normal_max': 100, 'unit': 'mg/dL'},
            'hdl': {'aliases': ['hdl cholesterol', 'high density lipoprotein'], 'normal_min': 40, 'normal_max': 200, 'unit': 'mg/dL'}
        }
    
    def extract_value_and_unit(self, text: str) -> Tuple[float, str]:
        """Extract numeric value and unit from text"""
        # Look for patterns like "15.2 g/dL" or "105 mg/dL"
        pattern = r'(\d+\.?\d*)\s*([a-zA-Z%/]*)'
        match = re.search(pattern, text.strip())
        
        if match:
            value_str = match.group(1)
            unit = match.group(2).strip()
            try:
                value = float(value_str)
                return value, unit
            except ValueError:
                return None, None
        return None, None
    
    def find_parameters(self, report_text: str) -> List[Dict[str, Any]]:
        """Find blood parameters in report text using pattern matching"""
        
        parameters = []
        report_lower = report_text.lower()
        
        # Split into lines and search for parameters
        lines = report_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Check each parameter
            for param_key, param_info in self.blood_markers.items():
                # Check if parameter name or aliases appear in line
                if param_key in line_lower:
                    # Extract value and unit
                    value, unit = self.extract_value_and_unit(line)
                    
                    if value is not None:
                        # Determine status
                        if value < param_info['normal_min']:
                            status = "LOW"
                            icon = "🔴"
                        elif value > param_info['normal_max']:
                            status = "HIGH"
                            icon = "🔴"
                        else:
                            status = "NORMAL"
                            icon = "✅"
                        
                        parameters.append({
                            'parameter': param_key.upper(),
                            'value': value,
                            'unit': unit or param_info['unit'],
                            'normal_min': param_info['normal_min'],
                            'normal_max': param_info['normal_max'],
                            'normal_range': f"{param_info['normal_min']}-{param_info['normal_max']}",
                            'status': status,
                            'icon': icon
                        })
                        break  # Move to next line
                
                # Also check aliases
                for alias in param_info.get('aliases', []):
                    if alias and alias in line_lower:
                        value, unit = self.extract_value_and_unit(line)
                        if value is not None:
                            if value < param_info['normal_min']:
                                status = "LOW"
                                icon = "🔴"
                            elif value > param_info['normal_max']:
                                status = "HIGH"
                                icon = "🔴"
                            else:
                                status = "NORMAL"
                                icon = "✅"
                            
                            parameters.append({
                                'parameter': param_key.upper(),
                                'value': value,
                                'unit': unit or param_info['unit'],
                                'normal_min': param_info['normal_min'],
                                'normal_max': param_info['normal_max'],
                                'normal_range': f"{param_info['normal_min']}-{param_info['normal_max']}",
                                'status': status,
                                'icon': icon
                            })
                            break
        
        return parameters
    
    def analyze(self, report_text: str) -> Dict[str, Any]:
        """
        Analyze blood report locally (no API required)
        
        Args:
            report_text: Blood report text
        
        Returns:
            Analysis results
        """
        
        if not report_text or len(report_text.strip()) < 10:
            return {
                "success": False,
                "error": "Invalid report text",
                "findings": []
            }
        
        try:
            # Extract parameters
            parameters = self.find_parameters(report_text)
            
            if not parameters:
                return {
                    "success": False,
                    "error": "No blood parameters found in report. Please ensure the report contains parameter names and values.",
                    "tips": [
                        "Report should contain parameter names like: Hemoglobin, Glucose, WBC, etc.",
                        "Values should be in numeric format with units",
                        "Example: 'Hemoglobin: 15.2 g/dL'"
                    ],
                    "findings": []
                }
            
            # Separate normal and abnormal
            abnormal = [p for p in parameters if p['status'] != 'NORMAL']
            normal = [p for p in parameters if p['status'] == 'NORMAL']
            
            # Generate findings
            findings = []
            for param in parameters:
                finding = f"{param['icon']} {param['parameter']}: {param['value']} {param['unit']} (Normal: {param['normal_range']}) - {param['status']}"
                findings.append(finding)
            
            # Generate interpretation
            if not abnormal:
                interpretation = f"✅ All {len(normal)} parameters are within normal ranges. No immediate concerns detected."
            elif len(abnormal) == 1:
                param = abnormal[0]
                interpretation = f"⚠️ {param['parameter']} is {param['status']} ({param['value']} {param['unit']}, normal: {param['normal_range']}). Consider follow-up testing."
            else:
                interpretation = f"⚠️ {len(abnormal)} abnormal parameter(s) detected: {', '.join([p['parameter'] for p in abnormal])}. Recommend medical consultation."
            
            return {
                "success": True,
                "parameters": parameters,
                "findings": findings,
                "abnormal_count": len(abnormal),
                "normal_count": len(normal),
                "total_count": len(parameters),
                "interpretation": interpretation,
                "recommendation": "⚠️ This is an automated analysis based on standard reference ranges. Please consult with a healthcare provider for official diagnosis and personalized treatment."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "findings": []
            }


# Test function
def test_analyzer():
    """Test the offline blood report analyzer"""
    
    test_report = """
    Blood Test Results - 2026-01-22
    Patient: John Doe
    
    Complete Blood Count:
    Hemoglobin: 15.2 g/dL
    Hematocrit: 45%
    WBC: 7.2 thousand/mcL
    Platelets: 250 thousand/mcL
    
    Metabolic Panel:
    Glucose: 105 mg/dL
    Creatinine: 0.9 mg/dL
    Potassium: 4.1 mEq/L
    Sodium: 138 mEq/L
    
    Lipid Panel:
    Total Cholesterol: 210 mg/dL
    LDL: 130 mg/dL
    HDL: 45 mg/dL
    Triglycerides: 145 mg/dL
    """
    
    analyzer = OfflineBloodReportAnalyzer()
    results = analyzer.analyze(test_report)
    
    print("\n" + "="*70)
    print("OFFLINE BLOOD REPORT ANALYSIS TEST")
    print("="*70)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    test_analyzer()
