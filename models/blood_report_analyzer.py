"""
Standalone Blood Report Analyzer - No Backend Required
Analyzes blood reports using Gemini AI directly
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv(override=True)

# Configure Gemini
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_KEY and 'your' not in GEMINI_KEY.lower():
    genai.configure(api_key=GEMINI_KEY)


class BloodReportAnalyzer:
    """Analyze blood reports without requiring backend API"""
    
    def __init__(self):
        """Initialize blood markers reference"""
        self.blood_markers = {
            'hemoglobin': {'normal_min': 13.5, 'normal_max': 17.5, 'unit': 'g/dL', 'name': 'Hemoglobin'},
            'hematocrit': {'normal_min': 41, 'normal_max': 53, 'unit': '%', 'name': 'Hematocrit'},
            'rbc': {'normal_min': 4.5, 'normal_max': 5.9, 'unit': 'million/mcL', 'name': 'Red Blood Cells'},
            'wbc': {'normal_min': 4.5, 'normal_max': 11.0, 'unit': 'thousand/mcL', 'name': 'White Blood Cells'},
            'platelets': {'normal_min': 150, 'normal_max': 400, 'unit': 'thousand/mcL', 'name': 'Platelets'},
            'glucose': {'normal_min': 70, 'normal_max': 100, 'unit': 'mg/dL', 'name': 'Glucose'},
            'creatinine': {'normal_min': 0.7, 'normal_max': 1.3, 'unit': 'mg/dL', 'name': 'Creatinine'},
            'bun': {'normal_min': 7, 'normal_max': 20, 'unit': 'mg/dL', 'name': 'Blood Urea Nitrogen'},
            'sodium': {'normal_min': 136, 'normal_max': 145, 'unit': 'mEq/L', 'name': 'Sodium'},
            'potassium': {'normal_min': 3.5, 'normal_max': 5.0, 'unit': 'mEq/L', 'name': 'Potassium'},
            'calcium': {'normal_min': 8.5, 'normal_max': 10.2, 'unit': 'mg/dL', 'name': 'Calcium'},
            'alt': {'normal_min': 7, 'normal_max': 35, 'unit': 'U/L', 'name': 'ALT (Liver)'},
            'ast': {'normal_min': 10, 'normal_max': 40, 'unit': 'U/L', 'name': 'AST (Liver)'},
            'cholesterol': {'normal_min': 0, 'normal_max': 200, 'unit': 'mg/dL', 'name': 'Total Cholesterol'},
            'triglycerides': {'normal_min': 0, 'normal_max': 150, 'unit': 'mg/dL', 'name': 'Triglycerides'},
            'ldl': {'normal_min': 0, 'normal_max': 100, 'unit': 'mg/dL', 'name': 'LDL Cholesterol'},
            'hdl': {'normal_min': 40, 'normal_max': 200, 'unit': 'mg/dL', 'name': 'HDL Cholesterol'}
        }
    
    def analyze(self, report_text: str) -> Dict[str, Any]:
        """
        Analyze blood report using Gemini
        
        Args:
            report_text: Blood report text or image-based report description
        
        Returns:
            Analysis results with parameters and findings
        """
        
        if not report_text or len(report_text.strip()) < 10:
            return {
                "success": False,
                "error": "Invalid report text",
                "findings": []
            }
        
        try:
            # Use Gemini to extract parameters
            prompt = f"""
You are a medical laboratory analyzer. Extract all blood test parameters from this report.

For each parameter, provide:
1. Parameter name
2. Patient's value
3. Unit of measurement
4. Normal range
5. Status (Normal/Low/High)
6. Clinical significance (1-2 sentences)

Format each as JSON:
{{
  "parameter": "name",
  "value": number,
  "unit": "unit",
  "normal_range": "range",
  "status": "Normal/Low/High",
  "significance": "brief clinical note"
}}

BLOOD REPORT:
{report_text}

Provide response as JSON array only, no other text.
"""
            
            model = genai.GenerativeModel('models/gemini-2.0-flash')  # Use latest available model
            response = model.generate_content(prompt)
            
            if not response or not response.text:
                return {
                    "success": False,
                    "error": "No response from AI",
                    "findings": []
                }
            
            # Parse Gemini response
            response_text = response.text.strip()
            
            # Handle markdown code blocks
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            response_text = response_text.strip()
            
            try:
                parameters = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON array from response
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    parameters = json.loads(json_match.group())
                else:
                    return {
                        "success": False,
                        "error": "Could not parse AI response",
                        "raw_response": response_text[:200],
                        "findings": []
                    }
            
            # Ensure it's a list
            if not isinstance(parameters, list):
                if isinstance(parameters, dict):
                    parameters = [parameters]
                else:
                    parameters = []
            
            # Build findings
            findings = []
            abnormalities = []
            normal_count = 0
            
            for param in parameters:
                if isinstance(param, dict):
                    param_name = param.get('parameter', '').lower()
                    value = param.get('value')
                    unit = param.get('unit', '')
                    status = param.get('status', 'Unknown').upper()
                    significance = param.get('significance', '')
                    
                    # Add finding
                    finding = f"{param_name}: {value} {unit} - {status}"
                    if significance:
                        finding += f" ({significance})"
                    findings.append(finding)
                    
                    # Track abnormalities
                    if 'LOW' in status or 'HIGH' in status:
                        abnormalities.append(finding)
                    elif 'NORMAL' in status:
                        normal_count += 1
            
            # Generate overall interpretation
            interpretation = self._generate_interpretation(findings, abnormalities)
            
            return {
                "success": True,
                "parameters": parameters,
                "findings": findings,
                "abnormalities": abnormalities,
                "normal_count": normal_count,
                "interpretation": interpretation,
                "recommendation": "Please consult with a healthcare provider for official diagnosis and treatment."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "findings": []
            }
    
    def _generate_interpretation(self, findings: List[str], abnormalities: List[str]) -> str:
        """Generate clinical interpretation"""
        
        if not findings:
            return "No parameters found in report."
        
        if not abnormalities:
            return "✅ All parameters are within normal ranges. No immediate concerns detected."
        
        if len(abnormalities) == 1:
            return f"⚠️ One parameter is abnormal: {abnormalities[0]}. Consider follow-up testing."
        
        return f"⚠️ Multiple parameters are abnormal ({len(abnormalities)} found). Recommend immediate medical consultation for detailed evaluation."


# Test function
def test_analyzer():
    """Test the blood report analyzer"""
    
    test_report = """
    Blood Test Results - 2026-01-22
    Patient: John Doe
    
    Complete Blood Count:
    Hemoglobin: 15.2 g/dL (Normal: 13.5-17.5)
    Hematocrit: 45% (Normal: 41-53%)
    WBC: 7.2 thousand/mcL (Normal: 4.5-11.0)
    Platelets: 250 thousand/mcL (Normal: 150-400)
    
    Metabolic Panel:
    Glucose: 105 mg/dL (Normal: 70-100) - SLIGHTLY ELEVATED
    Creatinine: 0.9 mg/dL (Normal: 0.7-1.3)
    Potassium: 4.1 mEq/L (Normal: 3.5-5.0)
    Sodium: 138 mEq/L (Normal: 136-145)
    
    Lipid Panel:
    Total Cholesterol: 210 mg/dL (Normal: <200) - HIGH
    LDL: 130 mg/dL (Normal: <100) - HIGH
    HDL: 45 mg/dL (Normal: >40)
    Triglycerides: 145 mg/dL (Normal: <150)
    """
    
    analyzer = BloodReportAnalyzer()
    results = analyzer.analyze(test_report)
    
    print("\n" + "="*70)
    print("BLOOD REPORT ANALYSIS TEST")
    print("="*70)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    test_analyzer()
