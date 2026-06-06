"""
Differential Diagnosis Engine
Structured differential diagnosis with Bayesian reasoning
Ranks diagnoses by likelihood, suggests next steps
Production-grade decision support
"""

import logging
from typing import List, Dict, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DifferentialDiagnosisEngine:
    """
    Structured differential diagnosis system
    Uses Bayesian reasoning and clinical likelihood ratios
    """
    
    def __init__(self):
        """Initialize differential diagnosis engine"""
        logger.info("Initializing Differential Diagnosis Engine...")
        
        # Disease prevalence database (base rates)
        self.prevalence_db = self._build_prevalence_database()
        
        # Symptom-disease associations
        self.symptom_disease_map = self._build_symptom_disease_map()
        
        # Diagnostic test information
        self.diagnostic_tests = self._build_diagnostic_tests()
        
        logger.info("✅ Differential Diagnosis Engine initialized")
        self.loaded = True
    
    def _build_prevalence_database(self) -> Dict[str, float]:
        """Build disease prevalence database (prior probabilities)"""
        return {
            # Common conditions
            'pneumonia': 0.08,
            'bronchitis': 0.12,
            'asthma': 0.06,
            'copd': 0.05,
            'influenza': 0.04,
            'common_cold': 0.15,
            'covid_19': 0.03,
            
            # Cardiac
            'acute_coronary_syndrome': 0.02,
            'myocardial_infarction': 0.01,
            'heart_failure': 0.03,
            'arrhythmia': 0.02,
            'pericarditis': 0.005,
            
            # GI
            'gastroenteritis': 0.10,
            'peptic_ulcer': 0.04,
            'gerd': 0.20,
            'appendicitis': 0.01,
            'pancreatitis': 0.002,
            
            # Metabolic
            'diabetes': 0.09,
            'thyroid_disorder': 0.05,
            'hypoglycemia': 0.003,
            
            # CNS
            'migraine': 0.15,
            'meningitis': 0.0002,
            'stroke': 0.001,
            'seizure': 0.003,
            
            # Other
            'anemia': 0.05,
            'urinary_tract_infection': 0.08,
            'acute_kidney_injury': 0.01,
        }
    
    def _build_symptom_disease_map(self) -> Dict[str, List[Dict[str, any]]]:
        """Build symptom-disease associations with likelihood ratios"""
        return {
            'fever': [
                {'disease': 'pneumonia', 'lr_plus': 2.5, 'lr_minus': 0.3},
                {'disease': 'influenza', 'lr_plus': 3.0, 'lr_minus': 0.2},
                {'disease': 'gastroenteritis', 'lr_plus': 2.0, 'lr_minus': 0.4},
                {'disease': 'meningitis', 'lr_plus': 5.0, 'lr_minus': 0.05},
                {'disease': 'urinary_tract_infection', 'lr_plus': 1.5, 'lr_minus': 0.5},
            ],
            'cough': [
                {'disease': 'pneumonia', 'lr_plus': 3.0, 'lr_minus': 0.2},
                {'disease': 'bronchitis', 'lr_plus': 2.8, 'lr_minus': 0.25},
                {'disease': 'asthma', 'lr_plus': 2.0, 'lr_minus': 0.4},
                {'disease': 'covid_19', 'lr_plus': 1.8, 'lr_minus': 0.45},
                {'disease': 'gerd', 'lr_plus': 1.5, 'lr_minus': 0.6},
            ],
            'chest_pain': [
                {'disease': 'acute_coronary_syndrome', 'lr_plus': 2.0, 'lr_minus': 0.3},
                {'disease': 'myocardial_infarction', 'lr_plus': 3.0, 'lr_minus': 0.2},
                {'disease': 'pericarditis', 'lr_plus': 4.0, 'lr_minus': 0.1},
                {'disease': 'pneumonia', 'lr_plus': 1.5, 'lr_minus': 0.5},
                {'disease': 'gerd', 'lr_plus': 1.2, 'lr_minus': 0.7},
            ],
            'shortness_of_breath': [
                {'disease': 'pneumonia', 'lr_plus': 2.8, 'lr_minus': 0.25},
                {'disease': 'asthma', 'lr_plus': 3.0, 'lr_minus': 0.2},
                {'disease': 'copd', 'lr_plus': 2.5, 'lr_minus': 0.3},
                {'disease': 'acute_coronary_syndrome', 'lr_plus': 1.5, 'lr_minus': 0.5},
                {'disease': 'heart_failure', 'lr_plus': 2.0, 'lr_minus': 0.4},
            ],
            'abdominal_pain': [
                {'disease': 'gastroenteritis', 'lr_plus': 2.5, 'lr_minus': 0.3},
                {'disease': 'appendicitis', 'lr_plus': 8.0, 'lr_minus': 0.05},
                {'disease': 'peptic_ulcer', 'lr_plus': 2.0, 'lr_minus': 0.4},
                {'disease': 'pancreatitis', 'lr_plus': 6.0, 'lr_minus': 0.1},
            ],
            'headache': [
                {'disease': 'migraine', 'lr_plus': 3.0, 'lr_minus': 0.2},
                {'disease': 'meningitis', 'lr_plus': 4.0, 'lr_minus': 0.15},
                {'disease': 'tension_headache', 'lr_plus': 2.0, 'lr_minus': 0.3},
                {'disease': 'influenza', 'lr_plus': 1.5, 'lr_minus': 0.5},
            ],
        }
    
    def _build_diagnostic_tests(self) -> Dict[str, Dict[str, any]]:
        """Build diagnostic test information"""
        return {
            'chest_xray': {
                'sensitivity': 0.85,
                'specificity': 0.90,
                'usefulness': 'HIGH',
                'cost': '$100-200',
                'time': '30 minutes',
                'diseases': ['pneumonia', 'tuberculosis', 'lung_cancer']
            },
            'cbc': {
                'sensitivity': 0.80,
                'specificity': 0.85,
                'usefulness': 'HIGH',
                'cost': '$15-30',
                'time': '1-2 hours',
                'diseases': ['anemia', 'infection', 'leukemia']
            },
            'blood_culture': {
                'sensitivity': 0.75,
                'specificity': 0.98,
                'usefulness': 'HIGH',
                'cost': '$50-100',
                'time': '24-48 hours',
                'diseases': ['sepsis', 'bacteremia']
            },
            'ekg': {
                'sensitivity': 0.95,
                'specificity': 0.90,
                'usefulness': 'HIGH',
                'cost': '$20-50',
                'time': '5 minutes',
                'diseases': ['myocardial_infarction', 'arrhythmia']
            },
            'troponin': {
                'sensitivity': 0.98,
                'specificity': 0.96,
                'usefulness': 'CRITICAL',
                'cost': '$10-20',
                'time': '30 minutes',
                'diseases': ['myocardial_infarction']
            },
            'ct_scan': {
                'sensitivity': 0.95,
                'specificity': 0.92,
                'usefulness': 'HIGH',
                'cost': '$500-1000',
                'time': '15 minutes',
                'diseases': ['appendicitis', 'pancreatitis', 'stroke']
            },
            'csf_analysis': {
                'sensitivity': 0.90,
                'specificity': 0.95,
                'usefulness': 'CRITICAL',
                'cost': '$150-200',
                'time': '1-2 hours',
                'diseases': ['meningitis', 'encephalitis']
            },
        }
    
    def generate_differential_diagnosis(self, symptoms: List[str], 
                                       vital_signs: Dict[str, float] = None,
                                       lab_results: Dict[str, any] = None) -> Dict[str, any]:
        """
        Generate differential diagnosis with Bayesian reasoning
        
        Args:
            symptoms (list): Patient symptoms
            vital_signs (dict): Vital signs if available
            lab_results (dict): Lab test results if available
            
        Returns:
            dict: Ranked differential diagnosis
        """
        try:
            # Calculate posterior probabilities for each disease
            disease_probabilities = {}
            
            for disease, prior in self.prevalence_db.items():
                # Start with prior probability
                posterior = prior
                
                # Apply likelihood ratios from symptoms
                for symptom in symptoms:
                    symptom_lower = symptom.lower().replace(' ', '_')
                    if symptom_lower in self.symptom_disease_map:
                        for assoc in self.symptom_disease_map[symptom_lower]:
                            if assoc['disease'] == disease:
                                # Bayesian update
                                posterior *= assoc['lr_plus']
            
                disease_probabilities[disease] = posterior
            
            # Normalize probabilities
            total = sum(disease_probabilities.values())
            if total > 0:
                disease_probabilities = {k: v/total for k, v in disease_probabilities.items()}
            
            # Sort by probability
            sorted_diagnoses = sorted(
                disease_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Build differential diagnosis list
            differential = []
            for rank, (disease, probability) in enumerate(sorted_diagnoses[:5], 1):
                differential.append({
                    'rank': rank,
                    'disease': disease,
                    'probability': round(probability * 100, 1),
                    'confidence': self._get_confidence_level(probability),
                    'severity': self._estimate_severity(disease),
                    'urgency': self._estimate_urgency(disease),
                    'recommended_tests': self._get_recommended_tests(disease)
                })
            
            return {
                'symptoms': symptoms,
                'differential_diagnoses': differential,
                'timestamp': datetime.now().isoformat(),
                'total_diseases_considered': len(disease_probabilities),
                'note': 'This is clinical decision support, not a diagnosis'
            }
        except Exception as e:
            logger.error(f"Error generating differential diagnosis: {str(e)}")
            return {"error": str(e)}
    
    def _get_confidence_level(self, probability: float) -> str:
        """Get confidence level from probability"""
        if probability >= 0.40:
            return "HIGH"
        elif probability >= 0.20:
            return "MODERATE"
        elif probability >= 0.05:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _estimate_severity(self, disease: str) -> str:
        """Estimate disease severity"""
        critical = ['meningitis', 'myocardial_infarction', 'sepsis', 'stroke']
        severe = ['pneumonia', 'acute_coronary_syndrome', 'pancreatitis']
        
        if disease in critical:
            return "CRITICAL"
        elif disease in severe:
            return "SEVERE"
        else:
            return "MODERATE"
    
    def _estimate_urgency(self, disease: str) -> str:
        """Estimate urgency of evaluation"""
        emergencies = ['meningitis', 'myocardial_infarction', 'sepsis', 'stroke', 'appendicitis']
        urgent = ['pneumonia', 'acute_coronary_syndrome', 'pancreatitis']
        
        if disease in emergencies:
            return "EMERGENCY (Call 911)"
        elif disease in urgent:
            return "URGENT (See doctor immediately)"
        else:
            return "SOON (See doctor within 24 hours)"
    
    def _get_recommended_tests(self, disease: str) -> List[Dict[str, str]]:
        """Get recommended diagnostic tests for a disease"""
        test_map = {
            'pneumonia': ['chest_xray', 'cbc', 'blood_culture'],
            'myocardial_infarction': ['ekg', 'troponin', 'cbc'],
            'meningitis': ['csf_analysis', 'cbc', 'blood_culture'],
            'appendicitis': ['ct_scan', 'cbc'],
            'pancreatitis': ['ct_scan', 'cbc'],
        }
        
        test_names = test_map.get(disease, ['cbc', 'chest_xray'])
        tests = []
        
        for test_name in test_names:
            if test_name in self.diagnostic_tests:
                test_info = self.diagnostic_tests[test_name].copy()
                test_info['name'] = test_name
                tests.append(test_info)
        
        return tests
    
    def suggest_next_test(self, current_diagnosis: str) -> Dict[str, any]:
        """
        Suggest next diagnostic test based on current diagnosis
        
        Args:
            current_diagnosis (str): Current suspected diagnosis
            
        Returns:
            dict: Recommended next test
        """
        if current_diagnosis not in self._get_recommended_tests(current_diagnosis):
            tests = self._get_recommended_tests(current_diagnosis)
            if tests:
                return {
                    'diagnosis': current_diagnosis,
                    'recommended_test': tests[0]['name'],
                    'test_details': tests[0],
                    'rationale': f"This test will help confirm or rule out {current_diagnosis}"
                }
        
        return {"message": "No additional tests recommended at this time"}
    
    def calculate_clinical_score(self, symptoms: List[str], 
                                vital_signs: Dict[str, float] = None) -> Dict[str, any]:
        """
        Calculate clinical severity score (0-100)
        
        Args:
            symptoms (list): Patient symptoms
            vital_signs (dict): Vital signs
            
        Returns:
            dict: Severity score and interpretation
        """
        score = 0
        
        # Symptom severity
        severe_symptoms = ['chest_pain', 'difficulty_breathing', 'unconscious', 'severe_bleeding']
        for symptom in symptoms:
            if symptom.lower() in severe_symptoms:
                score += 25
            else:
                score += 10
        
        # Vital signs
        if vital_signs:
            if vital_signs.get('heart_rate', 0) > 120:
                score += 10
            if vital_signs.get('respiratory_rate', 0) > 25:
                score += 10
            if vital_signs.get('systolic_bp', 0) < 90:
                score += 15
            if vital_signs.get('temperature', 0) > 39:
                score += 5
        
        score = min(100, score)
        
        return {
            'severity_score': score,
            'severity_level': 'CRITICAL' if score >= 80 else 'SEVERE' if score >= 60 else 'MODERATE' if score >= 40 else 'MILD',
            'recommendation': 'CALL 911' if score >= 80 else 'Go to ER' if score >= 60 else 'Urgent care' if score >= 40 else 'Contact doctor'
        }


def test_differential_diagnosis():
    """Test differential diagnosis engine"""
    engine = DifferentialDiagnosisEngine()
    
    print("\n" + "="*70)
    print("DIFFERENTIAL DIAGNOSIS ENGINE TEST")
    print("="*70)
    
    # Test case 1: Respiratory symptoms
    print("\n[1] Respiratory Symptoms:")
    symptoms1 = ['fever', 'cough', 'shortness_of_breath']
    result1 = engine.generate_differential_diagnosis(symptoms1)
    print(f"Symptoms: {symptoms1}\n")
    for dx in result1['differential_diagnoses']:
        print(f"  {dx['rank']}. {dx['disease'].replace('_', ' ').title()}")
        print(f"     Probability: {dx['probability']}%")
        print(f"     Severity: {dx['severity']}")
        print(f"     Urgency: {dx['urgency']}")
    
    # Test case 2: Chest pain
    print("\n[2] Chest Pain:")
    symptoms2 = ['chest_pain']
    result2 = engine.generate_differential_diagnosis(symptoms2)
    print(f"Symptoms: {symptoms2}\n")
    for dx in result2['differential_diagnoses'][:3]:
        print(f"  {dx['rank']}. {dx['disease'].replace('_', ' ').title()}")
        print(f"     Probability: {dx['probability']}%")
        print(f"     Recommended Tests: {[t['name'] for t in dx['recommended_tests']]}")
    
    # Test case 3: Severity scoring
    print("\n[3] Clinical Severity Score:")
    vital_signs = {
        'heart_rate': 130,
        'respiratory_rate': 28,
        'systolic_bp': 85,
        'temperature': 39.5
    }
    score = engine.calculate_clinical_score(symptoms1, vital_signs)
    print(f"Severity Score: {score['severity_score']}/100")
    print(f"Level: {score['severity_level']}")
    print(f"Recommendation: {score['recommendation']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_differential_diagnosis()
