"""
DrugBank Dataset Loader - Free Version with CSV and SDF support
Handles vocabulary and open structures data
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import sys

# Fix Unicode issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class DrugBankLoader:
    """Load and query DrugBank free version (vocabulary + structures)"""
    
    def __init__(self, data_dir: str = "ai_doctor_data/drugbank"):
        self.data_dir = Path(data_dir)
        self.vocabulary_file = self.data_dir / "drugbank_vocabulary.csv"
        self.structures_file = self.data_dir / "open_structures.sdf"
        
        self.vocabulary_df = None
        self.drugs_index = {}
        self.drug_cache = {}
        
        # Built-in medicine database for common drugs (no external file needed)
        self.builtin_medicines = self._load_builtin_medicines()
        
        print(f"DrugBank Loader initialized")
        print(f"  Vocabulary: {self.vocabulary_file}")
        print(f"  Structures: {self.structures_file}")
        print(f"  Built-in medicines: {len(self.builtin_medicines)} drugs")
    
    def _load_builtin_medicines(self) -> Dict[str, Dict]:
        """Built-in database of common medicines with details"""
        return {
            "paracetamol": {
                "name": "Paracetamol (Acetaminophen)",
                "brand_names": ["Tylenol", "Crocin", "Dolo", "Calpol"],
                "drug_class": "Analgesic, Antipyretic",
                "uses": ["Fever", "Mild to moderate pain", "Headache", "Muscle aches", "Arthritis", "Toothache"],
                "dosage": "Adults: 500-1000mg every 4-6 hours. Max 4g/day",
                "side_effects": ["Nausea", "Liver damage (overdose)", "Allergic reactions (rare)"],
                "contraindications": ["Liver disease", "Alcohol dependence", "G6PD deficiency"],
                "interactions": ["Warfarin (increased bleeding)", "Alcohol (liver damage)"],
                "pregnancy_category": "B (Generally safe)"
            },
            "ibuprofen": {
                "name": "Ibuprofen",
                "brand_names": ["Advil", "Motrin", "Brufen", "Nurofen"],
                "drug_class": "NSAID (Non-steroidal Anti-inflammatory Drug)",
                "uses": ["Pain relief", "Inflammation", "Fever", "Arthritis", "Menstrual cramps"],
                "dosage": "Adults: 200-400mg every 4-6 hours. Max 1200mg/day (OTC)",
                "side_effects": ["Stomach upset", "GI bleeding", "Kidney problems", "Cardiovascular risk"],
                "contraindications": ["GI ulcers", "Kidney disease", "Heart failure", "3rd trimester pregnancy"],
                "interactions": ["Aspirin", "Blood thinners", "ACE inhibitors", "Lithium"],
                "pregnancy_category": "C (D in 3rd trimester)"
            },
            "aspirin": {
                "name": "Aspirin (Acetylsalicylic Acid)",
                "brand_names": ["Bayer", "Ecosprin", "Disprin"],
                "drug_class": "NSAID, Antiplatelet",
                "uses": ["Pain", "Fever", "Heart attack prevention", "Stroke prevention", "Inflammation"],
                "dosage": "Pain: 325-650mg every 4-6 hours. Cardiac: 75-325mg daily",
                "side_effects": ["GI bleeding", "Tinnitus", "Allergic reactions", "Reye's syndrome (children)"],
                "contraindications": ["Bleeding disorders", "Children with viral infections", "Aspirin allergy"],
                "interactions": ["Blood thinners", "Methotrexate", "Other NSAIDs"],
                "pregnancy_category": "D (especially 3rd trimester)"
            },
            "amoxicillin": {
                "name": "Amoxicillin",
                "brand_names": ["Amoxil", "Mox", "Novamox"],
                "drug_class": "Penicillin antibiotic",
                "uses": ["Bacterial infections", "Ear infections", "Sinusitis", "UTI", "H. pylori"],
                "dosage": "Adults: 250-500mg every 8 hours or 500-875mg every 12 hours",
                "side_effects": ["Diarrhea", "Nausea", "Rash", "Allergic reactions"],
                "contraindications": ["Penicillin allergy", "Mononucleosis (causes rash)"],
                "interactions": ["Methotrexate", "Warfarin", "Oral contraceptives"],
                "pregnancy_category": "B"
            },
            "metformin": {
                "name": "Metformin",
                "brand_names": ["Glucophage", "Glycomet", "Glyciphage"],
                "drug_class": "Biguanide (Antidiabetic)",
                "uses": ["Type 2 Diabetes", "PCOS", "Prediabetes", "Weight management"],
                "dosage": "Start 500mg twice daily, max 2000-2550mg/day",
                "side_effects": ["GI upset", "Diarrhea", "Lactic acidosis (rare)", "B12 deficiency"],
                "contraindications": ["Kidney disease", "Liver disease", "Heart failure", "Alcoholism"],
                "interactions": ["Contrast dye", "Alcohol", "Cimetidine"],
                "pregnancy_category": "B"
            },
            "omeprazole": {
                "name": "Omeprazole",
                "brand_names": ["Prilosec", "Omez", "Losec"],
                "drug_class": "Proton Pump Inhibitor (PPI)",
                "uses": ["GERD", "Peptic ulcers", "H. pylori", "Zollinger-Ellison syndrome"],
                "dosage": "20-40mg once daily before meals",
                "side_effects": ["Headache", "Diarrhea", "B12 deficiency (long-term)", "Bone fractures (long-term)"],
                "contraindications": ["Hypersensitivity to PPIs"],
                "interactions": ["Clopidogrel", "Methotrexate", "Warfarin"],
                "pregnancy_category": "C"
            },
            "lisinopril": {
                "name": "Lisinopril",
                "brand_names": ["Zestril", "Prinivil", "Listril"],
                "drug_class": "ACE Inhibitor",
                "uses": ["Hypertension", "Heart failure", "Post-MI", "Diabetic nephropathy"],
                "dosage": "Start 5-10mg daily, max 40mg/day",
                "side_effects": ["Dry cough", "Hyperkalemia", "Dizziness", "Angioedema (rare)"],
                "contraindications": ["Pregnancy", "Angioedema history", "Bilateral renal artery stenosis"],
                "interactions": ["Potassium supplements", "NSAIDs", "Lithium"],
                "pregnancy_category": "D"
            },
            "atorvastatin": {
                "name": "Atorvastatin",
                "brand_names": ["Lipitor", "Atorva", "Tonact"],
                "drug_class": "HMG-CoA Reductase Inhibitor (Statin)",
                "uses": ["High cholesterol", "Cardiovascular prevention", "Dyslipidemia"],
                "dosage": "10-80mg once daily",
                "side_effects": ["Muscle pain", "Liver enzyme elevation", "Rhabdomyolysis (rare)"],
                "contraindications": ["Liver disease", "Pregnancy", "Breastfeeding"],
                "interactions": ["Grapefruit juice", "Clarithromycin", "Cyclosporine"],
                "pregnancy_category": "X"
            },
            "amlodipine": {
                "name": "Amlodipine",
                "brand_names": ["Norvasc", "Amlong", "Amlodac"],
                "drug_class": "Calcium Channel Blocker",
                "uses": ["Hypertension", "Angina", "Coronary artery disease"],
                "dosage": "2.5-10mg once daily",
                "side_effects": ["Ankle swelling", "Flushing", "Headache", "Dizziness"],
                "contraindications": ["Severe aortic stenosis", "Hypotension"],
                "interactions": ["Simvastatin (limit dose)", "CYP3A4 inhibitors"],
                "pregnancy_category": "C"
            },
            "losartan": {
                "name": "Losartan",
                "brand_names": ["Cozaar", "Losacar", "Repace"],
                "drug_class": "ARB (Angiotensin Receptor Blocker)",
                "uses": ["Hypertension", "Diabetic nephropathy", "Heart failure", "Stroke prevention"],
                "dosage": "25-100mg once daily",
                "side_effects": ["Dizziness", "Hyperkalemia", "Fatigue"],
                "contraindications": ["Pregnancy", "Bilateral renal artery stenosis"],
                "interactions": ["Potassium supplements", "NSAIDs", "Lithium"],
                "pregnancy_category": "D"
            },
            "metoprolol": {
                "name": "Metoprolol",
                "brand_names": ["Lopressor", "Betaloc", "Metolar"],
                "drug_class": "Beta Blocker",
                "uses": ["Hypertension", "Angina", "Heart failure", "Arrhythmias", "MI"],
                "dosage": "25-200mg twice daily (immediate release)",
                "side_effects": ["Bradycardia", "Fatigue", "Cold extremities", "Depression"],
                "contraindications": ["Severe bradycardia", "Heart block", "Decompensated heart failure"],
                "interactions": ["Verapamil", "Clonidine (withdrawal)", "MAOIs"],
                "pregnancy_category": "C"
            },
            "azithromycin": {
                "name": "Azithromycin",
                "brand_names": ["Zithromax", "Azithral", "Azee"],
                "drug_class": "Macrolide Antibiotic",
                "uses": ["Respiratory infections", "STIs", "Skin infections", "Traveler's diarrhea"],
                "dosage": "500mg day 1, then 250mg days 2-5, or 500mg x 3 days",
                "side_effects": ["GI upset", "QT prolongation", "Hearing loss (rare)"],
                "contraindications": ["Macrolide allergy", "Severe liver disease"],
                "interactions": ["Warfarin", "Antacids", "QT-prolonging drugs"],
                "pregnancy_category": "B"
            },
            "cetirizine": {
                "name": "Cetirizine",
                "brand_names": ["Zyrtec", "Cetzine", "Alerid"],
                "drug_class": "Antihistamine (2nd generation)",
                "uses": ["Allergies", "Hay fever", "Urticaria", "Itching"],
                "dosage": "Adults: 10mg once daily",
                "side_effects": ["Drowsiness (mild)", "Dry mouth", "Headache"],
                "contraindications": ["Severe kidney disease (dose adjustment)"],
                "interactions": ["Alcohol", "CNS depressants"],
                "pregnancy_category": "B"
            },
            "pantoprazole": {
                "name": "Pantoprazole",
                "brand_names": ["Protonix", "Pan", "Pantop"],
                "drug_class": "Proton Pump Inhibitor (PPI)",
                "uses": ["GERD", "Erosive esophagitis", "H. pylori", "Gastric ulcers"],
                "dosage": "40mg once daily",
                "side_effects": ["Headache", "Diarrhea", "Abdominal pain"],
                "contraindications": ["Hypersensitivity to PPIs"],
                "interactions": ["Methotrexate", "Warfarin", "Atazanavir"],
                "pregnancy_category": "B"
            },
            "ciprofloxacin": {
                "name": "Ciprofloxacin",
                "brand_names": ["Cipro", "Ciplox", "Cifran"],
                "drug_class": "Fluoroquinolone Antibiotic",
                "uses": ["UTI", "Respiratory infections", "GI infections", "Bone infections"],
                "dosage": "250-750mg twice daily",
                "side_effects": ["Tendon damage", "GI upset", "Photosensitivity", "CNS effects"],
                "contraindications": ["Children (except specific cases)", "Tendon disorders", "QT prolongation"],
                "interactions": ["Antacids", "Theophylline", "Warfarin", "NSAIDs"],
                "pregnancy_category": "C"
            },
            "prednisone": {
                "name": "Prednisone",
                "brand_names": ["Deltasone", "Wysolone", "Omnacortil"],
                "drug_class": "Corticosteroid",
                "uses": ["Inflammation", "Autoimmune diseases", "Allergies", "Asthma", "COPD"],
                "dosage": "Varies by condition: 5-60mg daily",
                "side_effects": ["Weight gain", "Mood changes", "Hyperglycemia", "Osteoporosis (long-term)"],
                "contraindications": ["Systemic fungal infections", "Live vaccines"],
                "interactions": ["NSAIDs", "Diabetes medications", "Warfarin"],
                "pregnancy_category": "C"
            },
            "gabapentin": {
                "name": "Gabapentin",
                "brand_names": ["Neurontin", "Gabapin", "Gralise"],
                "drug_class": "Anticonvulsant/Neuropathic pain agent",
                "uses": ["Neuropathic pain", "Epilepsy", "Restless legs", "Postherpetic neuralgia"],
                "dosage": "Start 300mg, increase to 900-3600mg/day in divided doses",
                "side_effects": ["Drowsiness", "Dizziness", "Edema", "Weight gain"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["Opioids", "Antacids (reduce absorption)", "CNS depressants"],
                "pregnancy_category": "C"
            },
            "sertraline": {
                "name": "Sertraline",
                "brand_names": ["Zoloft", "Serta", "Daxid"],
                "drug_class": "SSRI (Antidepressant)",
                "uses": ["Depression", "Anxiety", "OCD", "PTSD", "Panic disorder"],
                "dosage": "Start 50mg daily, max 200mg/day",
                "side_effects": ["Nausea", "Insomnia", "Sexual dysfunction", "Serotonin syndrome (rare)"],
                "contraindications": ["MAOIs (within 14 days)", "Pimozide use"],
                "interactions": ["MAOIs", "Tramadol", "Blood thinners", "St. John's Wort"],
                "pregnancy_category": "C"
            },
            "levothyroxine": {
                "name": "Levothyroxine",
                "brand_names": ["Synthroid", "Thyronorm", "Eltroxin"],
                "drug_class": "Thyroid Hormone",
                "uses": ["Hypothyroidism", "Thyroid cancer (suppression)", "Goiter"],
                "dosage": "Start 25-50mcg daily, adjust based on TSH",
                "side_effects": ["Palpitations", "Weight loss", "Insomnia", "Osteoporosis (excess)"],
                "contraindications": ["Untreated adrenal insufficiency", "Acute MI"],
                "interactions": ["Calcium", "Iron", "Antacids", "Warfarin"],
                "pregnancy_category": "A"
            },
            "insulin": {
                "name": "Insulin (various types)",
                "brand_names": ["Novolog", "Humalog", "Lantus", "Tresiba"],
                "drug_class": "Hormone (Antidiabetic)",
                "uses": ["Type 1 Diabetes", "Type 2 Diabetes", "DKA", "Hyperkalemia"],
                "dosage": "Individualized based on blood glucose",
                "side_effects": ["Hypoglycemia", "Weight gain", "Injection site reactions"],
                "contraindications": ["Hypoglycemia"],
                "interactions": ["Beta blockers (mask hypoglycemia)", "Alcohol", "ACE inhibitors"],
                "pregnancy_category": "B"
            },
            # Additional 30+ medicines
            "diclofenac": {
                "name": "Diclofenac",
                "brand_names": ["Voltaren", "Voveran", "Cataflam"],
                "drug_class": "NSAID",
                "uses": ["Pain", "Inflammation", "Arthritis", "Migraine", "Postoperative pain"],
                "dosage": "50mg 2-3 times daily, max 150mg/day",
                "side_effects": ["GI upset", "Cardiovascular risk", "Liver toxicity", "Renal impairment"],
                "contraindications": ["Heart disease", "GI bleeding", "Severe renal impairment"],
                "interactions": ["Anticoagulants", "Lithium", "Methotrexate", "ACE inhibitors"],
                "pregnancy_category": "C (D in 3rd trimester)"
            },
            "tramadol": {
                "name": "Tramadol",
                "brand_names": ["Ultram", "Tramal", "Contramal"],
                "drug_class": "Opioid Analgesic",
                "uses": ["Moderate to severe pain", "Chronic pain", "Postoperative pain"],
                "dosage": "50-100mg every 4-6 hours, max 400mg/day",
                "side_effects": ["Nausea", "Dizziness", "Constipation", "Seizures", "Dependence"],
                "contraindications": ["Seizure disorders", "MAOIs", "Severe respiratory depression"],
                "interactions": ["SSRIs (serotonin syndrome)", "MAOIs", "CNS depressants"],
                "pregnancy_category": "C"
            },
            "ranitidine": {
                "name": "Ranitidine",
                "brand_names": ["Zantac", "Aciloc", "Rantac"],
                "drug_class": "H2 Receptor Antagonist",
                "uses": ["GERD", "Peptic ulcers", "Zollinger-Ellison syndrome"],
                "dosage": "150mg twice daily or 300mg at bedtime",
                "side_effects": ["Headache", "Dizziness", "Constipation"],
                "contraindications": ["Acute porphyria"],
                "interactions": ["Ketoconazole", "Atazanavir"],
                "pregnancy_category": "B"
            },
            "montelukast": {
                "name": "Montelukast",
                "brand_names": ["Singulair", "Montair", "Montek"],
                "drug_class": "Leukotriene Receptor Antagonist",
                "uses": ["Asthma", "Allergic rhinitis", "Exercise-induced bronchoconstriction"],
                "dosage": "Adults: 10mg once daily at bedtime",
                "side_effects": ["Headache", "Abdominal pain", "Neuropsychiatric events (rare)"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["Phenobarbital", "Rifampin"],
                "pregnancy_category": "B"
            },
            "salbutamol": {
                "name": "Salbutamol (Albuterol)",
                "brand_names": ["Ventolin", "Asthalin", "ProAir"],
                "drug_class": "Beta-2 Agonist (Bronchodilator)",
                "uses": ["Asthma", "COPD", "Bronchospasm", "Exercise-induced asthma"],
                "dosage": "Inhaler: 1-2 puffs every 4-6 hours as needed",
                "side_effects": ["Tremor", "Tachycardia", "Headache", "Hypokalemia"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["Beta blockers (antagonism)", "Diuretics (hypokalemia)"],
                "pregnancy_category": "C"
            },
            "fluticasone": {
                "name": "Fluticasone",
                "brand_names": ["Flonase", "Flovent", "Flixotide"],
                "drug_class": "Corticosteroid (Inhaled/Nasal)",
                "uses": ["Asthma", "Allergic rhinitis", "COPD"],
                "dosage": "Varies by formulation; typically 1-2 sprays per nostril daily",
                "side_effects": ["Nasal irritation", "Headache", "Oral thrush (inhaled)", "Hoarseness"],
                "contraindications": ["Untreated fungal infections"],
                "interactions": ["CYP3A4 inhibitors (ritonavir)", "Ketoconazole"],
                "pregnancy_category": "C"
            },
            "clopidogrel": {
                "name": "Clopidogrel",
                "brand_names": ["Plavix", "Clopilet", "Plagril"],
                "drug_class": "Antiplatelet Agent",
                "uses": ["ACS", "MI prevention", "Stroke prevention", "Stent thrombosis prevention"],
                "dosage": "75mg once daily; loading dose 300-600mg",
                "side_effects": ["Bleeding", "Bruising", "GI upset", "Rash"],
                "contraindications": ["Active bleeding", "Severe liver disease"],
                "interactions": ["PPIs (omeprazole)", "Aspirin", "Anticoagulants"],
                "pregnancy_category": "B"
            },
            "warfarin": {
                "name": "Warfarin",
                "brand_names": ["Coumadin", "Warf", "Marevan"],
                "drug_class": "Anticoagulant (Vitamin K Antagonist)",
                "uses": ["DVT", "PE", "Atrial fibrillation", "Mechanical heart valves"],
                "dosage": "Individualized based on INR; typically 2-10mg daily",
                "side_effects": ["Bleeding", "Bruising", "Skin necrosis (rare)"],
                "contraindications": ["Active bleeding", "Pregnancy", "Severe liver disease"],
                "interactions": ["Aspirin", "NSAIDs", "Vitamin K foods", "Many antibiotics"],
                "pregnancy_category": "X"
            },
            "enoxaparin": {
                "name": "Enoxaparin",
                "brand_names": ["Lovenox", "Clexane"],
                "drug_class": "Low Molecular Weight Heparin",
                "uses": ["DVT prophylaxis", "DVT/PE treatment", "ACS", "Dialysis"],
                "dosage": "DVT prophylaxis: 40mg SC daily; Treatment: 1mg/kg SC q12h",
                "side_effects": ["Bleeding", "Thrombocytopenia", "Injection site hematoma"],
                "contraindications": ["Active major bleeding", "Thrombocytopenia with heparin"],
                "interactions": ["Antiplatelet agents", "Other anticoagulants"],
                "pregnancy_category": "B"
            },
            "furosemide": {
                "name": "Furosemide",
                "brand_names": ["Lasix", "Frusamide", "Frusemide"],
                "drug_class": "Loop Diuretic",
                "uses": ["Edema", "Heart failure", "Hypertension", "Acute pulmonary edema"],
                "dosage": "20-80mg once or twice daily; max 600mg/day",
                "side_effects": ["Hypokalemia", "Dehydration", "Hypotension", "Ototoxicity"],
                "contraindications": ["Anuria", "Severe hyponatremia/hypokalemia"],
                "interactions": ["Aminoglycosides (ototoxicity)", "Digoxin", "Lithium"],
                "pregnancy_category": "C"
            },
            "hydrochlorothiazide": {
                "name": "Hydrochlorothiazide",
                "brand_names": ["HCTZ", "Aquazide", "Microzide"],
                "drug_class": "Thiazide Diuretic",
                "uses": ["Hypertension", "Edema", "Heart failure"],
                "dosage": "12.5-50mg once daily",
                "side_effects": ["Hypokalemia", "Hyperuricemia", "Hyperglycemia", "Photosensitivity"],
                "contraindications": ["Anuria", "Sulfonamide allergy"],
                "interactions": ["Lithium", "Digoxin", "NSAIDs"],
                "pregnancy_category": "B"
            },
            "spironolactone": {
                "name": "Spironolactone",
                "brand_names": ["Aldactone", "Spiromide"],
                "drug_class": "Potassium-Sparing Diuretic (Aldosterone Antagonist)",
                "uses": ["Heart failure", "Cirrhosis ascites", "Hypertension", "Hirsutism"],
                "dosage": "25-100mg daily; max 400mg/day",
                "side_effects": ["Hyperkalemia", "Gynecomastia", "Menstrual irregularities"],
                "contraindications": ["Hyperkalemia", "Addison's disease", "Anuria"],
                "interactions": ["ACE inhibitors", "ARBs", "Potassium supplements"],
                "pregnancy_category": "C"
            },
            "digoxin": {
                "name": "Digoxin",
                "brand_names": ["Lanoxin", "Digitek"],
                "drug_class": "Cardiac Glycoside",
                "uses": ["Atrial fibrillation", "Heart failure"],
                "dosage": "0.125-0.25mg daily; monitor levels",
                "side_effects": ["Nausea", "Arrhythmias", "Visual disturbances", "Confusion"],
                "contraindications": ["Ventricular fibrillation", "AV block", "Hypokalemia"],
                "interactions": ["Amiodarone", "Verapamil", "Quinidine", "Diuretics"],
                "pregnancy_category": "C"
            },
            "amiodarone": {
                "name": "Amiodarone",
                "brand_names": ["Cordarone", "Pacerone"],
                "drug_class": "Class III Antiarrhythmic",
                "uses": ["Ventricular arrhythmias", "Atrial fibrillation", "VT/VF"],
                "dosage": "Loading: 800-1600mg/day; Maintenance: 200-400mg/day",
                "side_effects": ["Pulmonary toxicity", "Thyroid dysfunction", "Liver toxicity", "Photosensitivity"],
                "contraindications": ["Cardiogenic shock", "Sick sinus syndrome", "Iodine allergy"],
                "interactions": ["Warfarin", "Digoxin", "Statins", "QT-prolonging drugs"],
                "pregnancy_category": "D"
            },
            "carvedilol": {
                "name": "Carvedilol",
                "brand_names": ["Coreg", "Carvil"],
                "drug_class": "Beta Blocker (Non-selective with Alpha-1 blocking)",
                "uses": ["Heart failure", "Hypertension", "Post-MI"],
                "dosage": "Start 3.125mg BID, titrate to 25-50mg BID",
                "side_effects": ["Hypotension", "Bradycardia", "Dizziness", "Fatigue"],
                "contraindications": ["Decompensated heart failure", "Severe bradycardia", "Asthma"],
                "interactions": ["Digoxin", "Insulin", "Rifampin", "CYP2D6 inhibitors"],
                "pregnancy_category": "C"
            },
            "doxycycline": {
                "name": "Doxycycline",
                "brand_names": ["Vibramycin", "Doryx", "Doxy"],
                "drug_class": "Tetracycline Antibiotic",
                "uses": ["Respiratory infections", "Acne", "Malaria prophylaxis", "STIs", "Lyme disease"],
                "dosage": "100mg twice daily or 200mg once daily",
                "side_effects": ["GI upset", "Photosensitivity", "Esophagitis", "Tooth discoloration (children)"],
                "contraindications": ["Pregnancy", "Children under 8", "Severe liver disease"],
                "interactions": ["Antacids", "Iron", "Warfarin", "Oral contraceptives"],
                "pregnancy_category": "D"
            },
            "metronidazole": {
                "name": "Metronidazole",
                "brand_names": ["Flagyl", "Metrogyl"],
                "drug_class": "Nitroimidazole Antibiotic/Antiprotozoal",
                "uses": ["Anaerobic infections", "C. diff", "H. pylori", "Trichomoniasis", "Giardiasis"],
                "dosage": "250-500mg 3 times daily; varies by indication",
                "side_effects": ["Metallic taste", "Nausea", "Neuropathy (prolonged use)", "Dark urine"],
                "contraindications": ["First trimester pregnancy", "Alcohol use"],
                "interactions": ["Alcohol (disulfiram reaction)", "Warfarin", "Lithium"],
                "pregnancy_category": "B"
            },
            "clindamycin": {
                "name": "Clindamycin",
                "brand_names": ["Cleocin", "Dalacin"],
                "drug_class": "Lincosamide Antibiotic",
                "uses": ["Skin infections", "Bone infections", "Dental infections", "MRSA"],
                "dosage": "150-450mg every 6 hours",
                "side_effects": ["Diarrhea", "C. diff colitis", "Rash", "Liver toxicity"],
                "contraindications": ["History of C. diff", "Clindamycin/lincomycin allergy"],
                "interactions": ["Neuromuscular blocking agents", "Erythromycin"],
                "pregnancy_category": "B"
            },
            "fluconazole": {
                "name": "Fluconazole",
                "brand_names": ["Diflucan", "Forcan", "Zocon"],
                "drug_class": "Azole Antifungal",
                "uses": ["Candidiasis", "Cryptococcal meningitis", "Fungal prophylaxis"],
                "dosage": "150mg single dose (vaginal); 100-400mg daily (systemic)",
                "side_effects": ["Headache", "Nausea", "Liver toxicity", "QT prolongation"],
                "contraindications": ["QT prolongation", "Severe liver disease"],
                "interactions": ["Warfarin", "Phenytoin", "Cyclosporine", "Statins"],
                "pregnancy_category": "D"
            },
            "acyclovir": {
                "name": "Acyclovir",
                "brand_names": ["Zovirax", "Acivir"],
                "drug_class": "Antiviral (Nucleoside Analog)",
                "uses": ["Herpes simplex", "Herpes zoster (shingles)", "Chickenpox", "HSV encephalitis"],
                "dosage": "200-800mg 5 times daily depending on indication",
                "side_effects": ["Nausea", "Headache", "Renal toxicity (IV)", "Neurotoxicity"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["Probenecid", "Nephrotoxic drugs"],
                "pregnancy_category": "B"
            },
            "ondansetron": {
                "name": "Ondansetron",
                "brand_names": ["Zofran", "Ondem", "Emeset"],
                "drug_class": "5-HT3 Antagonist (Antiemetic)",
                "uses": ["Chemotherapy-induced nausea", "Postoperative nausea", "Radiation nausea"],
                "dosage": "4-8mg every 8 hours; max 24mg/day",
                "side_effects": ["Headache", "Constipation", "QT prolongation", "Dizziness"],
                "contraindications": ["Congenital long QT syndrome"],
                "interactions": ["QT-prolonging drugs", "Apomorphine"],
                "pregnancy_category": "B"
            },
            "domperidone": {
                "name": "Domperidone",
                "brand_names": ["Motilium", "Domstal"],
                "drug_class": "Dopamine Antagonist (Prokinetic)",
                "uses": ["Nausea", "Vomiting", "Gastroparesis", "Lactation stimulation"],
                "dosage": "10mg 3-4 times daily before meals",
                "side_effects": ["Dry mouth", "Headache", "QT prolongation", "Galactorrhea"],
                "contraindications": ["Prolactinoma", "GI obstruction", "QT prolongation"],
                "interactions": ["CYP3A4 inhibitors", "QT-prolonging drugs"],
                "pregnancy_category": "C"
            },
            "loperamide": {
                "name": "Loperamide",
                "brand_names": ["Imodium", "Eldoper"],
                "drug_class": "Opioid (Antidiarrheal)",
                "uses": ["Acute diarrhea", "Chronic diarrhea", "Traveler's diarrhea"],
                "dosage": "4mg initially, then 2mg after each loose stool; max 16mg/day",
                "side_effects": ["Constipation", "Abdominal cramps", "Dizziness"],
                "contraindications": ["Bloody diarrhea", "Bacterial enterocolitis", "Ileus"],
                "interactions": ["Ritonavir", "Quinidine", "P-gp inhibitors"],
                "pregnancy_category": "C"
            },
            "alprazolam": {
                "name": "Alprazolam",
                "brand_names": ["Xanax", "Alprax", "Trika"],
                "drug_class": "Benzodiazepine",
                "uses": ["Anxiety disorders", "Panic disorder", "Short-term anxiety"],
                "dosage": "0.25-0.5mg 3 times daily; max 4mg/day",
                "side_effects": ["Sedation", "Dependence", "Memory impairment", "Paradoxical reactions"],
                "contraindications": ["Acute narrow-angle glaucoma", "Severe respiratory depression"],
                "interactions": ["Opioids", "Alcohol", "CYP3A4 inhibitors"],
                "pregnancy_category": "D"
            },
            "diazepam": {
                "name": "Diazepam",
                "brand_names": ["Valium", "Calmpose"],
                "drug_class": "Benzodiazepine",
                "uses": ["Anxiety", "Seizures", "Muscle spasm", "Alcohol withdrawal", "Sedation"],
                "dosage": "2-10mg 2-4 times daily depending on indication",
                "side_effects": ["Sedation", "Dependence", "Respiratory depression", "Amnesia"],
                "contraindications": ["Severe respiratory insufficiency", "Sleep apnea", "Myasthenia gravis"],
                "interactions": ["Opioids", "Alcohol", "CNS depressants"],
                "pregnancy_category": "D"
            },
            "escitalopram": {
                "name": "Escitalopram",
                "brand_names": ["Lexapro", "Cipralex", "Nexito"],
                "drug_class": "SSRI (Antidepressant)",
                "uses": ["Depression", "Generalized anxiety disorder", "Social anxiety"],
                "dosage": "10mg once daily; max 20mg/day",
                "side_effects": ["Nausea", "Insomnia", "Sexual dysfunction", "QT prolongation"],
                "contraindications": ["MAOIs", "Pimozide", "QT prolongation"],
                "interactions": ["MAOIs", "NSAIDs", "Anticoagulants", "Tramadol"],
                "pregnancy_category": "C"
            },
            "duloxetine": {
                "name": "Duloxetine",
                "brand_names": ["Cymbalta", "Duzela"],
                "drug_class": "SNRI (Antidepressant)",
                "uses": ["Depression", "Anxiety", "Diabetic neuropathy", "Fibromyalgia", "Chronic pain"],
                "dosage": "30-60mg once daily; max 120mg/day",
                "side_effects": ["Nausea", "Dry mouth", "Constipation", "Dizziness", "Sexual dysfunction"],
                "contraindications": ["MAOIs", "Uncontrolled glaucoma", "Severe renal impairment"],
                "interactions": ["MAOIs", "CYP1A2 inhibitors", "Alcohol"],
                "pregnancy_category": "C"
            },
            "quetiapine": {
                "name": "Quetiapine",
                "brand_names": ["Seroquel", "Qutan"],
                "drug_class": "Atypical Antipsychotic",
                "uses": ["Schizophrenia", "Bipolar disorder", "Depression (adjunct)", "Insomnia"],
                "dosage": "25-800mg daily depending on indication",
                "side_effects": ["Sedation", "Weight gain", "Metabolic syndrome", "Orthostatic hypotension"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["CYP3A4 inhibitors", "CNS depressants", "Antihypertensives"],
                "pregnancy_category": "C"
            },
            "risperidone": {
                "name": "Risperidone",
                "brand_names": ["Risperdal", "Risdone"],
                "drug_class": "Atypical Antipsychotic",
                "uses": ["Schizophrenia", "Bipolar disorder", "Autism-related irritability"],
                "dosage": "1-6mg daily in 1-2 doses",
                "side_effects": ["Extrapyramidal symptoms", "Weight gain", "Prolactin elevation", "Sedation"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["CYP2D6 inhibitors", "Carbamazepine", "CNS depressants"],
                "pregnancy_category": "C"
            },
            "phenytoin": {
                "name": "Phenytoin",
                "brand_names": ["Dilantin", "Eptoin"],
                "drug_class": "Anticonvulsant (Hydantoin)",
                "uses": ["Epilepsy", "Seizures", "Status epilepticus"],
                "dosage": "200-400mg daily in divided doses; monitor levels",
                "side_effects": ["Gingival hyperplasia", "Hirsutism", "Nystagmus", "Ataxia"],
                "contraindications": ["Sinus bradycardia", "Heart block"],
                "interactions": ["Many drugs (CYP inducer)", "Warfarin", "Oral contraceptives"],
                "pregnancy_category": "D"
            },
            "carbamazepine": {
                "name": "Carbamazepine",
                "brand_names": ["Tegretol", "Zen Retard"],
                "drug_class": "Anticonvulsant",
                "uses": ["Epilepsy", "Trigeminal neuralgia", "Bipolar disorder"],
                "dosage": "200-1200mg daily in divided doses",
                "side_effects": ["Dizziness", "Drowsiness", "Aplastic anemia (rare)", "SJS (rare)"],
                "contraindications": ["AV block", "Bone marrow suppression", "MAOIs"],
                "interactions": ["Many drugs (CYP inducer)", "Oral contraceptives", "Warfarin"],
                "pregnancy_category": "D"
            },
            "levetiracetam": {
                "name": "Levetiracetam",
                "brand_names": ["Keppra", "Levera"],
                "drug_class": "Anticonvulsant",
                "uses": ["Epilepsy", "Partial seizures", "Myoclonic seizures"],
                "dosage": "500-1500mg twice daily",
                "side_effects": ["Drowsiness", "Behavioral changes", "Dizziness", "Weakness"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["Few significant interactions"],
                "pregnancy_category": "C"
            },
            "sildenafil": {
                "name": "Sildenafil",
                "brand_names": ["Viagra", "Revatio"],
                "drug_class": "PDE5 Inhibitor",
                "uses": ["Erectile dysfunction", "Pulmonary arterial hypertension"],
                "dosage": "ED: 25-100mg as needed; PAH: 20mg 3 times daily",
                "side_effects": ["Headache", "Flushing", "Dyspepsia", "Visual disturbances"],
                "contraindications": ["Nitrates", "Severe cardiovascular disease"],
                "interactions": ["Nitrates (severe hypotension)", "Alpha blockers", "CYP3A4 inhibitors"],
                "pregnancy_category": "B"
            },
            "tamsulosin": {
                "name": "Tamsulosin",
                "brand_names": ["Flomax", "Urimax"],
                "drug_class": "Alpha-1 Blocker",
                "uses": ["BPH (Benign Prostatic Hyperplasia)", "Urinary retention"],
                "dosage": "0.4mg once daily after meal",
                "side_effects": ["Dizziness", "Orthostatic hypotension", "Retrograde ejaculation"],
                "contraindications": ["Hypersensitivity"],
                "interactions": ["CYP3A4 inhibitors", "Other alpha blockers", "PDE5 inhibitors"],
                "pregnancy_category": "B (not for women)"
            },
            "finasteride": {
                "name": "Finasteride",
                "brand_names": ["Proscar", "Propecia", "Finax"],
                "drug_class": "5-Alpha Reductase Inhibitor",
                "uses": ["BPH", "Male pattern baldness"],
                "dosage": "BPH: 5mg daily; Hair loss: 1mg daily",
                "side_effects": ["Sexual dysfunction", "Gynecomastia", "Depression (rare)"],
                "contraindications": ["Pregnancy (teratogenic)", "Women of childbearing age"],
                "interactions": ["Few significant interactions"],
                "pregnancy_category": "X"
            },
            "allopurinol": {
                "name": "Allopurinol",
                "brand_names": ["Zyloprim", "Zyloric"],
                "drug_class": "Xanthine Oxidase Inhibitor",
                "uses": ["Gout", "Hyperuricemia", "Uric acid stones", "Tumor lysis syndrome"],
                "dosage": "100-300mg daily; max 800mg/day",
                "side_effects": ["Rash", "GI upset", "SJS/TEN (rare)", "Hepatotoxicity"],
                "contraindications": ["Acute gout attack"],
                "interactions": ["Azathioprine", "6-mercaptopurine", "Ampicillin"],
                "pregnancy_category": "C"
            },
            "colchicine": {
                "name": "Colchicine",
                "brand_names": ["Colcrys", "Goutnil"],
                "drug_class": "Anti-gout Agent",
                "uses": ["Acute gout", "Gout prophylaxis", "Familial Mediterranean fever"],
                "dosage": "Acute: 1.2mg then 0.6mg 1 hour later; Prophylaxis: 0.6mg daily",
                "side_effects": ["Diarrhea", "Nausea", "Bone marrow suppression (toxicity)"],
                "contraindications": ["Severe renal/hepatic impairment with P-gp inhibitors"],
                "interactions": ["CYP3A4 inhibitors", "P-gp inhibitors", "Statins"],
                "pregnancy_category": "C"
            }
        }
    
    def load_vocabulary(self) -> bool:
        """Load DrugBank vocabulary CSV"""
        try:
            if not self.vocabulary_file.exists():
                print("Error loading vocabulary: File not found")
                return False
            
            print("Loading vocabulary...")
            self.vocabulary_df = pd.read_csv(self.vocabulary_file)
            
            print(f"Loaded {len(self.vocabulary_df)} drugs")
            print(f"   Columns: {', '.join(self.vocabulary_df.columns.tolist())}")
            
            # Create quick lookup index - handle different column names
            drugbank_col = None
            name_col = None
            
            # Find correct column names (they may vary)
            for col in self.vocabulary_df.columns:
                if 'drug' in col.lower() and 'id' in col.lower():
                    drugbank_col = col
                if 'name' in col.lower() or 'common' in col.lower():
                    name_col = col
            
            if drugbank_col and name_col:
                for idx, row in self.vocabulary_df.iterrows():
                    drug_id = row.get(drugbank_col)
                    drug_name = row.get(name_col, '')
                    if drug_id and drug_name:
                        self.drugs_index[str(drug_name).lower()] = drug_id
                        self.drugs_index[str(drug_id)] = drug_name
            
            return True
            
        except Exception as e:
            print(f"Error loading vocabulary: {str(e)}")
            return False
    
    def get_drug_info(self, drug_name: str) -> Dict[str, Any]:
        """Get drug information from vocabulary"""
        try:
            if self.vocabulary_df is None:
                if not self.load_vocabulary():
                    return {"error": "Could not load vocabulary"}
            
            drug_name_lower = drug_name.lower()
            
            # Find the correct columns
            name_cols = [col for col in self.vocabulary_df.columns if 'name' in col.lower()]
            
            # Search in all name columns
            matches = pd.DataFrame()
            for col in name_cols:
                col_matches = self.vocabulary_df[
                    self.vocabulary_df[col].astype(str).str.lower().str.contains(drug_name_lower, na=False)
                ]
                matches = pd.concat([matches, col_matches]).drop_duplicates()
            
            if matches.empty:
                return {"error": f"Drug '{drug_name}' not found", "found": False}
            
            # Return best match
            drug = matches.iloc[0].to_dict()
            
            return {
                "found": True,
                "drugbank_id": drug.get('DrugBank ID'),
                "name": drug.get('Common name'),
                "synonyms": drug.get('Synonyms'),
                "cas": drug.get('CAS'),
                "full_data": drug
            }
            
        except Exception as e:
            return {"error": f"Error retrieving drug info: {str(e)}", "found": False}
    
    def search_drugs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for drugs by name or ID"""
        try:
            if self.vocabulary_df is None:
                if not self.load_vocabulary():
                    return []
            
            query_lower = query.lower()
            
            # Find name columns
            name_cols = [col for col in self.vocabulary_df.columns if 'name' in col.lower()]
            id_cols = [col for col in self.vocabulary_df.columns if 'id' in col.lower()]
            
            # Search across all relevant columns
            mask = pd.Series([False] * len(self.vocabulary_df))
            
            for col in name_cols + id_cols:
                col_mask = self.vocabulary_df[col].astype(str).str.lower().str.contains(query_lower, na=False)
                mask = mask | col_mask
            
            matches = self.vocabulary_df[mask].head(limit)
            
            results = []
            for _, drug in matches.iterrows():
                results.append({
                    "name": drug.get('Common name'),
                    "drugbank_id": drug.get('DrugBank ID'),
                    "synonyms": drug.get('Synonyms'),
                    "cas": drug.get('CAS')
                })
            
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def parse_sdf_file(self, max_compounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse SDF file for molecular structures
        Note: Large file - consider limiting max_compounds
        """
        try:
            if not self.structures_file.exists():
                print(f"Error: Structures file not found")
                return []
            
            print("Parsing molecular structures...")
            compounds = []
            current_compound = {}
            
            with open(self.structures_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                for i, line in enumerate(lines):
                    if line.startswith('> '):
                        # New compound metadata
                        if current_compound:
                            compounds.append(current_compound)
                            if max_compounds and len(compounds) >= max_compounds:
                                break
                        
                        # Extract metadata
                        parts = line[2:].strip().split()
                        current_compound = {
                            "index": i,
                            "metadata": line.strip()
                        }
            
            if current_compound:
                compounds.append(current_compound)
            
            print(f"Parsed {len(compounds)} compounds")
            return compounds[:max_compounds] if max_compounds else compounds
            
        except Exception as e:
            print(f"Error parsing SDF: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DrugBank statistics"""
        try:
            if self.vocabulary_df is None:
                if not self.load_vocabulary():
                    return {}
            
            stats = {
                "total_drugs": len(self.vocabulary_df),
                "columns": self.vocabulary_df.columns.tolist(),
                "unique_types": {}
            }
            
            # Count by type if available
            if 'structure_type' in self.vocabulary_df.columns:
                stats["unique_types"] = self.vocabulary_df['structure_type'].value_counts().to_dict()
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def check_drug_interactions(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """
        Check potential interactions between drugs
        Note: Full interaction database requires licensed access
        """
        info1 = self.get_drug_info(drug1)
        info2 = self.get_drug_info(drug2)
        
        if not info1.get('found') or not info2.get('found'):
            return {
                "status": "unknown",
                "message": "One or both drugs not found in database",
                "drug1": drug1,
                "drug2": drug2
            }
        
        return {
            "status": "interaction_check_requires_full_license",
            "drug1": info1.get('name'),
            "drug2": info2.get('name'),
            "note": "Interaction database available in licensed DrugBank version",
            "recommendation": "Consult healthcare provider for official interaction data"
        }
    
    def lookup_medicine(self, medicine_name: str) -> Dict[str, Any]:
        """
        Lookup medicine details from built-in database
        Returns comprehensive drug information including uses, dosage, side effects
        
        Args:
            medicine_name: Name of the medicine (generic or brand name)
        
        Returns:
            Dict with medicine details or error message
        """
        medicine_lower = medicine_name.lower().strip()
        
        # Direct match in built-in database
        if medicine_lower in self.builtin_medicines:
            return {
                "found": True,
                "source": "built-in database",
                **self.builtin_medicines[medicine_lower]
            }
        
        # Search by brand name
        for generic, info in self.builtin_medicines.items():
            brand_names_lower = [b.lower() for b in info.get("brand_names", [])]
            if medicine_lower in brand_names_lower:
                return {
                    "found": True,
                    "source": "built-in database (brand name match)",
                    "generic_name": generic,
                    **info
                }
        
        # Partial match
        for generic, info in self.builtin_medicines.items():
            if medicine_lower in generic or generic in medicine_lower:
                return {
                    "found": True,
                    "source": "built-in database (partial match)",
                    **info
                }
        
        # Try DrugBank vocabulary if available
        vocab_result = self.get_drug_info(medicine_name)
        if vocab_result.get("found"):
            return {
                "found": True,
                "source": "DrugBank vocabulary",
                "name": vocab_result.get("name"),
                "drugbank_id": vocab_result.get("drugbank_id"),
                "synonyms": vocab_result.get("synonyms"),
                "note": "Detailed information requires DrugBank license"
            }
        
        return {
            "found": False,
            "error": f"Medicine '{medicine_name}' not found",
            "suggestion": "Try generic name or common brand names"
        }
    
    def list_all_medicines(self) -> List[str]:
        """List all medicines in built-in database"""
        return list(self.builtin_medicines.keys())
    
    def search_by_use(self, condition: str) -> List[Dict[str, Any]]:
        """
        Search medicines by use/condition
        
        Args:
            condition: Medical condition to search for
        
        Returns:
            List of medicines that treat the condition
        """
        condition_lower = condition.lower()
        results = []
        
        for generic, info in self.builtin_medicines.items():
            uses_lower = [u.lower() for u in info.get("uses", [])]
            for use in uses_lower:
                if condition_lower in use or use in condition_lower:
                    results.append({
                        "name": info["name"],
                        "generic": generic,
                        "drug_class": info.get("drug_class"),
                        "matching_use": use
                    })
                    break
        
        return results


# Test function
def test_drugbank():
    """Test DrugBank loader"""
    
    print("\n" + "="*70)
    print("DRUGBANK LOADER TEST")
    print("="*70)
    
    loader = DrugBankLoader()
    
    # Load vocabulary
    print("\n1. Loading vocabulary...")
    if loader.load_vocabulary():
        stats = loader.get_statistics()
        print(f"   ✅ Statistics: {json.dumps(stats, indent=2)}")
    
    # Search for drugs
    print("\n2. Searching for 'aspirin'...")
    results = loader.search_drugs("aspirin", limit=5)
    for drug in results:
        print(f"   - {drug.get('name')} ({drug.get('drugbank_id')})")
    
    # Get drug info
    print("\n3. Getting info for 'aspirin'...")
    info = loader.get_drug_info("aspirin")
    if info.get('found'):
        print(f"   Name: {info.get('name')}")
        print(f"   ID: {info.get('drugbank_id')}")
        print(f"   Common Name: {info.get('common_name')}")
        print(f"   Molecular Weight: {info.get('molecular_weight')}")
    else:
        print(f"   ❌ {info.get('error')}")
    
    # Parse structures (sample only)
    print("\n4. Parsing molecular structures (first 5 compounds)...")
    compounds = loader.parse_sdf_file(max_compounds=5)
    print(f"   Found {len(compounds)} compounds in SDF file")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_drugbank()
