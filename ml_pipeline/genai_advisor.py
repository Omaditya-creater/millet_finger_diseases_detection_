import json

# Specialized Agronomic Knowledge Base for Finger Millet Diseases
DISEASE_KNOWLEDGE_BASE = {
    "Healthy Leaf": {
        "summary": "No active fungal or bacterial pathogens detected. Leaf tissue exhibits vibrant chlorophyll density and structural integrity.",
        "organic_remedies": ["Apply Panchagavya or Jeevamrutha spray monthly to maintain beneficial phyllosphere microflora."],
        "chemical_control": ["None required."],
        "dosage": "N/A",
        "phi_days": 0,
        "preventive_practices": [
            "Maintain optimal plant spacing (30cm x 10cm) for adequate sunlight penetration.",
            "Balance NPK fertilization (recommended 60:30:30 kg/ha)."
        ]
    },
    "Leaf Blast (Pyricularia oryzae)": {
        "summary": "Highly destructive blast fungal infection causing spindle-shaped brown lesions with greyish centers on leaf blades, nodes, and earhead collars.",
        "organic_remedies": [
            "Foliar spray of Pseudomonas fluorescens @ 10g/L or 2% Neem oil solution.",
            "Incorporate bio-fertilizer Trichoderma harzianum into soil during weeding."
        ],
        "chemical_control": [
            "Tricyclazole 75 WP or Carbendazim 50 WP for severe foliar outbreak.",
            "Isoprothiolane 40 EC as a secondary rotation fungicide."
        ],
        "dosage": "Tricyclazole @ 0.6 g/L water or Carbendazim @ 1.0 g/L water.",
        "phi_days": 21,
        "preventive_practices": [
            "Avoid excess nitrogenous fertilizer application during humid/cloudy spells.",
            "Perform seed treatment with Thiram @ 3g/kg seed before sowing.",
            "Adopt blast-resistant finger millet varieties (e.g., GPU 28, GPU 48, ML 365)."
        ]
    },
    "Cercospora Leaf Spot": {
        "summary": "Fungal spot disease producing reddish-brown circular to oval lesions with pale grey centers on mature lower leaves.",
        "organic_remedies": [
            "Foliar application of Cow urine (10% concentration) fermented with neem leaves.",
            "Copper oxychloride 50 WP organic formulation."
        ],
        "chemical_control": [
            "Mancozeb 75 WP or Difenoconazole 25 EC."
        ],
        "dosage": "Mancozeb @ 2.0 g/L water or Difenoconazole @ 0.5 mL/L water.",
        "phi_days": 14,
        "preventive_practices": [
            "Destroy infected crop residues post-harvest to reduce overwintering spore load.",
            "Ensure field drainage to prevent excessive canopy relative humidity (>85%)."
        ]
    },
    "Helminthosporium Blight": {
        "summary": "Bacterial/fungal blight causing dark brown elongated streaks along leaf veins, leading to premature leaf senescence.",
        "organic_remedies": [
            "Garlic extract (5%) + Neem kernel extract (NSKE 5%) spray."
        ],
        "chemical_control": [
            "Zineb 75 WP or Propiconazole 25 EC."
        ],
        "dosage": "Propiconazole @ 1.0 mL/L water.",
        "phi_days": 15,
        "preventive_practices": [
            "Practice crop rotation with legumes (pigeon pea, cowpea) every 2 seasons.",
            "Apply balanced potassium (K) fertilization to boost cell wall strength."
        ]
    },
    "Finger Millet Smut": {
        "summary": "Fungal infection affecting grain heads, transforming individual ovaries into enlarged dark green/black spore-filled galls.",
        "organic_remedies": [
            "Remove and destroy infected earheads manually before gall rupture."
        ],
        "chemical_control": [
            "Seed treatment with Carboxin 37.5% + Thiram 37.5% DS."
        ],
        "dosage": "Carboxin + Thiram @ 2.5 g/kg seed during land preparation.",
        "phi_days": 30,
        "preventive_practices": [
            "Use certified disease-free seeds.",
            "Avoid overhead sprinkler irrigation during flowering."
        ]
    }
}

class GenAIAgronomicAdvisor:
    """
    Generative AI Engine producing context-aware, personalized agricultural intervention plans
    for finger millet farmers.
    """
    def __init__(self):
        self.kb = DISEASE_KNOWLEDGE_BASE

    def generate_advisory(self, disease_class, confidence, lsi_percentage, growth_stage="Tillering/Flowering", soil_type="Red Loam", weather="28°C, 75% RH"):
        """
        Synthesizes a structured GenAI advisory report.
        """
        data = self.kb.get(disease_class, self.kb["Healthy Leaf"])
        
        # Risk assessment based on severity index LSI
        if lsi_percentage < 5.0:
            risk_level = "LOW / MILD"
            urgency = "Monitor field weekly; apply organic preventive measure."
        elif lsi_percentage < 20.0:
            risk_level = "MODERATE"
            urgency = "Initiate organic spray immediately; keep chemical fungicide on standby."
        else:
            risk_level = "HIGH / CRITICAL"
            urgency = "Immediate intervention required within 24-48 hours to prevent epidemic yield loss."
            
        advisory_report = {
            "disease_class": disease_class,
            "confidence_pct": round(confidence * 100, 2),
            "lesion_severity_index_pct": lsi_percentage,
            "risk_assessment": {
                "risk_level": risk_level,
                "urgency": urgency,
                "growth_stage": growth_stage,
                "soil_context": soil_type,
                "weather_context": weather
            },
            "pathological_summary": data["summary"],
            "treatment_protocol": {
                "organic_biological_remedies": data["organic_remedies"],
                "chemical_fungicide_control": data["chemical_control"],
                "recommended_dosage": data["dosage"],
                "pre_harvest_interval_days": data["phi_days"]
            },
            "preventive_agronomy_practices": data["preventive_practices"]
        }
        
        return advisory_report

    def format_markdown_report(self, advisory_dict):
        """
        Formats advisory dict into a publication-ready Markdown report.
        """
        md = f"""# Finger Millet Diagnostic & GenAI Advisory Report

**Diagnosed Condition:** `{advisory_dict['disease_class']}`  
**AI Model Confidence:** `{advisory_dict['confidence_pct']}%`  
**Lesion Severity Index (LSI):** `{advisory_dict['lesion_severity_index_pct']}%`  
**Risk Level:** **{advisory_dict['risk_assessment']['risk_level']}**

---

### Executive Pathological Summary
{advisory_dict['pathological_summary']}

> **Urgency Directive:** {advisory_dict['risk_assessment']['urgency']}

---

### Recommended Interventions

#### 1. Organic & Biological Control
"""
        for rem in advisory_dict['treatment_protocol']['organic_biological_remedies']:
            md += f"- [Organic] {rem}\n"
            
        md += f"""
#### 2. Chemical Control & Dosage
"""
        for chem in advisory_dict['treatment_protocol']['chemical_fungicide_control']:
            md += f"- [Chemical] {chem}\n"
            
        md += f"""
* **Recommended Dosage:** `{advisory_dict['treatment_protocol']['recommended_dosage']}`  
* **Pre-Harvest Interval (PHI):** `{advisory_dict['treatment_protocol']['pre_harvest_interval_days']} days`

---

### Long-Term Preventive Agronomy Practices
"""
        for prev in advisory_dict['preventive_agronomy_practices']:
            md += f"- [Preventive] {prev}\n"
            
        return md

if __name__ == "__main__":
    advisor = GenAIAgronomicAdvisor()
    rep = advisor.generate_advisory("Leaf Blast (Pyricularia oryzae)", confidence=0.984, lsi_percentage=18.4)
    print(advisor.format_markdown_report(rep))
