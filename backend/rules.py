def rule_based_prediction(symptoms):
    symptoms = symptoms.lower()

    if "fever" in symptoms and "cold" in symptoms:
        return {
            "disease": "Common Cold",
            "info": "Rest, hydration, and mild medication",
            "confidence": 95
        }

    if "chest pain" in symptoms and "shortness of breath" in symptoms:
        return {
            "disease": "Possible Heart Issue",
            "info": "Consult a doctor immediately",
            "confidence": 90
        }

    return None
