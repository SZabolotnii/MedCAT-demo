from medcat.cat import CAT

def load_medcat_model(model_pack_path: str) -> CAT:
    """
    Завантаження MedCAT v1 моделі з .zip паку.
    """
    cat = CAT.load_model_pack(model_pack_path)
    return cat

def extract_entities(cat: CAT, text: str):
    """
    Здійснити витяг медичних сутностей з тексту.
    Повертає словник результатів.
    """
    return cat.get_entities(text, only_cui=False)

if __name__ == "__main__":
    # Для тесту
    # model_path = "models/umls_sm_pt2ch_533bab5115c6c2d6.zip"
    model_path = "models/v2_Snomed2025_MIMIC_IV_bbe806e192df009f.zip"
    cat = load_medcat_model(model_path)
    sample_text = "The patient has diabetes, hypertension, and myocardial infarction."
    ents = extract_entities(cat, sample_text)
    print("Сутності:", ents)
