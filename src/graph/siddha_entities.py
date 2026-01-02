"""
Siddha Medical Entities Dictionary
Pre-defined terms for better entity extraction in RAG system.
"""

# Common Siddha herbs (English and Tamil names)
SIDDHA_HERBS = {
    # Popular herbs
    'neem', 'vembu', 'azadirachta',
    'turmeric', 'manjal', 'curcuma',
    'tulsi', 'thulasi', 'holy basil',
    'ginger', 'inji', 'zingiber',
    'pepper', 'milagu', 'piper',
    'ashwagandha', 'amukkara', 'withania',
    'brahmi', 'vallarai', 'bacopa',
    'amla', 'nellikkai', 'phyllanthus',
    'triphala', 'kadukkai', 'thanrikkai',
    'nilavembu', 'andrographis',
    'senna', 'nilavarai',
    'vetiver', 'vettiver', 'ramacham',
    'sandalwood', 'chandanam', 'santhana',
    'garcinia', 'kudampuli',
    'aloe', 'katrazhai', 'kumari',
    'cinnamon', 'pattai', 'ilavangam',
    'cardamom', 'elam', 'elaichi',
    'clove', 'kirambu', 'lavangam',
    'fenugreek', 'venthayam', 'methi',
    'cumin', 'jeeragam', 'cumin',
    'coriander', 'kothamalli', 'dhania',
    'curry leaves', 'karivepillai',
    'betel', 'vetrilai', 'paan',
    'moringa', 'murungai', 'drumstick',
    'noni', 'manjapazham',
    'hibiscus', 'chemparuthi',
    'curry', 'karivepila',
}

# Common diseases and conditions
SIDDHA_DISEASES = {
    # General conditions
    'fever', 'suram', 'jwara',
    'cold', 'cough', 'irumal',
    'headache', 'thalaivali',
    'diabetes', 'neerilivu', 'madhumegam',
    'arthritis', 'mootuvalai', 'vatham',
    'asthma', 'ilaippu', 'swasam',
    'skin', 'tholnoi', 'derma',
    'digestion', 'seeranamandratha',
    'constipation', 'malachikkal',
    'diarrhea', 'kazhichal',
    'ulcer', 'pun', 'viranam',
    'wound', 'kaya', 'pun',
    'inflammation', 'veekkam',
    'infection', 'thottruppu',
    'pain', 'vali', 'vedhanai',
    'joint', 'moottu', 'sandhi',
    'stomach', 'vayiru', 'vayu',
    'liver', 'eeral', 'yakrit',
    'kidney', 'siruneeragam',
    'heart', 'idhayam', 'hrudhaya',
    'blood', 'ratham', 'ratha',
    'pressure', 'rathaazhutham',
    'cholesterol',
    'obesity', 'udalparumai',
    'cancer', 'putrrunoi',
    'piles', 'moolam', 'arshas',
    'leprosy', 'kushta',
    'jaundice', 'manjal', 'kamaalai',
    'anemia', 'rathachogai',
}

# Siddha treatment types
SIDDHA_TREATMENTS = {
    # Internal medicines
    'chooranam', 'churnam', 'powder',
    'kudineer', 'kashayam', 'decoction',
    'lehyam', 'lehiyam', 'paste',
    'maathirai', 'mathirai', 'tablet',
    'nei', 'ghee', 'ghrita',
    'thailam', 'thylam', 'oil',
    'ennai', 'tailam',
    
    # Mineral preparations
    'parpam', 'bhasma', 'calcined',
    'chenduram', 'sinduram',
    'chunnam', 'calcium',
    'kalangu', 'metallic',
    'kattu', 'bound',
    
    # External treatments
    'patru', 'poultice',
    'podi', 'powder',
    'kuzhambu', 'paste',
    'pugai', 'fumigation',
    
    # Therapies
    'varmam', 'varma', 'vital points',
    'thokkanam', 'massage',
    'yoga', 'yogam',
    'pranayama', 'breathing',
    'dhyana', 'meditation',
}

# Siddha formulation names
SIDDHA_FORMULATIONS = {
    'nilavembu kudineer',
    'thoothuvalai kudineer',
    'triphala chooranam',
    'trikatu chooranam',
    'amukkara chooranam',
    'nellikkai lehyam',
    'chyawanprash',
    'agasthiyar kuzhampuis',
    'siddhar formulation',
    'siddha medicine',
    'siddha drug',
    'annabethi chenduram',
    'linga chenduram',
    'velvanga parpam',
    'thanga parpam',
    'kavikkal mathirai',
}

# All entities combined for quick lookup
ALL_SIDDHA_ENTITIES = SIDDHA_HERBS | SIDDHA_DISEASES | SIDDHA_TREATMENTS | SIDDHA_FORMULATIONS

def extract_siddha_entities(text: str) -> list:
    """
    Extract Siddha-specific entities from text using dictionary matching.
    
    Args:
        text: Input text to search
        
    Returns:
        List of matched entity strings
    """
    text_lower = text.lower()
    found = []
    
    # Check for exact matches
    for entity in ALL_SIDDHA_ENTITIES:
        if entity in text_lower:
            found.append(entity)
    
    # Also extract capitalized words that might be proper nouns (herb/disease names)
    import re
    capitalized = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    found.extend([w.lower() for w in capitalized if len(w) >= 4])
    
    return list(set(found))


def get_entity_type(entity: str) -> str:
    """Get the type of a Siddha entity."""
    entity_lower = entity.lower()
    
    if entity_lower in SIDDHA_HERBS:
        return "Herb"
    elif entity_lower in SIDDHA_DISEASES:
        return "Disease"
    elif entity_lower in SIDDHA_TREATMENTS:
        return "Treatment"
    elif entity_lower in SIDDHA_FORMULATIONS:
        return "Formulation"
    else:
        return "Entity"
