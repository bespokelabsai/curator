from mimesis import Person, Address
from mimesis.locales import Locale
from typing import List, Dict
import json

def generate_chinese_entities(num_entities: int = 10) -> List[Dict]:
    # Initialize providers with Chinese locale
    person = Person(locale=Locale.ZH)
    address = Address(locale=Locale.ZH)
    
    entities = []
    
    for _ in range(num_entities):
        # Generate a mix of Chinese and English data
        entity = {
            # Personal information
            "chinese_name": person.full_name(),
            "english_name": Person(Locale.EN).full_name(),
            "gender": person.gender(),
            "nationality": "中国",  # China
            "id_number": person.identifier(),  # Chinese ID
            
            # Contact information
            "phone": person.telephone(),
            "email": person.email(domains=['qq.com', '163.com', 'gmail.com']),
            
            # Address information
            "address": {
                "chinese": address.address(),
                "province": address.province(),
                "city": address.city(),
                "postal_code": address.postal_code()
            },
            
            # Professional information
            "occupation": person.occupation()
        }
        entities.append(entity)
    
    return entities

if __name__ == "__main__":
    # Generate 20 sets of Chinese PII entities
    entities = generate_chinese_entities(20)
    
    # Save to JSON file for later use
    with open('chinese_entities.json', 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    # Print first entity as example
    print("Example generated entity:")
    print(json.dumps(entities[0], ensure_ascii=False, indent=2))
    print(f"\nGenerated {len(entities)} entities and saved to chinese_entities.json")
