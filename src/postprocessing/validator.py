"""
Data Validation and Post-Processing Module
Cleans, validates, and formats extracted data
"""

import re
from typing import Dict, Optional, List


class DataValidator:
    """
    Validates and cleans extracted visiting card data
    Ensures data quality and consistency
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.min_email_length = config.get('min_email_length', 5)
        self.min_phone_length = config.get('min_phone_length', 7)
        self.max_phone_length = config.get('max_phone_length', 20)
    
    def validate_and_clean(self, extracted_data: Dict) -> Dict:
        """
        Main validation pipeline
        
        Args:
            extracted_data: Raw extracted data dictionary
            
        Returns:
            Cleaned and validated data dictionary
        """
        cleaned = {}
        
        # Validate and clean name
        if 'name' in extracted_data and extracted_data['name']:
            cleaned['name'] = self._clean_name(extracted_data['name'])
        else:
            cleaned['name'] = None
        
        # Validate and clean job position
        if 'job_position' in extracted_data and extracted_data['job_position']:
            cleaned['job_position'] = self._clean_job_position(extracted_data['job_position'])
        else:
            cleaned['job_position'] = None
        
        # Validate and clean company
        if 'company' in extracted_data and extracted_data['company']:
            cleaned['company'] = self._clean_company(extracted_data['company'])
        else:
            cleaned['company'] = None
        
        # Validate emails
        if 'emails' in extracted_data and extracted_data['emails']:
            cleaned['email'] = self._select_best_email(extracted_data['emails'])
        else:
            cleaned['email'] = None
        
        # Validate phones
        if 'phones' in extracted_data and extracted_data['phones']:
            cleaned['phone'] = self._select_best_phone(extracted_data['phones'])
            cleaned['alternate_phones'] = extracted_data['phones'][1:] if len(extracted_data['phones']) > 1 else []
        else:
            cleaned['phone'] = None
            cleaned['alternate_phones'] = []
        
        # Clean website
        if 'websites' in extracted_data and extracted_data['websites']:
            cleaned['website'] = extracted_data['websites'][0]
        else:
            cleaned['website'] = None
        
        # Clean address
        if 'address' in extracted_data and extracted_data['address']:
            cleaned['address'] = self._clean_address(extracted_data['address'])
        else:
            cleaned['address'] = None
        
        # Add other fields
        cleaned['fax'] = extracted_data.get('fax')
        cleaned['social_media'] = extracted_data.get('social_media', {})
        cleaned['company_ids'] = extracted_data.get('company_ids', {})
        
        return cleaned
    
    def _clean_name(self, name: str) -> str:
        """Clean and format person name"""
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Remove common prefixes if standalone
        prefixes = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.']
        for prefix in prefixes:
            if name.startswith(prefix + ' '):
                name = name[len(prefix)+1:]
        
        # Title case
        name = name.title()
        
        # Remove any trailing punctuation
        name = name.rstrip('.,;:')
        
        return name
    
    def _clean_job_position(self, position: str) -> str:
        """Clean and format job position"""
        # Remove extra whitespace
        position = ' '.join(position.split())
        
        # Title case for better readability
        position = position.title()
        
        # Remove trailing punctuation
        position = position.rstrip('.,;:')
        
        return position
    
    def _clean_company(self, company: str) -> str:
        """Clean and format company name"""
        # Remove extra whitespace
        company = ' '.join(company.split())
        
        # Remove trailing punctuation except for abbreviations
        company = re.sub(r'[,;:]+$', '', company)
        
        return company
    
    def _select_best_email(self, emails: List[str]) -> Optional[str]:
        """
        Select the most likely primary email
        Prefers company domains over generic ones
        """
        if not emails:
            return None
        
        # Score emails
        scored = []
        generic_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
        
        for email in emails:
            score = 0
            domain = email.split('@')[1].lower() if '@' in email else ''
            
            # Prefer company domains over generic
            if domain not in generic_domains:
                score += 10
            
            # Prefer shorter emails (usually primary)
            score += 5 / len(email)
            
            scored.append((email, score))
        
        # Return highest scored
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def _select_best_phone(self, phones: List[str]) -> Optional[str]:
        """
        Select the most likely primary phone number
        Prefers mobile numbers
        """
        if not phones:
            return None
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phones = []
        for phone in phones:
            phone_digits = re.sub(r'\D', '', phone)
            if phone_digits not in seen:
                seen.add(phone_digits)
                unique_phones.append(phone)
        
        # Score phones
        scored = []
        for phone in unique_phones:
            score = 0
            
            # Prefer numbers with country code
            if phone.startswith('+'):
                score += 5
            
            # Prefer mobile patterns (in India, starts with 6-9)
            digits = re.sub(r'\D', '', phone)
            if len(digits) == 10 and digits[0] in '6789':
                score += 10
            
            # Prefer properly formatted
            if re.match(r'^\+\d{1,3}[\s-]?\d{10}$', phone):
                score += 3
            
            scored.append((phone, score))
        
        # Return highest scored
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def _clean_address(self, address: str) -> str:
        """Clean and format address"""
        # Remove extra whitespace
        address = ' '.join(address.split())
        
        # Remove multiple commas
        address = re.sub(r',\s*,', ',', address)
        
        # Ensure proper spacing after commas
        address = re.sub(r',(\S)', r', \1', address)
        
        # Remove trailing punctuation
        address = address.rstrip('.,;:')
        
        return address
    
    def calculate_confidence_score(self, cleaned_data: Dict) -> Dict[str, float]:
        """
        Calculate confidence scores for each field
        
        Returns:
            Dictionary with field: confidence_score pairs
        """
        scores = {}
        
        # Name confidence
        if cleaned_data.get('name'):
            name = cleaned_data['name']
            score = 50
            if len(name.split()) >= 2:  # Full name
                score += 30
            if re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)+$', name):  # Proper format
                score += 20
            scores['name'] = min(score, 100)
        else:
            scores['name'] = 0
        
        # Email confidence
        if cleaned_data.get('email'):
            email = cleaned_data['email']
            score = 70
            if re.match(r'^[a-zA-Z]', email):  # Starts with letter
                score += 15
            if email.count('.') >= 1:  # Has domain extension
                score += 15
            scores['email'] = min(score, 100)
        else:
            scores['email'] = 0
        
        # Phone confidence
        if cleaned_data.get('phone'):
            phone = cleaned_data['phone']
            score = 60
            if phone.startswith('+'):  # International format
                score += 20
            if len(re.sub(r'\D', '', phone)) >= 10:  # Complete number
                score += 20
            scores['phone'] = min(score, 100)
        else:
            scores['phone'] = 0
        
        # Job position confidence
        if cleaned_data.get('job_position'):
            scores['job_position'] = 75
        else:
            scores['job_position'] = 0
        
        # Company confidence
        if cleaned_data.get('company'):
            scores['company'] = 70
        else:
            scores['company'] = 0
        
        # Address confidence
        if cleaned_data.get('address'):
            address = cleaned_data['address']
            score = 50
            if ',' in address:  # Multi-part address
                score += 25
            if re.search(r'\d{5,6}', address):  # Has postal code
                score += 25
            scores['address'] = min(score, 100)
        else:
            scores['address'] = 0
        
        # Overall confidence (weighted average)
        weights = {
            'name': 0.25,
            'email': 0.20,
            'phone': 0.20,
            'job_position': 0.15,
            'company': 0.15,
            'address': 0.05
        }
        
        overall = sum(scores.get(field, 0) * weight 
                     for field, weight in weights.items())
        scores['overall'] = round(overall, 2)
        
        return scores