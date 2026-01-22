"""
Helper Utilities Module
Common utility functions used across the application
"""

import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
import streamlit as st


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.warning(f"Config file not found: {config_path}. Using default configuration.")
        return get_default_config()
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration if config file is not found"""
    return {
        'preprocessing': {
            'target_dpi': 300,
            'resize_width': 1200,
            'denoise_strength': 7,
            'adaptive_threshold_block_size': 11,
            'adaptive_threshold_c': 2,
            'deskew_threshold': 0.5,
            'contrast_clip_limit': 2.0,
            'contrast_tile_grid_size': [8, 8]
        },
        'ocr': {
            'language': 'eng',
            'oem': 3,
            'psm': 6,
            'tesseract_config': '--oem 3 --psm 6',
            'confidence_threshold': 30
        },
        'nlp': {
            'min_name_length': 2,
            'max_name_length': 50,
            'min_name_words': 1,
            'max_name_words': 5,
            'common_titles': ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof'],
            'job_keywords': ['Manager', 'Director', 'CEO', 'Engineer', 'Designer']
        },
        'validation': {
            'min_email_length': 5,
            'min_phone_length': 7,
            'max_phone_length': 20
        }
    }


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image (numpy array)
    """
    # Convert PIL to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    numpy_image = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return opencv_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV format to PIL Image
    
    Args:
        cv2_image: OpenCV image (numpy array)
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    if len(cv2_image.shape) == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2_image
    
    # Convert to PIL
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file temporarily
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to saved file
    """
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    
    file_path = temp_dir / uploaded_file.name
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def format_phone_display(phone: str) -> str:
    """
    Format phone number for display
    
    Args:
        phone: Raw phone number string
        
    Returns:
        Formatted phone number
    """
    # Remove all non-digit characters except +
    cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # Format based on length
    if cleaned.startswith('+'):
        # International format
        return phone  # Keep as is
    elif len(cleaned) == 10:
        # Format as (XXX) XXX-XXXX
        return f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
    else:
        return phone


def create_vcard(data: Dict[str, Any]) -> str:
    """
    Create vCard format string from extracted data
    
    Args:
        data: Dictionary with contact information
        
    Returns:
        vCard formatted string
    """
    vcard = "BEGIN:VCARD\n"
    vcard += "VERSION:3.0\n"
    
    # Name
    if data.get('name'):
        name_parts = data['name'].split()
        if len(name_parts) >= 2:
            vcard += f"N:{name_parts[-1]};{' '.join(name_parts[:-1])}\n"
            vcard += f"FN:{data['name']}\n"
        else:
            vcard += f"FN:{data['name']}\n"
    
    # Job title and company
    if data.get('job_position'):
        vcard += f"TITLE:{data['job_position']}\n"
    if data.get('company'):
        vcard += f"ORG:{data['company']}\n"
    
    # Phone
    if data.get('phone'):
        vcard += f"TEL;TYPE=WORK,VOICE:{data['phone']}\n"
    
    # Alternate phones
    for alt_phone in data.get('alternate_phones', []):
        vcard += f"TEL;TYPE=WORK,VOICE:{alt_phone}\n"
    
    # Email
    if data.get('email'):
        vcard += f"EMAIL;TYPE=WORK:{data['email']}\n"
    
    # Website
    if data.get('website'):
        vcard += f"URL:{data['website']}\n"
    
    # Address
    if data.get('address'):
        vcard += f"ADR;TYPE=WORK:;;{data['address']}\n"
    
    vcard += "END:VCARD\n"
    
    return vcard


def get_extraction_summary(data: Dict, confidence_scores: Dict) -> str:
    """
    Generate human-readable summary of extraction
    
    Args:
        data: Extracted data dictionary
        confidence_scores: Confidence scores for each field
        
    Returns:
        Summary string
    """
    summary_parts = []
    
    # Count extracted fields
    extracted_count = sum(1 for v in data.values() 
                         if v and not isinstance(v, (dict, list)))
    
    summary_parts.append(f"✓ Extracted {extracted_count} fields")
    summary_parts.append(f"✓ Overall confidence: {confidence_scores.get('overall', 0):.1f}%")
    
    # High confidence fields
    high_conf = [k for k, v in confidence_scores.items() 
                 if v >= 80 and k != 'overall']
    if high_conf:
        summary_parts.append(f"✓ High confidence: {', '.join(high_conf)}")
    
    # Low confidence fields
    low_conf = [k for k, v in confidence_scores.items() 
                if 0 < v < 50 and k != 'overall']
    if low_conf:
        summary_parts.append(f"⚠ Low confidence: {', '.join(low_conf)}")
    
    return '\n'.join(summary_parts)


def validate_tesseract_installation() -> bool:
    """
    Check if Tesseract is properly installed
    
    Returns:
        True if Tesseract is available, False otherwise
    """
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def display_debug_images(debug_images: Dict, columns: int = 3):
    """
    Display debug images in Streamlit grid
    
    Args:
        debug_images: Dictionary of debug images
        columns: Number of columns in grid
    """
    if not debug_images:
        return
    
    st.subheader("🔍 Processing Steps")
    
    # Create columns
    cols = st.columns(columns)
    
    for idx, (step_name, image) in enumerate(debug_images.items()):
        col_idx = idx % columns
        with cols[col_idx]:
            # Format step name
            display_name = step_name.replace('_', ' ').title()
            st.image(image, caption=display_name, use_container_width=True)