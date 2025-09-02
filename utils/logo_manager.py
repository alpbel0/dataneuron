#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logo Manager for DataNeuron
============================

Manages random logo selection for dynamic branding experience.
Each session gets a randomly selected logo that persists throughout the session.
"""

import random
import hashlib
from pathlib import Path
from typing import Optional
from utils.logger import logger

class LogoManager:
    """Manages dynamic logo selection with session persistence."""
    
    def __init__(self, assets_dir: str = "assets"):
        self.assets_dir = Path(assets_dir)
        self.logo_files = [
            "DataNeuronAI_logo.png"  # Ana logo
        ]
        
    def get_session_logo(self, session_id: str) -> str:
        """
        Get consistent logo for a session based on session ID hash.
        Same session always gets same logo, but different sessions get random logos.
        """
        try:
            # Create hash from session ID for consistent selection
            session_hash = hashlib.md5(session_id.encode()).hexdigest()
            hash_int = int(session_hash[:8], 16)  # First 8 chars as hex int
            
            # Select logo based on hash
            logo_index = hash_int % len(self.logo_files)
            selected_logo = self.logo_files[logo_index]
            logo_path = self.assets_dir / selected_logo
            
            # Check if file exists
            if logo_path.exists():
                logger.debug(f"Selected logo for session {session_id[:8]}: {selected_logo}")
                return str(logo_path)
            else:
                logger.warning(f"Logo file not found: {logo_path}")
                return self._get_fallback_logo()
                
        except Exception as e:
            logger.error(f"Logo selection failed: {e}")
            return self._get_fallback_logo()
    
    def _get_fallback_logo(self) -> str:
        """Fallback to emoji if files not available."""
        return "🧠"  # Emoji fallback
    
    def get_available_logos(self) -> list[str]:
        """Get list of available logo files."""
        available = []
        for logo_file in self.logo_files:
            logo_path = self.assets_dir / logo_file
            if logo_path.exists():
                available.append(str(logo_path))
        return available
    
    def add_logo(self, filename: str) -> bool:
        """Add a new logo to the rotation."""
        try:
            if filename not in self.logo_files:
                self.logo_files.append(filename)
                logger.info(f"Added new logo to rotation: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add logo {filename}: {e}")
            return False