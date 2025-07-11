#!/usr/bin/env python3
"""
Test script to verify TopMost library functionality
"""

import pandas as pd
import numpy as np

try:
    import topmost
    print("✓ TopMost library imported successfully")
    
    # Test basic functionality
    print("Available TopMost models:", dir(topmost))
    
    # Test DETM
    try:
        detm = topmost.DETM(topic_num=5, epochs=5)
        print("✓ DETM model created successfully")
    except Exception as e:
        print(f"✗ DETM creation failed: {e}")
    
    # Test CFDTM
    try:
        cfdtm = topmost.CFDTM(topic_num=5, epochs=5)
        print("✓ CFDTM model created successfully")
    except Exception as e:
        print(f"✗ CFDTM creation failed: {e}")
    
    # Test with sample data
    print("\nTesting with sample data...")
    sample_docs = [
        "This is about travel to Thailand",
        "Banking and money exchange",
        "Food and restaurants in Thailand",
        "Visa requirements for Thailand",
        "Weather in Thailand is hot"
    ]
    
    time_slices = [5]  # All documents in one time slice
    
    try:
        detm = topmost.DETM(topic_num=3, epochs=5)
        detm.fit(sample_docs, time_slices)
        topics = detm.get_topics()
        print(f"✓ DETM fitted successfully, found {len(topics)} topics")
    except Exception as e:
        print(f"✗ DETM fitting failed: {e}")
    
    print("\nTopMost library test completed!")
    
except ImportError as e:
    print(f"✗ TopMost library not available: {e}")
except Exception as e:
    print(f"✗ Error testing TopMost: {e}") 