#!/usr/bin/env python3
"""
Test script for the journaling functionality in Voice Typer.
"""
import os
import sys
import tempfile
import numpy as np
from journaling import JournalingManager

def test_journaling():
    """Test the journaling functionality."""
    print("Testing Journaling Functionality")
    print("=" * 30)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize the journaling manager
        journal = JournalingManager(output_dir=temp_dir)
        
        # Create a test audio file (just random data for testing)
        test_audio = np.random.rand(16000).astype(np.float32)
        test_transcription = "This is a test journal entry. It contains multiple sentences for testing the journaling functionality."
        
        print("\nCreating a test journal entry...")
        try:
            # Create a journal entry
            entry = journal.create_journal_entry(
                transcription=test_transcription,
                audio_data=test_audio
            )
            
            print(f"Created entry with timestamp: {entry['timestamp']}")
            print(f"Audio file: {entry.get('audio_file', 'None')}")
            
            # Check if the journal file was created
            journal_file = os.path.join(temp_dir, "Journal.md")
            if os.path.exists(journal_file):
                print(f"\nJournal file created at: {journal_file}")
                with open(journal_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("\nJournal content:")
                    print("-" * 40)
                    print(content)
                    print("-" * 40)
            else:
                print("Error: Journal file was not created!")
                
        except Exception as e:
            print(f"Error during testing: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False
    
    print("\nJournaling tests completed successfully!")
    return True

if __name__ == "__main__":
    test_journaling()
