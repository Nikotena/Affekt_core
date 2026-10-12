#!/usr/bin/env python3
"""
Basic Test Interface for Kankei Engine
Implements the MVP requirement: Basic test interface
"""

import sys
import argparse
import os
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KankeiEngine import KankeiEngine

def run_basic_tests():
    """Run basic functionality tests"""
    print("=== Kankei Engine Basic Tests ===\n")
    
    # Test 1: Engine initialization
    try:
        engine = KankeiEngine()
        engine.initialize_modules()
        print("[PASS] Test 1: Engine initialized")
    except Exception as e:
        print(f"[FAIL] Test 1 FAILED: {e}")
        return False
    
    # Test 2: Vectorial Personality
    try:
        personality = engine.personality
        personality.set_temperament(2)  # Choleric
        result = personality.update_from_action([0.2, 0.3, 0.4, 0.5])
        
        state = personality.get_state()
        assert len(state["vector"]) == 4
        assert -1.0 <= state["vector"]["affective_valence"] <= 1.0
        print("[PASS] Test 2: Vectorial Personality calculation")
    except Exception as e:
        print(f"[FAIL] Test 2 FAILED: {e}")
        return False
    
    # Test 3: Memory Set
    try:
        memory = engine.memory
        memory.add_short_term_entry("NPC_0", "Greeted the player")
        memory.add_short_term_entry("NPC_0", "Asked about the quest")
        
        entries = memory.get_short_term_context("NPC_0")
        assert len(entries) == 2
        assert memory.get_short_term_context("NPC_1") == []
        print("[PASS] Test 3: Memory Set short-term storage")
    except Exception as e:
        print(f"[FAIL] Test 3 FAILED: {e}")
        return False
    
    # Test 4: Context Function
    try:
        context = engine.context
        triggers = context.should_trigger_script("sell sword to player", 
                                                 ["sell", "trade"])
        assert triggers is True
        payload = context.build_strict_payload("NPC_0", ["item: sword", "price: 100"])
        assert len(payload) > 0
        print("[PASS] Test 4: Context Function triggers and payload")
    except Exception as e:
        print(f"[FAIL] Test 4 FAILED: {e}")
        return False
    
    # Test 5: Full engine action processing
    try:
        engine.running = True
        results = engine.run(10)
        assert len(results) == 10
        assert all("response" in r for r in results)
        print("[PASS] Test 5: Full engine action processing")
    except Exception as e:
        print(f"[FAIL] Test 5 FAILED: {e}")
        return False
    
    print("\n=== All Basic Tests Passed ===")
    return True

def main():
    parser = argparse.ArgumentParser(description="Kankei Engine Basic Test Interface")
    parser.add_argument("--actions", type=int, default=20, help="Number of actions to process")
    args = parser.parse_args()
    
    if run_basic_tests():
        print(f"\nRunning {args.actions} actions...")
        engine = KankeiEngine()
        engine.initialize_modules()
        engine.running = True
        results = engine.run(args.actions)
        
        print(f"\nProcessed {args.actions} actions successfully")
        
        # Print summary of metrics
        total_payload_size = sum(r["payload_size"] for r in results)
        avg_payload_size = total_payload_size / len(results) if results else 0
        
        print(f"\n=== Engine Metrics ===")
        print(f"Total actions: {len(results)}")
        print(f"Average payload size: {avg_payload_size:.1f} tokens")
        
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
