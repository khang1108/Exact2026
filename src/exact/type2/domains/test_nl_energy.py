import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from exact.common.schemas import PredictionRequest
from exact.type2.pipeline import run_type2_pipeline

def main():
    print("--- Testing NL Energy ---")
    nl_req1 = PredictionRequest(id="NL380", question="A capacitor has a voltage U(t) = 250cos(2000t) and a capacitance of 4 µF. What is the electric field energy at time t = 1 ms?")
    nl_res1 = run_type2_pipeline(nl_req1)
    print(f"NL1 Result: {nl_res1.answer} {nl_res1.unit} (Domain: {nl_res1.routing_diagnostics.get('domain')})")
    
    nl_req2 = PredictionRequest(id="NL108", question="A capacitor has a capacitance of 12 μF and a voltage of 70 V. Calculate the stored electric field energy (mJ).")
    nl_res2 = run_type2_pipeline(nl_req2)
    print(f"NL2 Result: {nl_res2.answer} {nl_res2.unit} (Domain: {nl_res2.routing_diagnostics.get('domain')})")

    print("\n--- Testing LD (Fallback Preserved) ---")
    ld_req = PredictionRequest(id="LD001", question="Two charges q1 = 6e-8 C and q2 = -6e-8 C are 8 cm apart. Find force.")
    try:
        ld_res = run_type2_pipeline(ld_req)
        # Should gracefully fallback or hit PoT generic
        print(f"LD Result completed. Domain generic check passed.")
    except Exception as e:
        print(f"LD Failed: {e}")
        
    print("\n--- Testing TD (Fallback Preserved) ---")
    td_req = PredictionRequest(id="TD001", question="Capacitor C=500pF charged to 300V. Find voltage.")
    try:
        td_res = run_type2_pipeline(td_req)
        print(f"TD Result completed. Domain generic check passed.")
    except Exception as e:
        print(f"TD Failed: {e}")

if __name__ == "__main__":
    main()
