# main.py
import sys, os
# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import os
# path to the merged cert we just created
merged = os.path.abspath("merged_cacert.pem")
if os.path.exists(merged):
    os.environ["SSL_CERT_FILE"] = merged
    os.environ["REQUESTS_CA_BUNDLE"] = merged
    print("Using merged CA bundle at:", merged)
else:
    # fallback to certifi
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    print("Using certifi bundle at:", certifi.where())
    
    
    
from agents.coordinator_agent import build_and_run_graph

if __name__ == "__main__":
    CSV_PATH = "./data/loan_approval.csv"
    N_INSIGHTS = 4
    OUT_DIR = "reports"

    print("Starting Intelligent CSV Analyst pipeline...\n")

    try:
        result = build_and_run_graph(
            csv_path=CSV_PATH,
            n_insights=N_INSIGHTS,
            out_dir=OUT_DIR
        )
    except Exception as e:
        print("Pipeline execution failed.")
        print("Reason:", repr(e))
        raise

    print("\nPipeline completed successfully!")
    print(f"Report saved at: {result.get('report_path')}")
    print(f"Overall score: {result.get('overall_score')}")
    print(f"Retries: {result.get('retries')}")
    print("\nFull metadata:\n", result)