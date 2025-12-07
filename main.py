# main.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
merged = os.path.abspath("merged_cacert.pem")
if os.path.exists(merged):
    os.environ["SSL_CERT_FILE"] = merged
    os.environ["REQUESTS_CA_BUNDLE"] = merged
    print("Using merged CA bundle at:", merged)
else:

    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    print("Using certifi bundle at:", certifi.where())
import json
    
    
from agents.coordinator_agent import build_and_run_graph

if __name__ == "__main__":
    import sys

    N_INSIGHTS = 4
    OUT_DIR = "outputs"

    print("Starting Intelligent CSV Analyst pipeline...\n")
    CSV_PATH = input("Enter the dataset path: ").strip()

    if not CSV_PATH:
        print("No path provided. Exiting.")
        sys.exit(1)

    if not os.path.exists(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
        sys.exit(1)

    try:
        result = build_and_run_graph(
            csv_path=CSV_PATH,
            n_insights=N_INSIGHTS,
            out_dir=OUT_DIR
        )
    except Exception as e:
        print("Pipeline execution failed.")
        print("Reason:", repr(e))
        import traceback
        traceback.print_exc()
        sys.exit(2)

    if not result:
        print("Pipeline completed but returned no result object.")
        sys.exit(3)

    print("\nPipeline completed successfully!")
    print(f"Report saved at: {result.get('report_path')}")
    print(f"Overall score: {result.get('overall_score')}")
    print(f"Retries: {result.get('retries')}")
    if result.get("errors"):
        print("\nErrors encountered during run:")
        for err in result["errors"]:
            print("-", err)
    if result.get("warnings"):
        print("\nWarnings:")
        for w in result["warnings"]:
            print("-", w)

    print("\nFull metadata:\n", json.dumps(result, indent=2))