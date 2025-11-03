# main.py
# main entry point
from agents.coordinator_agent import CoordinatorAgent

if __name__ == "__main__":
    
    CSV_PATH = "./data/sample.csv"
    MODEL = "gpt-4o-mini"
    N_INSIGHTS = 4
    OUT_DIR = "reports"

    print("Starting Intelligent CSV Analyst pipeline...\n")

    coordinator = CoordinatorAgent(
        model=MODEL,
        auto_fix=True,
        fix_threshold=4.0,
        max_retries=1
    )

    result = coordinator.run_full_pipeline(
        path=CSV_PATH,
        n_insights=N_INSIGHTS,
        tone="concise and actionable",
        out_dir=OUT_DIR,
        use_llm_summary=True,
        polish_report=True,
        save_html=True
    )
    
    print("\nPipeline completed successfully!")
    print(f"Report saved at: {result.get('report_path')}")
    print(f"Overall score: {result.get('overall_score')}")
    print(f"Retries: {result.get('retries')}")
    print("\nFull metadata:\n", result)