from pipeline.generate_scenarios.generate_scenarios import ScenarioGenerator
from pipeline.responses_to_scenarios.responses_to_scenarios import ScenarioResponder
from pipeline.llm_as_judge.llm_as_judge import LLMAsJudge
from pipeline.generate_stats.generate_stats import GenerateStats

def main():
    # ScenarioGenerator().run()
    # print("Generated scenarios")
    # ScenarioResponder().run()
    # print("Got model responses to scenarios")
    # LLMAsJudge().run()
    # print("Judged model responses")
    GenerateStats().run()
    print("Generated stats")

if __name__ == "__main__":
    main()