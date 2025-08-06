from pipeline.generate_scenarios.generate_scenarios import ScenarioGenerator
from pipeline.responses_to_scenarios.responses_to_scenarios import ScenarioResponder

def main():
    # ScenarioGenerator().run()
    # print("Generated scenarios")
    ScenarioResponder().run()
    print("Got model responses to scenarios")

if __name__ == "__main__":
    main()