import os
import json
from typing import Any, Dict
from gensie.agents.stable_agent import StableAgent
from openai import OpenAI
from gensie.agent import GenSIEAgent, Participant, ParticipantInfo, PipelineInfo
from gensie.task import Task
from dotenv import load_dotenv
from logging import getLogger
from gensie.agents.experimental_agent import ExperimentalAgent
from gensie.agents.experimental_agent_v2 import ExperimentalAgentV2
from gensie.agents.stable_agent import StableAgent
load_dotenv()
logger = getLogger("gensie")


class BasicAgent(GenSIEAgent):
    """
    Reference implementation using OpenAI Structured Outputs.
    Configurable via environment variables:
    - OPENAI_BASE_URL: (Optional) Custom endpoint for local LLMs.
    - OPENAI_API_KEY: (Required) Your API key.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
        )

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        """
        Executes the extraction using OpenAI's response_format for strict schema compliance.
        """
        prompt = task.get_input_prompt()

        # Call OpenAI with the task's JSON schema
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction agent.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": task.target_schema,
                    "strict": True,
                },
            },
        )

        # Parse the structured JSON response
        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            # Fallback for unexpected API errors
            return {"error": f"Failed to parse model response: {str(e)}"}
        except Exception as e:
            logger.error(str(e))
            return {"error": str(e)}


class OfficialParticipant(Participant):
    """
    Standard entry point for the competition.
    Participants can configure up to 3 pipelines here.
    """

    def __init__(self):
        # Default pipeline using the reference BasicAgent
        self.pipelines = {
            "baseline": BasicAgent(),
            "stable": StableAgent(),
            "experimental": ExperimentalAgent(),
            "limited": ExperimentalAgentV2(),
        }

    def get_info(self) -> ParticipantInfo:
        return ParticipantInfo(
            team_name="Gradiant NLP Team",
            institution="Gradiant",
            pipelines=[
                PipelineInfo(
                    name="baseline",
                    description="Standard OpenAI agent using structured outputs.",
                ),
                PipelineInfo(
                    name="stable",
                    description="""
 This is a structured multi-stage inference pipeline that decomposes extraction into sequential SLM calls:
- Guideline Stage: produces extraction guidelines taking the text and schema as input without performing extraction
- Text Markup Stage: selects the entities in the text using the analysis guidelines without performing extraction)
- Self-consistent Stage extraction: it obtains the extraction using the guidelines and the text markup (repeated N times under a fixed time budget of the first 30 seconds)
- Final answer: it consolidates the final answer synthesizing from the previous answers
"""

                ),
                PipelineInfo(
                    name="limited",
                    description="Like experimental but enables reasoning_effort='low' for each strategy step when more than 40 s remain in the 60-second budget, falling back to no thinking once time is tight.",
                ),
                PipelineInfo(
                    name="experimental",
                    description="""
This is a experimental pipeline that plans and executes strategies based on categorization.

- The categorization is done taking into account the field types and clasify the fields in next categories: direct, categorical, soft_entities, fixed_entities and complex.

- The strategies are executed and also they have an estimated time to implement abetter planning in future implementations.

- Direct and categorical categories use the same strategy: direct call to de llm.

- Soft entities and fixed entities use also the same strategy: a direct prompt with diferent prompt than the previous one.

- Complex category use a two step strategy: first get candidates for the fields and then make a direct call with the candidates as hints.

- Finally each result is merged in a final result that is returned as output.

""",
                ),
            ],
        )

    def get_agent(self, pipeline_name: str) -> GenSIEAgent:
        if pipeline_name not in self.pipelines:
            # Fallback to default if pipeline not found, or raise error
            return self.pipelines["baseline"]
        return self.pipelines[pipeline_name]
