from langchain import PromptTemplate
from langchain.chains import LLMChain

class ChainCreator:
    def __init__(self, llm, persona_component, rubric_component):
        self.llm = llm
        self.persona_component = persona_component
        self.rubric_component = rubric_component

    # TODO: add a correlation definition between answer length and score
    def create(self):
        template = self.persona_component + "\n" + self.rubric_component + """

        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}

        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "score":"score", "perspective":"perspective on interview answer", "two_pieces_tactical_advice":"two distinct pieces of tactical advice for a better answer", "persona_note":"what the persona would note about the interviewee's answer", "stricter_score":"score"
        The above five keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL. The four keys in the JSON should be "score", "perspective", "two_pieces_tactical_advice", and "persona_note". The "stricture_score" should be a much more accurate score and much more strict than the "score".

        The JSON object:
        """

        prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"], template=template)

        return LLMChain(llm=self.llm, prompt=prompt)


class InterviewChains:
    def __init__(self):
        self.components = {
            'persona1_component':"Persona 1",
            'persona2_component':"Persona 2",
            'persona3_component':"Persona 3",
            'rubric1_component':"Rubric 1",
            'rubric2_component':"Rubric 2",
            'rubric3_component':"Rubric 3",
            'rubric4_component':"Rubric 4"
        }

        self.chain_ids = {
            'persona1_rubric1': "Persona 1 - Rubric 1",
            'persona1_rubric2': "Persona 1 - Rubric 2",
            'persona1_rubric3': "Persona 1 - Rubric 3",
            'persona1_rubric4': "Persona 1 - Rubric 4",
            'persona2_rubric1': "Persona 2 - Rubric 1",
            'persona2_rubric2': "Persona 2 - Rubric 2",
            'persona2_rubric3': "Persona 2 - Rubric 3",
            'persona2_rubric4': "Persona 2 - Rubric 4",
            'persona3_rubric1': "Persona 3 - Rubric 1",
            'persona3_rubric2': "Persona 3 - Rubric 2",
            'persona3_rubric3': "Persona 3 - Rubric 3",
            'persona3_rubric4': "Persona 3 - Rubric 4"
        }

    def set_component(self, component_name, new_text):
        if component_name in self.components:
            self.components[component_name] = new_text
        else:
            raise KeyError(f"{component_name} does not exist")

    def get_component(self, component_name):
        if component_name in self.components:
            return self.components[component_name]
        else:
            raise KeyError(f"{component_name} does not exist")

    def get_chains(self, llm):
        chains = {}

        for key, value in self.chain_ids.items():
            persona, rubric = key.split("_")
            persona_component = self.components[persona + "_component"]
            rubric_component = self.components[rubric + "_component"]

            chains[key] = ChainCreator(llm, persona_component, rubric_component).create() # use key instead of value

        return chains

