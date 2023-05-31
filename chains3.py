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
        "score":"score (1-100)", "perspective":"perspective on interview answer", "two_pieces_tactical_advice":"two distinct pieces of tactical advice for a better answer", "persona_note":"what the persona thinks about the interviewee based on their answer. this should be a personal opinion.", "stricter_score":"score (1-100)", "confidence":"confidence in score and evaluation of answer (1-5)"
        The above six keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL. The six keys in the JSON should be "score", "perspective", "two_pieces_tactical_advice", "persona_note", "stricter_score" and "confidence". The "stricture_score" should be a much more accurate score and much more strict than the "score". Both "score" & "stricter_score" should be a value from 1-100. The "confidence" should be a confidence score from 1-5 in the evaluation of the answer and score.

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
    
    def get_synthesis_chain(self, llm):
        template = """

        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}

        Persona 1 Opionions: {persona1_opinions}
        Persona 2 Opionions: {persona2_opinions}
        Persona 3 Opionions: {persona3_opinions}

        Persona 1 Tactical Advices: {persona1_tactical_advices}
        Persona 2 Tactical Advices: {persona2_tactical_advices}
        Persona 3 Tactical Advices: {persona3_tactical_advices}

        Above, you have ther interviewer question, and the interviewee answer. After it, you have the opinions of the three personas, and the tactical advices of the three personas. You're only job is to : 1) summarize as a one-liner the Persona 1 Opinions, Persona 2 Opinions, and Persona 3 Opinions; and, 2) Take all the tactical advice and synthesize it into a unique list of tactical advice for the interviewee. You can use the following template to synthesize a response:

        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "persona1_summarized_opinion":"summarized opinion from persona 1", "persona2_summarized_opinion":"summarized opinion from persona 2", "persona3_summarized_opinion":"summarized opinion from persona 3", "synthesized_tactical_advice_list":["synthesized unique tactical advice from all personas as a list"]
        The above four keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL. The four keys in the JSON should be "persona1_summarized_opinion", "persona2_summarized_opinion", "persona3_summarized_opinion", and "synthesized_tactical_advice_list". The "synthesized_tactical_advice_list" should be a list of unique tactical advice from all personas. The "persona1_summarized_opinion", "persona2_summarized_opinion", and "persona3_summarized_opinion" should be a one-liner summary of the opinions of the three personas.

        The JSON object:
        """

        prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response", "persona1_opinions", "persona2_opinions", "persona3_opinions", "persona1_tactical_advices", "persona2_tactical_advices", "persona3_tactical_advices"], template=template)

        return LLMChain(llm=self.llm, prompt=prompt)

