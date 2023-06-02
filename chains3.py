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
        "score":"score (1-10)", "perspective":"perspective on interview answer", "two_pieces_tactical_advice":"two distinct recommendations for a better answer based on your persona. make this specific, and very actionable.", "persona_note":"what the persona thinks about the interviewee based on their answer. this should be a personal opinion.", "stricter_score":"score (1-10)", "confidence":"confidence in score and evaluation of answer (1-5)"
        The above six keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL. The six keys in the JSON should be "score", "perspective", "two_pieces_tactical_advice", "persona_note", "stricter_score" and "confidence". The "stricture_score" should be a much more accurate score and much more strict than the "score". Both "score" & "stricter_score" should be a value from 1-10. The "confidence" should be a confidence score from 1-5 in the evaluation of the answer and score.

        The JSON object:
        """

        prompt = PromptTemplate(input_variables=[
                                "interviewer_question", "interviewee_response"], template=template)

        return LLMChain(llm=self.llm, prompt=prompt)


class InterviewChains:
    def __init__(self):
        self.components = {
            'persona1_component':
            "You are an experienced HR professional with several decades of experience interviewing candidates for roles. You have a talent for evaluating interview answers because you pay attention to its nuances and details. You are concerned with areas like professionalism, passion, teamwork, and emotional intelligence. You have very high standards when it comes to interview answers and only like candidates who give well-reasoned with supporting evidence.",

            'persona2_component':
            "You are a public speaking coach with several decades of experience helping people express themselves in public. You specialize in helping people speak in a clear, precise, and detailed manner. You are also understand how people can present themselves in a positive, impactful way and are capable of giving advice on how they can do better. You are concerned with communication style, speaking clarity, enthusiasm, and sophistication in speech. You have very high standards when it comes to interview answers and only like candidates who give well-reasoned answers with supporting evidence.",

            'persona3_component':
            "You are a professional mentor with several decades of experience helping people advance their careers. You specialize in helping people talk about themselves and their achievements in an effective and impactful way. You have seen many individuals grow and change in their career and are aware of what it takes to become a better professional. You are concerned with growth, learning on the job, building relationships, and being career-driven. You have very high standards when it comes to interview answers and only like candidates who give well-reasoned answers with supporting evidence.",

            'rubric1_component':
            "Evaluate the job interview response only on the following criteria and no other factors: The amount of effort made and adversity faced by the interviewee in their response Evaluate the number of distinct skills supported by specific evidence conveyed by the interviewee in their response The number of distinct positive personality traits supported by specific evidence conveyed by the interviewee in their response, giving a higher score the more there are The degree to which the interviewee’s actions had an impacted the situation",

            'rubric2_component':
            "Evaluate the job interview response only on the following criteria and no other factors: Situation: The clarity and specificity with which the situation or context of the anecdote is described Task: The clarity and specificity with which the specific tasks the interviewee had to accomplish are described Actions: The clarity and specificity with which the interviewee describes the actions taken to accomplish the task Rationale: The clarity and specificity with which the interviewee describes his or her thought process behind the actions taken Outcome: The clarity and specificity with which the interviewee describes the outcomes or results of the action that they took and the resolution of the situation Personal Takeaway: The clarity and specificity with which the interviewee describes the lessons learned in the process of tackling the situation in their response and the actions they took",

            'rubric3_component':
            "Evaluate the job interview response only on the following criteria and no other factors: The degree to which the interviewee presents themselves as proactive and self-driven in their answer The degree to which the interviewee presents themselves as autonomous and responsible in their decision-making in the answer The degree to which the interviewee’s actions impact or improve the situation as described in their answer",

            'rubric4_component':
            "Evaluate the job interview response only on the following criteria and no other factors: How organized the response is and whether the points flow logically The response’s relevance to the question asked The succinctness of the response; deduct points if it is too repetitive or wordy"
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

            # use key instead of value
            chains[key] = ChainCreator(
                llm, persona_component, rubric_component).create()

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

        Above, you have ther interviewer question, and the interviewee answer. After it, you have the opinions of the three personas, and the tactical advices of the three personas. You're only job is to : 1) summarize as a unique one-liner the Persona 1 Opinions, Persona 2 Opinions, and Persona 3 Opinions. each opinion should be unique for each; and, 2) Take all the tactical advice and synthesize it into a list of distinct recommendations for the interviewee. You can use the following template to synthesize a response:

        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "persona1_summarized_opinion":"opinion from persona 1, from the perspective of an HR professional", "persona2_summarized_opinion":"opinion from persona 2, from the perspective of apublic speaking coach", "persona3_summarized_opinion":"opinion from persona 3, from the perspective of a career coach", "synthesized_tactical_advice_list":["synthesized unique interview advice from all personas as a list"]
        The above four keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL. The four keys in the JSON should be "persona1_summarized_opinion", "persona2_summarized_opinion", "persona3_summarized_opinion", and "synthesized_tactical_advice_list". The "synthesized_tactical_advice_list" should be a list of unique specific advice from all personas. The "persona1_summarized_opinion", "persona2_summarized_opinion", and "persona3_summarized_opinion" should be a one-liner summary of the opinions of the three personas, and all should be different than each other. Do NOT duplicate the persona opinions or use the same ones. The "synthesized_tactical_advice_list" should be a list of distinct recommendations for improving interview response from all personas, with no duplicates. The one-liner summary of the opinions of the three personas should all be unique. Do not INCLUDE duplicate tactical advices in the "synthesized_tactical_advice_list" and DO NOT INCLUDE duplicate one-liner summaries of the opinions of the three personas. If you have duplicates, you will be penalized. 


        The JSON object:
        """

        prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response", "persona1_opinions", "persona2_opinions",
                                "persona3_opinions", "persona1_tactical_advices", "persona2_tactical_advices", "persona3_tactical_advices"], template=template)

        return LLMChain(llm=llm, prompt=prompt)
