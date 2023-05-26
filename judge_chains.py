# judge_chains.py
from langchain import PromptTemplate
from langchain.chains import LLMChain

class JudgeChains:
    def __init__(self, llm):
        self.components = {
            'starjudge_persona_component': "As the Head of People Operations focusing on Cultural Fit, your role is to critically review the Basic Scores provided by three separate AI Interview Scorers. Each Scorer has evaluated the same interview response based on the Basic Score criteria and provided a score out of 10, along with a single note explaining their considerations for the score. Remember, as the Head of People Operations, your expertise in assessing cultural fit is crucial in this task. Your critical eye and professional judgment are needed to ensure the most appropriate score is selected. After making your decision, respond with a JSON structure containing these five keys: 'chosen_ai_scorer':'ai scorer', 'chosen_score':score,'short_sentence_reason':'reason', 'short_piece_of_advice':'advice', 'positive_feedback':'feedback'.",
            'protagjudge_persona_component': "As a Career Coach acting as a judge, your role is to critically review the Protagonist Scores provided by three separate AI Interview Scorers. Each Scorer has evaluated the same interview response based on the Protagonist Score criteria and provided a score out of 10, along with a single note explaining their considerations for the score. Remember, as a Career Coach, your expertise in career development and job interviews is crucial in this task. Your critical eye and professional judgment are needed to ensure the most appropriate score is selected. After making your decision, respond with a JSON structure containing these five keys: 'chosen_ai_scorer':'ai scorer', 'chosen_score':score,'short_sentence_reason':'reason', 'short_piece_of_advice':'advice', 'positive_feedback':'feedback'",
            'structurejudge_persona_component': "As a Public Speaking Coach acting as a judge, your role is to critically review the Structure Scores provided by three separate AI Interview Scorers. Each Scorer has evaluated the same interview response based on the Structure Score criteria and provided a score out of 10, along with a single note explaining their considerations for the score. Remember, as a Public Speaking Coach, your expertise in effective communication and presentation is crucial in this task. Your critical eye and professional judgment are needed to ensure the most appropriate score is selected.After making your decision, respond with a JSON structure containing these five keys: 'chosen_ai_scorer':'ai scorer', 'chosen_score':score,'short_sentence_reason':'reason', 'short_piece_of_advice':'advice', 'positive_feedback':'feedback'",
            'star_rubric_component': "The Interview Scorers used this as their rubric:\n\nSituation: Evaluate the clarity and relevance of the situation described by the interviewee.\nTask: Assess the clarity and difficulty of the task the interviewee had to accomplish.\nAction: Analyze the strategy and execution of the actions taken by the interviewee.\nResult: Judge the outcome and learning derived from the result of the interviewee's actions.\nAfter scoring, provide a single note to the judge explaining what you took into consideration when giving the score.",
            'protag_rubric_component': "The Interview Scorers used this as their rubric:\n\nInitiative: Evaluate the proactivity and originality demonstrated by the interviewee.\nResponsibility: Assess the accountability and autonomy shown by the interviewee.\nImpact: Analyze the influence and improvement brought about by the interviewee's actions.\nAfter scoring, provide a single note to the judge explaining what you took into consideration when giving the score.",
            'structure_rubric_component': "The Structure Score criteria used by the AI Interview Scorers are as follows:\n\nCoherence: Examine the clarity and logical flow of the response.\n\nRelevance: Assess how well the response addresses the question and the overall importance of the answer.\n\nStructure: Evaluate the organization and completeness of the response.\n\nYou'll score each response on a scale of 1-10. After scoring, provide a single note to the judge about your scoring considerations. Do not provide separate feedback for each criterion. Aim to provide a balanced assessment of the response in all three aspects: coherence, relevance, and structure. Strict adherence to these guidelines is critical.",
            'judge_rubric_component':"""Before you respond, make sure your following this one clear rule:After making your decision, respond with a JSON structure containing these five keys: 'chosen_ai_scorer':'ai scorer', 'chosen_score':score,'short_sentence_reason':'reason', 'short_piece_of_advice':'advice', 'positive_feedback':'feedback'.Your task is to review each score and the accompanying note, and choose the one that you believe is most appropriate. In your evaluation, consider the following: \n Relevance: Does the score align with the Basic Score criteria? Does the note adequately explain the considerations for the score?
                                    Consistency: Is the score consistent with the quality of the interview response? Does the note reflect this consistency?
                                    Detail: Does the note provide specific details that justify the score? Are these details relevant and meaningful?
                                    Fairness: Does the score seem fair based on the interview response? Does the note reflect a fair and unbiased evaluation?
                                    Remember, as the Head of People Operations, your expertise in assessing cultural fit is crucial in this task. Your critical eye and professional judgment are needed to ensure the most appropriate score is selected. After making your decision, respond with the final chosen score only."""
        }

        self.chain_ids = {
            'hrjudgebasic1_chain': "STAR Category Judge",
            'ccjudgeprotag1_chain': "Protagonist Category Judge",
            'pscjudgestructure1_chain': "Structure Category Judge"
        }

        self.hrjudgebasic1_chain = self._create_hrjudgebasic1_chain(llm)
        self.ccjudgeprotag1_chain = self._create_ccjudgeprotag1_chain(llm)
        self.pscjudgestructure1_chain = self._create_pscjudgestructure1_chain(llm)


    def _create_hrjudgebasic1_chain(self, llm):
        # Define prompt templates and chains here
        prompt_HRJudge_basic_1_template = self.get_component('starjudge_persona_component') + "\n" +self.get_component('star_rubric_component') + "\n" +self.get_component('judge_rubric_component') + """
        Interviewer Question:
        {interviewer_question}

        Interviewee Answer:
        {interviewee_answer}

        Here are the three scores from the three AI Scorers-

        Interview Scorer A: {scorer_A}

        Interview Scorer B: {scorer_B}

        Interview Scorer C: {scorer_C}

        RREQUIRED: Return the following as a valid JSON object with structure following this format
        "chosen_ai_scorer":"ai scorer", "chosen_score":score, "short_sentence_reason":"reason", "short_piece_of_advice":"advice", "positive_feedback":"feedback"
        The above five keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The Structured JSON object structured with curly braces:
        """

        hrjudgebasic1_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_answer", "scorer_A", "scorer_B", "scorer_C"],
                            template=prompt_HRJudge_basic_1_template)
        hrjudgebasic1_chain = LLMChain(llm=llm, prompt=hrjudgebasic1_prompt)
        return hrjudgebasic1_chain
    
    def _create_ccjudgeprotag1_chain(self, llm):
        prompt_CCJudge_protag_1_template = self.get_component('protagjudge_persona_component') + "\n" +self.get_component('protag_rubric_component') + "\n" +self.get_component('judge_rubric_component') + """
        Interviewer Question:
        {interviewer_question}

        Interviewee Answer:
        {interviewee_answer}

        Here are the three scores from the three AI Scorers-

        Interview Scorer A: {scorer_A}

        Interview Scorer B: {scorer_B}

        Interview Scorer C: {scorer_C}

        RREQUIRED: Return the following as a valid JSON object with structure following this format
        "chosen_ai_scorer":"ai scorer", "chosen_score":score, "short_sentence_reason":"reason", "short_piece_of_advice":"advice", "positive_feedback":"feedback"
        The above five keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The Structured JSON object structured with curly braces:
        """

        ccjudgeprotag1_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_answer", "scorer_A", "scorer_B", "scorer_C"],
                            template=prompt_CCJudge_protag_1_template)
        ccjudgeprotag1_chain = LLMChain(llm=llm, prompt=ccjudgeprotag1_prompt)
        return ccjudgeprotag1_chain

    def _create_pscjudgestructure1_chain(self, llm):
        prompt_PSCJudge_structure_1_template = self.get_component('structurejudge_persona_component') + "\n" +self.get_component('structure_rubric_component') + "\n" +self.get_component('judge_rubric_component') + """

        Interviewer Question:
        {interviewer_question}

        Interviewee Answer:
        {interviewee_answer}

        Here are the three scores from the three AI Scorers-

        Interview Scorer A: {scorer_A}

        Interview Scorer B: {scorer_B}

        Interview Scorer C: {scorer_C}

        RREQUIRED: Return the following as a valid JSON object with structure following this format
        "chosen_ai_scorer":"ai scorer", "chosen_score":score, "short_sentence_reason":"reason", "short_piece_of_advice":"advice", "positive_feedback":"feedback"
        The above five keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The Structured JSON object structured with curly braces:
        """


        

        pscjudgestructure1_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_answer", "scorer_A", "scorer_B", "scorer_C"],
                            template=prompt_PSCJudge_structure_1_template)
        pscjudgestructure1_chain = LLMChain(llm=llm, prompt=pscjudgestructure1_prompt)
        return pscjudgestructure1_chain

    def get_hrjudgebasic1_chain(self):
        return self.hrjudgebasic1_chain

    def get_ccjudgeprotag1_chain(self):
        return self.ccjudgeprotag1_chain

    def get_pscjudgestructure1_chain(self):
        return self.pscjudgestructure1_chain
    
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