# chains.py
from langchain import PromptTemplate
from langchain.chains import LLMChain

class InterviewChains:
    def __init__(self):
        self.components = {
            'starscorer1_persona_component': "As an Experienced Manager, you have a keen eye for identifying the core elements of a situation and how individuals handle tasks and achieve results. Now, stepping into the role of an Interview Scorer, you are specifically focusing on the Basic Score category. Your years of experience in managing teams and projects give you a unique perspective on evaluating the basic elements of an interview response.",
            'starscorer2_persona_component': "As a Problem Solver, you have a knack for identifying issues and finding effective solutions. Now, stepping into the role of an Interview Scorer, you are specifically focusing on the Basic Score category. Your ability to analyze situations, tasks, actions, and results helps you evaluate how well an interviewee can handle challenges.",
            'starscorer3_persona_component': "As a Strategic Thinker, you understand the importance of clear goals and effective execution. Now, stepping into the role of an Interview Scorer, you are specifically focusing on the Basic Score category. Your ability to assess the clarity and relevance of a situation, the difficulty of a task, and the strategy and outcome of actions helps you evaluate an interviewee's strategic thinking skills.",
            'protagscorer1_persona_component': "As a Leadership Coach, you have spent your career helping individuals develop their leadership potential. Now, stepping into the role of an Interview Scorer, you are specifically focusing on the Protagonist Score category. Your understanding of the importance of initiative, responsibility, and impact in a leadership role allows you to critically evaluate how an interviewee demonstrates these qualities in their response.",
            'protagscorer2_persona_component': "As a Motivational Speaker, you inspire others to take initiative and make an impact. Now, stepping into the role of an Interview Scorer, you are specifically focusing on the Protagonist Score category. Your understanding of what drives people to take responsibility and make improvements helps you evaluate an interviewee's protagonist qualities.",
            'protagscorer3_persona_component': "As a Team Builder, you know the importance of initiative, responsibility, and impact in building successful teams. Now, stepping into the role of an Interview Scorer, you are specifically focusing on the Protagonist Score category. Your experience in fostering these qualities in team members helps you evaluate an interviewee's protagonist potential.",
            'structure1_persona_component': "As a Communication Expert, you bring a deep understanding of the critical role of clear, organized, and relevant communication. You know how the structure and relevance of a response can greatly impact its effectiveness. Now, in your role as an Interview Scorer, you are particularly assessing the Structure category. Your expertise is instrumental in discerning the organization and completeness of an interviewee's response and analyzing its alignment and importance in relation to the question asked.",
            'structure2_persona_component': "As an Organizational Psychologist, you possess a keen understanding of the essential role structure and coherence play in effective communication. In your capacity as an Interview Scorer, your primary focus shifts to evaluating the Structure and Coherence Score category. Your expertise in assessing the organization and completeness of a communication piece, along with evaluating its clarity and flow, comes into play.",
            'structure3_persona_component': "As a Content Strategist, your experience has sharpened your ability to recognize and appreciate clear, coherent, and relevant content. This expertise comes into play in your role as an Interview Scorer, where your primary focus is the Coherence and Relevance Score category. You'll utilize your skills to assess an interviewee's response to a given question.",
            'star_rubric_component': "You are tasked with scoring an interview response based on the following criteria:\n\nSituation: Evaluate the clarity and relevance of the situation described by the interviewee.\nTask: Assess the clarity and difficulty of the task the interviewee had to accomplish.\nAction: Analyze the strategy and execution of the actions taken by the interviewee.\nResult: Judge the outcome and learning derived from the result of the interviewee's actions.\nAfter scoring, provide a single note to the judge explaining what you took into consideration when giving the score. Do not provide feedback for each criterion separately. All these rules must be strictly followed, to ensure the effectiveness of the Basic Scorer.",
            'protag_rubric_component': "You are tasked with scoring an interview response based on the following criteria:\n\nInitiative: Evaluate the proactivity and originality demonstrated by the interviewee.\nResponsibility: Assess the accountability and autonomy shown by the interviewee.\nImpact: Analyze the influence and improvement brought about by the interviewee's actions.\nAfter scoring, provide a single note to the judge explaining what you took into consideration when giving the score. Do not provide feedback for each criterion separately. All these rules must be strictly followed, to ensure the effectiveness of the Protagonist Scorer.",
            'structure_rubric_component': "As a Structure Scorer, your rubric encompasses three core elements:\n\nCoherence: Examine the clarity and logical flow of the response.\n\nRelevance: Assess how well the response addresses the question and the overall importance of the answer.\n\nStructure: Evaluate the organization and completeness of the response.\n\nYou'll score each response on a scale of 1-10. After scoring, provide a single note to the judge about your scoring considerations. Do not provide separate feedback for each criterion. Aim to provide a balanced assessment of the response in all three aspects: coherence, relevance, and structure. Strict adherence to these guidelines is critical."
        }

        self.chain_ids = {
            'embasic1_chain': "STAR Scorer #1",
            'psbasic2_chain': "STAR Scorer #2",
            'stbasic3_chain': "STAR Scorer #3",
            'lcprotag1_chain': "Protagonist Scorer #1",
            'msprotag2_chain': "Protagonist Scorer #2",
            'tbprotag3_chain': "Protagonist Scorer #3",
            'cestructure1_chain': "Structure Scorer #1",
            'opstructure2_chain': "Structure Scorer #2",
            'csstructure3_chain': "Structure Scorer #3"
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

    def get_chains(self, llm, llm2, llm3):
        # Define prompt templates and chains here
        
        prompt_engineeringmanager_basic_1_template = self.get_component('starscorer1_persona_component') + "\n" +self.get_component('star_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "basic_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """

        embasic1_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                            template=prompt_engineeringmanager_basic_1_template)
        embasic1_chain = LLMChain(llm=llm, prompt=embasic1_prompt)

        prompt_problemsolver_basic_2_template = self.get_component('starscorer2_persona_component') + "\n" +self.get_component('star_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "basic_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        psbasic2_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                                template=prompt_problemsolver_basic_2_template)
        psbasic2_chain = LLMChain(llm=llm2, prompt=psbasic2_prompt)

        prompt_strategicthinker_basic_3_template = self.get_component('starscorer3_persona_component') + "\n" +self.get_component('star_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "basic_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        stbasic3_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                                template=prompt_strategicthinker_basic_3_template)
        stbasic3_chain = LLMChain(llm=llm3, prompt=stbasic3_prompt)



        prompt_leadershipcoach_protag_1_template = self.get_component('protagscorer1_persona_component') + "\n" +self.get_component('protag_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "protagonist_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        lcprotag1_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                                template=prompt_leadershipcoach_protag_1_template)
        lcprotag1_chain = LLMChain(llm=llm, prompt=lcprotag1_prompt)

        prompt_motivationalspeaker_protag_2_template = self.get_component('protagscorer2_persona_component') + "\n" +self.get_component('protag_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "protagonist_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        msprotag2_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                                template=prompt_motivationalspeaker_protag_2_template)
        msprotag2_chain = LLMChain(llm=llm2, prompt=msprotag2_prompt)

        prompt_teambuilder_protag_3_template = self.get_component('protagscorer3_persona_component') + "\n" +self.get_component('protag_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "protagonist_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        tbprotag3_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                                template=prompt_teambuilder_protag_3_template)
        tbprotag3_chain = LLMChain(llm=llm3, prompt=tbprotag3_prompt)

        prompt_communicationexpert_structure_1_template = self.get_component('structure1_persona_component') + "\n" +self.get_component('structure_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "structure_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        cestructure1_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                                template=prompt_communicationexpert_structure_1_template)
        cestructure1_chain = LLMChain(llm=llm, prompt=cestructure1_prompt)

        prompt_orgpsychologist_structure_2_template = self.get_component('structure2_persona_component') + "\n" +self.get_component('structure_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "structure_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        opstructure2_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                                template=prompt_orgpsychologist_structure_2_template)
        opstructure2_chain = LLMChain(llm=llm2, prompt=opstructure2_prompt)

        prompt_contentstrategist_structure_3_template = self.get_component('structure3_persona_component') + "\n" +self.get_component('structure_rubric_component') + """
        
        Interviewer Question: {interviewer_question}

        Interviewee Answer: {interviewee_response}
        
        REQUIRED: Return the following as a valid JSON object with structure following this format:
        "structure_score":"score", "note_to_judge":"note", "improvement_for_better_score":"improvement"
        The above three keys MUST be returned as a JSON object. THIS IS VERY IMPORTANT and CRITICAL.

        The JSON object:
        """
        csstructure3_prompt = PromptTemplate(input_variables=["interviewer_question", "interviewee_response"],
                            template=prompt_contentstrategist_structure_3_template)
        csstructure3_chain = LLMChain(llm=llm3, prompt=csstructure3_prompt)

        return {
            'embasic1_chain': embasic1_chain,
            'psbasic2_chain': psbasic2_chain,
            'stbasic3_chain': stbasic3_chain,
            'lcprotag1_chain': lcprotag1_chain,
            'msprotag2_chain': msprotag2_chain,
            'tbprotag3_chain': tbprotag3_chain,
            'cestructure1_chain': cestructure1_chain,
            'opstructure2_chain': opstructure2_chain,
            'csstructure3_chain': csstructure3_chain
        }
