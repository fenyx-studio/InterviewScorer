import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import concurrent.futures
import json
from questions import InterviewQuestions
from chains import InterviewChains
from judge_chains import JudgeChains
import re
import asyncio



threaded = False

# Initialize OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


llm = OpenAI(model_name='gpt-3.5-turbo',
             temperature=0)
llm2 = OpenAI(model_name='gpt-3.5-turbo',
             temperature=0.5)
llm3 = OpenAI(model_name='gpt-3.5-turbo',
             temperature=1)

# assuming all the chain objects and test variables are defined
def run_chain(chain, interviewer_question, interviewee_response):
    return chain.run(interviewer_question=interviewer_question, interviewee_response=interviewee_response)

async def async_run(chain, interviewer_question, interviewee_answer):
    return await chain.arun(interviewer_question=interviewer_question, interviewee_answer=interviewee_answer)

async def generate_concurrently(chains, interviewer_question, interviewee_answer):
    tasks = [async_run(chain, interviewer_question, interviewee_answer) for chain in chains.values()]
    return await asyncio.gather(*tasks)

# Get the current session state
state = st.session_state

# Initialize the state if it doesn't exist
if 'interview_chains' not in state:
    state.interview_chains = InterviewChains()

if 'judge_chains' not in state:
    state.judge_chains = JudgeChains(llm=llm)

chains = state.interview_chains.get_chains(llm, llm2, llm3)

chain_responses = []

test_interviewee_answer = """
At my previous job as a project manager at TechSoft, I once had to handle a situation where we were at risk of severely missing a delivery deadline for a key client, due to unforeseen technical challenges. This was a new feature that the client was eagerly anticipating and was critical to our year-end revenue target.

The challenge came when one of our key software components started failing consistently during testing. I took the initiative and immediately arranged a meeting with my team to thoroughly analyze the situation. I also took responsibility for communicating with the client, keeping them informed about the situation, and reassuring them that we were doing everything in our capacity to solve the issue.

Over the next few days, I led my team in troubleshooting the issue. I split them into smaller teams, each tasked with investigating a different aspect of the problem. With this approach, we managed to identify the bug and come up with a solution within two days.

Our response had a positive impact. Not only did we meet the deadline, but we also managed to improve the component's performance, making it 15% more efficient. The client was delighted with our proactive and transparent approach, and our management commended the team's effort. This experience taught me the value of clear communication, effective teamwork, and remaining calm under pressure.
"""

test_interviewer_question = "Can you tell me about a time when you faced a significant challenge at work and how you handled it?"

def select_score(responses, key):
    score_responses = []
    for response in responses:
        # Parse the response as JSON
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            continue

        # Check if the key exists
        if key in parsed_response:
            score_responses.append(parsed_response)

    return score_responses

# List of interview questions
questions = InterviewQuestions.get_questions()

# Sidebar with question selection
st.image('logo.png')
st.sidebar.title("Interview Questions")
selected_question = st.selectbox("Choose an interview question:", questions)
st.sidebar.markdown(f"## You selected: **{selected_question}**")


st.title("TheApply.AI Interview Scoring Tool")
test_interviewer_question = selected_question



expander = st.expander('Scoring Rubrics and Personas')

with expander:
    rubrics_tab, star_personas_tab, protag_personas_tab, structure_personas_tab, judges_personas_tab = st.tabs(["Scoring Rubrics", "STAR(R) Personas", "Protagonist Personas", "Structure Personas", "Judges Personas"])

    with rubrics_tab:
        star_score_rubric = rubrics_tab.text_area('STAR(T) Scoring Rubric', state.interview_chains.get_component('star_rubric_component'))
        protagonist_score_rubric = rubrics_tab.text_area('Protagonist Scoring Rubric', state.interview_chains.get_component('protag_rubric_component'))
        structure_score_rubric = rubrics_tab.text_area('Structure Scoring Rubric', state.interview_chains.get_component('structure_rubric_component'))

    with star_personas_tab:
        star_scorer_1 = star_personas_tab.text_area('STAR(T) Scorer #1 Persona', state.interview_chains.get_component('starscorer1_persona_component'))
        star_scorer_2 = star_personas_tab.text_area('STAR(T) Scorer #2 Persona', state.interview_chains.get_component('starscorer2_persona_component'))
        star_scorer_3 = star_personas_tab.text_area('STAR(T) Scorer #3 Persona', state.interview_chains.get_component('starscorer3_persona_component'))

    with protag_personas_tab:
        protagonist_scorer_1 = protag_personas_tab.text_area('Protagonist Scorer #1 Persona', state.interview_chains.get_component('protagscorer1_persona_component'))
        protagonist_scorer_2 = protag_personas_tab.text_area('Protagonist Scorer #2 Persona', state.interview_chains.get_component('protagscorer2_persona_component'))
        protagonist_scorer_3 = protag_personas_tab.text_area('Protagonist Scorer #3 Persona', state.interview_chains.get_component('protagscorer3_persona_component'))

    with structure_personas_tab:
        structure_scorer_1 = structure_personas_tab.text_area('Structure Scorer #1 Persona', state.interview_chains.get_component('structure1_persona_component'))
        structure_scorer_2 = structure_personas_tab.text_area('Structure Scorer #2 Persona', state.interview_chains.get_component('structure2_persona_component'))
        structure_scorer_3 = structure_personas_tab.text_area('Structure Scorer #3 Persona', state.interview_chains.get_component('structure3_persona_component'))

    if st.button('Update Model'):
        
        state.interview_chains.set_component('star_rubric_component', star_score_rubric)
        state.interview_chains.set_component('protag_rubric_component', protagonist_score_rubric)
        state.interview_chains.set_component('structure_rubric_component', structure_score_rubric)
        
        state.interview_chains.set_component('starscorer1_persona_component', star_scorer_1)
        state.interview_chains.set_component('starscorer2_persona_component', star_scorer_2)
        state.interview_chains.set_component('starscorer3_persona_component', star_scorer_3)
        
        state.interview_chains.set_component('protagscorer1_persona_component', protagonist_scorer_1)
        state.interview_chains.set_component('protagscorer2_persona_component', protagonist_scorer_2)
        state.interview_chains.set_component('protagscorer3_persona_component', protagonist_scorer_3)
        
        state.interview_chains.set_component('structure1_persona_component', structure_scorer_1)
        state.interview_chains.set_component('structure2_persona_component', structure_scorer_2)
        state.interview_chains.set_component('structure3_persona_component', structure_scorer_3)
        chains = state.interview_chains.get_chains(llm, llm2, llm3)
        st.write("Model has been updated!")


# Record the answer or enter long form text
answer_type = st.radio("Choose how to answer:", ["Text", "Audio"])
if answer_type == "Text":
    answer = st.text_area("Enter your answer here:")
    test_interviewee_answer = answer
else:
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        wav_file = open("audio.wav", "wb")
        wav_file.write(audio_bytes)

        audio_file = open("audio.wav", "rb")

        # Transcribe the answer using OpenAI Whisper API
        response = openai.Audio.transcribe("whisper-1", audio_file)
        test_interviewee_answer = response['text']
        st.markdown(f"## Transcribed text: **{response['text']}**")

# Define a function to select an emoji based on the score
def get_emoji(score):
    if score is None:
        return "😢"
    try:
        score = float(score)
    except ValueError:
        print(f"Couldn't convert score {score} to a number. Using default emoji.")
        return "😢"
    if score <= 2:
        return "😢"
    elif score <= 4:
        return "😔"
    elif score <= 6:
        return "😐"
    elif score <= 8:
        return "😊"
    else:
        return "😄"


# Score Card
if st.button("Submit Answer"):
    with st.spinner('Scoring Interview Answers...'):
        chain_results = {}
        if threaded:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Associate each future with its corresponding chain
                future_to_chain_id = {executor.submit(run_chain, chain, test_interviewer_question, test_interviewee_answer): chain_id for chain_id, chain in chains.items()}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_chain_id), start=1):
                    result = future.result()
                    chain_responses.append(result)
                    
                    # Get the chain_id that produced this future
                    chain_id = future_to_chain_id[future]
                    chain_role = state.interview_chains.chain_ids[chain_id]

                    # Parse the result as JSON
                    print(result)
                    chain_results[chain_id] = result

                    try:
                        parsed_result = json.loads(result)
                    except json.JSONDecodeError:
                        print("Attempting to parse as key-value pairs...")
                        parsed_result = {k: v.strip() for k, v in re.findall(r'(.*?):\s*(.*)', result)}
                    
                    # Extract the score and note to judge
                    score = parsed_result.get('basic_score') or parsed_result.get('protagonist_score') or parsed_result.get('structure_score')
                    note_to_judge = parsed_result.get('note_to_judge')

                    # Get the emoji for the score
                    emoji = get_emoji(score)

                    # Display the success message with the emoji
                    score_expander = st.expander(f"{chain_role}'s Score")
                    with score_expander:
                        st.sidebar.success(f"**{chain_role}'s Score is in!** \n\n**Note to judge:** {note_to_judge} \n\n**Score:** {score}/10", icon=emoji)
        else:

            async def async_run(chain, interviewer_question, interviewee_answer, chain_id):
                result = await chain.arun(interviewer_question=interviewer_question, interviewee_answer=interviewee_answer)
                return chain_id, result

            async def generate_concurrently(chains, interviewer_question, interviewee_answer):
                tasks = [async_run(chain, interviewer_question, interviewee_answer, chain_id) for chain_id, chain in chains.items()]
                for future in asyncio.as_completed(tasks):
                    chain_id, result = await future
                    chain_role = state.interview_chains.chain_ids[chain_id]
                    chain_responses.append(result)

                    # Parse the result as JSON
                    print(result)

                    try:
                        parsed_result = json.loads(result)
                    except json.JSONDecodeError:
                        print("Attempting to parse as key-value pairs...")
                        parsed_result = {k: v.strip() for k, v in re.findall(r'(.*?):\s*(.*)', result)}
                    
                    # Extract the score and note to judge
                    score = parsed_result.get('basic_score') or parsed_result.get('protagonist_score') or parsed_result.get('structure_score')
                    note_to_judge = parsed_result.get('note_to_judge')

                    # Get the emoji for the score
                    emoji = get_emoji(score)

                    # Display the success message with the emoji
                    score_expander = st.expander(f"{chain_role}'s Score")
                    with score_expander:
                        st.sidebar.success(f"**{chain_role}'s Score is in!** \n\n**Note to judge:** {note_to_judge} \n\n**Score:** {score}/10", icon=emoji)

            # Assuming chains is a dictionary where the keys are chain_ids and the values are the chains themselves
            asyncio.run(generate_concurrently(chains, test_interviewer_question, test_interviewee_answer))


    with st.spinner('Judges are deliberating...'):
        # associate each future with its corresponding chain
        async def async_run_judge(chain, interviewer_question, interviewee_answer, scorer_A, scorer_B, scorer_C):
            return await chain.arun(interviewer_question=interviewer_question, interviewee_answer=interviewee_answer, scorer_A=scorer_A, scorer_B=scorer_B, scorer_C=scorer_C)

        async def generate_concurrently_judge(chains, chain_results, interviewer_question, interviewee_answer):
            tasks = [
                async_run_judge(chains.hrjudgebasic1_chain, interviewer_question, interviewee_answer, chain_results['embasic1_chain'], chain_results['psbasic2_chain'], chain_results['stbasic3_chain']),
                async_run_judge(chains.ccjudgeprotag1_chain, interviewer_question, interviewee_answer, chain_results['lcprotag1_chain'], chain_results['msprotag2_chain'], chain_results['tbprotag3_chain']),
                async_run_judge(chains.pscjudgestructure1_chain, interviewer_question, interviewee_answer, chain_results['cestructure1_chain'], chain_results['opstructure2_chain'], chain_results['csstructure3_chain']),
            ]
            return await asyncio.gather(*tasks)

        # Run the judge chains concurrently
        star_winner, protagonist_winner, structure_winner = asyncio.run(generate_concurrently_judge(state.judge_chains, chain_results, test_interviewer_question, test_interviewee_answer))


    with st.expander("winning scores"):
        st.write(star_winner)
        st.write(protagonist_winner)
        st.write(structure_winner)

    # Extract scores and feedback
    basic_score = select_score(chain_responses, 'basic_score')[0]['basic_score']
    basic_feedback = select_score(chain_responses, 'basic_score')[0]['note_to_judge']
    protagonist_score = select_score(chain_responses, 'protagonist_score')[0]['protagonist_score']
    protagonist_feedback = select_score(chain_responses, 'protagonist_score')[0]['note_to_judge']
    structure_score = select_score(chain_responses, 'structure_score')[0]['structure_score']
    structure_feedback = select_score(chain_responses, 'structure_score')[0]['note_to_judge']

    with st.expander(f"JSON Dictionary Results"):
        st.write(chain_results)
    # Display the score card
    st.header("Score Card")

    with st.expander(f"## STAR Score: {get_emoji(basic_score)} {basic_score}/10"):
        if basic_score is None:
            basic_score = 0
        st.progress(float(basic_score)*10)
        st.markdown(f"**STAR Feedback:** {basic_feedback}")

    with st.expander(f"## Protagonist Score: {get_emoji(protagonist_score)} {protagonist_score}/10"):
        if protagonist_score is None:
            protagonist_score = 0
        st.progress(float(protagonist_score)*10)
        st.markdown(f"**Protagonist Feedback:** {protagonist_feedback}")

    with st.expander(f"## Coherence Score: {get_emoji(structure_score)} {structure_score}/10"):
        if structure_score is None:
            structure_score = 0
        st.progress(float(structure_score)*10)
        st.markdown(f"**Coherence Feedback:** {structure_feedback}")

