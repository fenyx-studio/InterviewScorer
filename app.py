import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
from streamlit_tags import st_tags
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import concurrent.futures
import json
from questions import InterviewQuestions
from chains import InterviewChains
import re


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


interview_chains = InterviewChains()
chains = interview_chains.get_chains(llm, llm2, llm3)

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
        star_score_rubric = rubrics_tab.text_area('STAR(T) Scoring Rubric', interview_chains.get_component('star_rubric_component'))
        protagonist_score_rubric = rubrics_tab.text_area('Protagonist Scoring Rubric', interview_chains.get_component('protag_rubric_component'))
        structure_score_rubric = rubrics_tab.text_area('Structure Scoring Rubric', interview_chains.get_component('structure_rubric_component'))

    with star_personas_tab:
        star_scorer_1 = star_personas_tab.text_area('STAR(T) Scorer #1 Persona', interview_chains.get_component('starscorer1_persona_component'))
        star_scorer_2 = star_personas_tab.text_area('STAR(T) Scorer #2 Persona', interview_chains.get_component('starscorer2_persona_component'))
        star_scorer_3 = star_personas_tab.text_area('STAR(T) Scorer #3 Persona', interview_chains.get_component('starscorer3_persona_component'))

    with protag_personas_tab:
        protagonist_scorer_1 = protag_personas_tab.text_area('Protagonist Scorer #1 Persona', interview_chains.get_component('protagscorer1_persona_component'))
        protagonist_scorer_2 = protag_personas_tab.text_area('Protagonist Scorer #2 Persona', interview_chains.get_component('protagscorer2_persona_component'))
        protagonist_scorer_3 = protag_personas_tab.text_area('Protagonist Scorer #3 Persona', interview_chains.get_component('protagscorer3_persona_component'))

    with structure_personas_tab:
        structure_scorer_1 = structure_personas_tab.text_area('Structure Scorer #1 Persona', interview_chains.get_component('structure1_persona_component'))
        structure_scorer_2 = structure_personas_tab.text_area('Structure Scorer #2 Persona', interview_chains.get_component('structure2_persona_component'))
        structure_scorer_3 = structure_personas_tab.text_area('Structure Scorer #3 Persona', interview_chains.get_component('structure3_persona_component'))

    if st.button('Update Model'):
        interview_chains.set_component('star_rubric_component', star_score_rubric)
        interview_chains.set_component('protag_rubric_component', protagonist_score_rubric)
        interview_chains.set_component('structure_rubric_component', structure_score_rubric)
        
        interview_chains.set_component('starscorer1_persona_component', star_scorer_1)
        interview_chains.set_component('starscorer2_persona_component', star_scorer_2)
        interview_chains.set_component('starscorer3_persona_component', star_scorer_3)
        
        interview_chains.set_component('protagscorer1_persona_component', protagonist_scorer_1)
        interview_chains.set_component('protagscorer2_persona_component', protagonist_scorer_2)
        interview_chains.set_component('protagscorer3_persona_component', protagonist_scorer_3)
        
        interview_chains.set_component('structure1_persona_component', structure_scorer_1)
        interview_chains.set_component('structure2_persona_component', structure_scorer_2)
        interview_chains.set_component('structure3_persona_component', structure_scorer_3)
        st.write("Model has been updated!")  # Add your update code here


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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_chain, chain, test_interviewer_question, test_interviewee_answer) for chain in chains]

            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
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
                score_expander = st.expander(f"Judge #{i}'s Score")
                with score_expander:
                    st.sidebar.success(f"**Judge #{i}'s Score is in!** \n\n**Note to judge:** {note_to_judge} \n\n**Score:** {score}/10", icon=emoji)


    # Extract scores and feedback
    basic_score = select_score(chain_responses, 'basic_score')[0]['basic_score']
    basic_feedback = select_score(chain_responses, 'basic_score')[0]['note_to_judge']
    protagonist_score = select_score(chain_responses, 'protagonist_score')[0]['protagonist_score']
    protagonist_feedback = select_score(chain_responses, 'protagonist_score')[0]['note_to_judge']
    structure_score = select_score(chain_responses, 'structure_score')[0]['structure_score']
    structure_feedback = select_score(chain_responses, 'structure_score')[0]['note_to_judge']

    st.header("Score Card")

    with st.expander(f"## STAR Score: {get_emoji(basic_score)} {basic_score}/10"):
        st.progress(basic_score*10)
        st.markdown(f"**STAR Feedback:** {basic_feedback}")

    with st.expander(f"## Protagonist Score: {get_emoji(protagonist_score)} {protagonist_score}/10"):
        st.progress(protagonist_score*10)
        st.markdown(f"**Protagonist Feedback:** {protagonist_feedback}")

    with st.expander(f"## Coherence Score: {get_emoji(structure_score)} {structure_score}/10"):
        st.progress(structure_score*10)
        st.markdown(f"**Coherence Feedback:** {structure_feedback}")

