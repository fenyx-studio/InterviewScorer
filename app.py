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
from chains3 import InterviewChains
from judge_chains import JudgeChains
import re
import asyncio

openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


llm = OpenAI(model_name='gpt-3.5-turbo',
             temperature=0)

# assuming all the chain objects and test variables are defined
def run_chain(chain, interviewer_question, interviewee_response):
    return chain.run(interviewer_question=interviewer_question, interviewee_response=interviewee_response)

async def async_run(chain, interviewer_question, interviewee_answer, chain_id):
    result = await chain.arun(interviewer_question=interviewer_question, interviewee_response=interviewee_answer)
    return chain_id, result

async def generate_concurrently(chains, interviewer_question, interviewee_answer):
    tasks = [async_run(chain, interviewer_question, interviewee_answer) for chain in chains.values()]
    return await asyncio.gather(*tasks)

# Get the current session state
state = st.session_state

# Initialize the state if it doesn't exist
if 'interview_chains' not in state:
    state.interview_chains = InterviewChains()

chains = state.interview_chains.get_chains(llm)

chain_responses = []

test_interviewee_answer = ""

test_interviewer_question = ""

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
    rubrics_tab, personas_tab = st.tabs(["Scoring Rubrics", "Personas"])
    with rubrics_tab:
        rubric1 = rubrics_tab.text_area('Rubric #1', state.interview_chains.get_component('rubric1_component'))
        rubric2 = rubrics_tab.text_area('Rubric #2', state.interview_chains.get_component('rubric2_component'))
        rubric3 = rubrics_tab.text_area('Rubric #3', state.interview_chains.get_component('rubric3_component'))
        rubric4 = rubrics_tab.text_area('Rubric #4', state.interview_chains.get_component('rubric4_component'))
    with personas_tab:
        persona1 = personas_tab.text_area('Persona #1', state.interview_chains.get_component('persona1_component'))
        persona2 = personas_tab.text_area('Persona #2', state.interview_chains.get_component('persona2_component'))
        persona3 = personas_tab.text_area('Persona #3', state.interview_chains.get_component('persona3_component'))
    
    if st.button('Update Model'):
        state.interview_chains.set_component('rubric1_component', rubric1)
        state.interview_chains.set_component('rubric2_component', rubric2)
        state.interview_chains.set_component('rubric3_component', rubric3)
        state.interview_chains.set_component('rubric4_component', rubric4)
        state.interview_chains.set_component('persona1_component', persona1)
        state.interview_chains.set_component('persona2_component', persona2)
        state.interview_chains.set_component('persona3_component', persona3)
        chains = state.interview_chains.get_chains(llm)
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

async def synthesis_run(chain, chain_results):
            # Extract perspectives and advices
            perspectives = {chain_id: result['perspective'] for chain_id, result in chain_results.items()}
            advices = {chain_id: result['two_pieces_tactical_advice'] for chain_id, result in chain_results.items()}

            result = await chain.arun(
                interviewer_question=test_interviewer_question,
                interviewee_response=test_interviewee_answer,
                persona1_opinions=perspectives['persona1_rubric1'],
                persona2_opinions=perspectives['persona2_rubric1'],
                persona3_opinions=perspectives['persona3_rubric1'],
                persona1_tactical_advices=advices['persona1_rubric1'],
                persona2_tactical_advices=advices['persona2_rubric1'],
                persona3_tactical_advices=advices['persona3_rubric1']
            )

            # Parse the result as JSON and return
            parsed_result = json.loads(result)
            return parsed_result


# Score Card
if st.button("Submit Answer"):
    with st.spinner('Scoring Interview Answers...'):
        chain_results = {}
        messages = {}  # New dictionary to hold messages
        async def async_run(chain, interviewer_question, interviewee_answer, chain_id):
            result = await chain.arun(interviewer_question=interviewer_question, interviewee_response=interviewee_answer)
            return chain_id, result

        async def generate_concurrently(chains, interviewer_question, interviewee_answer):
            tasks = [async_run(chain, interviewer_question, interviewee_answer, chain_id) for chain_id, chain in chains.items()]
            for future in asyncio.as_completed(tasks):
                chain_id, result = await future
                chain_role = chain_id # chain_id is already the key you want
                chain_responses.append(result)

                # Parse the result as JSON
                print(result)
                chain_results[chain_id] = result

                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    print("Attempting to parse as key-value pairs...")
                    parsed_result = {k: v.strip() for k, v in re.findall(r'(.*?):\s*(.*)', result)}
                
                if isinstance(parsed_result, dict):
                    chain_results[chain_id] = parsed_result
                else:
                    print(f"Invalid result format for chain_id {chain_id}: {result}")

                # Extract the score and note to judge
                score = parsed_result.get('score')
                perspective = parsed_result.get('perspective')

                # Get the emoji for the score
                emoji = get_emoji(score)

                messages[chain_role] = f"**{chain_role}'s Score is in!** \n\n**Perspective:** {perspective} \n\n**Score:** {score}/10"

            # Now all chains have finished, run the synthesis chain
            synthesis_chain = state.interview_chains.get_synthesis_chain(llm)  # Ensure you have defined get_synthesis_chain method
            synthesis_result = await synthesis_run(synthesis_chain, chain_results)
            # Here you can do something with the synthesis_result
            with st.expander(f"JSON Synthesis Results"):
                st.write(synthesis_result)

        # Assuming chains is a dictionary where the keys are chain_ids and the values are the chains themselves
        asyncio.run(generate_concurrently(chains, test_interviewer_question, test_interviewee_answer))

    # Sort the messages by chain_role and display them
    for chain_role in sorted(messages):
        # Display the success message with the emoji
        score_expander = st.expander(f"{chain_role}'s Score")
        with score_expander:
            st.sidebar.success(messages[chain_role], icon=get_emoji(chain_results[chain_role]['score']))

    # Extract scores and feedback
    chain_scores = {}
    chain_tactical_advice = {}
    chain_perspective = {}
    persona_notes = {}
    stricter_scores = {}
    confidence_scores = {}
    for chain_id, result in chain_results.items():
        if 'score' in result:
            chain_scores[chain_id] = result['score']
            chain_tactical_advice[chain_id] = result['two_pieces_tactical_advice']
            chain_perspective[chain_id] = result['perspective']
            persona_notes[chain_id] = result['persona_note']
            stricter_scores[chain_id] = result['stricter_score']
            confidence_scores[chain_id] = result['confidence']

    with st.expander(f"JSON Dictionary Results"):
        st.write(chain_results)
        
    # Display the score card
    st.header("Score Card")



    for chain_id in chain_results.keys():
        score = chain_scores.get(chain_id, 0) # default to 0 if no score
        tactica_advice = chain_tactical_advice.get(chain_id, 'No tactical advice')
        perspective = chain_perspective.get(chain_id, 'No perspective')
        persona_note = persona_notes.get(chain_id, 'No persona note')
        stricter_score = stricter_scores.get(chain_id, 0) # default to 0 if no score
        confidence = confidence_scores.get(chain_id, 0) # default to 0 if no score

        with st.expander(f"## {chain_id} Score: {get_emoji(score)} {score}/100"):
            st.markdown(f"** {chain_id} Perspective:** {perspective}")
            st.markdown(f"** {chain_id} Tactical Advice:** {tactica_advice}")
            st.markdown(f"** {chain_id} Persona Note:** {persona_note}")
            st.markdown(f"** {chain_id} Stricter Score:** {stricter_score}/100")
            st.markdown(f"** {chain_id} Confidence:** {confidence}/5")
            


