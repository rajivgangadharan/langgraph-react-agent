"""Default prompts used by the agent."""

SYSTEM_PROMPT = """
You are a helpful AI assistant.
System time: {system_time}"""

question_generation = """

Tool Names: {tool_names}
Tools: {tools}

You are a teacher creating a question for a student. You will use
generate the question through a series of steps 
Task, Thought, Action, Observation and Final Answer.

Task: Creating a Question
Thought: Your thought about the Task specifially which tool to use
Action: the action to take, should be one of [{tool_names}]
Action Input: {action_input}
Observation: the result of the action , {agent_scratchpad}
... (this Thought/Action/Observation can repeat 3 times)
Thought: I now know the final answer
Final Answer: {final_answer}

You will have to iterate Thought, Action and Observation with different tools 
in the following order 
- Retrieve Context
- Generate Question
The context from the first tool will be appended to the context and fed to the second tool

Generate a question from {topic} that is appropriate for the student's grade 
{grade} studying in {board} board, which is {difficulty}. The provided {context} 
has to be considered while formulating the question.

The question should be engaging and test the student's understanding 
of the topic.

Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Example Session:

Task: Generate a Question for the Topic pulleys based on the results
Thought: I need to generate a question about pulleys for a 10th-grade ICSE student by calling the necessary Action.
Action: Generate Question
Action Input: "topic": "pulleys", "difficulty": "hard", "board": "ICSE", "grade": "10th", "context": "Pulleys ICSE Class 10 Pulleys in the context of ICSE Class 10 Physics are devices used to lift heavy loads easily by changing the direction of the force applied. There are two main types of pulleys: fixed and movable. A fixed pulley has its axis of rotation stationary and is used to lift small loads such as a bucket of water. The mechanical advantage of a single fixed pulley is 1, meaning the effort required to lift the load is equal to the load itself, and it cannot be used as a force multiplier. The velocity ratio of a single fixed pulley is also 1. A movable pulley, on the other hand, has a movable axis of rotation and acts as a force multiplier. The mechanical advantage of a single movable pulley in an ideal case is 2. The velocity ratio of a single movable pulley is 2."
Observation: The question has been generated. 
Final Answer: Provide the question generated

Output only the question, without any additional text.

Your available tools are: 

Retrieve Context: 
    e.g. retrieve_context: [topic] [board] [difficulty]  [grade]  [context]
    Returns the [context] from the documents in the vector store

Generate Question:
    e.g. generate_question: [topic] [board] [difficulty]  [grade]  [context]
    Generates a question based on the Topic, Board, Grade , Difficulty and Context passed to it

{tools}

Question:"""

answer_evaluation = """
You are a teacher evaluating a student's answer.
Question: {question}
Student's Answer: {answer}

Evaluate the answer and provide:
1. A grade (A, B, C, D, or F).
2. Detailed feedback explaining why the answer is correct or incorrect.
3. Examples or additional information to help the student learn.

Output the evaluation in the following format:
Grade: <grade>
Feedback: <feedback>
Examples: <examples>

Evaluation:
"""

react = """
You are a helpful AI assistant. 

Tools:
{tool_names}


You run in a loop of Thought, Action, Observation and Final Answer

Task: {task}
Thought: {agent_scratchpad}
Action: {action}
Action Input: {action_input}
Observation: {observation}
Final Answer: {final_answer}

Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions. The answer will 
be 

Use the following tools to complete the task. 
Run through the process of identifying the Task, Thought, Action, 
Action Input, Obvervation and then finally produce the Final Answer:

Your available tools are: 

retrieve_context: 
    e.g. retrieve_context: Topic: [topic]
    Returns the context from the documents in the vector store

generate_question:
    e.g. generate_question: Context: Topic is [topic] for student studying in [board] which is [difficulty] for grade [grade]
    Generates a question based on the Topic, Board, Grade and Difficulty

search:
    e.g. search: Query: Topic is [topic] for student studying in [board] which is [difficulty] for grade [grade]
    returns search results from the internet as a result of this search

evaluate_answer:
e.g: evaluate_answer 
    Question: How does the refractive index of a medium related to it's real depth and apparent depth?
    Answer: Refractive Index = Real Depth / Apparent Depth

Example Sessions:

Task: Fetch the documents for the topic pulleys
Thought: I need to fetch the documents for the topic pulleys
Action: retrieve_context
Action Input: {"topic": "pulleys"}
Observation: The context for the topic has been fetched
Final Answer: {"context": "Pulleys ICSE Class 10 Pulleys in the context of ICSE Class 10 Physics are devices used to lift heavy loads easily by changing the direction of the force applied. There are two main types of pulleys: fixed and movable. A fixed pulley has its axis of rotation stationary and is used to lift small loads such as a bucket of water. The mechanical advantage of a single fixed pulley is 1, meaning the effort required to lift the load is equal to the load itself, and it cannot be used as a force multiplier. The velocity ratio of a single fixed pulley is also 1. A movable pulley, on the other hand, has a movable axis of rotation and acts as a force multiplier. The mechanical advantage of a single movable pulley in an ideal case is 2. The velocity ratio of a single movable pulley is 2." }

Task: Search the internet based on the contents
Thought: I need to seach the internet now that I have got the context
Action: search
Action Input: {"query": "Pulleys ICSE Class 10 Pulleys in the context of ICSE Class 10 Physics are devices used to lift heavy loads easily by changing the direction of the force applied. There are two main types of pulleys: fixed and movable. A fixed pulley has its axis of rotation stationary and is used to lift small loads such as a bucket of water. The mechanical advantage of a single fixed pulley is 1, meaning the effort required to lift the load is equal to the load itself, and it cannot be used as a force multiplier. The velocity ratio of a single fixed pulley is also 1. A movable pulley, on the other hand, has a movable axis of rotation and acts as a force multiplier. The mechanical advantage of a single movable pulley in an ideal case is 2. The velocity ratio of a single movable pulley is 2." }

Observation: The search results have been fetched, I will take the top one
Final Answer: {"results": "Pulleys ICSE Class 10 Pulleys in the context of ICSE Class 10 Physics are devices used to lift heavy loads easily by changing the direction of the force applied. There are two main types of pulleys: fixed and movable. A fixed pulley has its axis of rotation stationary and is used to lift small loads such as a bucket of water. The mechanical advantage of a single fixed pulley is 1, meaning the effort required to lift the load is equal to the load itself, and it cannot be used as a force multiplier. The velocity ratio of a single fixed pulley is also 1. A movable pulley, on the other hand, has a movable axis of rotation and acts as a force multiplier. The mechanical advantage of a single movable pulley in an ideal case is 2. The velocity ratio of a single movable pulley is 2." }

Task:  Generate a Question for the Topic pulleys based on the results
Thought: I need to generate a question about pulleys for a 10th-grade ICSE student.
Action: generate_question
Action Input: {"topic": "pulleys", "difficulty": "hard", "board":"ICSE", "grade": "10th", "context": "Pulleys ICSE Class 10 Pulleys in the context of ICSE Class 10 Physics are devices used to lift heavy loads easily by changing the direction of the force applied. There are two main types of pulleys: fixed and movable. A fixed pulley has its axis of rotation stationary and is used to lift small loads such as a bucket of water. The mechanical advantage of a single fixed pulley is 1, meaning the effort required to lift the load is equal to the load itself, and it cannot be used as a force multiplier. The velocity ratio of a single fixed pulley is also 1. A movable pulley, on the other hand, has a movable axis of rotation and acts as a force multiplier. The mechanical advantage of a single movable pulley in an ideal case is 2. The velocity ratio of a single movable pulley is 2." }
Observation: The question has been generated.
Final Answer: {"question": "What is the mechanical advantage of a compound pulley system?"}

Task: Evaluate the provided Answer for the question 
Thought: I need to evaluate the answer provided by the student
Action: evaluate_answer
Action Input: {"question": "How does the refractive index of a medium related to it's real depth and apparent depth?
, "answer": "Refractive Index = Real Depth / Apparent Depth"
}
Observation: The answer is evaluated
Final Answer: The Answer is correct, you scored a 10 out of possible 10 marks. 
"""
