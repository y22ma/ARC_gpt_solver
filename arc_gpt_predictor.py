from openai import OpenAI
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import csv
import time
import traceback
from io import StringIO
from helper_functions import *

# abstraction class to facilitate message array management and prompt loading
class BaseOpenAIChatApp:
    def __init__(self, sys_prompt_file):
        with open(sys_prompt_file, 'r') as file:
            self.sys_prompt = file.read()
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.reset()
        self.reasoning_modules = [
            "1. How could I devise an experiment to help solve that problem?",
            "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
            #"3. How could I measure progress on this problem?",
            "4. How can I simplify the problem so that it is easier to solve?",
            "5. What are the key assumptions underlying this problem?",
            "6. What are the potential risks and drawbacks of each solution?",
            "7. What are the alternative perspectives or viewpoints on this problem?",
            "8. What are the long-term implications of this problem and its solutions?",
            "9. How can I break down this problem into smaller, more manageable parts?",
            "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
            "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
            #"12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
            "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
            "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
            #"15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
            "16. What is the core issue or problem that needs to be addressed?",
            "17. What are the underlying causes or factors contributing to the problem?",
            "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
            "19. What are the potential obstacles or challenges that might arise in solving this problem?",
            "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
            "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
            "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
            "23. How can progress or success in solving the problem be measured or evaluated?",
            "24. What indicators or metrics can be used?",
            "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
            "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
            "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
            "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
            "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
            "30. Is the problem a design challenge that requires creative solutions and innovation?",
            "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
            "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
            "33. What kinds of solution typically are produced for this kind of problem specification?",
            "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
            "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
            "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
            "37. Ignoring the current best solution, create an entirely new solution to the problem."
            #"38. Let’s think step by step."
            "39. Let’s make a step by step plan and implement it with good notation and explanation."
        ]


    def clean_files(self):
        file_list = self.client.files.list()
        for fileobject in file_list.data:
            self.client.files.delete(file_id=fileobject.id)
            time.sleep(0.1)

    def reset(self):
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        #self.messages = []

    # function to send a chat message and manage the conversation message queues
    def chat(self, user_prompt):
        self.messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=self.messages)

        #self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        self.messages = [{"role": "system", "content": self.sys_prompt}, {"role": "assistant", "content": response.choices[0].message.content}]
        response_content = json.loads(response.choices[0].message.content)
        return response_content

# this class uses the assistant API instead to take advantage of code interpreter
class CodeTester():
    def test(self, python_code, task_data):
        error_msg = None
        try:
            # test out if function compiles
            local_scope = {}
            print(python_code)
            exec(python_code, globals())
            # Retrieve the compiled function from the local scope
            for i, question in enumerate(task_data["train"]):
                task_data["train"][i]["predicted_output"] = transform_grid(question["input"])
                task_data["train"][i]["passed"] = task_data["train"][i]["predicted_output"] == question["output"]
                if not task_data["train"][i]["passed"]:
                    print("predicted {} vs truth {}".format(task_data["train"][i]["predicted_output"], question["output"]))

            for i, question in enumerate(task_data["test"]):
                task_data["test"][i]["predicted_output"] = transform_grid(question["input"])

            return task_data
        except Exception as e:
            exception_traceback = StringIO()
            traceback.print_exc(file=exception_traceback)
            error_msg = exception_traceback.getvalue()
            print('Code did not work')
            print(error_msg)

        return {"error_message": error_msg}

solver = BaseOpenAIChatApp(sys_prompt_file="sys_prompt.txt")
code_tester = CodeTester()

# helper function to convert grids from 2D array format to the submission format
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# helper function to convert 2D int array to char grid view, suggested by the paper
def convert_int_grid_to_char(array2d):
    value_char_map = ['.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    char_array2d = []
    for row in array2d:
        char_row = [value_char_map[val] for val in row]
        char_array2d.append(char_row)
    return char_array2d

# helper function to convert 2D char grid array to int grid view, suggested by the paper
def convert_char_grid_to_int(array2d):
    char_value_map = {'.': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10}
    int_array2d = []
    for row in array2d:
        int_row = [char_value_map[char] for char in row]
        int_array2d.append(int_row)
    return int_array2d

def get_grid_size(array2d):
    return {"width": len(array2d[0]), "height": len(array2d)}

# a function that feeds the ARCOpenAISolver the task information and the output id and
# return the output
def solve_arc_question(task_json, task_file_path, output_id):
    modified_task_json = copy.deepcopy(task_json)

    for i in range(len(modified_task_json["train"])):
        modified_task_json["train"][i]["input"] = convert_int_grid_to_char(modified_task_json["train"][i]["input"])
        modified_task_json["train"][i]["input_grid_size"] = get_grid_size(modified_task_json["train"][i]["input"])
        modified_task_json["train"][i]["input_pixel_coords"] = get_pixel_coords(modified_task_json["train"][i]["input"])
        modified_task_json["train"][i]["input_objects"] = get_objects(modified_task_json["train"][i]["input"], more_info=False)
        modified_task_json["train"][i]["output"] = convert_int_grid_to_char(modified_task_json["train"][i]["output"])
        modified_task_json["train"][i]["output_grid_size"] = get_grid_size(modified_task_json["train"][i]["output"])
        modified_task_json["train"][i]["output_pixel_coords"] = get_pixel_coords(modified_task_json["train"][i]["output"])
        modified_task_json["train"][i]["output_objects"] = get_objects(modified_task_json["train"][i]["output"], more_info=False)

    for i in range(len(modified_task_json["test"])):
        modified_task_json["test"][i]["input"] = convert_int_grid_to_char(modified_task_json["test"][i]["input"])
        modified_task_json["test"][i]["input_grid_size"] = get_grid_size(modified_task_json["test"][i]["input"])
        modified_task_json["test"][i]["input_pixel_coords"] = get_pixel_coords(modified_task_json["test"][i]["input"])
        modified_task_json["test"][i]["input_objects"] = get_objects(modified_task_json["test"][i]["input"], more_info=False)
        if "output" in modified_task_json["test"][i]:
            del modified_task_json["test"][i]["output"]

    # SELECT step of Self Discovery
    task_description = "Your task is to find the transform between the input-output pairs in the 'train' section in this JSON:\n{}".format(modified_task_json) 
    user_prompt = "\n{}\nWhich of the following modules are relevant? Do elborate why\n\n{}\n".format(task_description, solver.reasoning_modules)
    user_prompt += '\nPlease respond in this JSON format: {"selected_modules": ["module_1", "module_2", ...]}'
    selected_modules = solver.chat(user_prompt=user_prompt)
    print(selected_modules)

    # ADAPT step of Self Discovery
    user_prompt = "Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{}\n\n{}".format(selected_modules, task_description)
    user_prompt += '\nPlease respond in this JSON format: {"adapted_modules": ["adapted module_1", "adapted module_2", ...]}'
    adapted_modules = solver.chat(user_prompt=user_prompt)
    print(adapted_modules)

    # IMPLEMENT
    user_prompt = "Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{}\n\n{}".format(adapted_modules, task_description)
    user_prompt += '\nPlease respond in this JSON format: {"reasoning_structure": ["structure_1 for adapted module_1", "structure_2 for adapted module_2", ...]}'
    reasoning_structure = solver.chat(user_prompt=user_prompt)
    print(reasoning_structure)

    # EXECUTE
    retries = 5
    solution = {"passed": False, "output": [[0]]}

    # provide environment feedback from code tester to the solver agent to ask for correction
    feedback_prompt = ""
    response_format = {
      "reflection": "reflect on the answer from the previous iteration if there is an error or failed to match prediction to groundtruth output",
      "pixel_changes": "describe the changes between the input and output pixels, focusing on movement or pattern changes",
      "object_changes": "describe the changes between the input and output objects, focusing on movement, object number, size, shape, position, value, cell count",
      "helper_functions": "list any relevant helper_functions for this task",
      "overall_pattern": "describe the simplest input-output relationship for all input-output pairs",
      "program_instructions": "Plan how to write the python function and what helper functions and conditions to use. Use a training input-output pair as an example to test your thought process",
      "python_program": "Python function named 'transform_grid' that takes in a 2D grid and generates a 2D grid. Output as a string in a single line with \n and \t."
    }
    user_prompt =  "With this reasoning structure:\n{}\n\n{}\n".format(reasoning_structure, task_description)
    user_prompt += "Please respond in this JSON format:\n{}".format(response_format)
    for i in range(retries):
        print("Solving for the {} time".format(i + 1))
        solver_response = solver.chat(user_prompt=user_prompt + feedback_prompt)
        print(solver_response)

        task_json_buf = copy.deepcopy(modified_task_json)
        result = code_tester.test(python_code=solver_response["python_program"], task_data=task_json_buf)
        if "error_message" in result:
            feedback_prompt = '\nPrevious Program:\n'+ solver_response["python_program"]
            feedback_prompt += '\nError message:\n' + result["error_message"]
            feedback_prompt += f'\nPrevious overall pattern: {solver_response["overall_pattern"]}.\n\nYour code had compilation errors. Correct it.'
            continue

        solution["passed"] = True
        for question in task_json_buf["train"]:
            solution["passed"] = question["passed"]
        output = convert_char_grid_to_int(result["test"][output_id]["predicted_output"])
        solution["output"] = output
        solution["solver_response"] = solver_response
        solution["python_program"] = solver_response["python_program"]

        if not solution["passed"]:
            feedback_prompt = "The program you provided fails to produce output predictions that match groundtruth for all input-output pairs in the train section."
            feedback_prompt += "Check the predicted_output vs the output entries in the JSON."
            feedback_prompt += '\nPrevious Program:\n'+ solver_response["python_program"]
            feedback_prompt += f'\nPrevious overall pattern: {solver_response["overall_pattern"]}.\n'
            feedback_prompt += f'\nJSON containing predicted output:\n{result}.\n'
            feedback_prompt += 'Please rethink your strategy.'
        else:
            break

        # attempt to avoid rate limit
        time.sleep(1)

    return solution

# helper function to plot and visualize the input/output pairs in train set
def show_image_from_json(task_json, input, predicted, groundtruth):
    train_data = task_json['train']
    test_data = task_json['test']
    fig, axs = plt.subplots(len(train_data) + 2, 2)
    for i, item in enumerate(train_data):
        input_image = np.array(item['input'])
        output_image = np.array(item['output'])
        axs[i, 0].imshow(input_image, cmap='viridis')
        axs[i, 0].set_title('Input Image')
        axs[i, 1].imshow(output_image, cmap='viridis')
        axs[i, 1].set_title('Output Image')

    input_image = np.array(input)
    predicted_image = np.array(predicted)
    axs[-2, 0].set_title('Test Input')
    axs[-2, 0].imshow(input_image, cmap='viridis')
    axs[-2, 1].set_title('Predicted')
    axs[-2, 1].imshow(predicted_image, cmap='viridis')

    if groundtruth:
      groundtruth_image = np.array(groundtruth)
      axs[-1, 0].set_title('Groundtruth')
      axs[-1, 0].imshow(groundtruth_image, cmap='viridis')

    plt.show()

# MAIN
if __name__ == "__main__":
    #folder = '/kaggle/input/abstraction-and-reasoning-challenge/test'
    #task_files = os.listdir(folder)
    folder = './evaluation/'
    task_files = os.listdir(folder)
    success_count = 0
    attempt_count = 0
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    output_ids = []
    outputs = []
    for json_file in task_files:
        attempt_count = attempt_count + 1
        task_file_path = os.path.join(folder, json_file)
        with open(task_file_path, 'r') as f:
            task_data = json.load(f)
            task_id = json_file.split('.')[0]

        solution_filename = os.path.join(result_folder, f"{task_id}_solution.json")
        if os.path.exists(solution_filename):
            success_count = success_count + 1
            print("Success count {}/{}".format(success_count, attempt_count))
            continue

        # for each question within each task, append a flattened output
        # with the corresponding {task_id}_{output_id}
        for output_id, question in enumerate(task_data["test"]):
            print("solving for task {} output_id {}".format(task_id, output_id))
            output = [[0]]
            try:
                solution = solve_arc_question(task_data, task_file_path, output_id)
                output = solution["output"]
            except Exception as err:
                print("Error encountered during answering for task {}, error: {}".format(task_id, err))
                #raise

            groundtruth = None
            if "output" in task_data["test"][output_id]:
                groundtruth = task_data["test"][output_id]["output"]
                task_success = flattener(output) == flattener(groundtruth)
                print("predicted {} vs groundtruth {}".format(flattener(output), flattener(groundtruth)))
                print("Pass? {}".format(task_success))

                if task_success:
                    with open(solution_filename, 'w') as solution_file:
                        json.dump({'solution': solution}, solution_file)
                    success_count = success_count + 1

            outputs.append(output)
            output_ids.append("{}_{}".format(task_id, output_id))
            print("Success count {}/{}".format(success_count, attempt_count))

            # uncomment to view dataset visualization
            #test_input = task_data["test"][output_id]["input"]
            #show_image_from_json(task_data, test_input, output, groundtruth)


        # write out the submission.csv according to ARC requirement
        with open('submission.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["output_id", "output"])
            for i in range(len(output_ids)):
                writer.writerow([output_ids[i], flattener(outputs[i])])
