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

    def clean_files(self):
        file_list = self.client.files.list()
        for fileobject in file_list.data:
            self.client.files.delete(file_id=fileobject.id)
            time.sleep(0.1)

    def reset(self):
        self.messages = [{"role": "system", "content": self.sys_prompt}]

    # function to send a chat message and manage the conversation message queues
    def chat(self, user_prompt):
        self.messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=self.messages)

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
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
            #transform_grid = local_scope['transform_grid']
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

    user_prompt = "Here is a JSON of the input-output pairs\n{}".format(modified_task_json)

    retries = 5
    solution = {"passed": False, "output": [[0]]}
    solver.reset()

    # provide environment feedback from code tester to the solver agent to ask for correction
    for i in range(retries):
        print("Solving for the {} time".format(i + 1))
        solver_response = solver.chat(user_prompt=user_prompt)
        print(solver_response)

        task_json_buf = copy.deepcopy(modified_task_json)
        result = code_tester.test(python_code=solver_response["python_program"], task_data=task_json_buf)
        if "error_message" in result:
            user_prompt += '\nPrevious Program:\n'+ solver_response["python_program"]
            user_prompt += '\nError message:\n' + result["error_message"]
            user_prompt += f'\nPrevious overall pattern: {solver_response["overall_pattern"]}.\n\nYour code had compilation errors. Correct it.'
            continue

        solution["passed"] = True
        for question in task_json_buf["train"]:
            solution["passed"] = question["passed"]
        output = convert_char_grid_to_int(result["test"][output_id]["predicted_output"])
        solution["output"] = output
        solution["solver_response"] = solver_response
        solution["python_program"] = solver_response["python_program"]

        if not solution["passed"]:
            user_prompt = "The program you provided fails to produce output predictions that match groundtruth for all input-output pairs in the train section."
            user_prompt += "Check the predicted_output vs the output entries in the JSON."
            user_prompt += '\nPrevious Program:\n'+ solver_response["python_program"]
            user_prompt += f'\nPrevious overall pattern: {solver_response["overall_pattern"]}.\n'
            user_prompt += f'\nJSON containing predicted output:\n{result}.\n'
            user_prompt += 'Please rethink your strategy.'
        else:
            break

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

    output_ids = []
    outputs = []
    for json_file in task_files:
        task_file_path = os.path.join(folder, json_file)
        with open(task_file_path, 'r') as f:
            task_data = json.load(f)
            task_id = json_file.split('.')[0]

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

                if not os.path.exists('results'):
                    os.makedirs('results')
                    
                if task_success:
                    solution_filename = os.path.join("results", f"{task_id}_solution.json")
                    with open(solution_filename, 'w') as solution_file:
                        json.dump({'solution': solution}, solution_file)
                    success_count = success_count + 1
                    

            outputs.append(output)
            output_ids.append("{}_{}".format(task_id, output_id))
            print("Success count {}/{}", success_count, len(task_files))

            # uncomment to view dataset visualization
            #test_input = task_data["test"][output_id]["input"]
            #show_image_from_json(task_data, test_input, output, groundtruth)
            

        # write out the submission.csv according to ARC requirement
        with open('submission.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["output_id", "output"])
            for i in range(len(output_ids)):
                writer.writerow([output_ids[i], flattener(outputs[i])])