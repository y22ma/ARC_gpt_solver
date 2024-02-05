from openai import OpenAI
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import csv
import time
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
class CodeTester(BaseOpenAIChatApp):
    def __init__(self, sys_prompt_file):
        super().__init__(sys_prompt_file)
        # code to create your own assistant using the system prompt. Please comment out the next line
        # that calls assistants.retrieve

        #helper_function_file = self.client.files.create(
        #    file=open("helper_functions.py", "rb"),
        #    purpose='assistants'
        #)
        #self.helper_function_fid = helper_function_file.id
        #self.assistant = self.client.beta.assistants.create(
        #    instructions=self.sys_prompt,
        #    model="gpt-4-turbo-preview",
        #    tools=[{"type": "code_interpreter"}],
        #    file_ids=[helper_function_file.id]
        #)
        self.assistant = self.client.beta.assistants.retrieve(
            assistant_id="asst_wms9TS1vJ4QP1o4dp9ZtJ9id")
        self.helper_function_fid = "asst_qBOR5eFp7Wz5RRCgblO5QfwB"

    # function to send a chat message and manage the conversation message queues
    def chat(self, user_prompt, file_name, assistant_timeout=600):
        file = self.client.files.create(
            file=open(file_name, "rb"),
            purpose='assistants'
        )
        user_prompt += "\nPlease see file with ID {} for the task JSON file".format(file.id)
        #user_prompt += "\nPlease see file with ID {} for the helper_functions.py, you can refer to it when building your transform function".format(self.helper_function_fid)
        user_prompt += "\nPLEASE NAME THE OUTPUT FILE task_result.json"
        self.messages = [{"role": "user", "content": user_prompt, "file_ids": [file.id]}]
        print(user_prompt)

        thread = self.client.beta.threads.create(
            messages=self.messages
        )
        run = self.client.beta.threads.runs.create(
          thread_id=thread.id,
          assistant_id=self.assistant.id
        )

        timeout = 0
        print("Awaiting on Assistant thread to finish")
        while timeout < assistant_timeout:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "completed":
                break
            
            time.sleep(1)
            timeout = timeout + 1
        self.client.files.delete(file_id=file.id)

        print("Fetching task_json output")
        file_list = self.client.files.list()
        for fileobject in file_list.data:
            print(fileobject)
            if fileobject.filename == '/mnt/data/task_result.json':
                task_file_content = self.client.files.retrieve_content(file_id=fileobject.id)
                task_json = json.loads(task_file_content)
                self.client.files.delete(file_id=fileobject.id)
                return task_json


solver = BaseOpenAIChatApp(sys_prompt_file="sys_prompt.txt")
code_tester = CodeTester(sys_prompt_file="code_tester_sys_prompt.txt")

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

    with open('/tmp/modified_task.json', 'w') as f:
        json.dump(modified_task_json, f)
    user_prompt = "Here is a JSON of the input-output pairs\n{}".format(modified_task_json)
    print(user_prompt)

    retries = 3
    retry_with_context = False
    solution = {}
    solver.reset()

    # provide environment feedback from code tester to the solver agent to ask for correction
    for i in range(retries):
        print("Solving for the {} time".format(i + 1))
        solver_response = solver.chat(user_prompt=user_prompt)

        code_tester.reset()
        code_tester_prompt = "Here is the description of the relationship between input and output 2D grids:\n{}\n".format(solver_response)
        code_response = code_tester.chat(user_prompt=code_tester_prompt, file_name='/tmp/modified_task.json')
        if code_response is None:
            solution["passed"] = false
            solution["output"] = [[0]]
        else:
            solution["passed"] = True
            for question in code_response["train"]:
                solution["passed"] = question["passed"]
            output = convert_char_grid_to_int(code_response["test"][output_id]["predicted_output"])
            solution["output"] = output
        print(solution)
        if not solution["passed"]:
            if retry_with_context:
                user_prompt = "Based on your description of the relationship between input-output pairs,"
                user_prompt += "a python problem is implemented, and it fails to produce output predictions that match groundtruth for all input-outpu pairs in the train section."
                user_prompt += "Here's the result:\n{}\nPlease try again".format(code_response)
            else:
                solver.reset()
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
                if solution["passed"]:
                    break
            except Exception as err:
                print("Error encountered during answering for task {}, error: {}".format(task_id, err))
                raise
                
            groundtruth = None
            if "output" in task_data["test"][output_id]:
                groundtruth = task_data["test"][output_id]["output"]
                print("predicted {} vs groundtruth {}".format(flattener(output), flattener(groundtruth)))
                print("Pass? {}".format(flattener(output) == flattener(groundtruth)))
            outputs.append(output)
            output_ids.append("{}_{}".format(task_id, output_id))

            # uncomment to view dataset visualization
            test_input = task_data["test"][output_id]["input"]
            show_image_from_json(task_data, test_input, output, groundtruth)

        # write out the submission.csv according to ARC requirement
        with open('submission.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["output_id", "output"])
            for i in range(len(output_ids)):
                writer.writerow([output_ids[i], flattener(outputs[i])])