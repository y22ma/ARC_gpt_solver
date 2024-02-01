from openai import OpenAI
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import csv
from typing import List
from pydantic import BaseModel, Field

# abstraction class to facilitate message array management and prompt loading
class ARCOpenAISolver:
    def __init__(self):
        with open('sys_prompt.txt', 'r') as file:
            self.sys_prompt = file.read()
        self.output_format_prompt = "Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.\n"
        self.output_format_prompt += "{'output': <2D array representing a rectangular grid of integers ranging from 0 to 9>}"
        self.num_tries = 3
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.reset()

    # function to send a chat message and manage the conversation message queues
    def chat(self, user_prompt):
        self.messages.append({"role": "user", "content": user_prompt})
        #print(self.messages)
        response = self.client.chat.completions.create(
            model="gpt-4-32k",
            messages=self.messages)

        self.messages= [{"role": "assistant", "content": response.choices[0].message.content}]
        return response
    
    # function to reset the conversation
    def reset(self):
        self.messages = [{"role": "system", "content": self.sys_prompt}]

    # utility function to check if the output json object matches with the expected schema
    def schema_match(self, json_obj):
        if "output" not in json_obj:
            return False
        if not isinstance(json_obj["output"], list) or not all(isinstance(row, list) for row in json_obj["output"]):
            return False
        if len(json_obj["output"]) == 0:
            return False
        row_length = len(json_obj["output"][0])
        for row in json_obj["output"]:
            if len(row) != row_length or not all(isinstance(element, int) for element in row):
                return False
        return True

    # extract the strict json object from the detailed answer given by GPT 
    def strict_json(self):
        complain = ""
        for i in range(self.num_tries):
            strict_json_prompt = complain + self.output_format_prompt
            self.messages.append({"role": "user", "content": strict_json_prompt})
            response = self.client.chat.completions.create(
                model="gpt-4-0125-preview",
                response_format={
                    "type": "json_object", 
                },
                messages=self.messages)
            try:
                response_content = json.loads(response.choices[0].message.content)
                print(response_content)
                if not self.schema_match(response_content):
                    raise json.JSONDecodeError("The JSON schema of the response does not match the expected schema.")
                return response_content
            except json.JSONDecodeError:
                print("{}th attempt at decoding for strict JSON output")
                complain = "The JSON returned is not valid, please "

chatApp = ARCOpenAISolver()

# helper function to convert grids from 2D array format to the submission format
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# a function that feeds the ARCOpenAISolver the task information and the output id and
# return the flattened representation of the output
def solve_arc_question(task_json, output_id):
    chatApp.reset()
    modified_task_json = copy.deepcopy(task_json)
    modified_task_json["test"][output_id]["output"] = "'to_be_filled'"
    user_prompt = "Here is a JSON of the input-output pairs\n{}".format(modified_task_json)
    verbose_response = chatApp.chat(user_prompt=user_prompt)
    print(verbose_response.choices[0].message.content)
    
    response = chatApp.strict_json()

    output = response["output"]
    return output 
        
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


if __name__ == "__main__":
    #folder = './traininig/'
    #train_files = os.listdir(folder)
    folder = './evaluation/'
    task_files = os.listdir(folder)

    output_ids = []
    outputs = []
    for json_file in task_files:
        with open(os.path.join(folder, json_file), 'r') as f:
            task_data = json.load(f)
            task_id = json_file.split('.')[0]

            # for each question within each task, append a flattened output
            # with the corresponding {task_id}_{output_id}
            for output_id, question in enumerate(task_data["test"]):
                print("solving for task {} output_id {}".format(task_id, output_id))
                try:
                    output = solve_arc_question(task_data, output_id)
                except Exception as err:
                    print("Error encountered during answering for task {}, error: {}".format(task_id, err))
                    output = [[0]]
                    #raise err
                groundtruth = None
                if "output" in task_data["test"][output_id]:
                    groundtruth = task_data["test"][output_id]["output"]
                    print("predicted {} vs groundtrudth {}".format(flattener(output), flattener(groundtruth)))
                    print("Pass? {}".format(flattener(output) == flattener(groundtruth)))
                outputs.append(output)
                output_ids.append("{}_{}".format(task_id, output_id))

                test_input = task_data["test"][output_id]["input"]
                show_image_from_json(task_data, test_input, output, groundtruth)


            # write out the submission.csv according to ARC requirement
            with open('submission.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["output_id", "output"])
                for i in range(len(output_ids)):
                    writer.writerow([output_ids[i], flattener(outputs[i])])