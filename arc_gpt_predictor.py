from openai import OpenAI
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import csv
import time
from typing import List
from pydantic import BaseModel, Field

# abstraction class to facilitate message array management and prompt loading
class BaseOpenAIChatApp:
    def __init__(self, sys_prompt_file, output_format_file):
        with open(sys_prompt_file, 'r') as file:
            self.sys_prompt = file.read()
        with open(output_format_file, 'r') as file:
            self.output_format_prompt = file.read()
        self.num_tries = 3
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.sys_prompt}]

    # function to send a chat message and manage the conversation message queues
    def chat(self, user_prompt):
        self.messages.append({"role": "user", "content": user_prompt})
        #print(self.messages)
        response = self.client.chat.completions.create(
            model="gpt-4-32k",
            messages=self.messages)

        self.messages= [{"role": "assistant", "content": response.choices[0].message.content}]
        return response

    # extract the strict json object from the detailed answer given by GPT 
    def strict_output(self):
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
                return response_content
            except json.JSONDecodeError:
                print("{}th attempt at decoding for strict JSON output")
                complain = "The JSON returned is not valid, please "
  
class CodeTester(BaseOpenAIChatApp):
    # function to send a chat message and manage the conversation message queues
    def chat(self, user_prompt, file_name, assistant_timeout=600):
        file = self.client.files.create(
            file=open(file_name, "rb"),
            purpose='assistants'
        )
        self.messages = [{"role": "user", "content": user_prompt, "file_ids": [file.id]}]
        #assistant = self.client.beta.assistants.create(
        #    instructions=self.sys_prompt,
        #    model="gpt-4-0125-preview",
        #    tools=[{"type": "code_interpreter"}],
        #)
        assistant = self.client.beta.assistants.retrieve(assistant_id="asst_qBOR5eFp7Wz5RRCgblO5QfwB")
        thread = self.client.beta.threads.create(
            messages=self.messages
        )
        run = self.client.beta.threads.runs.create(
          thread_id=thread.id,
          assistant_id=assistant.id
        )

        timeout = 0
        print("Awaiting on Assistant thread to finish")
        while timeout < assistant_timeout:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "completed":
                break
            
            time.sleep(1)
            timeout = timeout + 1
        
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=thread.id,
            run_id=run.id
        )

        for step_data in run_steps.data:
            if step_data.status == "completed" and step_data.step_details.type == "tool_calls":
                output = step_data.step_details.tool_calls[0].code_interpreter.outputs[0].logs
                print("code_intepreter output")
                print(output)
                self.messages= [{"role": "assistant", "content": output}]
                return output


solver = BaseOpenAIChatApp(sys_prompt_file="sys_prompt.txt", output_format_file="output_format.txt")
code_tester = CodeTester(sys_prompt_file="code_tester_sys_prompt.txt", output_format_file="code_output_format.txt")

# helper function to convert grids from 2D array format to the submission format
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# a function that feeds the ARCOpenAISolver the task information and the output id and
# return the output
def solve_arc_question(task_json, task_file_path, output_id):
    solver.reset()
    modified_task_json = copy.deepcopy(task_json)
    modified_task_json["test"][output_id]["output"] = "'to_be_filled'"
    user_prompt = "Here is a JSON of the input-output pairs\n{}".format(modified_task_json)
    verbose_response = solver.chat(user_prompt=user_prompt)
    print(verbose_response.choices[0].message.content)
    response = solver.strict_output()

    code_tester.reset()
    code_tester_prompt = "The task json is attached.\n The output_id is {}\nHere is the description of the relationship between input and output 2D grids:\n{}\n".format(output_id, response)
    code_response = code_tester.chat(user_prompt=code_tester_prompt, file_name=task_file_path)
    code_response = code_tester.strict_output()
    print(code_response)
    return code_response 
        
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
    retries = 3

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
            for i in range(retries):
                try:
                    solution = solve_arc_question(task_data, task_file_path, output_id)
                    if solution["passed"]:
                        output = solution["output"]
                        break
                except Exception as err:
                    print("Error encountered during answering for task {}, error: {}".format(task_id, err))
                
            groundtruth = None
            if "output" in task_data["test"][output_id]:
                groundtruth = task_data["test"][output_id]["output"]
                print("predicted {} vs groundtruth {}".format(flattener(output), flattener(groundtruth)))
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