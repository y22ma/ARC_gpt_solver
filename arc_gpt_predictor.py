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
        self.problem_types = [
            "Image reduction: input is a larger image, and patches of the image is aggregated (max reduction, min reduction, etc) and the output is recorded in the corresponding spot in the output",
            "Pixel value change: input image and output image has the same occupancy shape in the grid, but pixels with some values are changed to another value",
            "Pixel position: changing the position of some pixels. For examples, move all pixels of a certain color to row=0 or col=0",
            "Object change: transformations where one or more objects are translated, reflected, or rotated based on some conditions, or color changes based on some conditions",
            "Object counting: count object with the same shape, and output the grid of the object with a highest quantities",
            "Moving object: move objects by selecting an anchor and moving it to another location",
            "Reflecting the image: reflect some part of the image to create a larger image",
            "Rotate the image: rotate some part of the image by different degree amounts and tile them to create a larger image",
            "Copy the image: copy some part of the image as a tile to create a larger image",
            "Creating lines: create lines from an anchor or by connecting two coordinates",
            "Creating a border around the image with some color value",
            "Creating a border around the object with some color value",
            "Inpaint: filled in image content that's masked out by a value, using information from other parts of the image (symmetry from a reflection)",
        ]


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
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=self.messages)

        #self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        self.messages = [{"role": "system", "content": self.sys_prompt}, {"role": "assistant", "content": response.choices[0].message.content}]
        response_content = json.loads(response.choices[0].message.content)
        return response_content

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
def solve_arc_question(task_json):
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
    user_prompt = "\n{}\nDescribe the relationships between the input_grid_size and the output_grid_size each of the input-output pair in the 'train' section in this JSON".format(task_description)
    user_prompt += '\nPlease respond in this JSON format: {"grid_size_observation": ["grid_size changes for pair1", "grid_size_changes for pair2", ...]}'
    grid_size_obs = solver.chat(user_prompt=user_prompt)
    print(grid_size_obs)

    # ADAPT step of Self Discovery
    user_prompt = "\n{}\nDescribe the relashionships between input_objects and output_objects".format(task_description)
    user_prompt += '\nPlease respond in this JSON format: {"object_observations": ["object changes for pair1", "object changes for pair2", ...]}'
    obj_obs = solver.chat(user_prompt=user_prompt)
    print(obj_obs)

    user_prompt = "\n{}\nDescribe the relashionships between input_pixel_coords and output_pixel_coords".format(task_description)
    user_prompt += '\nPlease respond in this JSON format: {"pixel_coords_observations": ["pixel coords changes for pair1", "pixel coords changes for pair2", ...]}'
    pixel_coords_changes = solver.chat(user_prompt=user_prompt)
    print(pixel_coords_changes)

    user_prompt = "\n{}\nDescribe the relashionships between 'input' and 'output'".format(task_description)
    user_prompt += '\nPlease respond in this JSON format: {"grid_observations": ["grid changes for pair1", "grid changes for pair2", ...]}'
    grid_obs = solver.chat(user_prompt=user_prompt)
    print(grid_obs)

    user_prompt = "\n{}\nSummarize the following observations about the input-output pairs and infer the transformation between the pairs:\n{}\n{}\n{}".format(task_description, grid_size_obs, obj_obs, grid_obs)
    user_prompt += '\nUse all examples of "train" input-output pairs and please use the actual number.'
    user_prompt += '\nPlease respond in this JSON format: {"relationship": "...", "reasoning_on_pair": ["reasoning for how the proposed transformation transorms input 1 to output 1", ...]}'
    reasoning = solver.chat(user_prompt=user_prompt)
    print(reasoning)

    # EXECUTE
    retries = 5
    solution = {"passed": False, "output": [[0]]}

    # provide environment feedback from code tester to the solver agent to ask for correction
    feedback_prompt = ""
    response_format = {
      "reflection": "reflect on the answer from the previous iteration if there is an error or failed to match prediction to groundtruth output",
      "overall_pattern": "describe the simplest input-output relationship for all input-output pairs",
      "helper_functions": "list any relevant helper_functions for this task",
      "program_instructions": "Plan how to write the python function and what helper functions and conditions to use. Use a training input-output pair as an example to test your thought process",
      "python_program": "Python function named 'transform_grid' that takes in a 2D grid and generates a 2D grid. Output as a string in a single line with \n and \t."
    }
    user_prompt =  "With this reasoning:\n{}\n\n{}, write a program that transforms the input 2d grid to the output 2d grid for every pair of input-ouput in 'train'.\n".format(reasoning, task_description)
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
        solution["prediction"] = result["test"]
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
    #task_files = ["e57337a4.json", "f3e62deb.json"]
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
        print("solving for task {}".format(task_id))
        output = [[0]]
        try:
            solution = solve_arc_question(task_data)
        except Exception as err:
            print("Error encountered during answering for task {}, error: {}".format(task_id, err))
            raise

        for output_id, question in enumerate(task_data["test"]):
            output = convert_char_grid_to_int(solution["prediction"][output_id]["predicted_output"])
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
            test_input = task_data["test"][output_id]["input"]
            show_image_from_json(task_data, test_input, output, groundtruth)


        # write out the submission.csv according to ARC requirement
        with open('submission.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["output_id", "output"])
            for i in range(len(output_ids)):
                writer.writerow([output_ids[i], flattener(outputs[i])])
