from collections import defaultdict
import json
import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image
import random
from pprint import pprint
import pdb
import math
from PIL import ImageDraw, ImageFont

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import ast
import tqdm
import copy
import os
import numpy as np
import yaml
import sys
import shutil
import signal

from ai2thor.util.metrics import (
    get_shortest_path_to_object_type
)


class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutException("Operation timed out")


def with_timeout(timeout_seconds):
    """
    Context manager to add timeout to operations.
    Usage:
        with with_timeout(300):  # 5 minutes
            # your code here
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


def safe_stop_controller(controller, timeout_seconds=30):
    """
    Safely stop a controller with timeout protection.
    """
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        controller.stop()  # Fixed: was recursively calling itself
        signal.alarm(0)
    except TimeoutException:
        print(f"Warning: Controller stop timed out after {timeout_seconds} seconds")
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Error stopping controller: {e}")


def get_camera_state(controller):
    """Extract current camera position, rotation, and horizon from controller metadata."""
    metadata = controller.last_event.metadata
    agent = metadata['agent']
    
    return {
        'position': agent['position'],
        'rotation': agent['rotation']['y'],  # Yaw rotation
        'horizon': agent['cameraHorizon']  # Camera tilt
    }


def add_red_dot_with_text(image, position, text):
    # Load the image
    draw = ImageDraw.Draw(image)

    # Coordinates and radius of the dot
    x, y = position
    radius = 15  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    try:
        font = ImageFont.truetype("LiberationSans-Bold.ttf", 15)  # Adjust font and size as needed
    except IOError:
        font = ImageFont.load_default()

    # Calculate text width and height to center it
    text_width = draw.textlength(text, font=font)
    text_x = x - text_width / 2
    text_y = y

    # Draw the text
    draw.text((text_x, text_y), text, fill='white', font=font)

    return image


def get_current_state(controller):
    # Make sure to define objid2assetid in the scope where this function is called
    # This is a dependency from the original script that needs to be available
    global objid2assetid, assetid2desc
    
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=5).metadata["actionReturn"]
    
    # Filter objects
    filtered_nav_visible = []
    if nav_visible_objects:
        for obj_id in nav_visible_objects:
            if obj_id in objid2assetid and objid2assetid[obj_id] != "":
                filtered_nav_visible.append(obj_id)
    nav_visible_objects = filtered_nav_visible

    bboxes = controller.last_event.instance_detections2D
    vis_obj_to_size = {}
    if bboxes:
        for obj_id in bboxes:
            vis_obj_to_size[obj_id] = (bboxes[obj_id][2] - bboxes[obj_id][0]) * (bboxes[obj_id][3] - bboxes[obj_id][1])

    objid2info = {}
    objdesc2cnt = defaultdict(int)
    for obj_entry in controller.last_event.metadata['objects']:
        obj_name = obj_entry['name']
        obj_type = obj_entry['objectType']
        asset_id = obj_entry['assetId']
        obj_id = obj_entry['objectId']

        distance = obj_entry['distance']
        pos = np.array([obj_entry['position']['x'], obj_entry['position']['y'], obj_entry['position']['z']])
        rotation = obj_entry['rotation']
        desc = assetid2desc.get(asset_id, obj_type)
        moveable = obj_entry['moveable'] or obj_entry['pickupable']

        asset_size_xy = vis_obj_to_size.get(obj_entry['objectId'], 0)
        asset_pos_box = bboxes.get(obj_entry['objectId'], None)
        if asset_pos_box is not None:
            asset_pos_xy = [(asset_pos_box[0] + asset_pos_box[2]) / 2, (asset_pos_box[1] + asset_pos_box[3]) / 2]
        else:
            asset_pos_xy = None

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                if parent == "Floor":
                    parent = "Floor"
                elif parent in objid2assetid:
                    parent = objid2assetid[parent]
                else:
                    parent = None # Handle case where parent ID might not be in the map
        
        is_receptacle = obj_entry['receptacle']
        objid2info[obj_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size_xy, is_receptacle, asset_pos_xy)
        objdesc2cnt[obj_type] += 1

    moveable_visible_objs = []
    for objid in nav_visible_objects:
        if objid in objid2info and objid2info[objid][6] and objid2info[objid][8] > 1600:
            moveable_visible_objs.append(objid)

    return nav_visible_objects, objid2info, objdesc2cnt, moveable_visible_objs


def generate_variable_motion_commands(action_type, num_steps):
    """
    Generate variable motion commands based on action type and number of steps.
    Motion can vary between steps, including staying still.
    """
    commands = []
    
    if action_type == "Pan Left":
        for _ in range(num_steps):
            if random.random() < 0.8:  # 80% chance to rotate, 20% to stay still
                degrees = random.choice([15, 20, 25, 30])
                commands.append({"action": "RotateLeft", "degrees": degrees})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Pan Right":
        for _ in range(num_steps):
            if random.random() < 0.8:
                degrees = random.choice([15, 20, 25, 30])
                commands.append({"action": "RotateRight", "degrees": degrees})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Zoom In":  # Move Forward
        for _ in range(num_steps):
            if random.random() < 0.8:
                magnitude = random.choice([0.20, 0.25, 0.30, 0.35])
                commands.append({"action": "MoveAhead", "moveMagnitude": magnitude})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Zoom Out":  # Move Backward
        for _ in range(num_steps):
            if random.random() < 0.8:
                magnitude = random.choice([0.20, 0.25, 0.30, 0.35])
                commands.append({"action": "MoveBack", "moveMagnitude": magnitude})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Tilt Down":
        for _ in range(num_steps):
            if random.random() < 0.8:
                degrees = random.choice([10, 15, 20])
                commands.append({"action": "LookDown", "degrees": degrees})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Tilt Up":
        for _ in range(num_steps):
            if random.random() < 0.8:
                degrees = random.choice([10, 15, 20])
                commands.append({"action": "LookUp", "degrees": degrees})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Move Forward":
        for _ in range(num_steps):
            if random.random() < 0.8:
                magnitude = random.choice([0.20, 0.25, 0.30, 0.35])
                commands.append({"action": "MoveAhead", "moveMagnitude": magnitude})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Move Leftward":
        for _ in range(num_steps):
            if random.random() < 0.8:
                magnitude = random.choice([0.20, 0.25, 0.30, 0.35])
                commands.append({"action": "MoveLeft", "moveMagnitude": magnitude})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Move Rightward":
        for _ in range(num_steps):
            if random.random() < 0.8:
                magnitude = random.choice([0.20, 0.25, 0.30, 0.35])
                commands.append({"action": "MoveRight", "moveMagnitude": magnitude})
            else:
                commands.append({"action": "Pass"})
                
    elif action_type == "Still":
        for _ in range(num_steps):
            commands.append({"action": "Pass"})
            
    elif action_type == "Move Backward":
        for _ in range(num_steps):
            if random.random() < 0.8:
                magnitude = random.choice([0.20, 0.25, 0.30, 0.35])
                commands.append({"action": "MoveBack", "moveMagnitude": magnitude})
            else:
                commands.append({"action": "Pass"})
    
    return commands


# Global dictionaries needed by get_current_state
assetid2desc = {}
objid2assetid = {}

if __name__ == "__main__":

    split = "train"
    # Define paths - Update these to your local paths
    asset_info_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json"
    qa_im_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/cameramove_variable_{split}/'
    qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_cameramove_variable_qas_{split}.json'
    
    # Script control flags
    vis = True
    stats = False
    generate = True # Set to True to run generation
    load_progress = True # Set to True to load existing JSON

    # FOV options for randomization
    fov_options = [60, 70, 80, 90, 100, 110, 120]

    # Define all possible motion actions with simplified questions
    motion_actions = [
        {
            "name": "Pan Left",
            "type": "Pan Left",
            "question": "Does the camera pan to the left?",
            "negative_actions": ["Pan Right"]
        },
        {
            "name": "Pan Right",
            "type": "Pan Right",
            "question": "Does the camera pan to the right?",
            "negative_actions": ["Pan Left"]
        },
        {
            "name": "Zoom In",
            "type": "Zoom In",
            "question": "Does the camera zoom in?",
            "negative_actions": ["Zoom Out", "Move Backward"]
        },
        {
            "name": "Zoom Out",
            "type": "Zoom Out",
            "question": "Does the camera zoom out?",
            "negative_actions": ["Zoom In", "Move Forward"]
        },
        {
            "name": "Tilt Down",
            "type": "Tilt Down",
            "question": "Does the camera tilt downward?",
            "negative_actions": ["Tilt Up", "Move Rightward", "Move Leftward"]
        },
        {
            "name": "Tilt Up",
            "type": "Tilt Up",
            "question": "Does the camera tilt upward?",
            "negative_actions": ["Tilt Down", "Move Rightward", "Move Leftward"]
        },
        {
            "name": "Move Forward",
            "type": "Move Forward",
            "question": "Does the camera move forward?",
            "negative_actions": ["Zoom Out", "Move Backward"]
        },
        {
            "name": "Move Leftward",
            "type": "Move Leftward",
            "question": "Does the camera move leftward?",
            "negative_actions": ["Move Forward", "Move Rightward", "Move Backward"]
        },
        {
            "name": "Move Rightward",
            "type": "Move Rightward",
            "question": "Does the camera move rightward?",
            "negative_actions": ["Move Forward", "Move Leftward", "Move Backward"]
        },
        {
            "name": "Still",
            "type": "Still",
            "question": "Is the camera completely still?",
            "negative_actions": ["Pan Left", "Pan Right", "Zoom In", "Zoom Out", "Tilt Down", "Tilt Up", 
                                "Move Forward", "Move Leftward", "Move Rightward", "Move Backward"]
        },
        {
            "name": "Move Backward",
            "type": "Move Backward",
            "question": "Does the camera move backward?",
            "negative_actions": ["Zoom In", "Move Forward"]
        }
    ]


    if generate:
        if not os.path.exists(qa_im_path):
            os.makedirs(qa_im_path)

        # Load asset descriptions
        try:
            asset_id_desc_json = json.load(open(asset_info_path, "r"))
            for asset in asset_id_desc_json:
                entries = asset_id_desc_json[asset]
                captions = []
                for im, obj, desc in entries:
                    desc = desc.strip().lower().replace(".", "")
                    captions.append(desc)
                if captions:
                    assetid2desc[asset] = random.choice(captions)
        except FileNotFoundError:
            print(f"Warning: Asset info file not found at {asset_info_path}. Descriptions will be limited.")
            assetid2desc = {}
        except json.JSONDecodeError:
            print(f"Warning: Could not decode asset info file at {asset_info_path}.")
            assetid2desc = {}


        dataset = prior.load_dataset("procthor-10k")
        all_im_qas = []
        
        if load_progress:
            try:
                all_im_qas = json.load(open(qa_json_path, "r"))
                print(f"Loaded {len(all_im_qas)} existing QA pairs.")
            except FileNotFoundError:
                print("No existing QA file found, starting from scratch.")
            except json.JSONDecodeError:
                 print(f"Error reading {qa_json_path}, starting from scratch.")

        for house_ind, house in enumerate(tqdm.tqdm(dataset[split])):

            if load_progress:
                entry = all_im_qas[-1]  # Updated to account for new structure
                last_house_ind = entry[0]
                if house_ind <= last_house_ind:
                    continue
            
            house_json = house

            try:
                # Set timeout alarm for controller initialization (5 minutes)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minutes
                
                controller = Controller(scene=house, width=200, height=200, quality="Low", platform=CloudRendering)
                
                # Disable alarm after successful initialization
                signal.alarm(0)
            except TimeoutException:
                print(f"Timeout: Controller initialization for environment {house_ind} took too long (>5 min), skipping.")
                signal.alarm(0)  # Make sure to disable alarm
                continue
            except Exception as e:
                signal.alarm(0)  # Make sure to disable alarm
                print(f"Cannot render environment {house_ind}, continuing. Error: {e}")
                continue
            
            try:
                # Set timeout for getting reachable positions (5 minutes)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
                
                event = controller.step(action="GetReachablePositions")
                reachable_positions = event.metadata["actionReturn"]
                
                # Disable alarm
                signal.alarm(0)
            except TimeoutException:
                print(f"Timeout: GetReachablePositions for {house_ind} took too long (>5 min), skipping.")
                signal.alarm(0)
                safe_stop_controller(controller)
                continue
            except Exception as e:
                signal.alarm(0)
                print(f"Cannot get reachable positions for {house_ind}, continuing. Error: {e}")
                safe_stop_controller(controller)
                continue

            if not reachable_positions or len(reachable_positions) < 10:
                print(f"Not enough reachable positions in {house_ind}, continuing")
                safe_stop_controller(controller)
                continue
            
            random_positions = []
            # Sample more positions to find a good one
            num_samples = min(len(reachable_positions), 10)
            for cam_pos in random.sample(reachable_positions, num_samples):
                
                cam_rot = random.choice(range(360))
                try:
                    controller.step(action="Teleport", position=cam_pos, rotation=cam_rot, horizon=0) # Reset horizon
                except Exception as e:
                    print(f"Cannot teleport in {house_ind}, continuing. Error: {e}")
                    continue

                nav_visible_objects = controller.step(
                    "GetVisibleObjects",
                    maxDistance=5,
                ).metadata["actionReturn"]
                
                num_visible = len(nav_visible_objects) if nav_visible_objects else 0
                random_positions.append((cam_pos, cam_rot, num_visible))
            
            if len(random_positions) == 0:
                print(f"No objects visible in any sampled position for {house_ind}, continuing")
                safe_stop_controller(controller)
                continue

            # Sort by number of visible objects and take the best ones
            random_positions = sorted(random_positions, key=lambda x: x[2], reverse=True)
            
            safe_stop_controller(controller) # Stop the low-res controller

            house_img_dir = qa_im_path + f"{house_ind}"
            if not os.path.exists(house_img_dir):
                os.makedirs(house_img_dir)

            sample_count = 0
            # Try to generate from the top 3 best starting positions
            for cam_pos, cam_rot, _ in random_positions[:3]:
                if sample_count >= 1: # Only generate one sample per house for now
                    break

                # Loop over multiple FOVs for the same camera position
                # You can change this to use all FOVs or a subset
                # Option 1: Use all FOVs
                fovs_to_test = fov_options
                # Option 2: Use a random subset (e.g., 3 FOVs)
                # fovs_to_test = random.sample(fov_options, min(3, len(fov_options)))
                
                for selected_fov in fovs_to_test:
                    qa_pair_choices = []
                    
                    try:
                        # Set timeout alarm for controller initialization (5 minutes)
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(300)  # 5 minutes
                        
                        controller = Controller(
                            scene=house_json, 
                            width=512, 
                            height=512, 
                            quality="Ultra", 
                            platform=CloudRendering, 
                            renderInstanceSegmentation=True,
                            fieldOfView=selected_fov  # Loop over different FOVs
                        )
                        
                        # Disable alarm after successful initialization
                        signal.alarm(0)
                    except TimeoutException:
                        print(f"Timeout: High-res controller initialization for environment {house_ind} with FOV {selected_fov} took too long (>5 min), skipping.")
                        signal.alarm(0)  # Make sure to disable alarm
                        continue
                    except Exception as e:
                        signal.alarm(0)  # Make sure to disable alarm
                        print(f"Cannot render high-res environment {house_ind} with FOV {selected_fov}, continuing. Error: {e}")
                        continue
                    
                    try:
                        # Set timeout for teleport operation (2 minutes)
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(120)  # 2 minutes
                        
                        # Teleport to the starting position, reset horizon
                        controller.step(action="Teleport", position=cam_pos, rotation=cam_rot, horizon=0)
                        
                        # Disable alarm
                        signal.alarm(0)
                    except TimeoutException:
                        print(f"Timeout: Teleport for {house_ind} with FOV {selected_fov} took too long (>2 min), skipping.")
                        signal.alarm(0)
                        safe_stop_controller(controller)
                        continue
                    except Exception as e:
                        signal.alarm(0)
                        print(f"Cannot teleport high-res controller in {house_ind} with FOV {selected_fov}, continuing. Error: {e}")
                        safe_stop_controller(controller)
                        continue
                    
                    # Update objid2assetid map for this scene
                    objid2assetid = {}
                    for obj in controller.last_event.metadata['objects']:
                        objid2assetid[obj['objectId']] = obj['assetId']

                    # --- VARIABLE MOTION ACTION AND QA GENERATION LOGIC ---

                    # 1. Randomly pick a motion action
                    action_data = random.choice(motion_actions)
                    
                    # 2. Randomly determine number of frames (2-4 frames total, so 1-3 motion steps)
                    num_frames = random.randint(2, 4)
                    num_steps = num_frames - 1  # Number of motion steps
                    
                    # 3. Generate variable motion commands
                    action_commands = generate_variable_motion_commands(action_data["type"], num_steps)
                    
                    image_seq = []
                    camera_states = []  # Track camera state for each frame
                    all_actions_succeeded = True

                    # 4. Save initial frame (Frame 0) and capture camera state
                    img_view = Image.fromarray(controller.last_event.frame)
                    frame_path = os.path.join(house_img_dir, f"{sample_count}_fov{selected_fov}_0.jpg")
                    img_view.save(frame_path)
                    image_seq.append(frame_path)
                    
                    # Capture initial camera state
                    initial_state = get_camera_state(controller)
                    camera_states.append(initial_state)

                    # 5. Execute motion steps to get additional frames
                    try:
                        # Set timeout for action execution (5 minutes for all actions)
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(300)  # 5 minutes for entire action sequence
                        
                        for i, command in enumerate(action_commands):
                            try:
                                event = controller.step(**command)
                                # For "Pass" actions, success is always True
                                if not event.metadata["lastActionSuccess"] and command["action"] != "Pass":
                                    print(f"Action {command['action']} failed in {house_ind} with FOV {selected_fov}, stopping motion sequence.")
                                    all_actions_succeeded = False
                                    break
                                
                                # Save frame
                                img_view = Image.fromarray(controller.last_event.frame)
                                frame_path = os.path.join(house_img_dir, f"{sample_count}_fov{selected_fov}_{i+1}.jpg")
                                img_view.save(frame_path)
                                image_seq.append(frame_path)
                                
                                # Capture camera state after action
                                current_state = get_camera_state(controller)
                                camera_states.append(current_state)

                            except Exception as e:
                                print(f"Error during action {command.get('action', 'Unknown')} in {house_ind} with FOV {selected_fov}: {e}")
                                all_actions_succeeded = False
                                break
                        
                        # Disable alarm after actions complete
                        signal.alarm(0)
                        
                    except TimeoutException:
                        print(f"Timeout: Action execution for {house_ind} with FOV {selected_fov} took too long (>5 min), skipping.")
                        signal.alarm(0)  # Make sure to disable alarm
                        all_actions_succeeded = False
                    
                    # If motion failed or didn't produce expected number of frames, skip this sample
                    if not all_actions_succeeded or len(image_seq) != num_frames:
                        print(f"Skipping sample for {house_ind} with FOV {selected_fov} due to failed/incomplete action.")
                        safe_stop_controller(controller)
                        # Clean up partial images
                        for img_path in image_seq:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                        continue

                    # 6. Generate QA pairs (All Yes/No format)
                    
                    # QA Type 1: "Did [Correct Action] happen?" -> Yes
                    question_positive = action_data["question"]
                    answer_choices_positive = ["Yes", "No"]
                    correct_index_positive = 0  # "Yes"
                    qa_pair_choices.append((question_positive, image_seq, answer_choices_positive, correct_index_positive))

                    # QA Type 2: "Did [Wrong Action] happen?" -> No
                    # Pick a random negative action
                    negative_action_name = random.choice(action_data["negative_actions"])
                    
                    # Find the negative action's question
                    # For "Still" negatives, use lower degrees
                    if action_data["name"] == "Still":
                        # For still camera, negatives use lower degrees
                        # But the question is from another action's perspective
                        negative_action = next((a for a in motion_actions if a["name"] == negative_action_name), None)
                    else:
                        negative_action = next((a for a in motion_actions if a["name"] == negative_action_name), None)
                    
                    if negative_action:
                        question_negative = negative_action["question"]
                        answer_choices_negative = ["Yes", "No"]
                        correct_index_negative = 1  # "No"
                        qa_pair_choices.append((question_negative, image_seq, answer_choices_negative, correct_index_negative))

                    # --- END VARIABLE MOTION LOGIC ---

                    if len(qa_pair_choices) > 0:
                        # Save the (house_id, start_pos, start_rot, fov, camera_states, list_of_qa_tuples)
                        # Each qa_tuple is (question, image_seq_paths, [choice1, choice2], correct_index)
                        # camera_states is a list of dicts with position, rotation, horizon for each frame
                        all_im_qas.append((house_ind, cam_pos, cam_rot, selected_fov, camera_states, qa_pair_choices))
                    
                    sample_count += 1
                    safe_stop_controller(controller)
            
            # Save progress periodically
            if house_ind % 100 == 0 and len(all_im_qas) > 0:
                print(f"\nSaving progress: {len(all_im_qas)} total QA sets generated.")
                try:
                    json.dump(all_im_qas, open(qa_json_path, "w"), indent=2)
                except Exception as e:
                    print(f"Error saving progress to JSON: {e}")


    # Final save
    if generate and len(all_im_qas) > 0:
        print(f"\nFinished generation. Total QA sets: {len(all_im_qas)}")
        try:
            json.dump(all_im_qas, open(qa_json_path, "w"), indent=2)
            print(f"Successfully saved to {qa_json_path}")
        except Exception as e:
            print(f"Error on final save to JSON: {e}")
    elif generate:
        print("Generation finished, but no QA pairs were created.")