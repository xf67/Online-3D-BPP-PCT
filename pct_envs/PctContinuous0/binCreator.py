import numpy as np
import copy
import torch
import random

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)

class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])

class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        print("load data set successfully!")
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))
        self.box_trajs = torch.load(self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = np.array(self.box_trajs[self.index])
        self.boxes = self.boxes.tolist()
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([100, 100, 100])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1


class BinPackingGenerator:
    def __init__(self, container_size=(100, 100, 100)):
        self.container_size = container_size
        self.items = [container_size]
        
    def generate_items(self, min_items=10, max_items=50):
        """Generate a list of items using the splitting algorithm"""
        N = random.randint(min_items, max_items)
        
        while len(self.items) < N:
            # Pop an item randomly by volume
            volumes = [item[0] * item[1] * item[2] for item in self.items]
            probabilities = np.array(volumes) / sum(volumes)
            chosen_idx = np.random.choice(len(self.items), p=probabilities)
            item = self.items.pop(chosen_idx)
            
            # Choose an axis randomly by edge length
            edge_lengths = np.array(item)
            axis_probs = edge_lengths / sum(edge_lengths)
            split_axis = np.random.choice(3, p=axis_probs)
            
            # Choose split position (between 0.3 and 0.7 of the edge)
            edge_length = item[split_axis]
            min_ratio, max_ratio = 0.3, 0.7
            split_pos = random.uniform(
                edge_length * min_ratio, 
                edge_length * max_ratio
            )
            
            # Create two new items by splitting
            new_items = []
            for i in range(2):
                new_item = list(item)
                if i == 0:
                    new_item[split_axis] = split_pos
                else:
                    new_item[split_axis] = edge_length - split_pos
                
                # Random rotation (0-5 possible rotations in 3D)
                if random.random() < 0.5:  # 50% chance to rotate
                    rotation_type = random.randint(0, 5)
                    if rotation_type == 1:
                        new_item[0], new_item[1] = new_item[1], new_item[0]
                    elif rotation_type == 2:
                        new_item[0], new_item[2] = new_item[2], new_item[0]
                    elif rotation_type == 3:
                        new_item[1], new_item[2] = new_item[2], new_item[1]
                    elif rotation_type == 4:
                        new_item = [new_item[2], new_item[0], new_item[1]]
                    elif rotation_type == 5:
                        new_item = [new_item[1], new_item[2], new_item[0]]
                
                # Round dimensions to 2 decimal places
                new_item = tuple(round(x, 2) for x in new_item)
                new_items.append(new_item)
            
            # Add new items to the list
            self.items.extend(new_items)
        
        return self.items

    def reset(self):
        """Reset the generator to initial state"""
        self.items = [self.container_size]

class BoxCreatorFromGenerator(BoxCreator):
    def __init__(self, min_items=10, max_items=50, container_size=(100, 100, 100)):
        super().__init__()
        self.generator = BinPackingGenerator(container_size)
        self.min_items = min_items
        self.max_items = max_items
        
    def reset(self):
        """Reset both the box list and generator"""
        super().reset()
        self.generator.reset()
        # Generate new set of items
        self.box_set = self.generator.generate_items(self.min_items, self.max_items)
        
    def generate_box_size(self, **kwargs):
        """Add a box from the generated set to the box list"""
        if len(self.box_set) > 0:
            self.box_list.append(self.box_set.pop())
        else:
            # If we run out of boxes, generate new ones
            self.reset()
            self.box_list.append(self.box_set.pop())