import os
import numpy as np
import torch
import tools
import random
from tqdm import tqdm
from pct_envs.PctContinuous0.bin3D import PackingContinuousWithPreview
from model import DRL_GAT
import time
def evaluate(PCT_policy, eval_envs, timeStr, args, device, eval_freq = 100, factor = 1):
    PCT_policy.eval()
    obs = eval_envs.reset()
    obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
    all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                             args.internal_node_holder, args.leaf_node_holder)
    batchX = torch.arange(args.num_processes)
    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []

    while step_counter < eval_freq:
        with torch.no_grad():
            selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor = factor)
        selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
        items = eval_envs.packed
        obs, reward, done, infos = eval_envs.step(selected_leaf_node.cpu().numpy()[0][0:6])

        if done:
            print('Episode {} ends.'.format(step_counter))
            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])

            print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            all_episodes.append(items)
            step_counter += 1
            obs = eval_envs.reset()

        obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                                 args.internal_node_holder, args.leaf_node_holder)
        all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)

    result = "Evaluation using {} episodes\n" \
             "Mean ratio {:.5f}, mean length{:.5f}\n".format(len(episode_ratio), np.mean(episode_ratio), np.mean(episode_length))
    print(result)
    # Save the test trajectories.
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs.npy'), all_episodes)
    # Write the test results into local file.
    file = open(os.path.join('./logs/evaluation', timeStr, 'result.txt'), 'w')
    file.write(result)
    file.close()

class MCTSNode:
    def __init__(self, env_state:PackingContinuousWithPreview, available_boxes,droped_boxes=[], parent=None):
        self.env_state = env_state
        self.available_boxes = available_boxes # idx表示的盒子
        self.droped_boxes = droped_boxes # 已经装入的盒子
        self.parent = parent
        self.children = {}  # box_idx -> MCTSNode，字典
        self.visits = 0
        self.value = 0
        self.updated = False # env是drop过的还是直接从父节点拷贝的

class MCTS:
    def __init__(self, env, preview_boxes, PCT_policy, num_simulations, device, args, factor):
        # 传入的preview_boxes是盒子的idx
        self.root = MCTSNode(env, preview_boxes)
        self.num_simulations = num_simulations
        self.PCT_policy = PCT_policy
        self.device = device
        self.args = args
        self.factor = factor
        self.batchX = torch.arange(args.num_processes)

    def update_env(self, node:MCTSNode):
        if node.updated:
            return
        node.updated = True
        dd = node.droped_boxes[-1]
        node.env_state.rearrange([dd] + [i for i in range(len(node.available_boxes)) if i != dd]) # 把当前box和第0个box交换
        obs = torch.FloatTensor(node.env_state.cur_observation()).to(self.device).unsqueeze(dim=0)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, self.args.num_processes,
                                                            self.args.internal_node_holder, self.args.leaf_node_holder)
        all_nodes, leaf_nodes = all_nodes.to(self.device), leaf_nodes.to(self.device)
        selectedlogProb, selectedIdx, policy_dist_entropy, value = self.PCT_policy(all_nodes, True, normFactor = self.factor)
        selected_leaf_node = leaf_nodes[self.batchX, selectedIdx.squeeze()]
        obs, reward, done, infos = node.env_state.step(selected_leaf_node.cpu().numpy()[0][0:6])
        node.value += reward

    def search(self):
        for _ in tqdm(range(self.num_simulations)):
            node = self.select(self.root)
            if not node.children and node.available_boxes:
                node = self.expand(node)
            reward = self.simulate(node, self.factor)
            self.backpropagate(node, reward)
        return self.get_best_sequence(self.root)

    def select(self, node):
        # UCB1选择
        while node.children:
            if not all(child.visits > 0 for child in node.children.values()):
                # 如果有未访问的子节点，返回第一个未访问的节点
                return next(child for child in node.children.values() if child.visits == 0)
            # value/visits（平均价值）+ C*sqrt(ln(父节点访问次数)/子节点访问次数)
            node = max(node.children.values(),
                    key=lambda child: child.value/child.visits + 
                                    np.sqrt(2*np.log(node.visits)/child.visits))
        return node

    def expand(self,node):
        for i,box in enumerate(node.available_boxes):
            if box not in node.children:
                new_env = node.env_state.clone()
                new_boxes = node.available_boxes[:i] + node.available_boxes[i+1:]
                new_droped_boxes = node.droped_boxes + [node.available_boxes[i]]
                node.children[box] = MCTSNode(new_env, new_boxes, new_droped_boxes, parent=node)
        return random.choice(list(node.children.values()))

    def simulate(self, node, factor):
        self.update_env(node)
        env = node.env_state.clone()
        total_reward = 0
        for _ in range(len(node.available_boxes)): # 模拟到装这prev_size个箱结束
            obs = torch.FloatTensor(env.cur_observation()).to(self.device).unsqueeze(dim=0)
            all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, self.args.num_processes,
                                                                self.args.internal_node_holder, self.args.leaf_node_holder)
            all_nodes, leaf_nodes = all_nodes.to(self.device), leaf_nodes.to(self.device)
            selectedlogProb, selectedIdx, policy_dist_entropy, value = self.PCT_policy(all_nodes, True, normFactor = factor)
            selected_leaf_node = leaf_nodes[self.batchX, selectedIdx.squeeze()]
            obs, reward, done, infos = env.step(selected_leaf_node.cpu().numpy()[0][0:6])
            total_reward += reward
            if done:
                break
        return total_reward

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_best_sequence(self, root):
        sequence = []
        node = root
        while node.children:
            best_child_idx = max(node.children.items(),
                            key=lambda x: x[1].visits)[0]
            sequence.append(best_child_idx)
            node = node.children[best_child_idx]
        return sequence

def evaluate_mcts(PCT_policy:DRL_GAT, eval_envs:PackingContinuousWithPreview, timeStr, args, device, eval_freq = 10, factor = 1, prev_size=3, num_simulations=10):
    PCT_policy.eval()
    obs = eval_envs.reset()
    batchX = torch.arange(args.num_processes)
    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []

    while step_counter < eval_freq:
        preview_boxes = eval_envs.get_preview_boxes(prev_size) #返回的是盒子的xyz，返回preview_size个（超出则用[100,100,100]填充）
        arrange_ori = [i for i in range(prev_size)]
        a=time.time()
        print("running MCTS")
        mcts = MCTS(eval_envs.clone(), arrange_ori, PCT_policy, num_simulations, device, args, factor)
        best_sequence = mcts.search()
        print("MCTS finished: ",best_sequence,"time: ", time.time()-a)
        # best_sequence = arrange_ori
        eval_envs.rearrange(best_sequence)
        for _ in range(len(best_sequence)):
            with torch.no_grad():
                obs = torch.FloatTensor(eval_envs.cur_observation()).to(device).unsqueeze(dim=0)
                all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                                     args.internal_node_holder, args.leaf_node_holder)
                all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
                selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor = factor)
                selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
                
                obs, reward, done, infos = eval_envs.step(selected_leaf_node.cpu().numpy()[0][0:6])
                
                if done:
                    print('Episode {} ends.'.format(step_counter))
                    if 'ratio' in infos.keys():
                        episode_ratio.append(infos['ratio'])
                    if 'counter' in infos.keys():
                        episode_length.append(infos['counter'])
                    print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
                    print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
                    items = eval_envs.packed
                    all_episodes.append(items)
                    step_counter += 1
                    obs = eval_envs.reset()
                    break

    result = "Evaluation using {} episodes\n" \
             "Mean ratio {:.5f}, mean length{:.5f}\n".format(len(episode_ratio), np.mean(episode_ratio), np.mean(episode_length))
    print(result)
    # Save the test trajectories.
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs.npy'), all_episodes)
    # Write the test results into local file.
    file = open(os.path.join('./logs/evaluation', timeStr, 'result.txt'), 'w')
    file.write(result)
    file.close()