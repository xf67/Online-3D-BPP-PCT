import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from model import *
from tools import *
import gym
from pct_envs.PctContinuous0.bin3D import PackingContinuousWithPreview
from pct_envs.PctContinuous0.binCreator import CSVBoxCreator
import copy
import tools


# argss从get_args里获取，为第一个container的eval设定

class MultiContainerPacker:
    def __init__(self, container_sizes, model_paths, args):
        """
        初始化多容器打包器
        
        Args:
            container_sizes: 包含8种容器尺寸的列表，每个元素为 (length, width, height)
            model_paths: 对应每种容器的模型路径列表
        """
        self.container_sizes = container_sizes
        self.model_paths = model_paths
        self.models = []
        self.envs = []
        self.argss = []
        
        if args.no_cuda:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda', args.device)
            torch.cuda.set_device(args.device)
        
        # 为每种容器尺寸创建对应的环境和加载模型
        for i in range(len(container_sizes)):
            argss=copy.deepcopy(args)
            argss.container_size=container_sizes[i]
            argss.model_path=model_paths[i]
            argss.normFactor=1.0 / np.max(argss.container_size)
            self.argss.append(argss)

            env = gym.make(argss.id,
                    setting = argss.setting,
                    container_size=argss.container_size,
                    item_set=argss.item_size_set,
                    data_name=argss.dataset_path,
                    load_test_data = argss.load_dataset,
                    internal_node_holder=argss.internal_node_holder,
                    leaf_node_holder=argss.leaf_node_holder,
                    LNES = argss.lnes,
                    shuffle=argss.shuffle,
                    sample_from_distribution=argss.sample_from_distribution,
                    sample_left_bound=argss.sample_left_bound,
                    sample_right_bound=argss.sample_right_bound
                   )
            self.envs.append(env)
            
            PCT_policy = DRL_GAT(argss)  
            PCT_policy = PCT_policy.to(device)
            PCT_policy = load_policy(argss.model_path, PCT_policy)
            self.models.append(PCT_policy)

            print(f"Model {i} loaded from {argss.model_path}")
            
    def evaluate_container(self, eval_envs, PCT_policy, args):
        """评估单个容器的打包效果"""
        factor = args.normFactor
        if args.no_cuda:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda', args.device)
            torch.cuda.set_device(args.device)
        PCT_policy.eval()
        obs = eval_envs.reset()
        obs = torch.FloatTensor(obs).to(device).unsqueeze(dim=0)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                                args.internal_node_holder, args.leaf_node_holder)
        batchX = torch.arange(args.num_processes)
        
        # 按顺序打包一个order内的所有箱子（无MCTS）
        while True:
            # print("Packing box", eval_envs.box_creator.preview(1)[0])
            with torch.no_grad():
                obs = torch.FloatTensor(eval_envs.cur_observation()).to(device).unsqueeze(dim=0)
                all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, args.num_processes,
                                                                args.internal_node_holder, args.leaf_node_holder)
                selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor = factor)
                selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
                obs, reward, done, infos = eval_envs.step(selected_leaf_node.cpu().numpy()[0][0:6])
            # print("reward:",reward, "done:",done, "infos:",infos)
            if done:
                break

        return {
            'success': infos['finish'],
            'utilization': infos['ratio'],
            'solution': eval_envs.packed,
            'order_info': infos['order_info']['sta_code'],
        }
        
    def pack_order(self,):

        best_result = {
            'container_index': -1,
            'container_size': None,
            'utilization': 0,
            'packing_solution': None,
            'success': False,
            'order_info': None
        }
        
        # 遍历所有容器尺寸
        for i, (env, model, argss) in enumerate(zip(self.envs, self.models, self.argss)):
            # print("Running on container", env.bin_size)
            result = self.evaluate_container(env, model, argss) 
            if i == 0:
                best_result['order_info'] = result['order_info']
            # 更新最佳结果
            if result['success'] and result['utilization'] > best_result['utilization']:
                best_result = {
                    'container_index': i,
                    'container_size': self.container_sizes[i],
                    'utilization': result['utilization'],
                    'packing_solution': result['solution'],
                    'success': result['success'],
                    'order_info': result['order_info']
                }
                
        return best_result 
    
if __name__ == '__main__':
    registration_envs()
    args = get_args()
    container_sizes = [
        (35,23,13),
        (37,26,13),
        (38,26,13),
        (40,28,16),
        (42,30,18),
        (42,30,40),
        (52,40,17),
        (54,45,36),
    ]
    model_paths = [
        "logs/experiment/t352313-2024.12.27-09-39-41/PCT-t352313-2024.12.27-09-39-41_2024.12.27-09-39-45.pt",
        "logs/experiment/t372613-2024.12.27-02-24-30/PCT-t372613-2024.12.27-02-24-30_2024.12.27-02-24-34.pt",
        "logs/experiment/t382613-2024.12.27-03-23-39/PCT-t382613-2024.12.27-03-23-39_2024.12.27-03-23-43.pt",
        "logs/experiment/t402816-2024.12.27-04-24-55/PCT-t402816-2024.12.27-04-24-55_2024.12.27-04-24-59.pt",
        "logs/experiment/t423018-2024.12.27-05-26-12/PCT-t423018-2024.12.27-05-26-12_2024.12.27-05-26-16.pt",
        "logs/experiment/t423040-2024.12.27-06-26-14/PCT-t423040-2024.12.27-06-26-14_2024.12.27-06-26-18.pt",
        "logs/experiment/t524017-2024.12.27-07-29-47/PCT-t524017-2024.12.27-07-29-47_2024.12.27-07-29-51.pt",
        "logs/experiment/t544536-2024.12.27-08-31-16/PCT-t544536-2024.12.27-08-31-16_2024.12.27-08-31-20.pt",
    ]
    packer = MultiContainerPacker(container_sizes, model_paths, args)
    for i in range(6848):
        best_result = packer.pack_order()
        print(best_result)

