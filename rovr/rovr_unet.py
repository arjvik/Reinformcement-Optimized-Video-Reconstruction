import torch
import lpips
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from torchvision.models.optical_flow import raft_small
from tqdm import tqdm
from einops import rearrange

from video_ds import VideoDataset
from local_net import LocalNetwork
from action_lstm import ActionLSTM
from policy_net_1_unet import PolicyNetwork1UNet
from policy_net_2_unet import PolicyNetwork2UNet
from resnet_extractor import ResnetFeatureExtractor
from local_net_unet import LocalNetworkUNet ##CHANGED FROM VIT VERSION

from itertools import chain
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


class ROVR(nn.Module):
    def __init__(self, actor1 = None, critic1 = None, actor2 = None, critic2 = None, video_encoder = None, history_encoder = None, local_net = None, vid_length = 25, time_steps = 25, n_updates_per_ppo = 5):
        super(ROVR, self).__init__()
        
        print("INIT")
        
        video_encoder = ResnetFeatureExtractor(pretrained = True)
        actor1 = PolicyNetwork1UNet()
        actor2 = PolicyNetwork2UNet()
        history_encoder = ActionLSTM(hidden_dim = 1024, num_layers = 1, batch_size = 1)
        # local_net = LocalNetwork()
        critic1 = PolicyNetwork1UNet(is_critic = True)
        critic2 = PolicyNetwork2UNet(is_critic = True)
        
        local_net = LocalNetworkUNet(freeze = True)
        checkpoint = torch.load("runs/local_net/unet_mse/2023-06-06_11-31-21/checkpoints/98000.pt")
        local_net.load_state_dict(checkpoint['model_state_dict'])
        
        self.actor1 = actor1
        self.critic1 = critic1
        self.actor2 = actor2
        self.critic2 = critic2
        self.local_net = local_net

        self.vid_length = vid_length
        self.time_steps = time_steps
        self.num_updates_per_ppo = n_updates_per_ppo
        self.clip = 0.2
        self.lpips_mse_gamma = 1
        
        self.lpips = lpips.LPIPS(net='vgg')
        self.local_mse = torch.nn.MSELoss()
        self.video_encoder = video_encoder
        self.history_encoder = history_encoder

        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.actor2_optimizer = torch.optim.Adam(chain(self.actor2.parameters(), self.history_encoder.parameters()), lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-4)
        self.local_net_optimizer = torch.optim.Adam(self.local_net.parameters(), lr=1e-4)


        self.tensorboard_path = Path('runs') / 'rovr' / 'unet' / time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        (self.tensorboard_path / 'checkpoints').mkdir(parents=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_path, flush_secs=10)

    #this should be the main function that is called to train
    def train(self, video, org_video, i, device):
             
        obs_1, obs_2, acs_1, acs_2, log_prob_1, log_prob_2, rtg, optical_flow_by_frame, exp_optical_flow_by_frame, org_optical_flow_by_frame, corrupted_optical_flow_by_frame = self.forward(video, org_video, i)
        # print(torch.cuda.memory_allocated(device)/(1024 ** 3))

        print("Starting PPO on 2!")
        self.ppo(2, (obs_2, acs_2, log_prob_2, rtg), i, device)
        print("Starting PPO on 1!")
        self.ppo(1, (obs_1, acs_1, log_prob_1, rtg), i, device)
        
        return optical_flow_by_frame, exp_optical_flow_by_frame, org_optical_flow_by_frame, corrupted_optical_flow_by_frame

        
    def forward(self, video, org_video, i):
        with torch.autograd.set_detect_anomaly(True):
            b, s, c, h, w = video.shape
            curr_loss = self.lpips(video.squeeze(0).float(), org_video.squeeze(0).float(), normalize=True)

            reconstructed_video = video.clone()
            exp_reconstructed_video = video.clone()
            obs_1 = []
            obs_2 = []
            acs_1 = []
            acs_2 = []
            log_prob_1 = []
            log_prob_2 = []
            rewards = []

            org_optical_flow, org_optical_flow_by_frame = self.calculate_optical_flow(org_video.squeeze(0).float())
            corrupted_optical_flow, corrupted_optical_flow_by_frame = self.calculate_optical_flow(video.squeeze(0).float())

            local_device = video.device

            lstm_token = torch.zeros(b, 3, 80, 80).to(local_device) #reasonable default 
            
            encoded_frames = self.video_encoder(video) #output extracted features
            
            self.displays = []
            total_reward = 0
                        
            for j in tqdm(range(self.time_steps)):

                pn_1_top_frame, pn_1_log_prob = self.actor1(encoded_frames, lstm_token)
                
                # print("PN 1 Log Prob", pn_1_log_prob)

                obs_1.append((encoded_frames.detach(), lstm_token.detach()))
                acs_1.append(pn_1_top_frame)
                log_prob_1.append(pn_1_log_prob)

                pn_1_top_frame = pn_1_top_frame.unsqueeze(0)

                target_frame_index = pn_1_top_frame.item()

                target_frame = video[:, target_frame_index, :, :, :]

                pn_2_top_frames, pn_2_log_prob = self.actor2(encoded_frames.float(), target_frame.float(), pn_1_top_frame)
                
                # print("       PN 2 Log Prob", pn_2_log_prob)

                obs_2.append((encoded_frames.float().detach(), target_frame.float().detach(), pn_1_top_frame.detach()))
                acs_2.append(pn_2_top_frames)
                log_prob_2.append(pn_2_log_prob)
                #construct context package

                context_1_index = pn_2_top_frames[0, 0]
                context_2_index = pn_2_top_frames[0, 1]

                context_frame_1 = video[:, context_1_index, :, :, :]
                context_frame_2 = video[:, context_2_index, :, :, :]

#                 resized_context_1 = F.interpolate(context_frame_1, size = (128, 128), mode = "bilinear", align_corners = False)
#                 resized_context_2 = F.interpolate(context_frame_2, size = (128, 128), mode = "bilinear", align_corners = False)

#                 total_context = torch.cat((resized_context_1, resized_context_2), dim = 0).unsqueeze(0)


                total_context = torch.stack([context_frame_1, context_frame_2], dim = 1)
    
                    # Begin experiment
                exp_context_1_index = (target_frame_index - 2) % s
                exp_context_2_index = (target_frame_index - 1) % s

                exp_context_frame_1 = video[:, exp_context_1_index, :, :, :]
                exp_context_frame_2 = video[:, exp_context_2_index, :, :, :]

                exp_total_context = torch.stack([exp_context_frame_1, exp_context_frame_2], dim = 1)

                with torch.no_grad():
                    exp_decorrupted_image, _ = self.train_local_network(target_frame.float(), exp_total_context.float(), org_video[:, target_frame_index, :, :, :].float(), i, j)
                # End experiment


                #ship to local_net
                #rewards are ~0-1 for perceptual loss
                
                
                decorrupted_image, lpips_loss = self.train_local_network(target_frame.float(), total_context.float(), org_video[:, target_frame_index, :, :, :].float(), i, j)
                reward = lpips_loss
                # print("Decorrupted Image Range", decorrupted_image.min(), decorrupted_image.max())

                #lstm logic

                all_context_indices = torch.tensor([target_frame_index, context_1_index, context_2_index]).unsqueeze(0).to(local_device)

                lstm_patches = self.video_encoder.extract_patch(all_context_indices, encoded_frames)

                lstm_token = self.history_encoder(all_context_indices, lstm_patches)
                
                # print("LSTM Token Range", lstm_token.min(), lstm_token.max())

                reconstructed_video[:, target_frame_index, :, :, :] = decorrupted_image.squeeze()
                exp_reconstructed_video[:, target_frame_index, :, :, :] = exp_decorrupted_image.squeeze()


                encoded_frames = self.video_encoder.insert_encoded_frame_batch(pn_1_top_frame, target_frame, encoded_frames)   

                rewards.append(-(reward - curr_loss[target_frame_index]))
                total_reward += -(reward - curr_loss[target_frame_index])
                
                curr_loss[target_frame_index] = reward
            
            if i % 5 == 0:
                display = torch.cat(self.displays, dim=1)
                self.writer.add_image('Local_Net/Viz', display.detach(), i)
            
            #stack all of our lists
            obs_1 = (torch.stack([obs[0] for obs in obs_1]).squeeze(), torch.stack([obs[1] for obs in obs_1]).squeeze())
            obs_2 = (torch.stack([obs[0] for obs in obs_2]).squeeze(), torch.stack([obs[1] for obs in obs_2]).squeeze(), torch.stack([obs[2] for obs in obs_2]).squeeze().unsqueeze(-1))
            acs_1 = torch.stack(acs_1).squeeze()
            acs_2 = torch.stack(acs_2).squeeze()
            log_prob_1 = torch.stack(log_prob_1)
            log_prob_2 = torch.stack(log_prob_2)

            optical_flow, optical_flow_by_frame = self.calculate_optical_flow(reconstructed_video.squeeze(0).float())
            
            print("WORKING SHAPE", reconstructed_video.squeeze(0).float().shape)
            print("NOT WORKING SHAPE", exp_reconstructed_video.squeeze(0).float().shape)

            
            # Calculate optical flow for the experimental decorrupted image
            exp_optical_flow, exp_optical_flow_by_frame = self.calculate_optical_flow(exp_reconstructed_video.squeeze(0).float())
            

            # print("BIG BOY REWARD = ", abs(optical_flow - corrupted_optical_flow) - abs(org_optical_flow - optical_flow)) 
            #increase distance from corrupted optical flow and decrease distance from original optical flow
            # rewards[-1] = abs(optical_flow - corrupted_optical_flow) - abs(org_optical_flow - optical_flow)
            print(f"{optical_flow=}, {corrupted_optical_flow=}, {org_optical_flow=}")
            rtg = self.compute_rewards_to_go(rewards, lstm_patches.device)
            #### IF WE GET DATAPARALLEL DEVICE ERROR THIS IS THE CULPRIT
            
            #RESET LSTM STATES:
            self.history_encoder.reset_hidden_states()
            print("TOTAL REWARD", total_reward)

            return obs_1, obs_2, acs_1, acs_2, log_prob_1, log_prob_2, rtg, optical_flow_by_frame, exp_optical_flow_by_frame, org_optical_flow_by_frame, corrupted_optical_flow_by_frame

    #this function trains the local network and can be called independently for pretraining
    def train_local_network(self, images, conditioning, org_images, i, j):        
        y_hat = self.local_net(images, conditioning)
        context1, context2 = conditioning[:, 0], conditioning[:, 1]
        lpips_loss = self.lpips(y_hat, org_images, normalize=True)
        mse_loss = self.local_mse(rearrange(y_hat, 'b c h w -> b (c h w)'), rearrange(org_images, 'b c h w -> b (c h w)'))
        self.local_net_optimizer.zero_grad()
        # loss.backward()
        # self.local_net_optimizer.step()
        self.writer.add_scalar('Local_Net/mse_loss', mse_loss.detach(), self.time_steps * i + j)
        self.writer.add_scalar('Local_Net/lpips_loss', lpips_loss.detach(), self.time_steps * i + j)
        if i % 5 == 0:
            display = torch.cat([images, context1, context2, org_images, y_hat], dim=3).squeeze(0)
            self.displays.append(display.detach())
        return y_hat, lpips_loss.detach()

    #this function calculates rewards to go from marginal rewards
    def compute_rewards_to_go(self, rewards, device, gamma=1):
        b = len(rewards)  # Get the length of the sequence
        rewards_to_go_list = []  # Initialize a list to store rewards-to-go

        reward_to_go = 0  # Initialize the reward-to-go for the last step as 0
        for i in reversed(range(b)):  # Iterate through the sequence in reverse order
            reward_to_go = rewards[i] + gamma * reward_to_go  # Calculate the reward-to-go for the current step
            rewards_to_go_list.append(reward_to_go)  # Append the calculated reward-to-go to the list

        rewards_to_go_tensor = torch.tensor(rewards_to_go_list).to(device).view(-1, 1)  # Convert the list to a tensor and reshape it to b x 1
        return rewards_to_go_tensor

    #this function runs ppo on the selected network
    def ppo(self, net_num, info, i, device):                                                                
        if net_num == 1:
            actor = self.actor1
            critic = self.critic1
            actor_optim = self.actor1_optimizer
            critic_optim = self.critic1_optimizer

        elif net_num == 2:
            actor = self.actor2
            critic = self.critic2
            actor_optim = self.actor2_optimizer
            critic_optim = self.critic2_optimizer
        else:
            raise Exception("no bad")
        

        obs, acs, log_prob, rtgs = info
        # print("BEFORE CRITIC", torch.cuda.memory_allocated(device)/(1024 ** 3))
        V = critic(*obs, device)
        # print("AFTER CRITIC", torch.cuda.memory_allocated(device)/(1024 ** 3))

        A_k = rtgs - V.detach()                                                                  
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        print(f"{V.detach()[:3]=}")
        # print(f"{log_prob.shape=}")
        

        for j in range(self.num_updates_per_ppo): 
            # print(j, torch.cuda.memory_allocated(device)/(1024 ** 3))

            V = critic(*obs, device)
            curr_log_prob = actor.logprob(*obs, acs).unsqueeze(1)
            # print(f"{curr_log_prob.shape=}")
            ratio = torch.exp(curr_log_prob - log_prob)
            # print(f"{(ratio>0)[:3]=}")
            
            # print(f"{A_k.requires_grad=}, {rtgs.requires_grad=}, {obs.requires_grad=}")

            L1 = ratio * A_k
            L2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A_k
            # print(f"{L1=}, {L2=}")
            # print(f"{V[:3]=}")
            # print(f"{rtgs[:3]=}")
            actor_loss = -torch.min(L1, L2).mean()
            critic_loss = torch.nn.MSELoss()(V, rtgs.squeeze(1))
            
            critic_optim.zero_grad()
            critic_loss.backward(retain_graph = True)
            
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()


            critic_optim.step()

            self.writer.add_scalar(f'PPO/actor_{net_num}_loss', actor_loss.detach(), self.num_updates_per_ppo*i + j)
            self.writer.add_scalar(f'PPO/critic_{net_num}_loss', critic_loss.detach(), self.num_updates_per_ppo*i + j)


    def calculate_optical_flow(self, frames):
        model = raft_small(pretrained=True)
        model = model.eval().cuda() 

        def preprocess_image(image_tensor):
            image_tensor = Ft.resize(image_tensor, (256, 256))
            return image_tensor.cuda()

        b, _, _, _ = frames.shape

        # Preprocess frames
        frames_preprocessed = [preprocess_image(frames[i]) for i in range(b)]

        # Calculate optical flows
        flows = []
        scalar_flow_magnitudes = []  # New array to store scalar magnitudes of each optical flow
        for i in range(b - 1):
            with torch.no_grad():
                flow = model(frames_preprocessed[i].unsqueeze(0).float(), frames_preprocessed[i + 1].unsqueeze(0).float())
                flows.append(flow[-1])

                # Calculate scalar magnitude for this flow and add to our new array
                flow_magnitude = torch.sqrt(torch.sum(flow[-1]**2, dim=[1, 2, 3]))
                scalar_flow_magnitudes.append(flow_magnitude.item())

        # Calculate scalar magnitudes for the total flow
        total_magnitude = torch.sum(torch.stack([torch.tensor(f) for f in scalar_flow_magnitudes]))

        return total_magnitude.item(), scalar_flow_magnitudes  # Return the new array alongside the total magnitude
