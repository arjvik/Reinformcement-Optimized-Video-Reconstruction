import torch
import lpips
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small

class ROVR(nn.Module):
    def __init__(self, actor1, critic1, actor2, critic2, video_encoder, history_encoder, local_net, vid_length, time_steps, n_updates_per_ppo = 5):
        super(ROVR, self).__init__()
        
        print("INIT")
        
        self.actor1 = actor1
        self.critic1 = critic1
        self.actor2 = actor2
        self.critic2 = critic2
        self.local_net = local_net

        self.vid_length = vid_length
        self.time_steps = time_steps
        self.num_updates_per_ppo = n_updates_per_ppo

        self.lpips = lpips.LPIPS(net='alex')
        self.video_encoder = video_encoder
        self.history_encoder = history_encoder

        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-4)
        self.local_net_optimizer = torch.optim.Adam(self.local_net.parameters(), lr=1e-4)


        self.logger = {
                'actor1_losses': [],
                'critic1_losses': [],
                'actor2_losses': [],
                'critic2_losses': [],
                'local_net_losses': [],
        }

    #this should be the main function that is called to train
    def train(self, video, org_video):
        obs_1, obs_2, acs_1, ac_2, log_prob_1, log_prob_2, rtg = self.rollout(video, org_video)
        self.ppo(2, (obs_2, ac_2, log_prob_2, rtg))
        self.ppo(1, (obs_1, acs_1, log_prob_1, rtg))

    def forward(self, video):
        b, s, c, h, w = video.shape
        
        local_device = video.device

        
        lstm_token = torch.zeros(b, 3, 80, 80).to(local_device)

        encoded_frames = self.video_encoder(video)  

        
        
        for i in range(self.time_steps):
            
        
            pn_1_top_frame = self.actor1(encoded_frames, lstm_token).unsqueeze(0)
            
            target_frame_index = pn_1_top_frame.item()
        
            target_frame = video[:, target_frame_index, :, :, :]
                
            pn_2_top_frames, pn_2_log_prob = self.actor2(encoded_frames.float(), target_frame.float(), pn_1_top_frame)
                      
            #construct context package
            
            context_1_index = pn_2_top_frames[0, 0]
            context_2_index = pn_2_top_frames[0, 1]
              
            context_frame_1 = video[:, context_1_index, :, :, :]
            context_frame_2 = video[:, context_2_index, :, :, :]
              
            resized_context_1 = F.interpolate(context_frame_1, size = (128, 128), mode = "bilinear", align_corners = False)
            resized_context_2 = F.interpolate(context_frame_2, size = (128, 128), mode = "bilinear", align_corners = False)
        
            total_context = torch.cat((resized_context_1, resized_context_2), dim = 0).unsqueeze(0)

            #ship to local_net
                
            decorrupted_image = self.local_net(target_frame.float(), total_context.float())
            
            #lstm logic
            
            all_context_indices = torch.tensor([target_frame_index, context_1_index, context_2_index]).unsqueeze(0).to(local_device)
            
            lstm_patches = self.video_encoder.extract_patch(all_context_indices, encoded_frames)
                        
            lstm_token = self.history_encoder(all_context_indices, lstm_patches)
            
            video[:, target_frame_index, :, :, :] = decorrupted_image.squeeze()
            
            encoded_frames = self.video_encoder.insert_encoded_frame_batch(pn_1_top_frame, target_frame, encoded_frames)

        
        
    def rollout(self, video, org_video):
        curr_perceptual_loss = [self.lpips(y_star, y) for y_star, y in zip(video, org_video)]
        obs_1 = []
        obs_2 = []
        acs_1 = []
        ac_2 = []
        log_prob_1 = []
        log_prob_2 = []
        rewards = []
        org_optical_flow = self.calculate_optical_flow(org_video)
        corrupted_optical_flow = self.calculate_optical_flow(video)
        reconstructed_video = video
        for i in range(self.time_steps):
            #step through every model and get to image, conditioning to feed into local_net
            #append to obs_1, obs_2, acs_1, ac_2, log_prob_1, log_prob_2 using the two get_log_prob methods
            #TODO


            image = []
            conditioning = []
            #rewards are ~0-1 for perceptual loss
            y_hat, reward = self.train_local_network(image, conditioning, org_video["TARGET NUMBER"])
            reconstructed_video["TARGET NUMBER"] = y_hat
            rewards.append(-(reward - curr_perceptual_loss["TARGET NUMBER"]))
            curr_perceptual_loss[i] = reward
        
        optical_flow = self.calculate_optical_flow(reconstructed_video)
        #increase distance from corrupted optical flow and decrease distance from original optical flow
        rewards[-1] = abs(optical_flow - corrupted_optical_flow) - abs(org_optical_flow - optical_flow)
    
        rtg = self.compute_rewards_to_go(rewards)
        
        return obs_1, obs_2, acs_1, ac_2, log_prob_1, log_prob_2, rtg

    #this function trains the local network and can be called independently for pretraining
    def train_local_network(self, images, conditioning, org_images):
        y_hat = self.local_net(images, conditioning)
        loss = self.lpips(y_hat, org_images, normalize=True)
        self.local_net_optimizer.zero_grad()
        loss.backward()
        self.local_net_optimizer.step()
        self.logger['local_net_losses'].append(loss.detach())
        return y_hat, loss.detach()

    #this function calculates rewards to go from marginal rewards
    def compute_rewards_to_go(self, rewards, gamma=1):
        b, t = rewards.size()  # Get the batch size and sequence length
        rewards_to_go = torch.zeros_like(rewards)  # Initialize a tensor of rewards-to-go with the same shape as rewards

        for i in range(b):  # Iterate through each batch
            reward_to_go = 0  # Initialize the reward-to-go for the last step as 0
            for j in reversed(range(t)):  # Iterate through the rollout in reverse order (from the last step to the first step)
                reward_to_go = rewards[i, j] + gamma * reward_to_go  # Calculate the reward-to-go for the current step
                rewards_to_go[i, j] = reward_to_go  # Store the calculated reward-to-go in the rewards_to_go tensor

        return rewards_to_go

    #this function runs ppo on the selected network
    def ppo(self, net_num, info):                                                                
        if net_num == 1:
            actor = self.actor1
            critic = self.critic1
            actor_optim = self.actor1_optimizer
            critic_optim = self.critic1_optimizer
            actor_losses = self.logger['actor1_losses']
            critic_losses = self.logger['critic1_losses']

        elif net_num == 2:
            actor = self.actor2
            critic = self.critic2
            actor_optim = self.actor2_optimizer
            critic_optim = self.critic2_optimizer
            actor_losses = self.logger['actor2_losses']
            critic_losses = self.logger['critic2_losses']

        obs, acs, log_prob, rtgs = info                
        V = critic(*obs)
        A_k = rtgs - V.detach()                                                                  
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


        for _ in range(self.num_updates_per_ppo):                                                    
            V = critic(*obs)
            curr_log_prob = actor.get_log_prob(obs, acs)
            ratio = torch.exp(curr_log_prob - log_prob)

            L1 = ratio * A_k
            L2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = -torch.min(L1, L2).mean()
            critic_loss = torch.nn.MSELoss()(V, rtgs)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            actor_losses.append(actor_loss.detach())
            critic_losses.append(critic_loss.detach())

    #this function calculates the optical flow of a video
    def calculate_optical_flow(self, frames):
        model = raft_small(pretrained=True)
        model = model.eval().cuda() 

        def preprocess_image(image_tensor):
            image_tensor = F.resize(image_tensor, (256, 256))
            return image_tensor.cuda()
        
        b, _, _, _ = frames.shape

        # Preprocess frames
        frames_preprocessed = [preprocess_image(frames[i]) for i in range(b)]

        # Calculate optical flows
        flows = []
        for i in range(b - 1):
            with torch.no_grad():
                flow = model(frames_preprocessed[i], frames_preprocessed[i + 1])
                flows.append(flow[-1])

        # Calculate scalar magnitudes
        scalar_magnitudes = []
        for flow in flows:
            magnitude = torch.sqrt(torch.sum(flow**2, dim=[1, 2, 3]))  # Scalar magnitude
            scalar_magnitudes.append(magnitude)

        # Add scalar magnitudes together
        total_magnitude = torch.sum(torch.stack(scalar_magnitudes))

        return total_magnitude.item()
    
    
