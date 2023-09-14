"""
Learning from scratch using RL algorithm

Dual clip PPO

vmaf quality metric

"""
import os, logging
import torch
import torch.optim as optim
from torch.autograd import Variable
from .AC_net_vmaf import Actor, Critic
from .beta_vae_vmaf import BetaVAE
from .helper import save_models_ppo, compute_adv, load_ema
from .test_vmaf import valid

LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4
MAX_GRAD_NORM = 5.0
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6

torch.cuda.set_device(1)
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class PPO_agent:
    def __init__(self, args, br_dim, s_info, s_len, c_len, max_qoe, vae_c=2) -> None:
        # states and observation configures
        self.c_len = c_len
        self.s_info = s_info
        self.s_len = s_len
        self.steps_in_episode = args.ro_len
        self.default_quality = int(1)
        self.explo_bit_rate = self.default_quality
        self.done = True
        self.state_ = 0
        self.ob_ = 0
        self.action = 0
        self.value = 0

        # other configures
        self.args = args
        self.latent_dim = args.latent_dim
        self.br_dim = br_dim
        self.vae_c = vae_c

        # training parameters
        self.minibatch_size = args.batch_size  # BATCH_SIZE
        self.ppo_ups = args.ppo_ups
        self.clip = args.clip
        self.lc_alpha = args.lc_alpha
        self.lc_beta = args.lc_beta
        self.lc_gamma = args.lc_gamma
        self.lc_mu = args.lc_mu
        self.sample_num = int(args.sp_n)
        self.max_qoe = max_qoe

        # Define the vae model
        kld_beta, kld_lambda, recon_gamma = (
            self.args.kld_beta,
            self.args.kld_lambda,
            self.args.vae_gamma,
        )
        self.vae_net = BetaVAE(
            in_channels=self.vae_c,
            hist_dim=self.c_len,
            latent_dim=self.latent_dim,
            beta=kld_beta,
            delta=kld_lambda,
            gamma=recon_gamma,
        ).type(dtype)
        # vae_net.eval()
        self.vae_net_ori = BetaVAE(
            in_channels=self.vae_c,
            hist_dim=self.c_len,
            latent_dim=self.latent_dim,
            beta=kld_beta,
            delta=kld_lambda,
            gamma=recon_gamma,
        ).type(dtype)

        ## meta reinforcement learning with a latent-conditioned policy
        # establish the actor and critic model
        self.model_actor = Actor(
            self.br_dim, self.latent_dim, self.s_info, self.s_len
        ).type(dtype)
        self.model_actor_ori = Actor(
            self.br_dim, self.latent_dim, self.s_info, self.s_len
        ).type(dtype)
        self.model_actor_old = Actor(
            self.br_dim, self.latent_dim, self.s_info, self.s_len
        ).type(dtype)
        self.model_critic = Critic(
            self.br_dim, self.latent_dim, self.s_info, self.s_len
        ).type(dtype)
        self.model_critic_old = Critic(
            self.br_dim, self.latent_dim, self.s_info, self.s_len
        ).type(dtype)

        # exponential moving average for model parameters
        self.ema_actor = load_ema(self.model_actor, decay=args.ema)
        self.ema_vae = load_ema(self.vae_net, decay=args.ema)

        self.optimizer_vae = optim.Adam(
            self.vae_net.parameters(), lr=LEARNING_RATE_ACTOR
        )
        self.optimizer_actor = optim.Adam(
            self.model_actor.parameters(), lr=LEARNING_RATE_ACTOR
        )
        self.optimizer_critic = optim.Adam(
            self.model_critic.parameters(), lr=LEARNING_RATE_CRITIC
        )

        self.approx_critic = (
            args.vap or not args.from_il
        )  ## approximating the critic value by the current policy

    def finish_approx(self, summary_dir, add_str):
        "stopping approxiate the value function and save current model"
        self.approx_critic = False

        # save models
        # critic_save_path = summary_dir + '/' + add_str + '/' + 'models' + \
        #                 "/%s_%s_%s.model" %(str('critic'), add_str, str('base'))
        # if os.path.exists(critic_save_path): os.system('rm ' + critic_save_path)
        # torch.save(self.model_critic.state_dict(), critic_save_path)

    def initial(self, model_vae_para, model_actor_para, model_critic_para):
        if model_vae_para is not None:
            self.vae_net.load_state_dict(model_vae_para)
            self.vae_net_ori.load_state_dict(model_vae_para)

        ## load the pretrain model, initialize with sub-optimal models
        if model_actor_para is not None:
            self.model_actor.load_state_dict(model_actor_para)
            self.model_actor_ori.load_state_dict(model_actor_para)

        if model_critic_para is not None and not self.approx_critic:
            self.model_critic.load_state_dict(model_critic_para)
            print("critic model has been loaded!")

    def model_eval(self):
        # turn on the eval model
        self.model_actor.eval()
        self.model_critic.eval()
        self.vae_net.eval()

    def get_models(self):
        return self.model_actor, self.vae_net

    def explore(self, ob_, state_, action_mask_):
        with torch.no_grad():
            latent = self.vae_net.get_latent(ob_)
            prob = self.model_actor.forward(state_, latent)
            value = self.model_critic(state_, latent)
            # prob_ = prob * action_mask_ # It is easy to induce an error:
            # ##        RuntimeError: CUDA error: device-side assert triggered
            prob_ = prob  # + 1e-5 ## with replacement=False, not enough non-negative category to sample if there is an output being zero
            # prob_ = prob_ * action_mask_
            action = prob_.multinomial(num_samples=1)
            # action = prob.multinomial(num_samples=1)
        return value, action

    def collect_steps(self, args, memory, env):
        while memory.return_size() < args.explo_num * args.ro_len:
            # for _ in range(1):
            states = []
            obs = []
            actions = []
            rewards = []
            values = []

            for _ in range(self.steps_in_episode):
                # record the current state, observation and action
                if not self.done:
                    states.append(self.state_)
                    obs.append(self.ob_)
                    actions.append(self.action)
                    values.append(self.value)

                bit_rate = self.explo_bit_rate

                self.ob_, self.state_, reward_norm, self.done, action_mask_ = env.step(
                    bit_rate
                )
                rewards.append(reward_norm)

                self.value, self.action = self.explore(
                    self.ob_, self.state_, action_mask_
                )
                self.explo_bit_rate = int(self.action.squeeze().cpu().numpy())
                if self.done:
                    self.explo_bit_rate = self.default_quality
                    break

            # compute returns and GAE(lambda) advantages:
            if len(states) != len(rewards):
                if len(states) + 1 == len(rewards):
                    rewards = rewards[1:]
                else:
                    print("error in length of states!")
                    break
            advantages, returns = compute_adv(
                args, self.done, self.value, values, rewards
            )

            ## store usefull info:
            memory.push([states, obs, actions, returns, advantages])

    def train(self, memory):
        # policy grad updates:
        self.model_actor_old.load_state_dict(self.model_actor.state_dict())
        self.model_critic_old.load_state_dict(self.model_critic.state_dict())

        ## model update
        self.model_actor.train()
        self.vae_net.train()
        self.model_critic.train()
        self.model_actor_old.eval()
        self.model_critic_old.eval()

        vae_kld_loss_ = []
        policy_loss_ = []
        value_loss_ = []
        entropy_loss_ = []
        policy_mi_loss_ = []

        # ------------------- Meta DRL training ---------------------------
        for _ in range(self.ppo_ups):
            # new mini_batch
            # priority_batch_size = int(memory.get_capacity()/10)
            (
                batch_states,
                batch_obs,
                batch_actions,
                batch_returns,
                batch_advantages,
            ) = memory.sample_cuda(self.minibatch_size)

            # assert batch_states[:, 0:1, :].cpu().numpy().all() == batch_obs[:, -2:, 0:1].transpose(1, 2).cpu().numpy().all()
            # assert batch_states[:, -1:, :].cpu().numpy().all() == batch_obs[:, -2:, 1:2].transpose(1, 2).cpu().numpy().all()
            # ------------------ VAE case -----------------------
            batch_latents = self.vae_net.get_latent(batch_obs)
            batch_latents_ori = self.vae_net_ori.get_latent(batch_obs)

            x_train = batch_obs  # (N, C_LEN, in_channels)

            # fit the model
            z_mu, z_log_var = self.vae_net.forward(x_train)
            kld_loss = self.vae_net.loss_function(z_mu, z_log_var)
            kld_loss_ = kld_loss.detach().cpu().numpy()
            vae_kld_loss_.append(kld_loss_)

            # ------------------- Latent case ------------------------
            probs_ori = self.model_actor_ori(batch_states, batch_latents_ori).detach()

            # old_prob
            probs_old = self.model_actor_old(batch_states, batch_latents).detach()
            v_pre_old = self.model_critic_old(batch_states, batch_latents).detach()
            prob_value_old = torch.gather(
                probs_old, dim=1, index=batch_actions.type(dlongtype)
            ).detach()

            # new prob
            probs = self.model_actor(batch_states, batch_latents)
            v_pre = self.model_critic(batch_states, batch_latents.detach())
            prob_value = torch.gather(probs, dim=1, index=batch_actions.type(dlongtype))

            # ratio
            ratio = prob_value / (1e-6 + prob_value_old)

            # clip loss
            surr1 = ratio * batch_advantages.type(
                dtype
            )  # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * batch_advantages.type(
                dtype
            )
            loss_clip_ = torch.min(surr1, surr2)
            loss_clip_dual = torch.where(
                torch.lt(batch_advantages.type(dtype), 0.0),
                torch.max(loss_clip_, 3 * batch_advantages.type(dtype)),
                loss_clip_,
            )
            loss_clip_actor = -torch.mean(loss_clip_dual)
            # loss_clip_actor = -torch.mean(loss_clip_)

            # value loss
            vfloss1 = (v_pre - batch_returns.type(dtype)) ** 2

            # if not self.approx_critic:
            #     v_pred_clipped = v_pre_old + (v_pre - v_pre_old).clamp(-self.clip, self.clip)
            #     vfloss2 = (v_pred_clipped - batch_returns.type(dtype)) ** 2
            #     loss_value = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
            # else:
            loss_value = 0.5 * torch.mean(vfloss1)

            # entropy loss
            ent_latent = -torch.mean(probs * torch.log(probs + 1e-6))

            # mutual information loss
            # H(a|s)
            latent_samples = []
            for _ in range(self.sample_num):
                latent_samples.append(
                    torch.randn(self.minibatch_size, self.latent_dim).type(dtype)
                )  # .detach()
            probs_samples = torch.zeros(self.minibatch_size, self.br_dim, 1).type(dtype)
            for idx in range(self.sample_num):
                probs_ = self.model_actor(batch_states, latent_samples[idx])
                probs_ = probs_.unsqueeze(2)
                probs_samples = torch.cat((probs_samples, probs_), 2)
            probs_samples = probs_samples[:, :, 1:]
            probs_sa = torch.mean(
                probs_samples, dim=2
            )  # p(a|s) = 1/L * \sum p(a|s, z_i) p(z_i|s)
            probs_sa = Variable(probs_sa)
            ent_noLatent = -torch.mean(probs_sa * torch.log(probs_sa + 1e-6))

            # # H(a|s, c)
            # latent_samples = []
            # for _ in range(self.sample_num):
            #     batch_latents = self.vae_net.get_latent(batch_obs)
            #     latent_samples.append(batch_latents) #.detach()
            # probs_samples = torch.zeros(self.minibatch_size, self.br_dim, 1).type(dtype)
            # for idx in range(self.sample_num):
            #     probs_ = self.model_actor(batch_states, latent_samples[idx])
            #     probs_ = probs_.unsqueeze(2)
            #     probs_samples = torch.cat((probs_samples, probs_), 2)
            # probs_samples = probs_samples[:, :, 1:]
            # probs_sa = torch.mean(probs_samples, dim=2) # p(a|s) = 1/L * \sum p(a|s, z_i) p(z_i|s)
            # probs_sa = Variable(probs_sa)
            # ent_SLatent = - torch.mean(probs_sa * torch.log(probs_sa + 1e-6))

            # mutual_info = ent_noLatent - ent_SLatent

            mutual_info = ent_noLatent - ent_latent

            # cross entropy loss
            ce_latent = -torch.mean(probs_ori * torch.log(probs + 1e-6))

            # total loss
            loss_actor = (
                self.lc_alpha * loss_clip_actor
                - self.lc_gamma * mutual_info
                - self.lc_beta * ent_latent
            )
            # loss_actor = self.lc_alpha * loss_clip_actor - self.lc_gamma * mutual_info \
            #                 - self.lc_mu * ce_latent
            #     - lc_gamma * mutual_info
            # loss_actor = self.lc_alpha * loss_clip_actor - self.lc_mu * ce_latent -\
            # self.lc_gamma * mutual_info
            # loss_critic = loss_value
            # loss_critic = loss_value + kld_loss

            # record
            policy_loss_.append(loss_clip_actor.detach().cpu().numpy())
            entropy_loss_.append(ent_latent.detach().cpu().numpy())
            value_loss_.append(loss_value.detach().cpu().numpy())
            policy_mi_loss_.append(mutual_info.detach().cpu().numpy())

            # update parameters via backpropagation
            if not self.approx_critic:
                loss_total = loss_actor + kld_loss
                loss_critic = loss_value
                self.optimizer_actor.zero_grad()
                self.optimizer_vae.zero_grad()
                self.optimizer_critic.zero_grad()
                # loss_actor.backward(retain_graph=False)
                loss_critic.backward()
                loss_total.backward()
                # clip_grad_norm_(self.vae_net.parameters(), \
                #                     max_norm = MAX_GRAD_NORM, norm_type = 2)
                # clip_grad_norm_(self.model_actor.parameters(), \
                #                     max_norm = MAX_GRAD_NORM, norm_type = 2)
                # clip_grad_norm_(self.model_critic.parameters(), \
                #                     max_norm = MAX_GRAD_NORM, norm_type = 2)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                self.optimizer_vae.step()

                # -------- EMA update --------
                self.ema_actor.update(self.model_actor.parameters())
                self.ema_vae.update(self.vae_net.parameters())
            else:
                loss_critic = loss_value
                self.optimizer_critic.zero_grad()
                # loss_actor.backward(retain_graph=False)
                loss_critic.backward()
                # clip_grad_norm_(self.model_critic.parameters(), \
                #                     max_norm = MAX_GRAD_NORM, norm_type = 2)
                self.optimizer_critic.step()

        return vae_kld_loss_, policy_loss_, value_loss_, entropy_loss_, policy_mi_loss_

    def valid(self, args, env, epoch, test_log_file, save_folder, add_str):
        self.ema_actor.store(self.model_actor.parameters())
        self.ema_actor.copy_to(self.model_actor.parameters())
        self.ema_vae.store(self.vae_net.parameters())
        self.ema_vae.copy_to(
            self.vae_net.parameters()
        )  # copy the ema_adj value to model_adj
        mean_value = valid(
            args,
            env,
            self.model_actor,
            self.vae_net,
            epoch,
            test_log_file,
            save_folder,
            add_str,
        )
        save_models_ppo(
            logging,
            save_folder,
            add_str,
            self.model_actor,
            self.model_critic,
            self.vae_net,
            epoch,
            self.max_qoe,
            mean_value,
        )

        # -------- ema restore --------
        self.ema_actor.restore(self.model_actor.parameters())
        self.ema_vae.restore(self.vae_net.parameters())
        return mean_value

    def annealing(self):
        self.lc_beta = self.args.anneal_p * self.lc_beta
