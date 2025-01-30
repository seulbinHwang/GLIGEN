import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

class DDIMSampler(object):

    def __init__(self,
                 diffusion,
                 model,
                 schedule="linear",
                 alpha_generator_func=None,
                 set_alpha_scale=None):
        super().__init__()
        """
 # 'ldm.models.diffusion.ldm.LatentDiffusion'
    LatentDiffusion(DDPM)
        """
        self.diffusion = diffusion # 'ldm.models.diffusion.ldm.LatentDiffusion'

        self.model = model # 'ldm.modules.diffusionmodules.openaimodel.UNetModel'
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps # 1000
        self.schedule = schedule # "linear"
        """ alpha_generator(length, type)
    설명
        Gated Self attention 에서 사용하는데, v = v + alpha * (x 와 grounding condition token 의 attention 결과)   
        에서의 alpha 값을 의미합니다.    
        
        if length = 1000 step이면, 
            0 step ~ 1000 * 0.3 = 300 step까지는 alpha=1
            300 step ~ 300 step 구간에서는 linear decay from 1 to 0
            300 step ~ 1000 step까지는 alpha=0
        """
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self,
                      ddim_num_steps,
                      ddim_discretize="uniform",
                      ddim_eta=0.):
        """ self.ddim_timesteps
 [  1   5   9  13  , ... , 985 989 993 997]
 len(self.ddim_timesteps) = 250
        alphas_cumprod: shape (ddpm_num_timesteps) (1000,)

        ddim_sigmas, ddim_alphas, ddim_alphas_prev : shape (ddim_num_steps) (250,)
        """
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=False)
        # diffusion : 'ldm.models.diffusion.ldm.LatentDiffusion',
        # shape (ddpm_num_timesteps) (1000,)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[
            0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device
                                                                    )
        """
`register_buffer` 함수는 PyTorch의 `nn.Module` 클래스에서 사용되는 메서드로, 모듈에 버퍼를 추가하는 역할을 합니다. 
버퍼는 모델의 파라미터는 아니지만, 모델의 상태를 저장하는 데 사용됩니다. 
버퍼는 기본적으로 지속적(persistent)이며, 모델의 파라미터와 함께 저장됩니다. 
지속적이지 않은 버퍼는 `state_dict`에 포함되지 않습니다. 

이 함수의 주요 역할은 다음과 같습니다:
- 버퍼를 모듈의 속성으로 등록
- 버퍼의 지속성 여부 설정
- 버퍼를 `state_dict`에 포함할지 여부 결정
        """
        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        """
        ddim_sigmas, ddim_alphas, ddim_alphas_prev : shape (ddim_num_steps) (250,)
        ddim_sigmas 는 전부 0
        ddim_alphas 은 alphas_cumprod 에서 건너뛰면서 값을 가져옴
        """
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), # shape (ddpm_num_timesteps) (1000,)
            ddim_timesteps=self.ddim_timesteps, # shape (ddim_num_steps) (250,)
            eta=ddim_eta, # 0. -> DDIM deterministic sampling을 하겠다는 의미
            verbose=False)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               shape,
               input,
               uc=None,
               guidance_scale=1,
               mask=None,
               x0=None):
        """
    S = 250
    shape: (config.batch_size, model.in_channels, model.image_size,
         model.image_size) -> (1, 4, 64, 64)
    input = dict(
        x=starting_noise,
            None
        timesteps=None,
        context=context,
            (batch_size, 77, 768)
        grounding_input=grounding_input, (DICT)
            boxes: (batch_size, max_objs, 4)
            masks: (batch_size, max_objs)
            positive_embeddings: (batch_size, max_objs, 768)
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,)
    uc: (batch_size, 77, 768)
        - args.negative_prompt 로 부터 생성
'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    guidance_scale: config.guidance_scale = 7.5
    mask: inpainting_mask = None
    x0: z0 = None
        """
        """ self.ddim_timesteps
[  1   5   9  13  , ... , 985 989 993 997]
len(self.ddim_timesteps) = 250
        alphas_cumprod: shape (ddpm_num_timesteps) (1000,)

        ddim_sigmas, ddim_alphas, ddim_alphas_prev : shape (ddim_num_steps) (250,)
            ddim_sigmas 는 전부 0
            ddim_alphas 은 alphas_cumprod 에서 건너뛰면서 값을 가져옴
        """
        self.make_schedule(ddim_num_steps=S)
        return self.ddim_sampling(shape,
                                  input,
                                  uc,
                                  guidance_scale,
                                  mask=mask,
                                  x0=x0)

    @torch.no_grad()
    def ddim_sampling(self,
                      shape,
                      input,
                      uc,
                      guidance_scale=1,
                      mask=None,
                      x0=None):
        """
    shape: (config.batch_size, model.in_channels, model.image_size,
         model.image_size) -> (1, 4, 64, 64)
    input = dict(
        x=starting_noise,
            None
        timesteps=None,
        context=context,
            (batch_size, 77, 768)
        grounding_input=grounding_input, (DICT)
            boxes: (batch_size, max_objs, 4)
            masks: (batch_size, max_objs)
            positive_embeddings: (batch_size, max_objs, 768)
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,)
    uc: (batch_size, 77, 768)
        - args.negative_prompt 로 부터 생성
        - 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    guidance_scale: config.guidance_scale = 7.5
    mask: inpainting_mask = None
    x0: z0 = None
        """
        b = shape[0]

        img = input["x"]
        if img == None:
            # 각 요소는 평균이 0이고 표준편차가 1인 정규분포에서 무작위로 선택된 값
            # (batch_size, in_channels, image_size, image_size)
            img = torch.randn(shape, device=self.device)
            input["x"] = img
        """
self.ddim_timesteps
    [  1   5   9  13  , ... , 985 989 993 997]
     total_steps = len(self.ddim_timesteps) = 250
time_range
    [ 997 993 989 985  , ... , 13 9 5 1]
        """
        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]

        #iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        iterator = time_range

        if self.alpha_generator_func != None:
            """ alpha_generator(length, type)
            # grounding 정보를 얼마나 반영할 것인지에 대한 가중치
            denoising 초반에는 1로 설정하고, 그 후에는 0으로 설정
            
            if length = 250 step이면, 
            0 step ~ 250 * 0.3 step까지는 alpha=1
            250 * 0.3 step ~ 250 * 0.3 step 구간에서는 linear decay from 1 to 0
            250 * 0.3 step ~ 250 step까지는 alpha=0
            """
            # len(alphas) = 250
            alphas = self.alpha_generator_func(len(iterator))
        # i: 0, 1, ... 249
        # index: 249, 248, ... 0
        # step: 997, 993, ... 1
        for i, step in enumerate(iterator):
            # set alpha
            if self.alpha_generator_func != None:
                # self.model: ldm.modules.diffusionmodules.openaimodel.UNetModel
                # UNetModel의 gated self attention 에서 사용하는 alpha 값을 설정
                self.set_alpha_scale(self.model, alphas[i])
                if alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()

            # run
            index = total_steps - i - 1 # 249, 248, ... 0
            input["timesteps"] = torch.full((b,),
                                            step, # 997, 993, ... 1
                                            device=self.device,
                                            dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                """
                x0
                    # 각 요소는 평균이 0이고 표준편차가 1인 정규분포에서 무작위로 선택된 값
                    # (batch_size, in_channels, image_size, image_size)
                input["timesteps}]
                    # (batch_size)
                """
                #  'ldm.models.diffusion.ldm.LatentDiffusion'
                img_orig = self.diffusion.q_sample(x0, input["timesteps"])
                img = img_orig * mask + (1. - mask) * img
                input["x"] = img

            img, pred_x0 = self.p_sample_ddim(input,
                                              index=index, # index: 249, 248, ... 0
                                              uc=uc,
                                              guidance_scale=guidance_scale)
            input["x"] = img

        return img

    @torch.no_grad()
    def p_sample_ddim(self, input, index, uc=None, guidance_scale=1):
        """
    input = dict(
        x=starting_noise,
            # 각 요소는 평균이 0이고 표준편차가 1인 정규분포에서 무작위로 선택된 값
            # (batch_size, in_channels, image_size, image_size)
        timesteps= (batch_size) -> step 값으로 전부 채워짐
        context=context,
            (batch_size, 77, 768)
        grounding_input=grounding_input, (DICT)
            boxes: (batch_size, max_objs, 4)
            masks: (batch_size, max_objs)
            positive_embeddings: (batch_size, max_objs, 768)
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,)
    index: 249, 248, ... 0
    uc: (batch_size, 77, 768)
        - args.negative_prompt 로 부터 생성
        - 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    guidance_scale: config.guidance_scale = 7.5

        """
        # self.model: ldm.modules.diffusionmodules.openaimodel.UNetModel
        # e_t : (batch_size, c, image_size, image_size)
        e_t = self.model(input)
        if uc is not None and guidance_scale != 1:
            unconditional_input = dict(
                x=input["x"],
                timesteps=input["timesteps"],
                context=uc,
                inpainting_extra_input=input["inpainting_extra_input"],
                grounding_extra_input=input['grounding_extra_input'])
            e_t_uncond = self.model(unconditional_input)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        # select parameters corresponding to the currently considered timestep
        b = input["x"].shape[0]
        a_t = torch.full((b, 1, 1, 1),
                         self.ddim_alphas[index],
                         device=self.device)
        a_prev = torch.full((b, 1, 1, 1),
                            self.ddim_alphas_prev[index],
                            device=self.device)
        sigma_t = torch.full((b, 1, 1, 1),
                             self.ddim_sigmas[index],
                             device=self.device)
        # sqrt_one_minus_at = " root (1 - alpha) "
        sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                       self.ddim_sqrt_one_minus_alphas[index],
                                       device=self.device)

        # current prediction for x_0
        pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(input["x"])
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
