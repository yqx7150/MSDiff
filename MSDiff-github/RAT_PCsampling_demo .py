'''

Test for RAT dataset


'''
import os
import sys
import sampling as sampling
from sampling import ReverseDiffusionPredictor,LangevinCorrector,AnnealedLangevinDynamics ,EulerMaruyamaPredictor,AncestralSamplingPredictor
import aapm_sin_ncsnpp_120 as configs_120
import aapm_sin_ncsnpp_720 as configs_720
sys.path.append('..')
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import numpy as np
from utils import restore_checkpoint
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from sde_lib import VESDE, VPSDE, subVPSDE
import os.path as osp
if len(sys.argv) > 1:
  start = int(sys.argv[1])
  end = int(sys.argv[2])

checkpoint_num = [[24,28]]# 20,720
# print(checkpoint_num)
# assert False
def get_predict(num):
  if num == 0:
    return None
  elif num == 1:
    return EulerMaruyamaPredictor
  elif num == 2:
    return ReverseDiffusionPredictor

def get_correct(num):
  if num == 0:
    return None
  elif num == 1:
    return LangevinCorrector
  elif num == 2:
    return AnnealedLangevinDynamics

predicts = [2]
corrects = [1]
for predict in predicts:
  for correct in corrects:
    for check_num in checkpoint_num:
      sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
      if sde.lower() == 'vesde':
        ckpt_filename_120 = './exp_120/checkpoints/checkpoint_{}.pth'.format(check_num[0])
        ckpt_filename_720 = './exp_720/checkpoints/checkpoint_{}.pth'.format(check_num[1])
        assert os.path.exists(ckpt_filename_120)
        assert os.path.exists(ckpt_filename_720)
        config_120 = configs_120.get_config()
        config_720 = configs_720.get_config()
        sde_120 = VESDE(sigma_min=config_120.model.sigma_min, sigma_max=config_120.model.sigma_max, N=config_120.model.num_scales)
        sde_720 = VESDE(sigma_min=config_720.model.sigma_min, sigma_max=config_720.model.sigma_max, N=config_720.model.num_scales)
        sampling_eps = 1e-5


      # 120 model
      batch_size = 1 #@param {"type":"integer"}
      config_120.training.batch_size = batch_size
      config_120.eval.batch_size = batch_size

      random_seed = 0 #@param {"type": "integer"}

      sigmas = mutils.get_sigmas(config_120)
      v120_model = mutils.create_model(config_120)

      optimizer = get_optimizer(config_120, v120_model.parameters())
      ema = ExponentialMovingAverage(v120_model.parameters(),
                                    decay=config_120.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                  model=v120_model, ema=ema)

      state = restore_checkpoint(ckpt_filename_120, state, config_120.device)
      ema.copy_to(v120_model.parameters())

      # 4A model
      batch_size = 1 #@param {"type":"integer"}
      config_720.training.batch_size = batch_size
      config_720.eval.batch_size = batch_size

      random_seed = 0 #@param {"type": "integer"}

      sigmas = mutils.get_sigmas(config_720)
      v720_model = mutils.create_model(config_720)

      optimizer = get_optimizer(config_720, v720_model.parameters())
      ema = ExponentialMovingAverage(v720_model.parameters(),
                                    decay=config_720.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                  model=v720_model, ema=ema)

      state = restore_checkpoint(ckpt_filename_720, state, config_720.device)
      ema.copy_to(v720_model.parameters())

      #@title PC sampling
      # img_size = config_hh.data.image_size
      # channels = config_hh.data.num_channels
      # shape = (batch_size, channels, img_size, img_size)
      # predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      # corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
      predictor = get_predict(predict) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      corrector = get_correct(correct) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}

      snr = 0.16#0.16 #@param {"type": "number"}
      #snr = 0.20#0.16 #@param {"type": "number"}
      n_steps = 1#@param {"type": "integer"}
      probability_flow = False #@param {"type": "boolean"}
      sampling_fn = sampling.get_pc_sampler(sde_120,sde_720, predictor, corrector,
                                            None, snr, n_steps=n_steps,
                                            probability_flow=probability_flow,
                                            continuous_120=config_120.training.continuous,
                                            continuous_A=config_720.training.continuous,
                                            eps=sampling_eps, 
                                            device_120=config_120.device,
                                            device_720=config_720.device)

      sampling_fn(v120_model,v720_model,check_num,predict,correct)

