import torch
import yaml 
import hydra
from collections import OrderedDict

def yaml_safe_load(path_to_model_config):
    with open(path_to_model_config, "r") as stream:
        try:
            model_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return model_cfg

class Perturbed_model():   
    def __init__(self, 
                 model_config_path = 'perturbed_feats_extractor/configs/mask3d_scannet200.yaml', 
                 checkpoint_path = 'checkpoints/scannet200/scannet200_val.ckpt', 
                 perturbation_magnitude = 0.001, 
                 perturbation_type = 'static',
                 use_perturbed = True,
                 use_non_perturbed = False):
        
        self.model_cfg = yaml_safe_load(model_config_path) 
        self.use_non_perturbed_model = use_non_perturbed
        self.use_perturbed_model = use_perturbed
        self.perturbation_magnitude = perturbation_magnitude
        self.perturbation_type = perturbation_type
        self.checkpoint_path = checkpoint_path
        self.dataset_name = checkpoint_path.split('/')[1]
            
        self.perturbed_model = self.load_model()
        self.non_perturbed_model = self.load_model()
        
        if perturbation_type == 'static':
            perturbation = perturbation_magnitude
            perturbed_model_dict = self.perturbed_model.state_dict()
            for k,v in perturbed_model_dict.items():
                perturbed_model_dict[k] = v+perturbation
            self.perturbed_model.load_state_dict(OrderedDict(zip(self.perturbed_model.state_dict().keys(),perturbed_model_dict.values())))
            
    
    def __call__(self, x, target, raw_coord):
        self.device = target[0]['labels'].device
        
        if self.perturbation_type != 'static':
            self.update_perturbed_model()
        self.forward_pass(x, target, raw_coord)
        return {'perturbed_queries':self.perturbed_output["query_features"],
                'non_perturbed_queries':self.non_perturbed_output["query_features"],}
    
    def forward_pass(self, x, target, raw_coord):
        with torch.no_grad():
            
            if self.use_perturbed_model:
                self.perturbed_output = self.perturbed_model.to(self.device)(x,
                                        point2segment=[target[i]['point2segment'] for i in range(len(target))],
                                        raw_coordinates=raw_coord, is_eval=True)
            else:
                self.perturbed_output = {"query_features": 0}
                
            if self.use_non_perturbed_model:
                self.non_perturbed_output = self.non_perturbed_model.to(self.device)(x,
                                        point2segment=[target[i]['point2segment'] for i in range(len(target))],
                                        raw_coordinates=raw_coord, is_eval=True)
            else:
                self.non_perturbed_output = {"query_features": 0}
                
    
    def update_perturbed_model(self):
        if self.perturbation_type == 'random':
            perturbation = (torch.rand(1)*self.perturbation_magnitude).item()
        elif self.perturbation_type == 'signed_random':
            perturbation = ((2*torch.rand(1)-1)*self.perturbation_magnitude).item()
        elif self.perturbation_type == 'sign_flip':
            perturbation = (2*torch.bernoulli(torch.empty(1, 1).uniform_(0, 1))-1).item()*self.perturbation_magnitude
        elif self.perturbation_type == 'normal':
            perturbation = torch.normal(mean=0.0, std = 1/3, size=(1,1)).item()
        
        perturbed_model_dict = self.perturbed_model.state_dict()
        non_perturbed_model_dict = self.non_perturbed_model.state_dict()
        for k,v in non_perturbed_model_dict.items():
            # print(perturbation)
            perturbed_model_dict[k] = v+perturbation
        self.perturbed_model.load_state_dict(OrderedDict(zip(self.perturbed_model.state_dict().keys(),perturbed_model_dict.values())))
    
    def load_model(self):
        model_ = hydra.utils.instantiate(self.model_cfg)
        model_dict = torch.load(self.checkpoint_path)['state_dict']
        if self.dataset_name == 'scannet200':
            model_.load_state_dict(OrderedDict(zip(model_.state_dict().keys(),model_dict.values())))
        elif self.dataset_name == 'scannet':
            model_.load_state_dict(OrderedDict(zip(list(model_.state_dict().keys()),list(model_dict.values())[2:])))
        
        return model_