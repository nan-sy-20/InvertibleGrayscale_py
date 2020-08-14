import importlib

from .base_model import BaseModel

def find_model_using_name(model_name):
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls
    if model is None:
        raise NotImplementedError(
            '\u2755 In %s.py there should be a subclass of BaseModel with class name that matches %s in lowercase.' \
            % (model_filename, target_model_name))
    return model

def create_model(opt, verbose=True):
    net = find_model_using_name(opt.model)
    instance = net(opt)
    if verbose:
        print('\u270f Model [%s] was created.' % type(instance).__name__)
    return instance

def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options
