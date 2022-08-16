from easydict import EasyDict

cfg = EasyDict()

cfg.busi = EasyDict()
cfg.busi.label = 'data/bs_train.csv' #your BUSI datasets label file
cfg.busi.path = '' #your BUSI datasets image path

cfg.db = EasyDict()
cfg.db.label = 'data/db_train.csv' #your datasetb datasets label file
cfg.db.path = '' #your datasetb datasets image path

cfg.features = 8  #selected features num
